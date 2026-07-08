# =============================================================================
# hc_v6_lastday.py — 締切前に積める残り全部（時間対効果順）
#
#   posrow    : POS_CASH の行レベルモデル（未実装だった最後の row-level）
#               → キャッシュに特徴量追記。所要 ~10分
#   knnv      : KNN ターゲット特徴量のバリエーション (k=100 / k=1000 /
#               別特徴量空間) → キャッシュに追記。所要 ~15分
#   catnative : CatBoost をネイティブカテゴリ表現で学習（one-hot と別表現の
#               多様性源）→ oof_store に "cat_native" として追加。所要 ~1-2h
#   f10       : 主力 LGBM を 10-fold + lr0.005 で学習 → "lgb_gbdt_f10"。
#               所要 ~2-3h（夜に回す枠）
#   blend     : 複数 submission CSV の rank 平均（最後の保険。所要 ~1分）
#
# 推奨実行順（残り時間に応じて上から切る）:
#   python hc_v6_lastday.py posrow
#   python hc_v6_lastday.py knnv
#   rm oof_store/lgb_gbdt_full_* && python hc_campaign.py train lgb_gbdt_full
#   python hc_v6_lastday.py catnative
#   python hc_v6_lastday.py f10
#   python hc_v5_final_push.py ensemble2
#   python hc_v6_lastday.py blend submissions/a.csv submissions/b.csv ...
# =============================================================================

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb

from hc_ensemble_optuna import DATA_DIR, timer, USE_GPU

# 参考: hc_ensemble_optuna.USE_GPUはXGBoostのVRAMハング教訓でFalse固定だが、
# ここ(f10)は特徴量~217個と当時(4788個+DAE3072次元)よりずっと小さいのでGPUの
# 危険度は低い。試す場合はここをTrueにする(nvidia-smiで様子見しながら)。
USE_GPU_F10 = USE_GPU
# catnativeは高カーディナリティな生カテゴリ(ORGANIZATION_TYPE等)のCTR計算で
# GPUのVRAM消費がLightGBMと別の形で増えるため、より慎重に。既定はCPU。
USE_GPU_CATNATIVE = False

warnings.simplefilter(action="ignore", category=FutureWarning)

ROOT = Path(".")
CACHE = ROOT / "cache"
STORE = ROOT / "oof_store"
SUBS = ROOT / "submissions"
N_FOLDS = 5
SEED = 42


def load_features():
    train = pd.read_parquet(CACHE / "features_train.parquet")
    test = pd.read_parquet(CACHE / "features_test.parquet")
    feats = json.loads((CACHE / "feats.json").read_text())
    return train, test, feats


def save_features(train, test, feats):
    # 安全策: hc_campaign.pyのstage_features()と同じくinf/-infをNaNに変換してから保存。
    # ここで追加する特徴量の割り算は基本epsilonガード済みだが、将来の追記漏れ対策として
    # XGBoost("Input data contains inf")が再発しないよう保険をかけておく。
    for df in (train, test):
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    train.to_parquet(CACHE / "features_train.parquet")
    test.to_parquet(CACHE / "features_test.parquet")
    (CACHE / "feats.json").write_text(json.dumps(feats))


def merge_block(block, new_cols):
    train, test, feats = load_features()
    train = train.merge(block, on="SK_ID_CURR", how="left")
    test = test.merge(block, on="SK_ID_CURR", how="left")
    for df in (train, test):
        for c in new_cols:
            if c in df.columns:
                df[c] = df[c].astype("float32")
    feats = list(dict.fromkeys(feats + new_cols))
    save_features(train, test, feats)
    print(f"cache augmented: +{len(new_cols)} (total {len(feats)})")


# =============================================================================
# posrow: POS_CASH 行レベルモデル
# =============================================================================
def stage_posrow():
    df = pd.read_csv(f"{DATA_DIR}/POS_CASH_balance.csv",
                     usecols=["SK_ID_CURR", "MONTHS_BALANCE",
                              "CNT_INSTALMENT", "CNT_INSTALMENT_FUTURE",
                              "SK_DPD", "SK_DPD_DEF"],
                     dtype={"SK_ID_CURR": "int32",
                            "MONTHS_BALANCE": "int16"})
    # 返済の進捗率（残回数 / 総回数）: 完済間近か序盤かで意味が違う
    df["PROGRESS"] = 1 - df["CNT_INSTALMENT_FUTURE"] / (df["CNT_INSTALMENT"] + 1)
    df = df.sort_values(["SK_ID_CURR", "MONTHS_BALANCE"])
    df = df.groupby("SK_ID_CURR").tail(24).reset_index(drop=True)

    feats = ["MONTHS_BALANCE", "CNT_INSTALMENT", "CNT_INSTALMENT_FUTURE",
             "SK_DPD", "SK_DPD_DEF", "PROGRESS"]

    app = pd.read_csv(f"{DATA_DIR}/application_train.csv",
                      usecols=["SK_ID_CURR", "TARGET"])
    df = df.merge(app, on="SK_ID_CURR", how="left")
    tr_rows = df[df["TARGET"].notnull()]
    y_rows = tr_rows["TARGET"].values
    groups = tr_rows["SK_ID_CURR"].values

    params = {"objective": "binary", "metric": "auc", "verbosity": -1,
              "n_estimators": 1500, "learning_rate": 0.05, "num_leaves": 31,
              "colsample_bytree": 0.8, "subsample": 0.8, "subsample_freq": 1,
              "random_state": SEED, "n_jobs": -1}

    score = np.zeros(len(df))
    gkf = GroupKFold(n_splits=N_FOLDS)
    with timer("row-level POS model"):
        for tr_idx, va_idx in gkf.split(tr_rows[feats], y_rows, groups):
            m = lgb.LGBMClassifier(**params)
            m.fit(tr_rows[feats].iloc[tr_idx], y_rows[tr_idx],
                  eval_set=[(tr_rows[feats].iloc[va_idx], y_rows[va_idx])],
                  eval_metric="auc",
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            score[tr_rows.index[va_idx]] = \
                m.predict_proba(tr_rows[feats].iloc[va_idx])[:, 1]
            te_mask = df["TARGET"].isnull()
            score[te_mask.values] += \
                m.predict_proba(df.loc[te_mask, feats])[:, 1] / N_FOLDS

    df["ROW_SCORE"] = score
    agg = df.groupby("SK_ID_CURR").agg(
        POSROW_SCORE_MEAN=("ROW_SCORE", "mean"),
        POSROW_SCORE_MAX=("ROW_SCORE", "max"),
        POSROW_SCORE_TAIL6=("ROW_SCORE", lambda s: s.tail(6).mean()))
    merge_block(agg.reset_index(),
                ["POSROW_SCORE_MEAN", "POSROW_SCORE_MAX",
                 "POSROW_SCORE_TAIL6"])


# =============================================================================
# knnv: KNN ターゲット特徴量のバリエーション
#   - k=100（局所）と k=1000（大域）
#   - 別空間: EXT×金利×年齢（v5 の augment 後に実行すること）
# =============================================================================
def _knn_feature(train, test, space_cols, k, name):
    med = train[space_cols].median()
    tr_X = train[space_cols].fillna(med).values
    te_X = test[space_cols].fillna(med).values
    sc = StandardScaler().fit(tr_X)
    tr_X, te_X = sc.transform(tr_X), sc.transform(te_X)
    y = train["TARGET"].values

    oof = np.zeros(len(train))
    tst = np.zeros(len(test))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for tr_idx, va_idx in skf.split(tr_X, y):
        nn = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(tr_X[tr_idx])
        _, idx = nn.kneighbors(tr_X[va_idx])
        oof[va_idx] = y[tr_idx][idx].mean(axis=1)
        _, idx = nn.kneighbors(te_X)
        tst += y[tr_idx][idx].mean(axis=1) / N_FOLDS
    print(f"  [{name}] single AUC = {roc_auc_score(y, oof):.5f}")
    return oof, tst


def stage_knnv():
    train, test, feats = load_features()
    train["_CAR"] = train["AMT_CREDIT"] / (train["AMT_ANNUITY"] + 1)
    test["_CAR"] = test["AMT_CREDIT"] / (test["AMT_ANNUITY"] + 1)

    base_space = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "_CAR"]
    new_cols = []

    with timer("knn k=100"):
        o, t = _knn_feature(train, test, base_space, 100, "k100")
        train["NEW_KNN100_MEAN"], test["NEW_KNN100_MEAN"] = o, t
        new_cols.append("NEW_KNN100_MEAN")
    with timer("knn k=1000"):
        o, t = _knn_feature(train, test, base_space, 1000, "k1000")
        train["NEW_KNN1000_MEAN"], test["NEW_KNN1000_MEAN"] = o, t
        new_cols.append("NEW_KNN1000_MEAN")

    alt_space = [c for c in ["EXT_SOURCE_2", "EXT_SOURCE_3",
                             "NEW_EST_INT_ANNUAL", "DAYS_BIRTH",
                             "NEW_ANNUITY_INCOME_RATIO"]
                 if c in train.columns]
    if len(alt_space) >= 4:
        with timer("knn alt-space k=500"):
            o, t = _knn_feature(train, test, alt_space, 500, "alt500")
            train["NEW_KNN_ALT500_MEAN"], test["NEW_KNN_ALT500_MEAN"] = o, t
            new_cols.append("NEW_KNN_ALT500_MEAN")

    for df in (train, test):
        df.drop("_CAR", axis=1, inplace=True)
        for c in new_cols:
            df[c] = df[c].astype("float32")
    feats = list(dict.fromkeys(feats + new_cols))
    save_features(train, test, feats)
    print(f"cache augmented: +{len(new_cols)}")


# =============================================================================
# catnative: ネイティブカテゴリ CatBoost（別表現による多様性）
# =============================================================================
def stage_catnative():
    from catboost import CatBoostClassifier
    train, test, feats = load_features()

    # application の生カテゴリを文字列のまま接合
    cat_blocks = []
    for name in ["application_train.csv", "application_test.csv"]:
        raw = pd.read_csv(f"{DATA_DIR}/{name}")
        # 修正: dtype=="object"だけだとpandasのArrow-backed string型を
        # 見逃す(hc_ensemble_optuna系で3回踏んだのと同じバグ)ため追加判定
        obj_cols = [c for c in raw.columns
                    if raw[c].dtype == "object" or pd.api.types.is_string_dtype(raw[c])]
        cat_blocks.append(raw[["SK_ID_CURR"] + obj_cols])
    cats = pd.concat(cat_blocks, ignore_index=True)
    obj_cols = [c for c in cats.columns if c != "SK_ID_CURR"]
    cats[obj_cols] = cats[obj_cols].fillna("NA").astype(str)

    # 修正: 既にtrain/testに同名カラムがあるとmergeで_x/_yにsuffixされ、
    # 後段のtrain[c]がKeyErrorで落ちる。衝突分は既存の特徴量を信頼して除外する。
    collide = [c for c in obj_cols if c in train.columns]
    if collide:
        print(f"[skip] 既存カラムと衝突するため除外: {collide}")
        obj_cols = [c for c in obj_cols if c not in collide]
        cats = cats[["SK_ID_CURR"] + obj_cols]

    train = train.merge(cats, on="SK_ID_CURR", how="left")
    test = test.merge(cats, on="SK_ID_CURR", how="left")
    for c in obj_cols:
        train[c] = train[c].fillna("NA")
        test[c] = test[c].fillna("NA")

    use_feats = feats + obj_cols
    y = train["TARGET"].values

    params = {"iterations": 20000, "learning_rate": 0.02, "depth": 7,
              "l2_leaf_reg": 10, "eval_metric": "AUC", "random_seed": SEED,
              "od_type": "Iter", "od_wait": 200, "verbose": False,
              "allow_writing_files": False}
    if USE_GPU_CATNATIVE:
        params["task_type"] = "GPU"
        params["devices"] = "0"

    oof = np.zeros(len(train))
    pred = np.zeros(len(test))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(
            skf.split(train[use_feats], y)):
        m = CatBoostClassifier(**params, cat_features=obj_cols)
        m.fit(train[use_feats].iloc[tr_idx], y[tr_idx],
              eval_set=(train[use_feats].iloc[va_idx], y[va_idx]))
        oof[va_idx] = m.predict_proba(train[use_feats].iloc[va_idx])[:, 1]
        pred += m.predict_proba(test[use_feats])[:, 1] / N_FOLDS
        print(f"  [cat_native] fold {fold}: "
              f"AUC={roc_auc_score(y[va_idx], oof[va_idx]):.5f}")
    print(f"[cat_native] OOF AUC = {roc_auc_score(y, oof):.5f}")
    np.save(STORE / "cat_native_oof.npy", oof)
    np.save(STORE / "cat_native_pred.npy", pred)


# =============================================================================
# f10: 主力 LGBM を 10-fold + lr 0.005 で
# =============================================================================
def stage_f10():
    train, test, feats = load_features()
    p = CACHE / "best_params.json"
    best = json.loads(p.read_text()) if p.exists() else {}
    params = {"objective": "binary", "metric": "auc", "verbosity": -1,
              "n_estimators": 40000, "learning_rate": 0.005,
              "subsample_freq": 1, "random_state": SEED, "n_jobs": -1,
              **best}
    if USE_GPU_F10:
        params["device"] = "gpu"
    y = train["TARGET"].values

    oof = np.zeros(len(train))
    pred = np.zeros(len(test))
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train[feats], y)):
        m = lgb.LGBMClassifier(**params)
        m.fit(train[feats].iloc[tr_idx], y[tr_idx],
              eval_set=[(train[feats].iloc[va_idx], y[va_idx])],
              eval_metric="auc",
              callbacks=[lgb.early_stopping(300, verbose=False)])
        oof[va_idx] = m.predict_proba(train[feats].iloc[va_idx])[:, 1]
        pred += m.predict_proba(test[feats])[:, 1] / 10
        print(f"  [lgb_f10] fold {fold}: "
              f"AUC={roc_auc_score(y[va_idx], oof[va_idx]):.5f}")
    print(f"[lgb_f10] OOF AUC = {roc_auc_score(y, oof):.5f}")
    np.save(STORE / "lgb_gbdt_f10_oof.npy", oof)
    np.save(STORE / "lgb_gbdt_f10_pred.npy", pred)


# =============================================================================
# blend: 複数 submission の rank 平均（最終保険）
# =============================================================================
def stage_blend(paths):
    subs = [pd.read_csv(p) for p in paths]
    base = subs[0][["SK_ID_CURR"]].copy()
    ranks = np.column_stack([
        s.set_index("SK_ID_CURR").loc[base["SK_ID_CURR"], "TARGET"]
         .rank(pct=True).values for s in subs])
    base["TARGET"] = ranks.mean(axis=1)
    out = SUBS / "submission_blend_final.csv"
    base.to_csv(out, index=False)
    print(f"blended {len(paths)} subs -> {out}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "posrow":
        stage_posrow()
    elif cmd == "knnv":
        stage_knnv()
    elif cmd == "catnative":
        stage_catnative()
    elif cmd == "f10":
        stage_f10()
    elif cmd == "blend":
        stage_blend(sys.argv[2:])
    else:
        print("usage: python hc_v6_lastday.py "
              "[posrow|knnv|catnative|f10|blend <csv...>]")
