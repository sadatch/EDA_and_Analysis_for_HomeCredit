# =============================================================================
# hc_v5_final_push.py — 0.800 の壁を越えるための最終施策
#
# 現状診断: CV→Private オフセットは -0.0022 で安定(リークなし)。
#           3 提出が Private 0.7999x に収束 = 既存特徴量からの情報は飽和。
#           → 必要なのはブレンドの工夫ではなく「新しい情報源」。
#
# 施策:
#   [②] 現申込の金利推定(優勝解法の核心・未実装だった部分)
#       過去申込(prev)で CNT_PAYMENT 予測モデルを学習し、それを
#       「現在の申込」に適用して今回ローンの推定金利を作る。
#       "銀行がこの人に提示した金利水準" は外部信用スコアに匹敵する情報。
#   [⑤] Restacking — L2 スタッカーの入力に強い生特徴量を混ぜ、
#       「どの領域でどのモデルを信じるか」をメタモデルに学習させる。
#
# 使い方(既存の cache / oof_store を壊さない追記型):
#   python hc_v5_final_push.py augment    # キャッシュに金利特徴量を追記
#   rm oof_store/lgb_gbdt_full_*          # 主力 1-2 本だけ再学習
#   python hc_campaign.py train lgb_gbdt_full
#   python hc_campaign.py train lgb_dart
#   python hc_v5_final_push.py ensemble2  # restack 版アンサンブル
# =============================================================================

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

import lightgbm as lgb

from hc_ensemble_optuna import DATA_DIR, timer

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


# =============================================================================
# [②] 現申込の金利推定
#     ステップ 1: prev(Approved・CNT_PAYMENT>0)で
#                 CNT_PAYMENT ~ f(AMT_ANNUITY, AMT_CREDIT, AMT_GOODS_PRICE, 比率)
#                 の回帰モデルを学習
#     ステップ 2: application の同名カラムに適用 → 推定支払回数
#     ステップ 3: 推定金利 = AMT_ANNUITY × 推定回数 / AMT_CREDIT - 1 (年利換算)
#     ステップ 4: 過去金利との差分(条件が悪化したか)などの派生を作る
# =============================================================================
def current_app_interest_features():
    prev = pd.read_csv(
        f"{DATA_DIR}/previous_application.csv",
        usecols=["AMT_ANNUITY", "AMT_CREDIT", "AMT_GOODS_PRICE",
                 "CNT_PAYMENT", "NAME_CONTRACT_STATUS", "NAME_CONTRACT_TYPE"])
    prev = prev[(prev["NAME_CONTRACT_STATUS"] == "Approved") &
                (prev["CNT_PAYMENT"] > 0) &
                (prev["AMT_CREDIT"] > 0) &
                (prev["AMT_ANNUITY"] > 0)].copy()

    def add_ratio_feats(df):
        df = df.copy()
        df["F_ANNUITY_CREDIT"] = df["AMT_ANNUITY"] / (df["AMT_CREDIT"] + 1e-5)
        df["F_GOODS_CREDIT"] = df["AMT_GOODS_PRICE"] / (df["AMT_CREDIT"] + 1)
        df["F_LOG_CREDIT"] = np.log1p(df["AMT_CREDIT"])
        df["F_LOG_ANNUITY"] = np.log1p(df["AMT_ANNUITY"])
        return df

    prev = add_ratio_feats(prev)
    reg_feats = ["AMT_ANNUITY", "AMT_CREDIT", "AMT_GOODS_PRICE",
                 "F_ANNUITY_CREDIT", "F_GOODS_CREDIT",
                 "F_LOG_CREDIT", "F_LOG_ANNUITY"]

    # 現金ローンとリボ等で構造が違うため、Cash loans に絞って学習するのが
    # application(Cash が大半)への適用に一番素直
    prev_cash = prev[prev["NAME_CONTRACT_TYPE"] == "Cash loans"]
    if len(prev_cash) < 10000:
        prev_cash = prev  # 保険

    reg = lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.05,
                            num_leaves=63, colsample_bytree=0.8,
                            subsample=0.8, subsample_freq=1,
                            random_state=SEED, n_jobs=-1, verbosity=-1)
    with timer("CNT_PAYMENT regressor fit"):
        reg.fit(prev_cash[reg_feats], prev_cash["CNT_PAYMENT"])

    # application 側へ適用
    out = []
    for name in ["application_train.csv", "application_test.csv"]:
        app = pd.read_csv(f"{DATA_DIR}/{name}",
                          usecols=["SK_ID_CURR", "AMT_ANNUITY", "AMT_CREDIT",
                                   "AMT_GOODS_PRICE"])
        app = add_ratio_feats(app)
        est_cnt = reg.predict(app[reg_feats]).clip(4, 84)  # 支払回数を常識範囲に
        app["NEW_EST_CNT_PAYMENT"] = est_cnt
        # 安全策: AMT_CREDITは実データ上ほぼ常に>0だが、他の割り算箇所で
        # epsilonガード漏れがinf/XGBoostクラッシュを繰り返し起こしてきたため、
        # ここでも念のためepsilonを入れておく
        total_int = (app["AMT_ANNUITY"] * est_cnt / (app["AMT_CREDIT"] + 1e-5)) - 1
        app["NEW_EST_INT_TOTAL"] = total_int.clip(-0.1, 2.0)
        app["NEW_EST_INT_ANNUAL"] = (total_int / est_cnt * 12).clip(-0.1, 1.5)
        out.append(app[["SK_ID_CURR", "NEW_EST_CNT_PAYMENT",
                        "NEW_EST_INT_TOTAL", "NEW_EST_INT_ANNUAL"]])
    return pd.concat(out, ignore_index=True)


def stage_augment():
    train, test, feats = load_features()
    block = current_app_interest_features()

    train = train.merge(block, on="SK_ID_CURR", how="left")
    test = test.merge(block, on="SK_ID_CURR", how="left")
    new_feats = ["NEW_EST_CNT_PAYMENT", "NEW_EST_INT_TOTAL",
                 "NEW_EST_INT_ANNUAL"]

    # 派生: 過去に借りた金利との比較(条件悪化シグナル)と EXT との相互作用
    for df in (train, test):
        if "PREVINT_INT_ANNUAL_MEAN" in df.columns:
            df["NEW_EST_INT_VS_HIST"] = (df["NEW_EST_INT_ANNUAL"] -
                                         df["PREVINT_INT_ANNUAL_MEAN"])
        if "PREVINT_INT_ANNUAL_LAST" in df.columns:
            df["NEW_EST_INT_VS_LAST"] = (df["NEW_EST_INT_ANNUAL"] -
                                         df["PREVINT_INT_ANNUAL_LAST"])
        if "NEW_EXTSOURCE_MEAN" in df.columns:
            df["NEW_EST_INT_X_EXT"] = (df["NEW_EST_INT_ANNUAL"] *
                                       (1 - df["NEW_EXTSOURCE_MEAN"]))
    for c in ["NEW_EST_INT_VS_HIST", "NEW_EST_INT_VS_LAST",
              "NEW_EST_INT_X_EXT"]:
        if c in train.columns:
            new_feats.append(c)

    # float32 へ揃える(既存キャッシュとの一貫性)
    for df in (train, test):
        for c in new_feats:
            df[c] = df[c].astype("float32")

    # 安全策: hc_campaign.pyのstage_features()と同じくinf/-infをNaNに変換してから保存
    for df in (train, test):
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

    feats = list(dict.fromkeys(feats + new_feats))
    train.to_parquet(CACHE / "features_train.parquet")
    test.to_parquet(CACHE / "features_test.parquet")
    (CACHE / "feats.json").write_text(json.dumps(feats))
    print(f"augmented: +{len(new_feats)} features "
          f"(total {len(feats)})")
    print("次: rm oof_store/lgb_gbdt_full_* 等で主力モデルを消してから "
          "hc_campaign.py train で再学習してください")


# =============================================================================
# [⑤] Restacking アンサンブル
#     L2 の入力 = 全モデルの OOF(rank) + 強い生特徴量(rank)。
#     生特徴量は「モデル間の得意領域の違い」をメタモデルが読むための座標。
# =============================================================================
RESTACK_RAW_FEATS = [
    "NEW_EXTSOURCE_MEAN", "NEW_TARGET_NEIGHBORS_500_MEAN",
    "INSTAL_DPD_MEAN", "NEW_DAYS_BIRTH", "NEW_EST_INT_ANNUAL",
    "AMT_CREDIT", "NEW_ANNUITY_INCOME_RATIO",
]


def stage_ensemble2(tag="restack"):
    train, test, _ = load_features()
    y = train["TARGET"].values

    oofs = {p.stem[:-4]: np.load(p) for p in STORE.glob("*_oof.npy")}
    preds = {p.stem[:-5]: np.load(p) for p in STORE.glob("*_pred.npy")}
    names = sorted(set(oofs) & set(preds))
    print(f"models: {names}")

    L1o = [pd.Series(oofs[n]).rank(pct=True).values for n in names]
    L1p = [pd.Series(preds[n]).rank(pct=True).values for n in names]

    # 生特徴量を rank 変換して追加(欠損は中央値埋め)
    raw_used = []
    for f in RESTACK_RAW_FEATS:
        if f not in train.columns:
            continue
        med = train[f].median()
        L1o.append(train[f].fillna(med).rank(pct=True).values)
        L1p.append(test[f].fillna(med).rank(pct=True).values)
        raw_used.append(f)
    print(f"restack raw feats: {raw_used}")

    Xo = np.column_stack(L1o)
    Xp = np.column_stack(L1p)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + 7)
    meta_oof = {"logreg": np.zeros(len(y)), "lgbm": np.zeros(len(y))}
    meta_pred = {"logreg": np.zeros(len(Xp)), "lgbm": np.zeros(len(Xp))}
    for tr_idx, va_idx in skf.split(Xo, y):
        lr = LogisticRegression(C=0.5, max_iter=2000)
        lr.fit(Xo[tr_idx], y[tr_idx])
        meta_oof["logreg"][va_idx] = lr.predict_proba(Xo[va_idx])[:, 1]
        meta_pred["logreg"] += lr.predict_proba(Xp)[:, 1] / N_FOLDS

        gm = lgb.LGBMClassifier(objective="binary", n_estimators=600,
                                learning_rate=0.02, num_leaves=15,
                                min_child_samples=1000,
                                colsample_bytree=0.7, subsample=0.8,
                                subsample_freq=1, random_state=SEED,
                                n_jobs=-1, verbosity=-1)
        gm.fit(Xo[tr_idx], y[tr_idx])
        meta_oof["lgbm"][va_idx] = gm.predict_proba(Xo[va_idx])[:, 1]
        meta_pred["lgbm"] += gm.predict_proba(Xp)[:, 1] / N_FOLDS

    results = {}
    for k in meta_oof:
        auc = roc_auc_score(y, meta_oof[k])
        results[k] = auc
        print(f"  restack L2 {k}: OOF AUC = {auc:.5f}")

    # 2 つの rank 平均も候補に
    blend_oof = (pd.Series(meta_oof["logreg"]).rank(pct=True).values +
                 pd.Series(meta_oof["lgbm"]).rank(pct=True).values) / 2
    blend_pred = (pd.Series(meta_pred["logreg"]).rank(pct=True).values +
                  pd.Series(meta_pred["lgbm"]).rank(pct=True).values) / 2
    blend_auc = roc_auc_score(y, blend_oof)
    print(f"  restack L3 blend: OOF AUC = {blend_auc:.5f}")

    cands = {"logreg": (results["logreg"], meta_pred["logreg"]),
             "lgbm": (results["lgbm"], meta_pred["lgbm"]),
             "blend": (blend_auc, blend_pred)}
    best = max(cands, key=lambda k: cands[k][0])
    cv, final = cands[best]
    print(f"[restack] final = {best} (OOF {cv:.5f})")

    SUBS.mkdir(exist_ok=True)
    sub = test[["SK_ID_CURR"]].copy()
    sub["TARGET"] = final
    path = SUBS / f"submission_{tag}_{best}_cv{cv:.5f}.csv"
    sub.to_csv(path, index=False)
    print(f"saved: {path}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "augment"
    if cmd == "augment":
        stage_augment()
    elif cmd == "ensemble2":
        stage_ensemble2(sys.argv[2] if len(sys.argv) > 2 else "restack")
    else:
        print("usage: python hc_v5_final_push.py [augment|ensemble2]")
