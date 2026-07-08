# =============================================================================
# Home Credit Default Risk — v3: 優勝解法の深部 + 2019年以降の知見
#
# v2 に対する追加（すべて出典あり、詳細はチャット参照）:
#   [H] KNN ターゲット特徴量 neighbors_target_mean_500
#       … 優勝解法で最重要特徴量。EXT_SOURCE×CREDIT_ANNUITY_RATIO 空間の
#         近傍 500 人のデフォルト率。fold-safe 実装必須。
#   [I] 2 段モデル（row-level auxiliary model）
#       … previous_application / installments の「行」に TARGET を結合して
#         行レベルの補助モデルを学習し、予測スコアを顧客単位に集計。
#         1st place・17th place 解法の核心。SK_ID_CURR 単位の GroupKFold で
#         リークを遮断する。
#   [J] EXT_SOURCE 欠損の回帰補完 + 補完フラグ
#   [K] EMA（指数移動平均）特徴量 … 優勝解法の Weighted Moving Average
#   [L] last vs mean ラグ特徴量 … AmEx 2022 上位解法の定番(「直近の変化」検出)
#   [M] OpenFE 自動特徴量生成フック（ICML'23、trainのみでfitしてリーク回避）
#   [N] TabM / NN を OOF に混ぜるためのインターフェース
#
# 実行: v1, v2 と同じディレクトリに置いて python hc_ensemble_optuna_v3.py
# =============================================================================

import gc
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb

from hc_ensemble_optuna import (
    DATA_DIR, N_FOLDS, SEED, timer, kfold_train,
    make_lgb_fn, make_xgb_fn, make_cat_fn, tune_lgb,
)
from hc_ensemble_optuna_v2 import (
    build_dataset_v2, null_importance_selection, stack_predictions,
    USE_NULL_IMPORTANCE, SEED_AVERAGING,
)

warnings.simplefilter(action="ignore", category=FutureWarning)

USE_OPENFE = False   # pip install openfe が必要。時間コスト大。


# =============================================================================
# [H] neighbors_target_mean_500（KNN ターゲット特徴量）
#     リーク防止: 各 fold の学習データだけで KNN を fit し、
#     検証 fold / test の近傍を引く。fold 分割は本学習と同一 seed。
# =============================================================================
def knn_target_feature(train, test, k=500):
    base_feats = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    # CREDIT_ANNUITY_RATIO（優勝解法の定義）
    for df in (train, test):
        df["_CAR"] = df["AMT_CREDIT"] / (df["AMT_ANNUITY"] + 1)
    feats = base_feats + ["_CAR"]

    # KNN は欠損を扱えないので中央値埋め + 標準化（この特徴量専用の一時処理）
    med = train[feats].median()
    tr_X = train[feats].fillna(med).values
    te_X = test[feats].fillna(med).values
    scaler = StandardScaler().fit(tr_X)
    tr_X, te_X = scaler.transform(tr_X), scaler.transform(te_X)

    y = train["TARGET"].values
    oof_feat = np.zeros(len(train))
    test_feat = np.zeros(len(test))

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for tr_idx, va_idx in skf.split(tr_X, y):
        nn = NearestNeighbors(n_neighbors=k, n_jobs=-1)
        nn.fit(tr_X[tr_idx])
        # 検証 fold: 学習 fold 内の近傍 500 人のターゲット平均
        _, idx = nn.kneighbors(tr_X[va_idx])
        oof_feat[va_idx] = y[tr_idx][idx].mean(axis=1)
        # test: fold ごとの平均を取る
        _, idx = nn.kneighbors(te_X)
        test_feat += y[tr_idx][idx].mean(axis=1) / N_FOLDS

    train["NEW_TARGET_NEIGHBORS_500_MEAN"] = oof_feat
    test["NEW_TARGET_NEIGHBORS_500_MEAN"] = test_feat
    train.drop("_CAR", axis=1, inplace=True)
    test.drop("_CAR", axis=1, inplace=True)
    print(f"[knn target] OOF single-feature AUC = "
          f"{roc_auc_score(y, oof_feat):.5f}")
    return train, test


# =============================================================================
# [I] 2 段モデル: previous_application の行レベル補助モデル
#     各行（過去の 1 申込）に「その顧客が今回デフォルトしたか」を教師として
#     つけ、行レベルで学習 → 顧客単位に予測スコアを集計して特徴量化。
#     GroupKFold(SK_ID_CURR) で自分の行が自分の学習に入らないようにする。
# =============================================================================
def row_level_prev_score(train_ids_target, test_ids):
    df = pd.read_csv(f"{DATA_DIR}/previous_application.csv")
    # 修正: object dtypeだけでなくpandasのstring dtype(arrow-backed含む)も対象にする
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col], _ = pd.factorize(df[col])
    for c in ["DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION",
              "DAYS_LAST_DUE", "DAYS_TERMINATION"]:
        # 修正: 連鎖代入のinplace=TrueはCopy-on-Write環境でno-opになるため代入形に変更
        df[c] = df[c].replace(365243, np.nan)

    feats = [c for c in df.columns if c not in ["SK_ID_CURR", "SK_ID_PREV"]]

    # train 行に TARGET を結合
    df = df.merge(train_ids_target, on="SK_ID_CURR", how="left")
    tr_rows = df[df["TARGET"].notnull()].reset_index(drop=True)
    all_rows = df.reset_index(drop=True)

    params = {"objective": "binary", "metric": "auc", "verbosity": -1,
              "n_estimators": 2000, "learning_rate": 0.05, "num_leaves": 63,
              "colsample_bytree": 0.6, "subsample": 0.8, "subsample_freq": 1,
              "random_state": SEED, "n_jobs": -1}

    row_score = np.zeros(len(all_rows))
    gkf = GroupKFold(n_splits=N_FOLDS)
    y_rows = tr_rows["TARGET"].values
    groups = tr_rows["SK_ID_CURR"].values

    with timer("row-level prev model"):
        for tr_idx, va_idx in gkf.split(tr_rows[feats], y_rows, groups):
            m = lgb.LGBMClassifier(**params)
            m.fit(tr_rows[feats].iloc[tr_idx], y_rows[tr_idx],
                  eval_set=[(tr_rows[feats].iloc[va_idx], y_rows[va_idx])],
                  eval_metric="auc",
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            # OOF 分（train 行）
            va_ids = tr_rows.index[va_idx]
            row_score[va_ids] = m.predict_proba(tr_rows[feats].iloc[va_idx])[:, 1]
            # test 顧客の行は fold 平均
            te_mask = all_rows["TARGET"].isnull()
            row_score[te_mask.values] += (
                m.predict_proba(all_rows.loc[te_mask, feats])[:, 1] / N_FOLDS)

    all_rows["ROW_SCORE"] = row_score
    all_rows["_LAST_RANK"] = all_rows.groupby("SK_ID_CURR")["DAYS_DECISION"] \
                                     .rank(ascending=False)

    agg = all_rows.groupby("SK_ID_CURR").agg(
        PREVROW_SCORE_MEAN=("ROW_SCORE", "mean"),
        PREVROW_SCORE_MAX=("ROW_SCORE", "max"),
        PREVROW_SCORE_MIN=("ROW_SCORE", "min"),
        PREVROW_SCORE_STD=("ROW_SCORE", "std"))
    last = (all_rows[all_rows["_LAST_RANK"] == 1]
            .set_index("SK_ID_CURR")["ROW_SCORE"].rename("PREVROW_SCORE_LAST"))
    agg = agg.join(last, how="left")

    del df, tr_rows, all_rows
    gc.collect()
    return agg


# =============================================================================
# [I'] 2 段モデル: installments 行レベル（返済 1 回単位）
#      行数が多い（~13M）ので直近 24 回/顧客にサブサンプルして学習。
# =============================================================================
def row_level_installments_score(train_ids_target):
    # ★ メモリ節約: 使うカラムだけ読み込み + float32 化（元は float64 全カラム
    #   読み込みで 13M行 x 8列 相当が無駄に倍のメモリを取っていた）
    usecols = ["SK_ID_CURR", "NUM_INSTALMENT_VERSION", "NUM_INSTALMENT_NUMBER",
               "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT",
               "AMT_PAYMENT"]
    dtype_map = {c: "float32" for c in usecols if c != "SK_ID_CURR"}
    dtype_map["SK_ID_CURR"] = "int32"
    df = pd.read_csv(f"{DATA_DIR}/installments_payments.csv",
                     usecols=usecols, dtype=dtype_map)
    df["DPD"] = (df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"]).clip(lower=0)
    df["DBD"] = (df["DAYS_INSTALMENT"] - df["DAYS_ENTRY_PAYMENT"]).clip(lower=0)
    df["PAYMENT_RATIO"] = df["AMT_PAYMENT"] / (df["AMT_INSTALMENT"] + 1)

    # 直近 24 回に限定（計算量と「直近性」の両取り）
    df = df.sort_values(["SK_ID_CURR", "DAYS_INSTALMENT"])
    df = df.groupby("SK_ID_CURR").tail(24).reset_index(drop=True)

    feats = ["NUM_INSTALMENT_VERSION", "NUM_INSTALMENT_NUMBER",
             "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT",
             "AMT_PAYMENT", "DPD", "DBD", "PAYMENT_RATIO"]

    df = df.merge(train_ids_target, on="SK_ID_CURR", how="left")
    tr_rows = df[df["TARGET"].notnull()]
    y_rows = tr_rows["TARGET"].values
    groups = tr_rows["SK_ID_CURR"].values

    params = {"objective": "binary", "metric": "auc", "verbosity": -1,
              "n_estimators": 1500, "learning_rate": 0.05, "num_leaves": 63,
              "colsample_bytree": 0.8, "subsample": 0.8, "subsample_freq": 1,
              "random_state": SEED, "n_jobs": -1}

    score = np.zeros(len(df))
    gkf = GroupKFold(n_splits=N_FOLDS)
    with timer("row-level installments model"):
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
        INSROW_SCORE_MEAN=("ROW_SCORE", "mean"),
        INSROW_SCORE_MAX=("ROW_SCORE", "max"),
        INSROW_SCORE_TAIL5=("ROW_SCORE",
                            lambda s: s.tail(5).mean()))
    del df, tr_rows
    gc.collect()
    return agg


# =============================================================================
# [J] EXT_SOURCE 欠損の回帰補完（+補完フラグ）
#     欠損数自体のシグナル(NEW_EXTSOURCE_NA_CNT)は v1 で保持済みなので、
#     ここでは値を埋めて KNN 特徴量や NN の質を上げるのが目的。
# =============================================================================
def impute_ext_sources(train, test):
    from lightgbm import LGBMRegressor
    # ★ メモリ/速度対策: train・test は既に大量の df[col]=... 代入で断片化して
    #   いるため、concat 直後に copy() で連続領域に再配置する。
    #   これを怠ると、この関数内の 812 列超フレームへの列選択・列追加のたびに
    #   pandas 内部でブロック走査が発生し、本来数分の処理が数十分に化ける
    #   （実測: 15.5s → 2852.3s の主因はこれ）。
    all_df = pd.concat([train, test], ignore_index=True).copy()
    predictors = ["DAYS_BIRTH", "DAYS_EMPLOYED", "AMT_INCOME_TOTAL",
                  "AMT_CREDIT", "AMT_ANNUITY", "NEW_DAYS_EMPLOYED_RATIO",
                  "REGION_POPULATION_RELATIVE", "DAYS_ID_PUBLISH"]
    predictors = [c for c in predictors if c in all_df.columns]

    for tgt in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
        flag = f"NEW_{tgt}_IMPUTED"
        all_df[flag] = all_df[tgt].isnull().astype(int)
        known = all_df[all_df[tgt].notnull()]
        unknown = all_df[all_df[tgt].isnull()]
        if len(unknown) == 0:
            continue
        reg = LGBMRegressor(n_estimators=300, learning_rate=0.05,
                            num_leaves=31, random_state=SEED, n_jobs=-1,
                            verbosity=-1)
        reg.fit(known[predictors], known[tgt])
        all_df.loc[all_df[tgt].isnull(), tgt] = reg.predict(unknown[predictors])
    all_df = all_df.copy()  # ループ中の再断片化を後始末

    n_tr = len(train)
    return (all_df.iloc[:n_tr].reset_index(drop=True),
            all_df.iloc[n_tr:].reset_index(drop=True))


# =============================================================================
# [K][L] EMA + last-vs-mean 特徴量（POS / installments / credit card）
#     優勝解法の Weighted Moving Average と AmEx 上位解法の
#     「最新値 − 平均値」（直近の行動変化）をまとめて実装。
# =============================================================================
def ema_and_lag_features():
    out = []

    # --- installments: DPD と支払比率の EMA / last-vs-mean ---
    ins = pd.read_csv(f"{DATA_DIR}/installments_payments.csv",
                      usecols=["SK_ID_CURR", "DAYS_INSTALMENT",
                               "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT",
                               "AMT_PAYMENT"])
    ins["DPD"] = (ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]).clip(lower=0)
    ins["PAYRATIO"] = ins["AMT_PAYMENT"] / (ins["AMT_INSTALMENT"] + 1)
    ins = ins.sort_values(["SK_ID_CURR", "DAYS_INSTALMENT"])

    g = ins.groupby("SK_ID_CURR")
    ema = g[["DPD", "PAYRATIO"]].apply(
        lambda x: x.ewm(halflife=6).mean().iloc[-1])
    ema.columns = ["INS_EMA_DPD", "INS_EMA_PAYRATIO"]

    last = g[["DPD", "PAYRATIO"]].last()
    mean = g[["DPD", "PAYRATIO"]].mean()
    lag = (last - mean)
    lag.columns = ["INS_LASTvMEAN_DPD", "INS_LASTvMEAN_PAYRATIO"]
    out += [ema, lag]
    del ins, g
    gc.collect()

    # --- credit card: 利用率の EMA / last-vs-mean ---
    cc = pd.read_csv(f"{DATA_DIR}/credit_card_balance.csv",
                     usecols=["SK_ID_CURR", "MONTHS_BALANCE", "AMT_BALANCE",
                              "AMT_CREDIT_LIMIT_ACTUAL"])
    cc["UTIL"] = cc["AMT_BALANCE"] / (cc["AMT_CREDIT_LIMIT_ACTUAL"] + 1)
    cc = cc.sort_values(["SK_ID_CURR", "MONTHS_BALANCE"])
    g = cc.groupby("SK_ID_CURR")
    ema = g["UTIL"].apply(lambda x: x.ewm(halflife=3).mean().iloc[-1]) \
                   .rename("CC_EMA_UTIL").to_frame()
    lag = (g["UTIL"].last() - g["UTIL"].mean()) \
        .rename("CC_LASTvMEAN_UTIL").to_frame()
    out += [ema, lag]
    del cc, g
    gc.collect()

    # --- POS: 残回数の消化ペース（進捗率の last-vs-mean）---
    pos = pd.read_csv(f"{DATA_DIR}/POS_CASH_balance.csv",
                      usecols=["SK_ID_CURR", "MONTHS_BALANCE", "SK_DPD_DEF"])
    pos = pos.sort_values(["SK_ID_CURR", "MONTHS_BALANCE"])
    g = pos.groupby("SK_ID_CURR")
    lag = (g["SK_DPD_DEF"].last() - g["SK_DPD_DEF"].mean()) \
        .rename("POS_LASTvMEAN_DPDDEF").to_frame()
    out.append(lag)
    del pos, g
    gc.collect()

    return pd.concat(out, axis=1)


# =============================================================================
# [M] OpenFE 自動特徴量生成（オプション）
#     ICML'23。注意: 公式実装は train+test 全体で候補評価する使い方をすると
#     リーク疑義がある（Medium の批判記事）。ここでは train のみで fit し、
#     transform だけを test に適用する安全側の使い方に固定する。
# =============================================================================
def openfe_features(train, test, feats, n_jobs=8, top_k=50):
    from openfe import OpenFE, transform
    ofe = OpenFE()
    with timer("OpenFE fit (train only)"):
        new_feats = ofe.fit(data=train[feats], label=train["TARGET"],
                            n_jobs=n_jobs)
    with timer("OpenFE transform"):
        train, test = transform(train, test, new_feats[:top_k], n_jobs=n_jobs)
    return train, test


# =============================================================================
# main
# =============================================================================
def main():
    train, test, feats = build_dataset_v2()
    y_target = train[["SK_ID_CURR", "TARGET"]]

    # --- [J] EXT_SOURCE 補完（KNN 特徴量の質を上げるため先に）---
    with timer("ext_source imputation"):
        train, test = impute_ext_sources(train, test)
        feats += [c for c in train.columns
                  if c.endswith("_IMPUTED") and c not in feats]

    # --- [H] KNN ターゲット特徴量 ---
    with timer("knn target feature"):
        train, test = knn_target_feature(train, test, k=500)
        feats.append("NEW_TARGET_NEIGHBORS_500_MEAN")

    # --- [I] 2 段モデル ---
    prev_score = row_level_prev_score(y_target, test["SK_ID_CURR"])
    train = train.merge(prev_score, on="SK_ID_CURR", how="left")
    test = test.merge(prev_score, on="SK_ID_CURR", how="left")
    feats += list(prev_score.columns)

    ins_score = row_level_installments_score(y_target)
    train = train.merge(ins_score, on="SK_ID_CURR", how="left")
    test = test.merge(ins_score, on="SK_ID_CURR", how="left")
    feats += list(ins_score.columns)

    # --- [K][L] EMA / last-vs-mean ---
    with timer("ema and lag features"):
        el = ema_and_lag_features()
        train = train.merge(el, on="SK_ID_CURR", how="left")
        test = test.merge(el, on="SK_ID_CURR", how="left")
        feats += list(el.columns)

    # --- [M] OpenFE（オプション）---
    if USE_OPENFE:
        train, test = openfe_features(train, test, feats)
        feats = [c for c in train.columns if c not in
                 ["TARGET", "SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV", "index"]]

    train = train.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x))
    test = test.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x))
    feats = [re.sub("[^A-Za-z0-9_]+", "", f) for f in feats]
    feats = list(dict.fromkeys(feats))  # 重複除去・順序維持
    print(f"n_features (v3): {len(feats)}")

    if USE_NULL_IMPORTANCE:
        feats = null_importance_selection(train, feats)

    with timer("optuna_lgb"):
        best_lgb = tune_lgb(train, feats)

    # LGBM シード平均
    oof_lgb = np.zeros(len(train))
    pred_lgb = np.zeros(len(test))
    for s in SEED_AVERAGING:
        def fn(tr_x, tr_y, va_x, va_y, _s=s):
            p = {"objective": "binary", "metric": "auc", "verbosity": -1,
                 "n_estimators": 20000, "learning_rate": 0.01,
                 "subsample_freq": 1, "random_state": _s, "n_jobs": -1,
                 **best_lgb}
            m = lgb.LGBMClassifier(**p)
            m.fit(tr_x, tr_y, eval_set=[(va_x, va_y)], eval_metric="auc",
                  callbacks=[lgb.early_stopping(200, verbose=False)])
            return m
        o, p_, _ = kfold_train(train, test, feats, fn, f"LGB(s{s})")
        oof_lgb += o / len(SEED_AVERAGING)
        pred_lgb += p_ / len(SEED_AVERAGING)
    print(f"[LGB seed-avg] OOF AUC = {roc_auc_score(train['TARGET'], oof_lgb):.5f}")

    with timer("xgb_cv"):
        oof_xgb, pred_xgb, _ = kfold_train(train, test, feats,
                                           make_xgb_fn(best_lgb), "XGB")
    with timer("cat_cv"):
        oof_cat, pred_cat, _ = kfold_train(train, test, feats,
                                           make_cat_fn(), "CAT")

    # --- [N] NN(TabM 等) の OOF があればここに追加する ---
    # oof_nn = np.load("oof_tabm.npy"); pred_nn = np.load("pred_tabm.npy")
    oofs = [oof_lgb, oof_xgb, oof_cat]
    preds = [pred_lgb, pred_xgb, pred_cat]

    y = train["TARGET"].values
    final_pred = stack_predictions(oofs, preds, y)

    submission = test[["SK_ID_CURR"]].copy()
    submission["TARGET"] = final_pred
    submission.to_csv("submission.csv", index=False)
    pd.DataFrame({"SK_ID_CURR": train["SK_ID_CURR"], "oof_lgb": oof_lgb,
                  "oof_xgb": oof_xgb, "oof_cat": oof_cat, "target": y}
                 ).to_csv("oof_predictions.csv", index=False)
    print("done.")


if __name__ == "__main__":
    main()
