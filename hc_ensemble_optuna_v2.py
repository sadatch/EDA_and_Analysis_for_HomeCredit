# =============================================================================
# Home Credit Default Risk — v2: 0.80 の壁を攻める追加実装
#
# v1 (hc_ensemble_optuna.py) に対する追加:
#   [A] 金利近似特徴量（1st place の核心アイデアの簡易版）
#   [B] 時間窓集計（直近 90/180/365/730 日）— 上位解法の共通パターン
#   [C] last-k 集計（直近 1/3/5 件の申込・返済に限定した集計）
#   [D] トレンド特徴量（直近 vs 全期間の差分 = 行動が悪化しているか）
#   [E] グループ統計量（職業・組織内での相対的位置。ターゲット不使用なのでリーク無し）
#   [F] Null Importance による特徴量選択（ノイズ特徴量の枝刈り）
#   [G] スタッキング（LogisticRegression メタモデル）+ シード平均
#
# 実行: v1 と同じディレクトリに置いて python hc_ensemble_optuna_v2.py
# =============================================================================

import gc
import os
import re
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

import lightgbm as lgb

# v1 の前処理・学習基盤を再利用
from hc_ensemble_optuna import (
    DATA_DIR, N_FOLDS, SEED, timer, one_hot,
    preprocess_application, preprocess_bureau, preprocess_prev,
    preprocess_pos, preprocess_installments, preprocess_credit_card,
    tune_lgb, kfold_train, make_lgb_fn, make_xgb_fn, make_cat_fn,
    optimize_weights,
)

warnings.simplefilter(action="ignore", category=FutureWarning)

USE_NULL_IMPORTANCE = True   # 時間がなければ False（LGBM 学習 ~40 回分のコスト）
NULL_IMP_RUNS = int(os.environ.get("HC_NULLIMP_RUNS", 30))
# null importance 用の行サブサンプル率（actual/null の両方に同一適用するので
# 比較の公平性は保たれる。0.5 にすると実行時間はほぼ半分）
NULL_IMP_SAMPLE = float(os.environ.get("HC_NULLIMP_SAMPLE", 1.0))
SEED_AVERAGING = [42, 2025, 777]  # LGBM のみシード平均


# =============================================================================
# [A] 金利近似特徴量（previous_application）
#     Home Credit は金利カラムをほぼ削除しているが、
#     CNT_PAYMENT × AMT_ANNUITY と AMT_CREDIT から逆算できる。
#     1st place はこれを外部モデルで精密に推定した。ここでは単利近似。
# =============================================================================
def prev_interest_features():
    df = pd.read_csv(f"{DATA_DIR}/previous_application.csv",
                     usecols=["SK_ID_CURR", "SK_ID_PREV", "AMT_ANNUITY",
                              "AMT_CREDIT", "CNT_PAYMENT", "DAYS_DECISION",
                              "NAME_CONTRACT_STATUS"])
    df = df[(df["CNT_PAYMENT"] > 0) & (df["AMT_CREDIT"] > 0)]

    # 総支払額 / 元本 - 1 = 期間全体の金利負担率
    df["TOTAL_PAYMENT"] = df["AMT_ANNUITY"] * df["CNT_PAYMENT"]
    df["INTEREST_TOTAL"] = df["TOTAL_PAYMENT"] / df["AMT_CREDIT"] - 1
    # 月あたり単利（CNT_PAYMENT は月数）→ 年利換算
    df["INTEREST_ANNUAL"] = df["INTEREST_TOTAL"] / df["CNT_PAYMENT"] * 12

    # 非現実的な値（データノイズ）はクリップ
    df["INTEREST_ANNUAL"] = df["INTEREST_ANNUAL"].clip(-0.1, 1.5)

    agg = df.groupby("SK_ID_CURR").agg(
        INT_ANNUAL_MEAN=("INTEREST_ANNUAL", "mean"),
        INT_ANNUAL_MAX=("INTEREST_ANNUAL", "max"),
        INT_ANNUAL_MIN=("INTEREST_ANNUAL", "min"),
        INT_ANNUAL_STD=("INTEREST_ANNUAL", "std"),
        INT_TOTAL_MEAN=("INTEREST_TOTAL", "mean"),
        CNT_PAYMENT_MEAN=("CNT_PAYMENT", "mean"),
        CNT_PAYMENT_SUM=("CNT_PAYMENT", "sum"),
    )
    # 直近の申込ほど現在の信用状態を反映する → 最後の 1 件の金利
    last = (df.sort_values("DAYS_DECISION")
              .groupby("SK_ID_CURR").tail(1)
              .set_index("SK_ID_CURR")[["INTEREST_ANNUAL", "CNT_PAYMENT"]])
    last.columns = ["INT_ANNUAL_LAST", "CNT_PAYMENT_LAST"]
    agg = agg.join(last, how="left")
    agg.columns = [f"PREVINT_{c}" for c in agg.columns]

    del df, last
    gc.collect()
    return agg


# =============================================================================
# [B][C] previous_application の時間窓 + last-k 集計
# =============================================================================
def prev_window_features():
    cols = ["SK_ID_CURR", "DAYS_DECISION", "AMT_CREDIT", "AMT_ANNUITY",
            "AMT_APPLICATION", "AMT_DOWN_PAYMENT", "NAME_CONTRACT_STATUS"]
    df = pd.read_csv(f"{DATA_DIR}/previous_application.csv", usecols=cols)
    df["REFUSED"] = (df["NAME_CONTRACT_STATUS"] == "Refused").astype(int)
    df["APPROVED"] = (df["NAME_CONTRACT_STATUS"] == "Approved").astype(int)

    out = []
    # 時間窓: 直近 1 年 / 2 年に申込がどれだけ集中しているか（資金繰り悪化のシグナル）
    for days, tag in [(365, "1Y"), (730, "2Y")]:
        sub = df[df["DAYS_DECISION"] >= -days]
        a = sub.groupby("SK_ID_CURR").agg(
            **{f"PREVW{tag}_COUNT": ("DAYS_DECISION", "size"),
               f"PREVW{tag}_AMT_CREDIT_SUM": ("AMT_CREDIT", "sum"),
               f"PREVW{tag}_REFUSED_MEAN": ("REFUSED", "mean"),
               f"PREVW{tag}_REFUSED_SUM": ("REFUSED", "sum")})
        out.append(a)

    # last-k: 直近 k 件だけ見た却下率・金額（古い履歴で薄まらない）
    df = df.sort_values(["SK_ID_CURR", "DAYS_DECISION"])
    for k in [1, 3, 5]:
        sub = df.groupby("SK_ID_CURR").tail(k)
        a = sub.groupby("SK_ID_CURR").agg(
            **{f"PREVL{k}_REFUSED_MEAN": ("REFUSED", "mean"),
               f"PREVL{k}_AMT_CREDIT_MEAN": ("AMT_CREDIT", "mean"),
               f"PREVL{k}_AMT_ANNUITY_MEAN": ("AMT_ANNUITY", "mean"),
               f"PREVL{k}_DAYS_DECISION_MAX": ("DAYS_DECISION", "max")})
        out.append(a)

    agg = pd.concat(out, axis=1)
    del df, out
    gc.collect()
    return agg


# =============================================================================
# [B][D] installments の時間窓 + トレンド
#     「昔は普通に払えていたが最近遅れ始めた」を捉えるのが目的。
# =============================================================================
def installments_window_trend():
    df = pd.read_csv(f"{DATA_DIR}/installments_payments.csv",
                     usecols=["SK_ID_CURR", "DAYS_INSTALMENT",
                              "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT"])
    df["DPD"] = (df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"]).clip(lower=0)
    df["PAYMENT_RATIO"] = df["AMT_PAYMENT"] / (df["AMT_INSTALMENT"] + 1)
    df["LATE"] = (df["DPD"] > 0).astype(int)

    out = []
    # 全期間ベースライン
    base = df.groupby("SK_ID_CURR").agg(
        INS_ALL_DPD_MEAN=("DPD", "mean"),
        INS_ALL_LATE_MEAN=("LATE", "mean"),
        INS_ALL_PAYRATIO_MEAN=("PAYMENT_RATIO", "mean"))
    out.append(base)

    # 時間窓
    for days, tag in [(90, "90D"), (180, "180D"), (365, "1Y"), (730, "2Y")]:
        sub = df[df["DAYS_INSTALMENT"] >= -days]
        a = sub.groupby("SK_ID_CURR").agg(
            **{f"INSW{tag}_DPD_MEAN": ("DPD", "mean"),
               f"INSW{tag}_DPD_MAX": ("DPD", "max"),
               f"INSW{tag}_LATE_MEAN": ("LATE", "mean"),
               f"INSW{tag}_PAYRATIO_MEAN": ("PAYMENT_RATIO", "mean")})
        out.append(a)

    # last-k 回の返済
    df = df.sort_values(["SK_ID_CURR", "DAYS_INSTALMENT"])
    for k in [3, 5, 10]:
        sub = df.groupby("SK_ID_CURR").tail(k)
        a = sub.groupby("SK_ID_CURR").agg(
            **{f"INSL{k}_DPD_MEAN": ("DPD", "mean"),
               f"INSL{k}_LATE_MEAN": ("LATE", "mean"),
               f"INSL{k}_PAYRATIO_MEAN": ("PAYMENT_RATIO", "mean")})
        out.append(a)

    agg = pd.concat(out, axis=1)

    # トレンド = 直近 − 全期間（正なら悪化中）
    agg["INS_TREND_DPD_1Y"] = agg["INSW1Y_DPD_MEAN"] - agg["INS_ALL_DPD_MEAN"]
    agg["INS_TREND_LATE_1Y"] = agg["INSW1Y_LATE_MEAN"] - agg["INS_ALL_LATE_MEAN"]
    agg["INS_TREND_DPD_90D"] = agg["INSW90D_DPD_MEAN"] - agg["INS_ALL_DPD_MEAN"]
    agg["INS_TREND_PAYRATIO_1Y"] = agg["INSW1Y_PAYRATIO_MEAN"] - agg["INS_ALL_PAYRATIO_MEAN"]

    del df, out
    gc.collect()
    return agg


# =============================================================================
# [B][D] credit_card の時間窓 + 利用率トレンド
# =============================================================================
def credit_card_trend():
    df = pd.read_csv(f"{DATA_DIR}/credit_card_balance.csv",
                     usecols=["SK_ID_CURR", "MONTHS_BALANCE", "AMT_BALANCE",
                              "AMT_CREDIT_LIMIT_ACTUAL", "AMT_DRAWINGS_CURRENT",
                              "SK_DPD"])
    df["UTIL"] = df["AMT_BALANCE"] / (df["AMT_CREDIT_LIMIT_ACTUAL"] + 1)

    base = df.groupby("SK_ID_CURR").agg(
        CCT_ALL_UTIL_MEAN=("UTIL", "mean"),
        CCT_ALL_UTIL_MAX=("UTIL", "max"),
        CCT_ALL_DPD_MEAN=("SK_DPD", "mean"))

    recent = df[df["MONTHS_BALANCE"] >= -6].groupby("SK_ID_CURR").agg(
        CCT_6M_UTIL_MEAN=("UTIL", "mean"),
        CCT_6M_UTIL_MAX=("UTIL", "max"),
        CCT_6M_DRAWINGS_MEAN=("AMT_DRAWINGS_CURRENT", "mean"),
        CCT_6M_DPD_MAX=("SK_DPD", "max"))

    agg = base.join(recent, how="left")
    # 利用率が上がってきている = 資金繰りが厳しくなっている
    agg["CCT_UTIL_TREND"] = agg["CCT_6M_UTIL_MEAN"] - agg["CCT_ALL_UTIL_MEAN"]

    del df, base, recent
    gc.collect()
    return agg


# =============================================================================
# [B] bureau の時間窓（直近の新規借入は強い危険シグナル）
# =============================================================================
def bureau_window_features():
    df = pd.read_csv(f"{DATA_DIR}/bureau.csv",
                     usecols=["SK_ID_CURR", "DAYS_CREDIT", "AMT_CREDIT_SUM",
                              "AMT_CREDIT_SUM_DEBT", "CREDIT_ACTIVE"])
    out = []
    for days, tag in [(365, "1Y"), (730, "2Y")]:
        sub = df[df["DAYS_CREDIT"] >= -days]
        a = sub.groupby("SK_ID_CURR").agg(
            **{f"BUROW{tag}_COUNT": ("DAYS_CREDIT", "size"),
               f"BUROW{tag}_AMT_SUM": ("AMT_CREDIT_SUM", "sum"),
               f"BUROW{tag}_DEBT_SUM": ("AMT_CREDIT_SUM_DEBT", "sum")})
        out.append(a)

    # 最後にローンを組んでからの日数、借入間隔の平均
    df = df.sort_values(["SK_ID_CURR", "DAYS_CREDIT"])
    df["CREDIT_GAP"] = df.groupby("SK_ID_CURR")["DAYS_CREDIT"].diff()
    a = df.groupby("SK_ID_CURR").agg(
        BURO_DAYS_LAST_CREDIT=("DAYS_CREDIT", "max"),
        BURO_CREDIT_GAP_MEAN=("CREDIT_GAP", "mean"),
        BURO_CREDIT_GAP_MIN=("CREDIT_GAP", "min"))
    out.append(a)

    agg = pd.concat(out, axis=1)
    del df, out
    gc.collect()
    return agg


# =============================================================================
# [E] グループ統計量（application 内、ターゲット不使用）
#     「同じ職業の中で収入が低い」等の相対情報。GBDT は絶対値の分割は得意だが
#     グループ内相対位置は明示的に与えたほうが効く。
# =============================================================================
def add_group_stats(df):
    specs = [
        ("OCCUPATION_TYPE", "AMT_INCOME_TOTAL"),
        ("ORGANIZATION_TYPE", "AMT_INCOME_TOTAL"),
        ("NAME_EDUCATION_TYPE", "AMT_INCOME_TOTAL"),
        ("OCCUPATION_TYPE", "EXT_SOURCE_2"),
        ("OCCUPATION_TYPE", "AMT_CREDIT"),
        ("NAME_EDUCATION_TYPE", "AMT_CREDIT"),
        ("ORGANIZATION_TYPE", "DAYS_EMPLOYED"),
    ]
    # one-hot 前のカラムを raw から復元する必要があるため、
    # この関数は preprocess_application の one_hot 前に呼ぶ設計にしてある
    for group, target in specs:
        if group not in df.columns or target not in df.columns:
            continue
        med = df.groupby(group)[target].transform("median")
        std = df.groupby(group)[target].transform("std")
        df[f"NEW_GRP_{target}_BY_{group}_RATIO"] = df[target] / (med + 1e-6)
        df[f"NEW_GRP_{target}_BY_{group}_Z"] = (df[target] - med) / (std + 1e-6)
    # ★ 断片化対策: 逐次列追加で分断された内部ブロックを連続領域に再配置
    return df.copy()


# =============================================================================
# [F] Null Importance 特徴量選択
#     ターゲットをシャッフルして得た importance の分布と、実ターゲットでの
#     importance を比較し、シャッフルに勝てない特徴量（=ノイズ）を落とす。
# =============================================================================
def null_importance_selection(train, feats, n_runs=NULL_IMP_RUNS,
                              sample_frac=NULL_IMP_SAMPLE):
    """★ 高速化版（結果のセマンティクスは従来と同一: split importance を
    実ターゲット vs シャッフルターゲットの75パーセンタイルで比較）。

    従来はsklearnラッパーで毎回 fit しており、31回の学習それぞれで
    (1) 2GB超のDataFrame→内部表現コピー (2) 特徴量ビニングの再計算 が発生し
    WSL2ではスワップも誘発して実測 466s/run（計約3.9時間）かかっていた。

    対策:
      - X を float32 の numpy 配列に一度だけ変換（メモリ半減・コピー排除）
      - lgb.Dataset を1回だけ構築し set_label() でラベルだけ差し替え
        → ビニング再計算を31回→1回に削減
      - HC_NULLIMP_SAMPLE<1.0 で actual/null 両方に同一の行サブサンプルを適用可
    """
    y_full = train["TARGET"].values.astype(np.float32)

    rng = np.random.RandomState(SEED)
    if sample_frac < 1.0:
        idx = rng.choice(len(y_full), int(len(y_full) * sample_frac),
                         replace=False)
        X = train[feats].iloc[idx].to_numpy(dtype=np.float32)
        y = y_full[idx]
        print(f"[null importance] row subsample: {len(y)} / {len(y_full)}")
    else:
        X = train[feats].to_numpy(dtype=np.float32)
        y = y_full

    params = {"objective": "binary", "verbosity": -1,
              "learning_rate": 0.05, "num_leaves": 63,
              "feature_fraction": 0.5, "bagging_fraction": 0.8,
              "bagging_freq": 1, "num_threads": os.cpu_count()}

    dtrain = lgb.Dataset(X, label=y, free_raw_data=False)

    def get_imp(target, seed):
        dtrain.set_label(target)
        booster = lgb.train({**params, "seed": seed}, dtrain,
                            num_boost_round=300)
        return booster.feature_importance("split").astype(float)

    with timer("null_importance: actual"):
        actual = get_imp(y, SEED)

    null_imps = np.zeros((n_runs, len(feats)))
    with timer(f"null_importance: {n_runs} shuffled runs"):
        for i in range(n_runs):
            null_imps[i] = get_imp(rng.permutation(y), SEED + i)
            if i % 5 == 0:
                gc.collect()  # WSL2 のメモリ逼迫対策: 定期的に明示解放

    del dtrain, X
    gc.collect()

    # 実 importance が null 分布の 75 パーセンタイルを超えない特徴量を除去
    threshold = np.percentile(null_imps, 75, axis=0)
    keep = [f for f, a, t in zip(feats, actual, threshold) if a > t]
    dropped = len(feats) - len(keep)
    print(f"[null importance] kept {len(keep)} / dropped {dropped}")
    return keep


# =============================================================================
# [G] スタッキング
# =============================================================================
def stack_predictions(oofs, preds, y):
    """OOF を特徴量にしたロジスティック回帰メタモデル。
    重みブレンドと比較して OOF AUC が高いほうを採用。"""
    oof_rank = np.column_stack([pd.Series(o).rank(pct=True) for o in oofs])
    pred_rank = np.column_stack([pd.Series(p).rank(pct=True) for p in preds])

    # メタモデル自体も CV で OOF 評価（メタレベルのリークを防ぐ）
    meta_oof = np.zeros(len(y))
    meta_pred = np.zeros(pred_rank.shape[0])
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + 1)
    for tr_idx, va_idx in skf.split(oof_rank, y):
        meta = LogisticRegression(C=1.0, max_iter=1000)
        meta.fit(oof_rank[tr_idx], y[tr_idx])
        meta_oof[va_idx] = meta.predict_proba(oof_rank[va_idx])[:, 1]
        meta_pred += meta.predict_proba(pred_rank)[:, 1] / N_FOLDS

    stack_auc = roc_auc_score(y, meta_oof)
    print(f"[stacking] LogReg meta OOF AUC = {stack_auc:.5f}")

    # 重みブレンドと比較
    weights = optimize_weights(oofs, y)
    blend_oof = sum(w * pd.Series(o).rank(pct=True).values
                    for w, o in zip(weights, oofs))
    blend_auc = roc_auc_score(y, blend_oof)

    if stack_auc >= blend_auc:
        print(f"[ensemble] stacking wins ({stack_auc:.5f} vs {blend_auc:.5f})")
        return meta_pred
    print(f"[ensemble] weight blend wins ({blend_auc:.5f} vs {stack_auc:.5f})")
    return sum(w * pd.Series(p).rank(pct=True).values
               for w, p in zip(weights, preds))


# =============================================================================
# データセット構築 v2
# =============================================================================
def build_dataset_v2():
    # application はグループ統計を one-hot 前に差し込むため raw から作り直す
    with timer("application(+group stats)"):
        train = pd.read_csv(f"{DATA_DIR}/application_train.csv")
        test = pd.read_csv(f"{DATA_DIR}/application_test.csv")
        raw = pd.concat([train, test], ignore_index=True)
        raw = add_group_stats(raw)
        grp_cols = [c for c in raw.columns if c.startswith("NEW_GRP_")]
        grp = raw[["SK_ID_CURR"] + grp_cols]
        del train, test, raw
        gc.collect()

        df = preprocess_application()
        df = df.merge(grp, on="SK_ID_CURR", how="left")
        del grp

    with timer("bureau"):
        df = df.merge(preprocess_bureau(), on="SK_ID_CURR", how="left")
    with timer("bureau windows"):
        df = df.merge(bureau_window_features(), on="SK_ID_CURR", how="left")
    with timer("previous_application"):
        df = df.merge(preprocess_prev(), on="SK_ID_CURR", how="left")
    with timer("prev interest"):
        df = df.merge(prev_interest_features(), on="SK_ID_CURR", how="left")
    with timer("prev windows/last-k"):
        df = df.merge(prev_window_features(), on="SK_ID_CURR", how="left")
    with timer("pos_cash"):
        df = df.merge(preprocess_pos(), on="SK_ID_CURR", how="left")
    with timer("installments"):
        df = df.merge(preprocess_installments(), on="SK_ID_CURR", how="left")
    with timer("installments windows/trends"):
        df = df.merge(installments_window_trend(), on="SK_ID_CURR", how="left")
    with timer("credit_card"):
        df = df.merge(preprocess_credit_card(), on="SK_ID_CURR", how="left")
    with timer("credit_card trends"):
        df = df.merge(credit_card_trend(), on="SK_ID_CURR", how="left")

    # テーブル横断の相互作用
    df["NEW_INSTAL_DPD_X_EXT_MEAN"] = df["INSTAL_DPD_MEAN"] * (1 - df["NEW_EXTSOURCE_MEAN"])
    df["NEW_DEBT_INCOME_RATIO"] = df["BURO_AMT_CREDIT_SUM_DEBT_SUM"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["NEW_TOTAL_CREDIT_INCOME"] = (df["AMT_CREDIT"] + df["BURO_AMT_CREDIT_SUM_DEBT_SUM"].fillna(0)) / (df["AMT_INCOME_TOTAL"] + 1)
    # 過去の金利 × 現在の返済負担: 高金利でしか借りられない人 × 重い負担
    df["NEW_INT_X_ANNUITY_RATIO"] = df["PREVINT_INT_ANNUAL_MEAN"] * df["NEW_ANNUITY_INCOME_RATIO"]
    # 直近悪化トレンド × EXT スコアの低さ
    df["NEW_TREND_X_EXT"] = df["INS_TREND_LATE_1Y"] * (1 - df["NEW_EXTSOURCE_MEAN"])

    df = df.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x))
    nunique = df.nunique()
    df.drop(nunique[nunique <= 1].index, axis=1, inplace=True)

    train = df[df["TARGET"].notnull()].reset_index(drop=True)
    test = df[df["TARGET"].isnull()].reset_index(drop=True)
    del df
    gc.collect()

    # ★ メモリ節約: 大量の df[col]=... 代入で内部的に断片化したブロックを
    #   1回の copy() で連続領域に再配置する（fragmented frame 警告の対策）。
    #   さらに float64 → float32 にダウンキャストしてメモリを概ね半減させる。
    train = train.copy()
    test = test.copy()
    for df_ in (train, test):
        float_cols = df_.select_dtypes("float64").columns
        df_[float_cols] = df_[float_cols].astype("float32")
    gc.collect()

    feats = [c for c in train.columns if c not in
             ["TARGET", "SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV", "index"]]
    print(f"train: {train.shape}, test: {test.shape}, n_features: {len(feats)}")
    print(f"memory: train={train.memory_usage(deep=True).sum()/1e9:.2f}GB, "
          f"test={test.memory_usage(deep=True).sum()/1e9:.2f}GB")
    return train, test, feats


# =============================================================================
# main
# =============================================================================
def main():
    train, test, feats = build_dataset_v2()

    # --- [F] Null importance でノイズ特徴量を枝刈り ---
    if USE_NULL_IMPORTANCE:
        feats = null_importance_selection(train, feats)

    # --- Optuna（LGBM）---
    with timer("optuna_lgb"):
        best_lgb = tune_lgb(train, feats)

    # --- LGBM: シード平均つき 5-fold OOF ---
    oof_lgb = np.zeros(len(train))
    pred_lgb = np.zeros(len(test))
    imp_all = pd.DataFrame()
    for s in SEED_AVERAGING:
        params = dict(best_lgb)
        with timer(f"lgb_cv seed={s}"):
            # kfold 内の分割 seed は固定、モデル seed だけ変える
            def fn(tr_x, tr_y, va_x, va_y, _s=s, _p=params):
                p = {"objective": "binary", "metric": "auc", "verbosity": -1,
                     "n_estimators": 20000, "learning_rate": 0.01,
                     "subsample_freq": 1, "random_state": _s, "n_jobs": -1, **_p}
                m = lgb.LGBMClassifier(**p)
                m.fit(tr_x, tr_y, eval_set=[(va_x, va_y)], eval_metric="auc",
                      callbacks=[lgb.early_stopping(200, verbose=False)])
                return m
            o, p_, imp = kfold_train(train, test, feats, fn, f"LGB(s{s})")
            oof_lgb += o / len(SEED_AVERAGING)
            pred_lgb += p_ / len(SEED_AVERAGING)
            imp_all = pd.concat([imp_all, imp])
    print(f"[LGB seed-avg] OOF AUC = {roc_auc_score(train['TARGET'], oof_lgb):.5f}")

    # --- XGB / CatBoost ---
    with timer("xgb_cv"):
        oof_xgb, pred_xgb, _ = kfold_train(train, test, feats, make_xgb_fn(best_lgb), "XGB")
    with timer("cat_cv"):
        oof_cat, pred_cat, _ = kfold_train(train, test, feats, make_cat_fn(), "CAT")

    # --- Feature importance 保存 ---
    (imp_all.groupby("feature")["importance"].mean()
        .sort_values(ascending=False).head(80)
        .to_csv("feature_importance_top80.csv"))

    # --- [G] スタッキング vs 重みブレンド ---
    y = train["TARGET"].values
    final_pred = stack_predictions(
        [oof_lgb, oof_xgb, oof_cat],
        [pred_lgb, pred_xgb, pred_cat], y)

    submission = test[["SK_ID_CURR"]].copy()
    submission["TARGET"] = final_pred
    submission.to_csv("submission.csv", index=False)

    pd.DataFrame({"SK_ID_CURR": train["SK_ID_CURR"], "oof_lgb": oof_lgb,
                  "oof_xgb": oof_xgb, "oof_cat": oof_cat, "target": y}
                 ).to_csv("oof_predictions.csv", index=False)
    print("done.")


if __name__ == "__main__":
    main()
