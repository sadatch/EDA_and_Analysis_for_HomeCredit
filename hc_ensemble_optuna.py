# =============================================================================
# Home Credit Default Risk
# LightGBM + XGBoost + CatBoost アンサンブル（Optuna チューニング付き）
#
# MyHomeCreditDefaultRisk_1〜7.ipynb の特徴量エンジニアリングをベースに:
#   - nb7 のバグ修正（STATUS_SCORE の ^ 演算子 / bureau 二重マージ /
#     NAME_EDUCATION_TYPE の ORGANIZATION_TYPE 誤参照）
#   - installments_payments / credit_card_balance の本格的な集計を追加
#     （nb7 では COUNT のみだった最大の伸びしろ）
#   - ホールドアウト → StratifiedKFold 5-fold OOF に変更
#   - Optuna で LightGBM をチューニング → XGB / CatBoost にも展開
#   - OOF に対する重み最適化でアンサンブル
# =============================================================================

import gc
import os
import re
import time
import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import optuna
from scipy.optimize import minimize

warnings.simplefilter(action="ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# -----------------------------------------------------------------------------
# 設定
# -----------------------------------------------------------------------------
# 修正: このプロジェクトの実際のraw配置は ./data/raw (config.pyのRAW_DIRと同じ)。
# 元の "../input/home-credit-default-risk" は存在しないパスで、hc_campaign.py
# 経由のどのstageも FileNotFoundError で即死する状態だったため修正。
DATA_DIR = os.environ.get("HC_DATA_DIR", "./data/raw")  # 環境変数で上書き可（スモークテスト用）
N_FOLDS = 5
SEED = 42
N_TRIALS_LGB = 50          # Optuna 試行回数（時間がなければ 20 程度に）
TUNE_SAMPLE_FRAC = 0.35    # チューニング時は行をサンプリングして高速化
# 修正: 現行パイプラインの本番runでXGBoost(GPU)がVRAM不足で17時間ハングした実績があるため、
# このセカンドパイプラインはデフォルトで安全なCPU側に倒す。GPUが必要なら明示的にTrueへ。
USE_GPU = False            # RTX 3070 Ti は8GB VRAM。特徴量が多いとGPU hist法がVRAM超過で
                           # ページング/ハングする実績があるため、まずはCPUで様子見を推奨


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] {time.time() - t0:.1f}s")


def one_hot(df, nan_as_category=True):
    """カテゴリ変数をダミー化し、新規カラム名を返す（nb7 の定型処理を関数化）"""
    original = list(df.columns)
    # 修正: dtype=="object"だけだとpandasがCSV文字列列をArrow-backedの
    # string dtypeで読み込む環境（本環境で発生）を取りこぼし、後段のgroupby.agg(mean)等が
    # "dtype 'str' does not support operation 'mean'" で落ちる。is_string_dtypeも含めて判定。
    cat_cols = [c for c in df.columns
                if df[c].dtype == "object" or pd.api.types.is_string_dtype(df[c])]
    df = pd.get_dummies(df, columns=cat_cols, dummy_na=nan_as_category)
    new_cols = [c for c in df.columns if c not in original]
    return df, new_cols


# =============================================================================
# 1. application_train / test（nb5〜7 の集大成 + バグ修正）
# =============================================================================
def preprocess_application():
    train = pd.read_csv(f"{DATA_DIR}/application_train.csv")
    test = pd.read_csv(f"{DATA_DIR}/application_test.csv")
    df = pd.concat([train, test], ignore_index=True)  # train/test でカラムを揃える（nb2 の head(1).append より安全）
    del train, test

    # --- 欠損・異常値対応（nb2〜） ---
    df.loc[df["CODE_GENDER"] == "XNA", "CODE_GENDER"] = "F"
    df.loc[df["NAME_FAMILY_STATUS"] == "Unknown", "NAME_FAMILY_STATUS"] = "Married"
    # 修正: df[col].replace(..., inplace=True)は連鎖代入でCopy-on-Write環境では
    # 実際にはdfを更新しないサイレントなno-opになる(ChainedAssignmentError警告が出るだけ)。
    # 365243センチネル→NaN変換が効かないまま学習される致命的なバグだったため代入形に修正。
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
    # 異常値だったこと自体を特徴量として残す（missingness シグナル）
    df["NEW_DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"].isnull()).astype(int)

    # --- カテゴリ統合（nb4） ---
    df.loc[df["NAME_INCOME_TYPE"] == "Businessman", "NAME_INCOME_TYPE"] = "Commercial associate"
    df.loc[df["NAME_INCOME_TYPE"] == "Maternity leave", "NAME_INCOME_TYPE"] = "Pensioner"
    df.loc[df["NAME_INCOME_TYPE"] == "Student", "NAME_INCOME_TYPE"] = "State servant"
    df.loc[df["NAME_INCOME_TYPE"] == "Unemployed", "NAME_INCOME_TYPE"] = "Pensioner"

    org = df["ORGANIZATION_TYPE"]
    df["ORGANIZATION_TYPE"] = np.select(
        [
            org.str.contains("Business Entity", na=False),
            org.str.contains("Industry", na=False),
            org.str.contains("Trade", na=False),
            org.str.contains("Transport", na=False),
            org.isin(["School", "Kindergarten", "University"]),
            org.isin(["Emergency", "Police", "Medicine", "Postal", "Military",
                      "Security Ministries", "Legal Services", "Goverment"]),
            org.isin(["Bank", "Insurance"]),
            org.isin(["Realtor", "Housing"]),
            org.isin(["Hotel", "Restaurant", "Services"]),
            org.isin(["Cleaning", "Electricity", "Telecom", "Mobile",
                      "Advertising", "Religion", "Culture"]),
        ],
        ["Business_Entity", "Industry", "Trade", "Transport", "Education",
         "Official", "Finance", "Realty", "TourismFoodSector", "Other"],
        default=org,
    )

    occ = df["OCCUPATION_TYPE"]
    df["OCCUPATION_TYPE"] = np.select(
        [
            occ.isin(["Low-skill Laborers", "Cooking staff", "Security staff",
                      "Private service staff", "Cleaning staff", "Waiters/barmen staff"]),
            occ.isin(["IT staff", "High skill tech staff"]),
            occ.isin(["Secretaries", "HR staff", "Realty agents"]),
        ],
        ["Low_skill_staff", "High_skill_staff", "Others"],
        default=occ,
    )

    # NAME_TYPE_SUITE: 1% 未満は Rare に（nb4）
    tmp = df["NAME_TYPE_SUITE"].value_counts() / len(df)
    df.loc[df["NAME_TYPE_SUITE"].isin(tmp[tmp < 0.01].index), "NAME_TYPE_SUITE"] = "Rare"

    # ★ nb4-7 バグ修正: 元コードは NAME_EDUCATION_TYPE に ORGANIZATION_TYPE の値を
    #   代入してしまっていた（np.where の第3引数ミス）。学歴情報が消えていた。
    df.loc[df["NAME_EDUCATION_TYPE"] == "Academic degree", "NAME_EDUCATION_TYPE"] = "Higher education"

    # 二値カテゴリのラベルエンコーディング（nb4）
    for c in ["NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY"]:
        df[c], _ = pd.factorize(df[c])

    # 地域不一致フラグ・提出書類フラグの合算（nb4）
    reg_cols = ["REG_REGION_NOT_LIVE_REGION", "REG_REGION_NOT_WORK_REGION",
                "LIVE_REGION_NOT_WORK_REGION", "REG_CITY_NOT_LIVE_CITY",
                "REG_CITY_NOT_WORK_CITY", "LIVE_CITY_NOT_WORK_CITY"]
    df["NEW_REGION"] = df[reg_cols].sum(axis=1)
    df.drop(reg_cols, axis=1, inplace=True)

    doc_cols = [c for c in df.columns if "FLAG_DOC" in c]
    df["NEW_DOCUMENT"] = df[doc_cols].sum(axis=1)
    df.drop(doc_cols, axis=1, inplace=True)

    # --- 計算特徴量（nb3〜） ---
    df["NEW_DAYS_EMPLOYED_RATIO"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
    df["NEW_INCOME_CREDIT_RATIO"] = df["AMT_INCOME_TOTAL"] / df["AMT_CREDIT"]
    df["NEW_INCOME_PER_RATIO"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]
    df["NEW_ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["NEW_PAYMENT_RATIO"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]
    df["NEW_GOODS_CREDIT_RATIO"] = df["AMT_GOODS_PRICE"] / df["AMT_CREDIT"]
    df["NEW_GOODS_CREDIT_DIFF"] = df["AMT_GOODS_PRICE"] - df["AMT_CREDIT"]
    df["NEW_GOODS_CREDIT_DIFF_RATIO"] = df["NEW_GOODS_CREDIT_DIFF"] / df["AMT_INCOME_TOTAL"]
    df["NEW_INCOME_BIRTH_RATIO"] = df["AMT_INCOME_TOTAL"] / df["DAYS_BIRTH"]
    df["NEW_DAYS_BIRTH"] = round(df["DAYS_BIRTH"] * -1 / 365)
    # 修正: リボルビングローンはAMT_ANNUITY=0になるケースがあり、無epsilonの除算だと
    # infが生じてXGBoostがQuantileDMatrix生成時にエラー落ちする(LightGBMはinfを許容するため
    # 気づかれずにいた)。epsilonガードを追加。
    df["NEW_CREDIT_TERM"] = df["AMT_CREDIT"] / (df["AMT_ANNUITY"] + 1e-5)  # 返済年数の逆視点

    # EXT_SOURCE 系（nb3 + 拡張: 1st-place 系解法で最重要ブロック）
    ext = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
    df["NEW_EXTSOURCE_MEAN"] = ext.mean(axis=1)
    df["NEW_EXTSOURCES_WPOINT"] = ext.prod(axis=1)          # 積（欠損があると NaN）
    df["NEW_EXTSOURCE_STD"] = ext.std(axis=1)
    df["NEW_EXTSOURCE_MIN"] = ext.min(axis=1)
    df["NEW_EXTSOURCE_MAX"] = ext.max(axis=1)
    df["NEW_EXTSOURCE_NA_CNT"] = ext.isnull().sum(axis=1)
    df["NEW_EXT2_X_EXT3"] = df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
    df["NEW_EXT_MEAN_X_DAYS_EMPLOYED_RATIO"] = df["NEW_EXTSOURCE_MEAN"] * df["NEW_DAYS_EMPLOYED_RATIO"]
    df["NEW_EXT3_DIV_BIRTH"] = df["EXT_SOURCE_3"] / (df["NEW_DAYS_BIRTH"] + 1)

    # 周期エンコーディング（nb3）
    weekday_dict = {"MONDAY": 1, "TUESDAY": 2, "WEDNESDAY": 3, "THURSDAY": 4,
                    "FRIDAY": 5, "SATURDAY": 6, "SUNDAY": 7}
    df["WEEKDAY_APPR_PROCESS_START"] = df["WEEKDAY_APPR_PROCESS_START"].map(weekday_dict)
    df["NEW_WEEKDAY_SIN"] = np.sin(2 * np.pi * df["WEEKDAY_APPR_PROCESS_START"] / 7)
    df["NEW_WEEKDAY_COS"] = np.cos(2 * np.pi * df["WEEKDAY_APPR_PROCESS_START"] / 7)
    df["NEW_HOUR_SIN"] = np.sin(2 * np.pi * df["HOUR_APPR_PROCESS_START"] / 24)
    df["NEW_HOUR_COS"] = np.cos(2 * np.pi * df["HOUR_APPR_PROCESS_START"] / 24)

    # ビン化（nb5）
    df["NEW_SEGMENT_AGE"] = pd.cut(df["NEW_DAYS_BIRTH"], bins=[0, 34, 54, 200],
                                   labels=["Young", "Middle_Age", "Old"]).astype(object)
    df["NEW_SEGMENT_INCOME"] = pd.cut(df["AMT_INCOME_TOTAL"], bins=[0, 112500, 225000, np.inf],
                                      labels=["Low", "Middle", "High"]).astype(object)
    df["NEW_DEF_30_60_SOCIAL_CIRCLE"] = (
        (df["DEF_30_CNT_SOCIAL_CIRCLE"].fillna(0) + df["DEF_60_CNT_SOCIAL_CIRCLE"].fillna(0)) > 0
    ).astype(int)

    # --- 不要変数の削除（nb4-5） ---
    drop_cols = ["FONDKAPREMONT_MODE", "WALLSMATERIAL_MODE", "HOUSETYPE_MODE",
                 "EMERGENCYSTATE_MODE", "FLAG_MOBIL", "FLAG_EMP_PHONE",
                 "FLAG_WORK_PHONE", "FLAG_CONT_MOBILE", "FLAG_EMAIL",
                 "OBS_30_CNT_SOCIAL_CIRCLE", "OBS_60_CNT_SOCIAL_CIRCLE"]
    df.drop(drop_cols, axis=1, inplace=True)

    df, _ = one_hot(df, nan_as_category=False)
    gc.collect()
    return df


# =============================================================================
# 2. bureau + bureau_balance（nb7 ベース + バグ修正）
# =============================================================================
def preprocess_bureau():
    bb = pd.read_csv(f"{DATA_DIR}/bureau_balance.csv")
    bureau = pd.read_csv(f"{DATA_DIR}/bureau.csv")

    # bureau_balance の STATUS をダミー化して SK_ID_BUREAU 集計
    bb, bb_cat = one_hot(bb, nan_as_category=False)
    agg_list = {"MONTHS_BALANCE": ["min", "max", "size"]}
    for c in bb_cat:
        agg_list[c] = ["mean", "sum"]
    bb_agg = bb.groupby("SK_ID_BUREAU").agg(agg_list)
    bb_agg.columns = pd.Index([f"{a}_{b.upper()}" for a, b in bb_agg.columns])

    # ★ nb7 バグ修正: 元コードは `x ^ 2`（XOR）になっていた。延滞の重み付けスコアは累乗ではなく
    #   単純な重み付き和のほうが意図に沿うので、係数掛けの線形和で再定義。
    for s in ["1", "2", "3", "4", "5"]:
        col = f"STATUS_{s}_SUM"
        if col not in bb_agg.columns:
            bb_agg[col] = 0
    bb_agg["NEW_STATUS_SCORE"] = (bb_agg["STATUS_1_SUM"] * 1 + bb_agg["STATUS_2_SUM"] * 2 +
                                  bb_agg["STATUS_3_SUM"] * 3 + bb_agg["STATUS_4_SUM"] * 4 +
                                  bb_agg["STATUS_5_SUM"] * 5)

    bureau = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")
    bureau.drop("SK_ID_BUREAU", axis=1, inplace=True)
    del bb, bb_agg
    gc.collect()

    # カテゴリ整備（nb7）
    bureau.drop("CREDIT_CURRENCY", axis=1, inplace=True)
    # 修正: object dtypeだけでなくpandasのstring dtype(arrow-backed含む)も対象にする
    for col in bureau.select_dtypes(include=["object", "string"]).columns:
        tmp = bureau[col].value_counts() / len(bureau)
        rare = tmp[tmp < 0.2].index
        bureau[col] = np.where(bureau[col].isin(rare), "Rare", bureau[col])
    bureau["CREDIT_ACTIVE"] = bureau["CREDIT_ACTIVE"].replace("Rare", "Active")

    # 計算特徴量（nb7）
    bureau["NEW_EARLY_ACTIVE"] = ((bureau["CREDIT_ACTIVE"] == "Active") &
                                  (bureau["DAYS_CREDIT_ENDDATE"] < 0)).astype(int)
    bureau["NEW_CNT_CREDIT_PROLONG_CAT"] = (bureau["CNT_CREDIT_PROLONG"] > 0).astype(int)
    bureau["NEW_DEBT_RATIO"] = bureau["AMT_CREDIT_SUM_DEBT"] / (bureau["AMT_CREDIT_SUM"] + 1)
    bureau["NEW_OVERDUE_RATIO"] = bureau["AMT_CREDIT_SUM_OVERDUE"] / (bureau["AMT_CREDIT_SUM"] + 1)

    loan_types = bureau.groupby("SK_ID_CURR")["CREDIT_TYPE"].nunique().rename("NEW_BUREAU_LOAN_TYPES")

    bureau, bureau_cat = one_hot(bureau, nan_as_category=False)

    num_agg = {
        "DAYS_CREDIT": ["min", "max", "mean", "var"],
        "DAYS_CREDIT_ENDDATE": ["min", "max", "mean"],
        "DAYS_CREDIT_UPDATE": ["mean"],
        "CREDIT_DAY_OVERDUE": ["max", "mean"],
        "DAYS_ENDDATE_FACT": ["min", "max", "mean"],
        "AMT_CREDIT_MAX_OVERDUE": ["mean", "max"],
        "AMT_CREDIT_SUM": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_DEBT": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_OVERDUE": ["mean", "sum"],
        "AMT_CREDIT_SUM_LIMIT": ["mean", "sum"],
        "AMT_ANNUITY": ["max", "mean"],
        "CNT_CREDIT_PROLONG": ["sum"],
        "MONTHS_BALANCE_MIN": ["min"],
        "MONTHS_BALANCE_MAX": ["max"],
        "MONTHS_BALANCE_SIZE": ["mean", "sum"],
        "NEW_STATUS_SCORE": ["min", "mean", "max", "sum"],
        "NEW_DEBT_RATIO": ["min", "max", "mean"],
        "NEW_OVERDUE_RATIO": ["max", "mean"],
        "NEW_EARLY_ACTIVE": ["mean", "sum"],
        "NEW_CNT_CREDIT_PROLONG_CAT": ["mean"],
    }
    cat_agg = {c: ["mean"] for c in bureau_cat}

    agg = bureau.groupby("SK_ID_CURR").agg({**num_agg, **cat_agg})
    agg.columns = pd.Index([f"BURO_{a}_{b.upper()}" for a, b in agg.columns])
    agg["BURO_COUNT"] = bureau.groupby("SK_ID_CURR").size()
    agg = agg.join(loan_types, how="left")

    # Active / Closed 限定集計（nb7）
    for flag, prefix in [("CREDIT_ACTIVE_Active", "ACTIVE"), ("CREDIT_ACTIVE_Closed", "CLOSED")]:
        if flag in bureau.columns:
            sub = bureau[bureau[flag] == 1].groupby("SK_ID_CURR").agg(num_agg)
            sub.columns = pd.Index([f"{prefix}_{a}_{b.upper()}" for a, b in sub.columns])
            agg = agg.join(sub, how="left")
            del sub

    del bureau
    gc.collect()
    return agg


# =============================================================================
# 3. previous_application（nb7 ベース）
# =============================================================================
def preprocess_prev():
    df = pd.read_csv(f"{DATA_DIR}/previous_application.csv")

    # 修正: object dtypeだけでなくpandasのstring dtype(arrow-backed含む)も対象にする
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df.loc[df[col].isin(["XNA", "XAP"]), col] = np.nan
        tmp = df[col].value_counts() / len(df)
        rare = tmp[tmp < 0.01].index
        df[col] = np.where(df[col].isin(rare), "Rare", df[col])

    for c in ["DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION",
              "DAYS_LAST_DUE", "DAYS_TERMINATION"]:
        # 修正: 連鎖代入のinplace=TrueはCopy-on-Write環境でno-opになるため代入形に変更
        df[c] = df[c].replace(365243, np.nan)

    df.drop(["RATE_INTEREST_PRIMARY", "RATE_INTEREST_PRIVILEGED",
             "NAME_CASH_LOAN_PURPOSE", "CODE_REJECT_REASON",
             "FLAG_LAST_APPL_PER_CONTRACT", "NFLAG_LAST_APPL_IN_DAY",
             "SELLERPLACE_AREA"], axis=1, inplace=True)

    # 計算特徴量（nb7）
    # 修正: AMT_ANNUITY/AMT_GOODS_PRICEが0またはNaNのケースでinfが発生しXGBoostが
    # エラー落ちしていたためepsilonガードを追加。
    df["NEW_AMT_CREDIT_RATIO"] = df["AMT_APPLICATION"] / (df["AMT_CREDIT"] + 1e-5)
    df["NEW_HOW_PAID_YEARS"] = df["AMT_CREDIT"] / (df["AMT_ANNUITY"] + 1e-5)
    df["NEW_GOODS_RATIO"] = df["AMT_APPLICATION"] / (df["AMT_GOODS_PRICE"] + 1e-5)
    df["NEW_LATE_DAYS"] = df["DAYS_LAST_DUE_1ST_VERSION"] - df["DAYS_FIRST_DUE"]
    df["NEW_FLAG_LATE_DAYS"] = ((df["DAYS_LAST_DUE_1ST_VERSION"] - df["DAYS_LAST_DUE"]) >= 0).astype(float)
    df["NEW_DOWN_PAYMENT_RATIO"] = df["AMT_DOWN_PAYMENT"] / (df["AMT_APPLICATION"] + 1)

    df, cat_cols = one_hot(df, nan_as_category=True)

    num_list = [c for c in df.columns
                if c not in cat_cols + ["SK_ID_CURR", "SK_ID_PREV"]]
    num_agg = {c: ["min", "max", "mean"] for c in num_list}
    cat_agg = {c: ["mean"] for c in cat_cols}

    agg = df.groupby("SK_ID_CURR").agg({**num_agg, **cat_agg})
    agg.columns = pd.Index([f"PREV_{a}_{b.upper()}" for a, b in agg.columns])
    agg["PREV_COUNT"] = df.groupby("SK_ID_CURR").size()

    # Approved / Refused 限定集計（nb7）
    for flag, prefix in [("NAME_CONTRACT_STATUS_Approved", "APPROVED"),
                         ("NAME_CONTRACT_STATUS_Refused", "REFUSED")]:
        if flag in df.columns:
            sub = df[df[flag] == 1].groupby("SK_ID_CURR").agg(num_agg)
            sub.columns = pd.Index([f"{prefix}_{a}_{b.upper()}" for a, b in sub.columns])
            agg = agg.join(sub, how="left")
            del sub

    # 却下率（Refused / 総申込数）は強い行動シグナル
    if "PREV_NAME_CONTRACT_STATUS_Refused_MEAN" in agg.columns:
        agg["NEW_REFUSED_RATIO"] = agg["PREV_NAME_CONTRACT_STATUS_Refused_MEAN"]

    del df
    gc.collect()
    return agg


# =============================================================================
# 4. POS_CASH_balance（nb7 ベース + 直近性の追加）
# =============================================================================
def preprocess_pos():
    df = pd.read_csv(f"{DATA_DIR}/POS_CASH_balance.csv")
    df, cat_cols = one_hot(df, nan_as_category=True)

    aggregations = {
        "MONTHS_BALANCE": ["max", "mean", "size"],
        "CNT_INSTALMENT": ["max", "mean", "std", "min", "median"],
        "CNT_INSTALMENT_FUTURE": ["max", "mean", "sum", "min", "median", "std"],
        "SK_DPD": ["max", "mean", "sum"],
        "SK_DPD_DEF": ["max", "mean", "sum"],
    }
    for c in cat_cols:
        aggregations[c] = ["mean"]

    agg = df.groupby("SK_ID_CURR").agg(aggregations)
    agg.columns = pd.Index([f"POS_{a}_{b.upper()}" for a, b in agg.columns])
    agg["POS_COUNT"] = df.groupby("SK_ID_CURR").size()

    # 直近 12 ヶ月に限定した延滞集計（時系列の直近性を反映）
    recent = df[df["MONTHS_BALANCE"] >= -12]
    r_agg = recent.groupby("SK_ID_CURR").agg({"SK_DPD": ["max", "mean"],
                                              "SK_DPD_DEF": ["max", "mean"]})
    r_agg.columns = pd.Index([f"POS_RECENT_{a}_{b.upper()}" for a, b in r_agg.columns])
    agg = agg.join(r_agg, how="left")

    del df, recent, r_agg
    gc.collect()
    return agg


# =============================================================================
# 5. installments_payments
#    ★ nb7 では COUNT のみだった。ここが今回最大の伸びしろ。
#    実際の返済行動（遅延日数・支払額の過不足）はデフォルト予測に直結する。
# =============================================================================
def preprocess_installments():
    df = pd.read_csv(f"{DATA_DIR}/installments_payments.csv")

    # 支払比率と差額（予定 vs 実績）
    df["PAYMENT_RATIO"] = df["AMT_PAYMENT"] / df["AMT_INSTALMENT"]
    df["PAYMENT_DIFF"] = df["AMT_INSTALMENT"] - df["AMT_PAYMENT"]
    # 遅延日数 DPD（正なら遅延）と前倒し日数 DBD（正なら前倒し）
    df["DPD"] = (df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"]).clip(lower=0)
    df["DBD"] = (df["DAYS_INSTALMENT"] - df["DAYS_ENTRY_PAYMENT"]).clip(lower=0)
    df["LATE_FLAG"] = (df["DPD"] > 0).astype(int)
    df["UNDERPAID_FLAG"] = (df["PAYMENT_DIFF"] > 0).astype(int)

    aggregations = {
        "NUM_INSTALMENT_VERSION": ["nunique"],
        "DPD": ["max", "mean", "sum"],
        "DBD": ["max", "mean", "sum"],
        "PAYMENT_RATIO": ["max", "mean", "min", "std"],
        "PAYMENT_DIFF": ["max", "mean", "sum", "std"],
        "AMT_INSTALMENT": ["max", "mean", "sum"],
        "AMT_PAYMENT": ["min", "max", "mean", "sum"],
        "DAYS_ENTRY_PAYMENT": ["max", "mean", "min"],
        "LATE_FLAG": ["mean", "sum"],
        "UNDERPAID_FLAG": ["mean", "sum"],
    }
    agg = df.groupby("SK_ID_CURR").agg(aggregations)
    agg.columns = pd.Index([f"INSTAL_{a}_{b.upper()}" for a, b in agg.columns])
    agg["INSTAL_COUNT"] = df.groupby("SK_ID_CURR").size()

    # 直近 365 日の返済行動
    recent = df[df["DAYS_INSTALMENT"] >= -365]
    r_agg = recent.groupby("SK_ID_CURR").agg({"DPD": ["max", "mean"],
                                              "PAYMENT_RATIO": ["mean"],
                                              "LATE_FLAG": ["mean"]})
    r_agg.columns = pd.Index([f"INSTAL_RECENT_{a}_{b.upper()}" for a, b in r_agg.columns])
    agg = agg.join(r_agg, how="left")

    del df, recent, r_agg
    gc.collect()
    return agg


# =============================================================================
# 6. credit_card_balance
#    ★ nb7 では COUNT のみだった。リボ残高・利用率は強力な信用シグナル。
# =============================================================================
def preprocess_credit_card():
    df = pd.read_csv(f"{DATA_DIR}/credit_card_balance.csv")
    df.drop("SK_ID_PREV", axis=1, inplace=True)

    # 利用率（残高 / 限度額）: クレジットスコアリングの定番指標
    df["UTILIZATION"] = df["AMT_BALANCE"] / (df["AMT_CREDIT_LIMIT_ACTUAL"] + 1)
    df["DRAWING_RATIO"] = df["AMT_DRAWINGS_CURRENT"] / (df["AMT_CREDIT_LIMIT_ACTUAL"] + 1)
    # 最低支払額に対する実支払
    df["MIN_PAYMENT_RATIO"] = df["AMT_PAYMENT_TOTAL_CURRENT"] / (df["AMT_INST_MIN_REGULARITY"] + 1)

    df, cat_cols = one_hot(df, nan_as_category=True)

    num_cols = [c for c in df.columns if c not in cat_cols + ["SK_ID_CURR"]]
    aggregations = {c: ["min", "max", "mean", "var"] for c in num_cols}
    for c in cat_cols:
        aggregations[c] = ["mean"]

    agg = df.groupby("SK_ID_CURR").agg(aggregations)
    agg.columns = pd.Index([f"CC_{a}_{b.upper()}" for a, b in agg.columns])
    agg["CC_COUNT"] = df.groupby("SK_ID_CURR").size()

    del df
    gc.collect()
    return agg

# =============================================================================
# 7. データ統合
#    ★ nb6/7 バグ修正: bureau が 2 回マージされていた（_x/_y 重複カラムが発生し、
#      片方は LGBM に無視されずノイズとして学習されていた）。1 回に修正。
# =============================================================================
def build_dataset():
    with timer("application"):
        df = preprocess_application()
    with timer("bureau"):
        df = df.merge(preprocess_bureau(), on="SK_ID_CURR", how="left")
    with timer("previous_application"):
        df = df.merge(preprocess_prev(), on="SK_ID_CURR", how="left")
    with timer("pos_cash"):
        df = df.merge(preprocess_pos(), on="SK_ID_CURR", how="left")
    with timer("installments"):
        df = df.merge(preprocess_installments(), on="SK_ID_CURR", how="left")
    with timer("credit_card"):
        df = df.merge(preprocess_credit_card(), on="SK_ID_CURR", how="left")

    # テーブル横断の相互作用特徴量
    df["NEW_INSTAL_DPD_X_EXT_MEAN"] = df["INSTAL_DPD_MEAN"] * (1 - df["NEW_EXTSOURCE_MEAN"])
    df["NEW_DEBT_INCOME_RATIO"] = df["BURO_AMT_CREDIT_SUM_DEBT_SUM"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["NEW_TOTAL_CREDIT_INCOME"] = (df["AMT_CREDIT"] + df["BURO_AMT_CREDIT_SUM_DEBT_SUM"].fillna(0)) / (df["AMT_INCOME_TOTAL"] + 1)

    # カラム名整備（nb1 から共通）
    df = df.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x))

    # 定数カラムの削除
    nunique = df.nunique()
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        df.drop(const_cols, axis=1, inplace=True)
        print(f"dropped {len(const_cols)} constant cols")

    train = df[df["TARGET"].notnull()].reset_index(drop=True)
    test = df[df["TARGET"].isnull()].reset_index(drop=True)
    del df
    gc.collect()

    feats = [c for c in train.columns if c not in
             ["TARGET", "SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV", "index"]]
    print(f"train: {train.shape}, test: {test.shape}, n_features: {len(feats)}")
    return train, test, feats


# =============================================================================
# 8. Optuna チューニング（LightGBM）
#    - 行サンプリング + 3-fold で 1 trial を軽くする
#    - 探索空間は Home Credit 系で効くと知られる範囲に絞る
# =============================================================================
def tune_lgb(train, feats, n_trials=N_TRIALS_LGB):
    sample = train.sample(frac=TUNE_SAMPLE_FRAC, random_state=SEED)
    X, y = sample[feats], sample["TARGET"]

    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "n_estimators": 10000,
            "learning_rate": 0.02,   # 探索中は固定、本番は 0.01 に下げて木を増やす
            "num_leaves": trial.suggest_int("num_leaves", 16, 96),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 500),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-2, 60, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 0.8),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.1),
            "random_state": SEED,
            "n_jobs": -1,
        }
        if USE_GPU:
            params["device"] = "gpu"

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        aucs = []
        for tr_idx, va_idx in skf.split(X, y):
            model = lgb.LGBMClassifier(**params)
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx],
                      eval_set=[(X.iloc[va_idx], y.iloc[va_idx])],
                      eval_metric="auc",
                      callbacks=[lgb.early_stopping(100, verbose=False)])
            aucs.append(roc_auc_score(y.iloc[va_idx],
                                      model.predict_proba(X.iloc[va_idx])[:, 1]))
        return float(np.mean(aucs))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"[Optuna LGB] best AUC={study.best_value:.5f}")
    print(f"[Optuna LGB] best params={study.best_params}")
    return study.best_params


# =============================================================================
# 9. K-Fold 学習（3 モデル共通の OOF フレームワーク）
# =============================================================================
def kfold_train(train, test, feats, model_fn, name):
    """model_fn(tr_x, tr_y, va_x, va_y) -> fitted model with predict_proba"""
    oof = np.zeros(len(train))
    preds = np.zeros(len(test))
    importances = pd.DataFrame()

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train[feats], train["TARGET"])):
        tr_x, tr_y = train[feats].iloc[tr_idx], train["TARGET"].iloc[tr_idx]
        va_x, va_y = train[feats].iloc[va_idx], train["TARGET"].iloc[va_idx]

        model = model_fn(tr_x, tr_y, va_x, va_y)
        oof[va_idx] = model.predict_proba(va_x)[:, 1]
        preds += model.predict_proba(test[feats])[:, 1] / N_FOLDS

        if hasattr(model, "feature_importances_"):
            imp = pd.DataFrame({"feature": feats,
                                "importance": model.feature_importances_,
                                "fold": fold})
            importances = pd.concat([importances, imp])

        print(f"  [{name}] fold {fold}: AUC={roc_auc_score(va_y, oof[va_idx]):.5f}")
        del model, tr_x, tr_y, va_x, va_y
        gc.collect()

    total_auc = roc_auc_score(train["TARGET"], oof)
    print(f"[{name}] OOF AUC = {total_auc:.5f}")
    return oof, preds, importances


def make_lgb_fn(best_params):
    params = {
        "objective": "binary", "metric": "auc", "verbosity": -1,
        "n_estimators": 20000, "learning_rate": 0.01,
        "subsample_freq": 1, "random_state": SEED, "n_jobs": -1,
        **best_params,
    }
    if USE_GPU:
        params["device"] = "gpu"

    def fn(tr_x, tr_y, va_x, va_y):
        model = lgb.LGBMClassifier(**params)
        model.fit(tr_x, tr_y, eval_set=[(va_x, va_y)], eval_metric="auc",
                  callbacks=[lgb.early_stopping(200, verbose=False)])
        return model
    return fn


def make_xgb_fn(lgb_params):
    """LGBM の最適値を XGB の対応パラメータに写像（フル再探索より効率的）"""
    params = {
        "objective": "binary:logistic", "eval_metric": "auc",
        "n_estimators": 20000, "learning_rate": 0.01,
        "max_depth": min(lgb_params.get("max_depth", 8), 10),
        "min_child_weight": lgb_params.get("min_child_weight", 40),
        "subsample": lgb_params.get("subsample", 0.85),
        "colsample_bytree": lgb_params.get("colsample_bytree", 0.5),
        "reg_alpha": lgb_params.get("reg_alpha", 0.5),
        "reg_lambda": lgb_params.get("reg_lambda", 0.5),
        "gamma": lgb_params.get("min_split_gain", 0.02),
        "random_state": SEED, "n_jobs": -1,
        "tree_method": "hist",
        "early_stopping_rounds": 200,
    }
    if USE_GPU:
        params["device"] = "cuda"

    def fn(tr_x, tr_y, va_x, va_y):
        model = xgb.XGBClassifier(**params)
        model.fit(tr_x, tr_y, eval_set=[(va_x, va_y)], verbose=False)
        return model
    return fn


def make_cat_fn():
    params = {
        "iterations": 20000, "learning_rate": 0.02,
        "depth": 7, "l2_leaf_reg": 10,
        "eval_metric": "AUC", "random_seed": SEED,
        "od_type": "Iter", "od_wait": 200, "verbose": False,
        "allow_writing_files": False,
    }
    if USE_GPU:
        params["task_type"] = "GPU"

    def fn(tr_x, tr_y, va_x, va_y):
        model = CatBoostClassifier(**params)
        model.fit(tr_x, tr_y, eval_set=(va_x, va_y))
        return model
    return fn


# =============================================================================
# 10. アンサンブル（OOF 上での重み最適化）
# =============================================================================
def optimize_weights(oofs, y):
    """OOF AUC を最大化する凸結合の重みを scipy で求める（rank 平均後）"""
    ranked = [pd.Series(o).rank(pct=True).values for o in oofs]

    def neg_auc(w):
        w = np.abs(w) / np.abs(w).sum()
        blend = sum(wi * ri for wi, ri in zip(w, ranked))
        return -roc_auc_score(y, blend)

    n = len(oofs)
    best = None
    for init in [np.ones(n) / n, np.array([0.5, 0.3, 0.2])[:n]]:
        res = minimize(neg_auc, init, method="Nelder-Mead")
        if best is None or res.fun < best.fun:
            best = res
    w = np.abs(best.x) / np.abs(best.x).sum()
    print(f"[ensemble] weights={np.round(w, 3)}, OOF AUC={-best.fun:.5f}")
    return w


# =============================================================================
# main
# =============================================================================
def main():
    train, test, feats = build_dataset()

    # --- Optuna（LGBM のみフル探索、他モデルは写像 or 手動）---
    with timer("optuna_lgb"):
        best_lgb = tune_lgb(train, feats)

    # --- 3 モデルを 5-fold OOF で学習 ---
    with timer("lgb_cv"):
        oof_lgb, pred_lgb, imp = kfold_train(train, test, feats, make_lgb_fn(best_lgb), "LGB")
    with timer("xgb_cv"):
        oof_xgb, pred_xgb, _ = kfold_train(train, test, feats, make_xgb_fn(best_lgb), "XGB")
    with timer("cat_cv"):
        oof_cat, pred_cat, _ = kfold_train(train, test, feats, make_cat_fn(), "CAT")

    # --- Feature Importance 上位を保存 ---
    mean_imp = (imp.groupby("feature")["importance"].mean()
                   .sort_values(ascending=False))
    mean_imp.head(60).to_csv("feature_importance_top60.csv")
    print(mean_imp.head(30))

    # --- アンサンブル ---
    y = train["TARGET"].values
    weights = optimize_weights([oof_lgb, oof_xgb, oof_cat], y)
    ranked_preds = [pd.Series(p).rank(pct=True).values
                    for p in [pred_lgb, pred_xgb, pred_cat]]
    final_pred = sum(w * r for w, r in zip(weights, ranked_preds))

    # 単体 OOF との比較を出力
    for name, o in [("LGB", oof_lgb), ("XGB", oof_xgb), ("CAT", oof_cat)]:
        print(f"  {name} single OOF AUC = {roc_auc_score(y, o):.5f}")

    # --- 提出 ---
    submission = test[["SK_ID_CURR"]].copy()
    submission["TARGET"] = final_pred
    submission.to_csv("submission.csv", index=False)

    # OOF も保存（後で hillclimb や stacking に使い回せる）
    pd.DataFrame({"SK_ID_CURR": train["SK_ID_CURR"], "oof_lgb": oof_lgb,
                  "oof_xgb": oof_xgb, "oof_cat": oof_cat, "target": y}
                 ).to_csv("oof_predictions.csv", index=False)
    print("done: submission.csv / oof_predictions.csv / feature_importance_top60.csv")


if __name__ == "__main__":
    main()

