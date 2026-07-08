"""
追加特徴量（時系列トレンド・加重平均・申込ベロシティ・書類提出・周期エンコーディング・異常度）。

feature_engineering.py / domain_features.py / oof_features.py / extra_features.py で
既にカバーされていない「時間方向の変化率そのもの」「多様性」「頻度」を特徴量化する。
出典と位置づけ:

  TREND_*（groupby_slope）
      各サブテーブルの月次系列（bureau_balance/POS_CASH/credit_card/installments）に
      対して、線形回帰の傾きを解析的に算出（groupby.applyより大幅に高速）。
      既存の「期間別集約（全期間/6M/1Y/3M）」「直近スナップショット」は "水準" の違いしか
      見ないが、傾きは "変化の速さ・方向" を直接捉える。AmEx Default Prediction上位解法で
      多用された「時系列トレンド特徴」の考え方を踏襲。
  WMA_*（groupby_wma）
      直近ほど指数的に重みが大きい加重平均。Home Credit **1位**解法の
      "weighted moving average" 相当（単純な期間カットではなく連続的な時間減衰）。
  BUREAU_CREDIT_TYPE_*
      信用情報機関に登録された債務の「種類の多様性」。AMT_REQ_CREDIT_BUREAU_*（照会件数）
      とは別に、実際に成立した債務の種類数で見るクレジットハンガーの別シグナル。
  PREV_APP_INTERVAL_* / PREV_LAST3_REFUSED_RATIO
      Home Credit自身への過去申込の間隔（申込ベロシティ）と直近謝絶率。
  DOC_*
      本人確認書類の提出数。「提出書類パターンそのものが情報」という公開kernelの定石。
  HOUR_*/WEEKDAY_*
      申込時刻・曜日の周期エンコーディング（sin/cos）+ オフタイム申込×地域リスクの交互作用。
  ISO_ANOMALY_SCORE
      IsolationForestによる教師なし異常度スコア（target非依存）。extra_features.pyのk-means
      距離とは異なる「分割ベース」の異常検知を追加し、多様性を持たせる。

すべて「元になる列が存在するときだけ」作る（合成データ/実データのどちらでも落ちない）。
"""
import numpy as np
import pandas as pd

EPS = 1e-5


# =====================================================================
# 0. 汎用ヘルパ（ベクトル化: groupby.applyを使わず高速に計算）
# =====================================================================
def groupby_slope(df: pd.DataFrame, group_col: str, x_col: str, y_col: str) -> pd.Series:
    """
    group_colごとに y = a + slope*x の最小二乗傾きを解析的に求める。
    n<2 または分散0のグループは NaN。戻り値は group_col をindexとするSeries。
    """
    d = df[[group_col, x_col, y_col]].dropna()
    if d.empty:
        return pd.Series(dtype=np.float64)
    x = d[x_col].astype(np.float64)
    y = d[y_col].astype(np.float64)
    xy = x * y
    xx = x * x
    grp = d[group_col]

    n = grp.groupby(grp).size().astype(np.float64)
    sx = x.groupby(grp).sum()
    sy = y.groupby(grp).sum()
    sxy = xy.groupby(grp).sum()
    sxx = xx.groupby(grp).sum()

    denom = n * sxx - sx * sx
    slope = (n * sxy - sx * sy) / denom.replace(0.0, np.nan)
    slope[n < 2] = np.nan
    return slope


def groupby_wma(df: pd.DataFrame, group_col: str, x_col: str, y_col: str,
                 tau: float = 12.0) -> pd.Series:
    """
    group_colごとに直近（x_colが0に近い、通常はMONTHS_BALANCEのような負の経過月）ほど
    重みが指数的に大きい加重平均を求める。tauは減衰の緩さ（大きいほど過去も重視）。
    """
    d = df[[group_col, x_col, y_col]].dropna()
    if d.empty:
        return pd.Series(dtype=np.float64)
    x = d[x_col].astype(np.float64)
    y = d[y_col].astype(np.float64)
    grp = d[group_col]
    w = np.exp(x / tau)
    wy = w * y
    num = wy.groupby(grp).sum()
    den = w.groupby(grp).sum()
    return num / (den + EPS)


# =====================================================================
# 1. bureau_balance: STATUS_NUMの傾き・加重平均（SK_ID_BUREAU単位）
# =====================================================================
def bureau_balance_trend(bb: pd.DataFrame) -> pd.DataFrame:
    """bb_aggにmergeして使う想定。既存のBB_STATUS_MEAN等（水準）を、傾き/加重平均で補完する。"""
    needed = {"SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS_NUM"}
    if not needed.issubset(bb.columns):
        return pd.DataFrame(columns=["SK_ID_BUREAU"])
    slope = groupby_slope(bb, "SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS_NUM")
    wma = groupby_wma(bb, "SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS_NUM", tau=12.0)
    if slope.empty and wma.empty:
        return pd.DataFrame(columns=["SK_ID_BUREAU"])
    out = pd.DataFrame({"BB_TREND_STATUS_SLOPE": slope, "BB_WMA_STATUS": wma})
    out.index.name = "SK_ID_BUREAU"
    return out.reset_index()


def bureau_balance_recency(bb: pd.DataFrame) -> pd.DataFrame:
    """
    「最後の延滞から何ヶ月経過したか」の正規化された再帰性(recency)特徴（P3）。
    MONTHS_BALANCEは0=直近、負の値ほど過去。STATUS_NUM>0(延滞月)のうち最大のMONTHS_BALANCE
    (=現在に最も近い延滞月)を取り、符号反転して「現在から何ヶ月前に最後の延滞があったか」
    という直感的な尺度にする。延滞が一度も無いローンはNaN+フラグ列で区別する
    （bb_agg経由でSK_ID_CURR単位にmean/max/min/sumで自動集約される）。
    """
    needed = {"SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS_NUM"}
    if not needed.issubset(bb.columns):
        return pd.DataFrame(columns=["SK_ID_BUREAU"])
    dpd = bb[bb["STATUS_NUM"] > 0]
    if dpd.empty:
        return pd.DataFrame(columns=["SK_ID_BUREAU"])
    last_dpd_month = dpd.groupby("SK_ID_BUREAU")["MONTHS_BALANCE"].max()  # 0に近いほど直近
    recency = (-last_dpd_month).rename("BB_MONTHS_SINCE_LAST_DPD")
    out = recency.reset_index()
    all_ids = bb[["SK_ID_BUREAU"]].drop_duplicates()
    out = all_ids.merge(out, on="SK_ID_BUREAU", how="left")
    out["BB_NEVER_DPD"] = out["BB_MONTHS_SINCE_LAST_DPD"].isna().astype(np.int8)
    return out


# =====================================================================
# 2. POS_CASH: SK_DPDの傾き（SK_ID_PREV単位 -> SK_ID_CURRへ集約）
# =====================================================================
def pos_cash_trend_features(pos: pd.DataFrame) -> pd.DataFrame:
    needed = {"SK_ID_PREV", "SK_ID_CURR", "MONTHS_BALANCE", "SK_DPD"}
    if not needed.issubset(pos.columns):
        return pd.DataFrame(columns=["SK_ID_CURR"])
    slope = groupby_slope(pos, "SK_ID_PREV", "MONTHS_BALANCE", "SK_DPD")
    if slope.empty:
        return pd.DataFrame(columns=["SK_ID_CURR"])
    prev_to_curr = pos.drop_duplicates("SK_ID_PREV").set_index("SK_ID_PREV")["SK_ID_CURR"]
    slope_df = pd.DataFrame({"SK_ID_PREV": slope.index, "POS_LOAN_DPD_SLOPE": slope.values})
    slope_df["SK_ID_CURR"] = slope_df["SK_ID_PREV"].map(prev_to_curr)
    slope_df = slope_df.dropna(subset=["SK_ID_CURR"])
    if slope_df.empty:
        return pd.DataFrame(columns=["SK_ID_CURR"])
    agg = slope_df.groupby("SK_ID_CURR")["POS_LOAN_DPD_SLOPE"].agg(["mean", "max"])
    agg.columns = ["POS_DPD_SLOPE_mean", "POS_DPD_SLOPE_max"]
    return agg.reset_index()


# =====================================================================
# 3. credit_card_balance: 利用率(UTILIZATION)の傾き（SK_ID_PREV単位 -> SK_ID_CURRへ集約）
# =====================================================================
def credit_card_trend_features(cc: pd.DataFrame) -> pd.DataFrame:
    needed = {"SK_ID_PREV", "SK_ID_CURR", "MONTHS_BALANCE", "UTILIZATION"}
    if not needed.issubset(cc.columns):
        return pd.DataFrame(columns=["SK_ID_CURR"])
    slope = groupby_slope(cc, "SK_ID_PREV", "MONTHS_BALANCE", "UTILIZATION")
    if slope.empty:
        return pd.DataFrame(columns=["SK_ID_CURR"])
    prev_to_curr = cc.drop_duplicates("SK_ID_PREV").set_index("SK_ID_PREV")["SK_ID_CURR"]
    slope_df = pd.DataFrame({"SK_ID_PREV": slope.index, "CC_UTIL_SLOPE": slope.values})
    slope_df["SK_ID_CURR"] = slope_df["SK_ID_PREV"].map(prev_to_curr)
    slope_df = slope_df.dropna(subset=["SK_ID_CURR"])
    if slope_df.empty:
        return pd.DataFrame(columns=["SK_ID_CURR"])
    agg = slope_df.groupby("SK_ID_CURR")["CC_UTIL_SLOPE"].agg(["mean", "max"])
    agg.columns = ["CC_UTIL_SLOPE_mean", "CC_UTIL_SLOPE_max"]
    return agg.reset_index()


# =====================================================================
# 4. installments_payments: PAYMENT_RATIOの傾き（SK_ID_PREV単位 -> SK_ID_CURRへ集約）
# =====================================================================
def installments_trend_features(ins: pd.DataFrame) -> pd.DataFrame:
    needed = {"SK_ID_PREV", "SK_ID_CURR", "DAYS_INSTALMENT", "PAYMENT_RATIO"}
    if not needed.issubset(ins.columns):
        return pd.DataFrame(columns=["SK_ID_CURR"])
    slope = groupby_slope(ins, "SK_ID_PREV", "DAYS_INSTALMENT", "PAYMENT_RATIO")
    if slope.empty:
        return pd.DataFrame(columns=["SK_ID_CURR"])
    prev_to_curr = ins.drop_duplicates("SK_ID_PREV").set_index("SK_ID_PREV")["SK_ID_CURR"]
    slope_df = pd.DataFrame({"SK_ID_PREV": slope.index, "INS_PAYRATIO_SLOPE": slope.values})
    slope_df["SK_ID_CURR"] = slope_df["SK_ID_PREV"].map(prev_to_curr)
    slope_df = slope_df.dropna(subset=["SK_ID_CURR"])
    if slope_df.empty:
        return pd.DataFrame(columns=["SK_ID_CURR"])
    # meanは全体傾向、minは「最も悪化した1本」を検出（複数ローンのうち1本が急速に焦げ付く例を拾う）
    agg = slope_df.groupby("SK_ID_CURR")["INS_PAYRATIO_SLOPE"].agg(["mean", "min"])
    agg.columns = ["INS_PAYRATIO_SLOPE_mean", "INS_PAYRATIO_SLOPE_min"]
    return agg.reset_index()


# =====================================================================
# 5. bureau: 信用の種類の多様性（クレジットハンガーの別シグナル）
# =====================================================================
def bureau_credit_type_diversity(bureau: pd.DataFrame) -> pd.DataFrame:
    """OHE+sum/meanでは失われる「distinct値の個数」を明示的に特徴化する。"""
    if "SK_ID_CURR" not in bureau.columns:
        return pd.DataFrame(columns=["SK_ID_CURR"])
    feats = {}
    if "CREDIT_TYPE" in bureau.columns:
        feats["BUREAU_CREDIT_TYPE_NUNIQUE"] = bureau.groupby("SK_ID_CURR")["CREDIT_TYPE"].nunique()
    if "CREDIT_ACTIVE" in bureau.columns and "CREDIT_TYPE" in bureau.columns:
        active = bureau[bureau["CREDIT_ACTIVE"] == "Active"]
        if not active.empty:
            feats["BUREAU_ACTIVE_CREDIT_TYPE_NUNIQUE"] = (
                active.groupby("SK_ID_CURR")["CREDIT_TYPE"].nunique()
            )
    if "CREDIT_CURRENCY" in bureau.columns:
        feats["BUREAU_CURRENCY_NUNIQUE"] = bureau.groupby("SK_ID_CURR")["CREDIT_CURRENCY"].nunique()
    if not feats:
        return pd.DataFrame(columns=["SK_ID_CURR"])
    out = pd.DataFrame(feats)
    out.index.name = "SK_ID_CURR"
    return out.reset_index()


# =====================================================================
# 6. previous_application: 自社申込の間隔（ベロシティ）と直近謝絶率
# =====================================================================
def previous_application_velocity(prev: pd.DataFrame) -> pd.DataFrame:
    """
    bureauのAMT_REQ_CREDIT_BUREAU_*（他社照会件数）とは別に、Home Credit自身への
    申込頻度そのものを見る。間隔が短いほど直近で頻繁に借入を試みている＝与信ハンガーの兆候。
    """
    needed = {"SK_ID_CURR", "DAYS_DECISION"}
    if not needed.issubset(prev.columns):
        return pd.DataFrame(columns=["SK_ID_CURR"])
    p = prev.sort_values(["SK_ID_CURR", "DAYS_DECISION"])
    p = p.copy()
    p["_INTERVAL"] = p.groupby("SK_ID_CURR")["DAYS_DECISION"].diff()
    interval_agg = p.groupby("SK_ID_CURR")["_INTERVAL"].agg(["mean", "min", "std"])
    interval_agg.columns = ["PREV_APP_INTERVAL_mean", "PREV_APP_INTERVAL_min", "PREV_APP_INTERVAL_std"]
    out = interval_agg.reset_index()

    if "NAME_CONTRACT_STATUS" in prev.columns:
        p2 = prev.sort_values(["SK_ID_CURR", "DAYS_DECISION"], ascending=[True, False])
        last3 = p2.groupby("SK_ID_CURR").head(3)
        last3_refused = (
            (last3["NAME_CONTRACT_STATUS"] == "Refused").groupby(last3["SK_ID_CURR"]).mean()
        )
        last3_refused = last3_refused.rename("PREV_LAST3_REFUSED_RATIO").reset_index()
        out = out.merge(last3_refused, on="SK_ID_CURR", how="outer")
    return out


# =====================================================================
# 7. application: 提出書類数（情報の乏しさ・提出パターンそのものがシグナル）
# =====================================================================
def add_document_features(df: pd.DataFrame) -> int:
    doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
    if not doc_cols:
        return 0
    df["DOC_SUBMIT_COUNT"] = df[doc_cols].sum(axis=1).astype(np.int16)
    if "FLAG_DOCUMENT_3" in doc_cols:
        other_docs = [c for c in doc_cols if c != "FLAG_DOCUMENT_3"]
        if other_docs:
            # FLAG_DOCUMENT_3はほぼ全員提出する定番書類。それ以外の提出有無=非定型パターン
            df["DOC_SUBMIT_COUNT_EXCL3"] = df[other_docs].sum(axis=1).astype(np.int16)
    return len(doc_cols)


# =====================================================================
# 8. application: 申込時刻/曜日の周期エンコーディング
# =====================================================================
WEEKDAY_ORDER = {
    "MONDAY": 0, "TUESDAY": 1, "WEDNESDAY": 2, "THURSDAY": 3,
    "FRIDAY": 4, "SATURDAY": 5, "SUNDAY": 6,
}


def add_cyclical_time_features(df: pd.DataFrame) -> int:
    n = 0
    if "HOUR_APPR_PROCESS_START" in df.columns:
        h = pd.to_numeric(df["HOUR_APPR_PROCESS_START"], errors="coerce")
        df["HOUR_SIN"] = np.sin(2 * np.pi * h / 24.0).astype(np.float32)
        df["HOUR_COS"] = np.cos(2 * np.pi * h / 24.0).astype(np.float32)
        df["HOUR_IS_OFFHOURS"] = ((h < 8) | (h >= 20)).astype(np.int8)
        n += 3
        if "REGION_RATING_CLIENT" in df.columns:
            df["HOUR_OFF_x_REGION_RATING"] = (
                df["HOUR_IS_OFFHOURS"].astype(np.float32) *
                pd.to_numeric(df["REGION_RATING_CLIENT"], errors="coerce")
            ).astype(np.float32)
            n += 1
    if "WEEKDAY_APPR_PROCESS_START" in df.columns:
        wd = df["WEEKDAY_APPR_PROCESS_START"].astype(str).str.upper().map(WEEKDAY_ORDER)
        df["WEEKDAY_NUM"] = wd.astype("float32")
        df["WEEKDAY_SIN"] = np.sin(2 * np.pi * wd / 7.0).astype(np.float32)
        df["WEEKDAY_COS"] = np.cos(2 * np.pi * wd / 7.0).astype(np.float32)
        df["IS_WEEKEND_APPR"] = wd.isin([5, 6]).astype(np.int8)
        n += 4
    return n


# =====================================================================
# 9. application: IsolationForest 教師なし異常度スコア（target非依存）
# =====================================================================
ISO_DEFAULT_COLS = [
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "EXT_SOURCE_MEAN",
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "DAYS_BIRTH",
    "DAYS_EMPLOYED", "CREDIT_ANNUITY_RATIO",
]


def add_isolation_forest_anomaly(app_train: pd.DataFrame, app_test: pd.DataFrame,
                                  cols=None, contamination: float = 0.05,
                                  seed: int = 42) -> tuple:
    """
    extra_features.pyのk-meansクラスタ距離（密度・重心ベース）とは異なる「分割ベース」の
    異常検知を追加し、異常検知の観点に多様性を持たせる。教師なしなのでリークなし。
    """
    try:
        from sklearn.ensemble import IsolationForest
    except Exception:
        print("  [iso_forest] scikit-learnのIsolationForestが利用不可のためスキップ")
        return app_train, app_test

    use_cols = cols or ISO_DEFAULT_COLS
    use_cols = [c for c in use_cols if c in app_train.columns and c in app_test.columns]
    if len(use_cols) < 3:
        print("  [iso_forest] 対象列が不足のためスキップ")
        return app_train, app_test

    n_tr = len(app_train)
    full = pd.concat([app_train[use_cols], app_test[use_cols]], axis=0, ignore_index=True)
    full = full.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    full = full.fillna(full.median())

    iso = IsolationForest(n_estimators=200, contamination=contamination,
                           random_state=seed, n_jobs=-1)
    iso.fit(full.values)
    score = -iso.score_samples(full.values)  # 大きいほど異常

    app_train["ISO_ANOMALY_SCORE"] = score[:n_tr].astype(np.float32)
    app_test["ISO_ANOMALY_SCORE"] = score[n_tr:].astype(np.float32)
    print(f"  [iso_forest] IsolationForest異常度スコアを付与 (cols={len(use_cols)})")
    return app_train, app_test


def add_application_level_all(app_train: pd.DataFrame, app_test: pd.DataFrame) -> tuple:
    """application直付け特徴（書類数・周期エンコーディング・IsolationForest）をまとめて実行。"""
    n_doc = add_document_features(app_train)
    add_document_features(app_test)
    n_time = add_cyclical_time_features(app_train)
    add_cyclical_time_features(app_test)
    print(f"  [trend/velocity] 書類提出特徴 {n_doc > 0 and 'あり' or 'なし'}, "
          f"周期特徴 {n_time}個")
    app_train, app_test = add_isolation_forest_anomaly(app_train, app_test)
    return app_train, app_test
