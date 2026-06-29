"""
金融ドメイン特徴量（クレジットリスク実務の指標をHCデータに落とし込んだもの）。

特徴は7カテゴリに分け、列名プレフィックスで識別できるようにしてある（後でLightGBM重要度を
カテゴリ別に集計して「どのカテゴリが効くか」を一個ずつ検証できるようにするため）:

  DOM_CAP_*  返済能力 / DTI（全債務を横断合算）
  DOM_LEV_*  レバレッジ / 与信妥当性
  DOM_DLQ_*  延滞の深刻度・直近トレンド
  DOM_VEL_*  申込ベロシティ / クレジットハンガー
  DOM_UTL_*  カード利用・キャッシング苦境シグナル
  DOM_PAY_*  返済行動（installments）
  DOM_STB_*  安定性・外部スコア交互作用

すべての特徴は「元になる列が存在するときだけ」作る（合成データ/実データのどちらでも落ちない）。
本パイプラインのテーブル集約後（BUREAU_/PREV_/POS_/INS_/CC_ プレフィックス）の列も参照する。
"""
import numpy as np
import pandas as pd

EPS = 1e-5

# カテゴリ説明（feature_importance.py のロールアップで使う）
DOMAIN_GROUPS = {
    "DOM_CAP": "返済能力/DTI(全債務横断)",
    "DOM_LEV": "レバレッジ/与信妥当性",
    "DOM_DLQ": "延滞の深刻度/トレンド",
    "DOM_VEL": "申込ベロシティ/クレジットハンガー",
    "DOM_UTL": "カード利用/キャッシング苦境",
    "DOM_PAY": "返済行動(installments)",
    "DOM_STB": "安定性/外部スコア交互作用",
}


def _col(df, name):
    """列があればSeries(float)で返す。無ければNone。"""
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").astype("float64")
    return None


def _first(df, names):
    """候補名のうち最初に存在する列を返す（集約名の揺れ吸収）。"""
    for n in names:
        s = _col(df, n)
        if s is not None:
            return s
    return None


def _rowmax(df, names):
    cols = [n for n in names if n in df.columns]
    if not cols:
        return None
    return df[cols].apply(pd.to_numeric, errors="coerce").max(axis=1)


def _add_for_df(df: pd.DataFrame) -> int:
    feats = {}

    income = _col(df, "AMT_INCOME_TOTAL")
    credit = _col(df, "AMT_CREDIT")
    annuity = _col(df, "AMT_ANNUITY")
    fam = _col(df, "CNT_FAM_MEMBERS")
    birth = _col(df, "DAYS_BIRTH")

    # ---------- 1. 返済能力 / DTI（全債務横断） ----------
    bureau_debt = _first(df, ["BUREAU_AMT_CREDIT_SUM_DEBT_sum"])
    bureau_credit = _first(df, ["BUREAU_AMT_CREDIT_SUM_sum"])
    prev_annuity = _first(df, ["PREV_AMT_ANNUITY_sum", "PREV_AMT_ANNUITY_mean"])

    if bureau_debt is not None and income is not None:
        feats["DOM_CAP_BUREAU_DEBT_TO_INCOME"] = bureau_debt / (income + EPS)
    if bureau_debt is not None and credit is not None:
        feats["DOM_CAP_BUREAU_DEBT_TO_CREDIT"] = bureau_debt / (credit + EPS)
    if bureau_credit is not None and credit is not None and income is not None:
        feats["DOM_CAP_TOTAL_EXPOSURE_TO_INCOME"] = (credit + bureau_credit) / (income + EPS)
    if annuity is not None and income is not None:
        # 全返済負担（当該 + 他社annuity推計）/ 収入
        total_ann = annuity + (prev_annuity if prev_annuity is not None else 0.0)
        feats["DOM_CAP_TOTAL_ANNUITY_TO_INCOME"] = total_ann / (income + EPS)
        feats["DOM_CAP_RESIDUAL_INCOME"] = income - annuity
        if fam is not None:
            feats["DOM_CAP_RESIDUAL_PER_PERSON"] = (income - annuity) / (fam + EPS)

    # ---------- 2. レバレッジ / 与信妥当性 ----------
    overdue = _first(df, ["BUREAU_AMT_CREDIT_SUM_OVERDUE_sum"])
    if overdue is not None and bureau_debt is not None:
        feats["DOM_LEV_OVERDUE_DEBT_RATIO"] = overdue / (bureau_debt.abs() + EPS)
    downpay = _first(df, ["PREV_RATE_DOWN_PAYMENT_mean", "PREV_AMT_DOWN_PAYMENT_mean"])
    if downpay is not None:
        feats["DOM_LEV_DOWNPAY"] = downpay
    app_credit_ratio = _first(df, ["PREV_APP_CREDIT_RATIO_mean"])
    if app_credit_ratio is not None:
        feats["DOM_LEV_PREV_APP_CREDIT_RATIO"] = app_credit_ratio

    # ---------- 3. 延滞の深刻度 / トレンド ----------
    max_dpd = _rowmax(df, ["POS_SK_DPD_max", "POS_SK_DPD_DEF_max", "CC_SK_DPD_max",
                            "POS_3M_SK_DPD_max", "BUREAU_CREDIT_DAY_OVERDUE_max"])
    if max_dpd is not None:
        feats["DOM_DLQ_MAX_DPD"] = max_dpd
    if overdue is not None:
        feats["DOM_DLQ_HAS_BUREAU_OVERDUE"] = (overdue > 0).astype(np.int8)
    # installmentsの遅延・過小払いの「直近 vs 生涯」トレンド
    late_all = _first(df, ["INS_ALL_IS_LATE_mean"])
    late_1yr = _first(df, ["INS_1YR_IS_LATE_mean"])
    if late_all is not None and late_1yr is not None:
        feats["DOM_DLQ_LATE_TREND_1YR"] = late_1yr - late_all
    def_all = _first(df, ["INS_ALL_PAYMENT_DEFICIT_mean"])
    def_1yr = _first(df, ["INS_1YR_PAYMENT_DEFICIT_mean"])
    if def_all is not None and def_1yr is not None:
        feats["DOM_DLQ_DEFICIT_TREND_1YR"] = def_1yr - def_all

    # ---------- 4. 申込ベロシティ / クレジットハンガー ----------
    req_yr = _col(df, "AMT_REQ_CREDIT_BUREAU_YEAR")
    req_qrt = _col(df, "AMT_REQ_CREDIT_BUREAU_QRT")
    req_mon = _col(df, "AMT_REQ_CREDIT_BUREAU_MON")
    req_week = _col(df, "AMT_REQ_CREDIT_BUREAU_WEEK")
    req_day = _col(df, "AMT_REQ_CREDIT_BUREAU_DAY")
    short_parts = [s for s in [req_day, req_week, req_mon] if s is not None]
    if short_parts:
        feats["DOM_VEL_INQ_SHORT"] = sum(short_parts)
    if req_qrt is not None and req_yr is not None:
        feats["DOM_VEL_INQ_RECENT_RATIO"] = req_qrt / (req_yr + EPS)  # 直近四半期への集中度
    active_ratio = _first(df, ["BUREAU_CREDIT_ACTIVE_Active_mean"])
    if active_ratio is not None:
        feats["DOM_VEL_ACTIVE_RATIO"] = active_ratio

    # ---------- 5. カード利用 / キャッシング苦境 ----------
    util_mean = _first(df, ["CC_UTILIZATION_mean"])
    util_max = _first(df, ["CC_UTILIZATION_max"])
    if util_mean is not None:
        feats["DOM_UTL_MEAN"] = util_mean
    if util_max is not None:
        feats["DOM_UTL_MAX"] = util_max
    atm = _first(df, ["CC_AMT_DRAWINGS_ATM_CURRENT_sum"])
    draw = _first(df, ["CC_AMT_DRAWINGS_CURRENT_sum"])
    if atm is not None and draw is not None:
        feats["DOM_UTL_CASH_ADVANCE_RATIO"] = atm / (draw.abs() + EPS)  # ATM現金引出依存
    min_def = _first(df, ["CC_MIN_PAYMENT_DEFICIT_mean"])
    if min_def is not None:
        feats["DOM_UTL_MINPAY_DEFICIT"] = min_def

    # ---------- 6. 返済行動（installments） ----------
    pay_all = _first(df, ["INS_ALL_AMT_PAYMENT_sum"])
    inst_all = _first(df, ["INS_ALL_AMT_INSTALMENT_sum"])
    if pay_all is not None and inst_all is not None:
        feats["DOM_PAY_COVERAGE"] = pay_all / (inst_all + EPS)  # 生涯の支払充足率
    delay_max = _first(df, ["INS_ALL_PAYMENT_DELAY_max", "INS_1YR_PAYMENT_DELAY_max"])
    if delay_max is not None:
        feats["DOM_PAY_DELAY_MAX"] = delay_max
    if late_all is not None:
        feats["DOM_PAY_LATE_RATE"] = late_all

    # ---------- 7. 安定性 / 外部スコア交互作用 ----------
    ext_mean = _col(df, "EXT_SOURCE_MEAN")
    region = _col(df, "REGION_RATING_CLIENT")
    if ext_mean is not None and region is not None:
        feats["DOM_STB_EXT_x_REGION"] = ext_mean * region
    phone = _col(df, "DAYS_LAST_PHONE_CHANGE")
    if phone is not None and birth is not None:
        feats["DOM_STB_PHONE_TO_AGE"] = phone / (birth.abs() + EPS)
    def30 = _col(df, "DEF_30_CNT_SOCIAL_CIRCLE")
    obs30 = _col(df, "OBS_30_CNT_SOCIAL_CIRCLE")
    if def30 is not None and obs30 is not None:
        feats["DOM_STB_SOCIAL_DEF_RATIO"] = def30 / (obs30 + EPS)  # 社会的圏のデフォルト率

    # まとめて代入（断片化回避）。inf を NaN に。
    if feats:
        add = pd.DataFrame(feats, index=df.index).replace([np.inf, -np.inf], np.nan)
        for c in add.columns:
            df[c] = add[c].astype(np.float32)
    return len(feats)


def add_domain_features(app_train: pd.DataFrame, app_test: pd.DataFrame) -> tuple:
    """train/test両方に金融ドメイン特徴を付与。作成できた特徴数を表示する。"""
    n_tr = _add_for_df(app_train)
    _add_for_df(app_test)
    # trainにあってtestに無い/その逆を防ぐ（存在列が揃わない場合に備える）
    only_tr = [c for c in app_train.columns if c.startswith("DOM_") and c not in app_test.columns]
    only_te = [c for c in app_test.columns if c.startswith("DOM_") and c not in app_train.columns]
    for c in only_tr:
        app_test[c] = np.float32(np.nan)
    for c in only_te:
        app_train[c] = np.float32(np.nan)
    made = [c for c in app_train.columns if c.startswith("DOM_")]
    by_grp = {}
    for c in made:
        grp = "_".join(c.split("_")[:2])  # DOM_CAP 等
        by_grp[grp] = by_grp.get(grp, 0) + 1
    print(f"  [domain] 金融ドメイン特徴 {len(made)}個を付与: " +
          ", ".join(f"{g}:{n}" for g, n in sorted(by_grp.items())))
    return app_train, app_test
