"""
Home Credit Default Risk 1位ソリューションdiscussion（Bojan/Olivier/Ryan/Phil/Yang/
Michael Jahrer, Kaggle 2018）でスコア寄与が明記された特徴量のうち、既存パイプライン
（feature_engineering.py本体 + domain_features.py + oof_features.py + extra_features.py +
trend_velocity_features.py）にまだ無かったものを追加する。

  YEARLY_INTEREST_RATE / MONTHLY_INTEREST_RATE
      現在の申込の年利率をNewton法でIRR近似。Olivierが「最もスコアに効いた特徴の一つ」と明言。
      申込時点ではCNT_PAYMENT(分割回数)が無いため、同一顧客の過去ローン平均回数
      (PREV_CNT_PAYMENT_mean)を代理変数に使う（無ければtrain中央値でフォールバック）。
  EXT3_DIV_*
      EXT_SOURCE_3による除算特徴量（AMT_CREDIT/AMT_ANNUITY/DAYS_BIRTH/AMT_INCOME_TOTAL）。
      EXT_SOURCE_MEANとの交互作用はextra_features.pyにあるが、EXT_SOURCE_3単体は無かった。
  AGE_INT
      年齢の離散化（連続値のDAYS_BIRTHより離散化した方が効くとの報告）。
  INCOME_ANNUITY_RATIO / ANNUITY_TO_MAX_INSTALLMENT_RATIO
      収入/年間返済の逆数比、および申込annuityと過去installments最大支払額との比。
  BUREAU_LAST_ACTIVE_DAYS_CREDIT / BUREAU_ACTIVE_DEBT_SUM
      アクティブローンに絞った直近性・残債合計（既存の期間別集約は全ローン対象だった）。
  PREV_LAST3S_*/PREV_LAST5S_*/PREV_FIRST2S_*/PREV_FIRST4S_* / PREV_LAST_PRODUCT_COMBINATION
      previous_applicationの直近N件・最初N件スライス集約と最新申込の商品カテゴリ。
  INS_60D_*/INS_90D_*/INS_180D_*/INS_1000D_* / INS_NUM{1,2,3,4}_*
      installmentsの期間別集約の細分化、および支払回次別（初回〜4回目）の集約。

すべて「元になる列が存在するときだけ」作る（合成データ/実データのどちらでも落ちない）。
"""
import numpy as np
import pandas as pd

from utils import flatten_agg_columns

EPS = 1e-5


# =====================================================================
# 0. CNT_PAYMENT予測モデル（年利率精度向上のP1）
# =====================================================================
# previous_applicationとapplication_train/testの両方に同じ意味で存在する列だけを使う
# （カテゴリ列はドメインが違う=previousはConsumer loans等を含むがapplicationには無い、
#   等の食い違いがあるため数値列のみに絞ってクロスドメイン予測を安全にする）。
CNT_PAYMENT_NUM_COLS = ["AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]


def _prep_cnt_payment_features(df: pd.DataFrame) -> pd.DataFrame:
    credit = pd.to_numeric(df["AMT_CREDIT"], errors="coerce")
    annuity = pd.to_numeric(df["AMT_ANNUITY"], errors="coerce")
    goods = pd.to_numeric(df["AMT_GOODS_PRICE"], errors="coerce") if "AMT_GOODS_PRICE" in df.columns \
        else pd.Series(np.nan, index=df.index)
    feats = pd.DataFrame({
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity,
        "AMT_GOODS_PRICE": goods,
        "CREDIT_TERM_RAW": annuity / (credit + EPS),
        "CREDIT_GOODS_RATIO_RAW": credit / (goods + EPS),
    }, index=df.index)
    return feats.replace([np.inf, -np.inf], np.nan)


def _train_cnt_payment_model(prev: pd.DataFrame, seed: int = 42):
    """previous_applicationでCNT_PAYMENT(分割回数)を回帰学習する。
    1位解法discussionの核心「申込ごとのCNT_PAYMENT予測→Newton法で金利逆算」の予測部分。
    LightGBM未インストール/データ不足の場合はNoneを返し、呼び出し側は既存フォールバックを使う。
    """
    try:
        import lightgbm as lgb
    except ImportError:
        return None
    if not (set(CNT_PAYMENT_NUM_COLS) | {"CNT_PAYMENT"}).issubset(prev.columns):
        return None
    d = prev[prev["CNT_PAYMENT"].notna() & (prev["CNT_PAYMENT"] > 0)]
    if len(d) < 200:
        return None
    X = _prep_cnt_payment_features(d)
    y = pd.to_numeric(d["CNT_PAYMENT"], errors="coerce")
    mask = y.notna()
    X, y = X[mask], y[mask]
    X_filled = X.fillna(X.median())
    model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                              min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
                              random_state=seed, verbose=-1)
    model.fit(X_filled, y)
    return model


def add_cnt_payment_prediction(app_train: pd.DataFrame, app_test: pd.DataFrame,
                                raw_dir=None, raw_filename: str = None) -> tuple:
    """
    previous_application.csvを（get_aggregated_previousとは別に軽量に）読み込んでCNT_PAYMENT
    回帰モデルを学習し、現在の申込（AMT_CREDIT/AMT_ANNUITY/AMT_GOODS_PRICEは申込・過去ローン
    どちらにも同じ意味で存在するためドメインを跨いでそのまま使える）に適用してPRED_CNT_PAYMENT
    列を付与する。add_yearly_interest_rateがこれをPREV_CNT_PAYMENT_meanより優先して使う。
    """
    import config
    raw_dir = raw_dir or config.RAW_DIR
    raw_filename = raw_filename or config.RAW_FILES["previous"]
    path = raw_dir / raw_filename
    if not path.exists():
        print(f"  [cnt_payment] {path} が見つからないためスキップ")
        return app_train, app_test

    usecols = list(set(CNT_PAYMENT_NUM_COLS) | {"CNT_PAYMENT"})
    prev = pd.read_csv(path, usecols=lambda c: c in usecols)
    model = _train_cnt_payment_model(prev)
    del prev
    if model is None:
        print("  [cnt_payment] 学習データ不足/LightGBM無しのためスキップ（PREV_CNT_PAYMENT_meanフォールバックのまま）")
        return app_train, app_test

    for df in (app_train, app_test):
        if not set(CNT_PAYMENT_NUM_COLS).issubset(df.columns):
            continue
        X_app = _prep_cnt_payment_features(df)
        pred = model.predict(X_app.fillna(X_app.median()))
        df["PRED_CNT_PAYMENT"] = np.clip(pred, 1, 120).astype(np.float32)
    print("  [cnt_payment] CNT_PAYMENT予測モデル(LightGBM回帰)を適用 -> PRED_CNT_PAYMENT")
    return app_train, app_test


# =====================================================================
# 1. 年利率（Newton法によるIRR近似）
# =====================================================================
def _estimate_rate_newton(credit: np.ndarray, annuity: np.ndarray, n_payments: np.ndarray,
                           iters: int = 40) -> np.ndarray:
    """
    P = A * (1-(1+r)^-n) / r （元利均等返済の現在価値公式）を満たす月利rをNewton法で解く。
    解析微分が煩雑なため中心差分の数値微分で代用（ベクトル化しているので実データ規模でも高速）。
    """
    credit = np.asarray(credit, dtype=np.float64)
    annuity = np.asarray(annuity, dtype=np.float64)
    n = np.asarray(n_payments, dtype=np.float64)
    valid = (credit > 0) & (annuity > 0) & (n >= 1) & np.isfinite(credit) & np.isfinite(annuity) & np.isfinite(n)
    r = np.full_like(credit, 0.02, dtype=np.float64)  # 初期値: 月2%程度から出発

    def f(rr):
        with np.errstate(all="ignore"):
            out = annuity * (1.0 - np.power(1.0 + rr, -n)) / rr - credit
        near_zero = np.abs(rr) < 1e-8
        out = np.where(near_zero, annuity * n - credit, out)  # r->0 の極限（単純合計-元本）
        return out

    eps = 1e-6
    for _ in range(iters):
        f0 = f(r)
        f1 = f(r + eps)
        deriv = (f1 - f0) / eps
        deriv = np.where(np.abs(deriv) < 1e-9, 1e-9, deriv)
        step = np.clip(f0 / deriv, -0.5, 0.5)  # 発散防止
        r = np.clip(r - step, -0.99, 5.0)
    return np.where(valid, r, np.nan)


def add_yearly_interest_rate(app_train: pd.DataFrame, app_test: pd.DataFrame) -> tuple:
    """
    現在の申込の年利率を付与する。previous_application集約（PREV_CNT_PAYMENT_mean）が
    マージ済みである必要があるため、table_builders実行後に呼ぶこと。

    分割回数(n)の優先順位（P1: CNT_PAYMENT予測モデルによる精度向上）:
      1. PRED_CNT_PAYMENT   … add_cnt_payment_predictionが学習した回帰モデルの申込ごとの予測値（最も精度が高い）
      2. PREV_CNT_PAYMENT_mean … 同一顧客の過去ローンの平均回数（次点のフォールバック）
      3. train中央値         … どちらも無い場合の最終フォールバック
    """
    if not {"AMT_CREDIT", "AMT_ANNUITY"}.issubset(app_train.columns):
        return app_train, app_test

    if "PRED_CNT_PAYMENT" in app_train.columns:
        n_col = "PRED_CNT_PAYMENT"
    elif "PREV_CNT_PAYMENT_mean" in app_train.columns:
        n_col = "PREV_CNT_PAYMENT_mean"
    else:
        n_col = None
    fallback_n = 12.0
    if n_col is not None:
        med = pd.to_numeric(app_train[n_col], errors="coerce").median()
        if np.isfinite(med) and med > 0:
            fallback_n = float(med)

    for df in (app_train, app_test):
        credit = pd.to_numeric(df["AMT_CREDIT"], errors="coerce")
        annuity = pd.to_numeric(df["AMT_ANNUITY"], errors="coerce")
        if n_col is not None and n_col in df.columns:
            n_est = pd.to_numeric(df[n_col], errors="coerce").fillna(fallback_n)
        else:
            n_est = pd.Series(fallback_n, index=df.index)
        n_est = n_est.clip(lower=1, upper=120)  # 月次分割を想定し最大10年程度に丸める

        r_month = _estimate_rate_newton(credit.values, annuity.values, n_est.values)
        with np.errstate(all="ignore"):
            r_year = np.power(1.0 + r_month, 12) - 1.0
        df["MONTHLY_INTEREST_RATE"] = r_month.astype(np.float32)
        df["YEARLY_INTEREST_RATE"] = r_year.astype(np.float32)
        # 派生: 与信額×予測金利、過去平均金利との比（このローンが過去より高利率か）
        df["CREDIT_x_YEARLY_RATE"] = (credit * df["YEARLY_INTEREST_RATE"]).astype(np.float32)
        if "PREV_ESTIMATED_INTEREST_RATE_mean" in df.columns:
            prev_rate = pd.to_numeric(df["PREV_ESTIMATED_INTEREST_RATE_mean"], errors="coerce")
            df["RATE_VS_PREV_MEAN_RATIO"] = (
                df["YEARLY_INTEREST_RATE"] / (prev_rate + EPS)
            ).replace([np.inf, -np.inf], np.nan).astype(np.float32)
    print(f"  [interest_rate] 年利率(Newton法IRR近似, n_col={n_col})を付与")
    return app_train, app_test


# =====================================================================
# 2. EXT_SOURCE_3 除算特徴量
# =====================================================================
def add_ext3_ratios(app_train: pd.DataFrame, app_test: pd.DataFrame) -> tuple:
    n_made = 0
    for df in (app_train, app_test):
        if "EXT_SOURCE_3" not in df.columns:
            continue
        e3 = pd.to_numeric(df["EXT_SOURCE_3"], errors="coerce")
        made = 0
        for col, name in [("AMT_CREDIT", "CREDIT"), ("AMT_ANNUITY", "ANNUITY"),
                           ("DAYS_BIRTH", "DAYS_BIRTH"), ("AMT_INCOME_TOTAL", "INCOME")]:
            if col in df.columns:
                ratio = pd.to_numeric(df[col], errors="coerce") / (e3 + EPS)
                df[f"EXT3_DIV_{name}"] = ratio.replace([np.inf, -np.inf], np.nan).astype(np.float32)
                made += 1
        n_made = made
    print(f"  [ext3] EXT_SOURCE_3除算特徴 {n_made}個を付与")
    return app_train, app_test


# =====================================================================
# 3. 年齢の離散化
# =====================================================================
def add_age_int(app_train: pd.DataFrame, app_test: pd.DataFrame) -> tuple:
    if "DAYS_BIRTH" not in app_train.columns:
        return app_train, app_test
    for df in (app_train, app_test):
        age_years = -pd.to_numeric(df["DAYS_BIRTH"], errors="coerce") / 365.0
        df["AGE_INT"] = np.floor(age_years).astype(np.float32)
    print("  [age] AGE_INT(年齢の離散化)を付与")
    return app_train, app_test


# =====================================================================
# 4. その他の比率（収入/annuity逆数、annuity/過去最大支払額）
# =====================================================================
def add_extra_ratios(app_train: pd.DataFrame, app_test: pd.DataFrame) -> tuple:
    n_made = 0
    max_inst_col = "INS_ALL_AMT_INSTALMENT_max"
    for df in (app_train, app_test):
        made = 0
        if {"AMT_INCOME_TOTAL", "AMT_ANNUITY"}.issubset(df.columns):
            df["INCOME_ANNUITY_RATIO"] = (
                pd.to_numeric(df["AMT_INCOME_TOTAL"], errors="coerce") /
                (pd.to_numeric(df["AMT_ANNUITY"], errors="coerce") + EPS)
            ).replace([np.inf, -np.inf], np.nan).astype(np.float32)
            made += 1
        if "AMT_ANNUITY" in df.columns and max_inst_col in df.columns:
            df["ANNUITY_TO_MAX_INSTALLMENT_RATIO"] = (
                pd.to_numeric(df["AMT_ANNUITY"], errors="coerce") /
                (pd.to_numeric(df[max_inst_col], errors="coerce") + EPS)
            ).replace([np.inf, -np.inf], np.nan).astype(np.float32)
            made += 1
        n_made = made
    print(f"  [extra_ratio] 追加比率特徴 {n_made}個を付与")
    return app_train, app_test


def add_post_merge_priority_features(app_train: pd.DataFrame, app_test: pd.DataFrame) -> tuple:
    """table_builders(bureau/prev/POS/installments/CC)マージ後に呼ぶ必要があるものをまとめる。"""
    app_train, app_test = add_cnt_payment_prediction(app_train, app_test)
    app_train, app_test = add_yearly_interest_rate(app_train, app_test)
    app_train, app_test = add_extra_ratios(app_train, app_test)
    return app_train, app_test


def add_application_level_priority_features(app_train: pd.DataFrame, app_test: pd.DataFrame) -> tuple:
    """table_buildersマージ前でも計算できる、application自身の列だけで作れるものをまとめる。"""
    app_train, app_test = add_ext3_ratios(app_train, app_test)
    app_train, app_test = add_age_int(app_train, app_test)
    return app_train, app_test


# =====================================================================
# 5. bureau: アクティブローンに絞った直近性・残債合計
# =====================================================================
def bureau_last_active_snapshot(bureau: pd.DataFrame) -> pd.DataFrame:
    needed = {"SK_ID_CURR", "CREDIT_ACTIVE", "DAYS_CREDIT"}
    if not needed.issubset(bureau.columns):
        return pd.DataFrame(columns=["SK_ID_CURR"])
    active = bureau[bureau["CREDIT_ACTIVE"] == "Active"]
    if active.empty:
        return pd.DataFrame(columns=["SK_ID_CURR"])
    out = (active.groupby("SK_ID_CURR")["DAYS_CREDIT"].max()
           .rename("BUREAU_LAST_ACTIVE_DAYS_CREDIT").reset_index())
    if "AMT_CREDIT_SUM_DEBT" in bureau.columns:
        debt = (active.groupby("SK_ID_CURR")["AMT_CREDIT_SUM_DEBT"].sum()
                .rename("BUREAU_ACTIVE_DEBT_SUM").reset_index())
        out = out.merge(debt, on="SK_ID_CURR", how="outer")
    return out


# =====================================================================
# 6. previous_application: 直近N件・最初N件スライス集約 + 最新PRODUCT_COMBINATION
# =====================================================================
PREV_SLICE_COLS = ["AMT_CREDIT", "AMT_ANNUITY", "AMT_DOWN_PAYMENT", "AMT_APPLICATION",
                    "APP_CREDIT_RATIO", "ESTIMATED_INTEREST_RATE", "DAYS_DECISION"]


def previous_application_slices(prev: pd.DataFrame) -> pd.DataFrame:
    """
    直近3/5件・最初2/4件のスライス集約。全期間集約や直近1件スナップショットだけでは
    失われる「申込履歴の順序」を明示的に特徴化する（1位解法でCV効果確認済みとの記載）。
    """
    if not {"SK_ID_CURR", "DAYS_DECISION"}.issubset(prev.columns):
        return pd.DataFrame(columns=["SK_ID_CURR"])
    cols = [c for c in PREV_SLICE_COLS if c in prev.columns]
    if not cols:
        return pd.DataFrame(columns=["SK_ID_CURR"])

    prev_sorted = prev.sort_values(["SK_ID_CURR", "DAYS_DECISION"])  # 昇順=古い→新しい

    def _agg_slice(df_slice, prefix):
        agg = df_slice.groupby("SK_ID_CURR")[cols].agg(["mean", "max", "min"])
        agg = flatten_agg_columns(agg, prefix)
        return agg.reset_index()

    out = None
    slices = [
        (prev_sorted.groupby("SK_ID_CURR").tail(3), "PREV_LAST3S"),
        (prev_sorted.groupby("SK_ID_CURR").tail(5), "PREV_LAST5S"),
        (prev_sorted.groupby("SK_ID_CURR").head(2), "PREV_FIRST2S"),
        (prev_sorted.groupby("SK_ID_CURR").head(4), "PREV_FIRST4S"),
    ]
    for df_slice, prefix in slices:
        if df_slice.empty:
            continue
        agg = _agg_slice(df_slice, prefix)
        out = agg if out is None else out.merge(agg, on="SK_ID_CURR", how="outer")

    if "PRODUCT_COMBINATION" in prev.columns:
        last_pc = (prev_sorted.groupby("SK_ID_CURR").tail(1)[["SK_ID_CURR", "PRODUCT_COMBINATION"]]
                   .rename(columns={"PRODUCT_COMBINATION": "PREV_LAST_PRODUCT_COMBINATION"}))
        out = last_pc if out is None else out.merge(last_pc, on="SK_ID_CURR", how="outer")

    return out if out is not None else pd.DataFrame(columns=["SK_ID_CURR"])


# =====================================================================
# 7. installments: 期間別集約の細分化（60/90/180/1000日）+ 回次別集約（初回〜4回目）
#    + 深掘り特徴（指数減衰加重DPD/早期完済/直近-全期間比率）(P4)
# =====================================================================
INS_PERIOD_DAYS = [60, 90, 180, 1000]


def installments_advanced_features(ins: pd.DataFrame) -> pd.DataFrame:
    """
    installments深掘り特徴(P4)。既存の期間別集約(mean/sum等の水準)を補完する3種類:
      - INS_DPD_EWM_SUM      : 指数減衰加重(tau=180日)したDPD(延滞日数)合計。
                               「直近ほど重い今の返済ストレス」を1本の連続値に要約する。
      - INS_EARLY_PAY_RATIO / INS_EARLY_PAY_DAYS_mean : 前倒し払い(早期完済)の比率と平均日数。
                               延滞の逆シグナルを明示的に特徴化（良好な顧客ほど高い）。
    直近 vs 全期間の「比率」(既存INS_TREND_LATEは差分)は get_aggregated_installments 側で
    ins_agg構築後に追加する（このスライスからだけでは全期間水準にアクセスできないため）。
    """
    needed = {"SK_ID_CURR", "DAYS_INSTALMENT", "PAYMENT_DELAY"}
    if not needed.issubset(ins.columns):
        return pd.DataFrame(columns=["SK_ID_CURR"])
    d = ins[["SK_ID_CURR", "DAYS_INSTALMENT", "PAYMENT_DELAY"]].dropna().copy()
    if d.empty:
        return pd.DataFrame(columns=["SK_ID_CURR"])

    # 指数減衰加重DPD合計：DAYS_INSTALMENTは負値で0に近いほど直近 -> 重みが大きくなる
    dpd = d["PAYMENT_DELAY"].clip(lower=0).astype(np.float64)
    w = np.exp(d["DAYS_INSTALMENT"].astype(np.float64) / 180.0)
    ewm_sum = (w * dpd).groupby(d["SK_ID_CURR"]).sum().rename("INS_DPD_EWM_SUM")

    # 早期完済(前倒し払い)の比率と平均日数
    is_early = (d["PAYMENT_DELAY"] < 0)
    early_ratio = is_early.astype(np.float64).groupby(d["SK_ID_CURR"]).mean().rename("INS_EARLY_PAY_RATIO")
    early_days = (-d["PAYMENT_DELAY"]).where(is_early)
    early_days_mean = early_days.groupby(d["SK_ID_CURR"]).mean().rename("INS_EARLY_PAY_DAYS_mean")

    out = pd.concat([ewm_sum, early_ratio, early_days_mean], axis=1)
    out.index.name = "SK_ID_CURR"
    return out.reset_index()


def installments_period_slices(ins: pd.DataFrame) -> pd.DataFrame:
    """既存は全期間/1年のみだった期間別集約を60/90/180/1000日に細分化する。"""
    if not {"SK_ID_CURR", "DAYS_INSTALMENT"}.issubset(ins.columns):
        return pd.DataFrame(columns=["SK_ID_CURR"])
    rules = {}
    for c, aggs in [("PAYMENT_DEFICIT", ["mean", "sum"]), ("PAYMENT_RATIO", ["mean", "min"]),
                     ("IS_LATE", ["mean", "sum"]), ("AMT_INSTALMENT", ["mean", "sum"])]:
        if c in ins.columns:
            rules[c] = aggs
    if not rules:
        return pd.DataFrame(columns=["SK_ID_CURR"])

    out = None
    for days in INS_PERIOD_DAYS:
        subset = ins[ins["DAYS_INSTALMENT"] >= -days]
        if subset.empty:
            continue
        agg = subset.groupby("SK_ID_CURR").agg(rules)
        agg = flatten_agg_columns(agg, f"INS_{days}D").reset_index()
        out = agg if out is None else out.merge(agg, on="SK_ID_CURR", how="outer")
    return out if out is not None else pd.DataFrame(columns=["SK_ID_CURR"])


def installments_by_number(ins: pd.DataFrame, max_n: int = 4) -> pd.DataFrame:
    """初回〜max_n回目の支払行動別集約（初回延滞は強いデフォルトシグナルとされる）。"""
    if not {"SK_ID_CURR", "NUM_INSTALMENT_NUMBER"}.issubset(ins.columns):
        return pd.DataFrame(columns=["SK_ID_CURR"])
    rules = {}
    for c in ["PAYMENT_RATIO", "IS_LATE", "PAYMENT_DEFICIT"]:
        if c in ins.columns:
            rules[c] = ["mean"]
    if not rules:
        return pd.DataFrame(columns=["SK_ID_CURR"])

    out = None
    for n in range(1, max_n + 1):
        subset = ins[ins["NUM_INSTALMENT_NUMBER"] == n]
        if subset.empty:
            continue
        agg = subset.groupby("SK_ID_CURR").agg(rules)
        agg = flatten_agg_columns(agg, f"INS_NUM{n}").reset_index()
        out = agg if out is None else out.merge(agg, on="SK_ID_CURR", how="outer")
    return out if out is not None else pd.DataFrame(columns=["SK_ID_CURR"])
