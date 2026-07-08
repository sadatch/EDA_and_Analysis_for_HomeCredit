"""
ギャップ特徴量（チャット提案分の実装）。HC_FE_GAP=0 で丸ごと無効化できる。

カテゴリ（列名プレフィックスで識別）:
  GAP_TL_   時間軸インターリーブ系
            - 同時進行ローンの最大件数（bureauの区間重なりsweep line）
            - 最後の完済からの経過日数 / 借入開始間隔の最大値（借金空白期間）
            - アクティブローンの残存月数
            - 直近の謝絶/承認からの経過日数、最新決定が謝絶だったかフラグ
  GAP_BEH_  行動の質系
            - POS: 予定より早く完済した契約の数・比率（CNT_INSTALMENT_FUTURE>0のままCompleted）
            - CC: 支払いが最低額に張り付いている月の比率（全期間/直近12M）
  GAP_PRES_ テーブル存在・欠損パターン系
            - bureau/prev/POS/INS/CC 履歴の有無フラグ + 欠けているテーブル数
            - 行単位のNaN数・比率（集約特徴の欠損パターン自体を情報として使う）
  GAP_CON_  テーブル間整合性・自己申告の怪しさ系
            - 収入のキリ番フラグ・末尾ゼロ数（自己申告の水増しシグナル）
            - 全債務横断の月次返済負担 / 収入（bureau年金＋今回annuity）
            - bureau残債があるのに直近1年の照会が0の不整合フラグ
            - 新規ローン期間と既存債務残存期間のオーバーラップ月数・比率
  COMBO_ / FREQ_  カテゴリ組合せ列（OOF TEはfeature_engineering側で付与）と
            その出現頻度エンコーディング（train+test結合で算出、target非依存）

すべて「必要な列が存在するときだけ」作るので、合成データ（スモーク）でも実データでも落ちない。
"""
import gc

import numpy as np
import pandas as pd

EPS = 1e-5

# カテゴリ組合せの候補（存在するものだけ使う）
COMBO_PAIRS = [
    ("ORGANIZATION_TYPE", "OCCUPATION_TYPE"),
    ("NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE"),
    ("CODE_GENDER", "NAME_EDUCATION_TYPE"),
]

# テーブル存在フラグ: 列プレフィックス -> フラグ名
PRESENCE_PREFIXES = {
    "BUREAU_": "GAP_PRES_HAS_BUREAU",
    "PREV_": "GAP_PRES_HAS_PREV",
    "POS_": "GAP_PRES_HAS_POS",
    "INS_": "GAP_PRES_HAS_INS",
    "CC_": "GAP_PRES_HAS_CC",
}


def _first(df: pd.DataFrame, names):
    """候補リストのうち最初に存在する列をSeriesで返す（無ければNone）。"""
    for n in names:
        if n in df.columns:
            return pd.to_numeric(df[n], errors="coerce")
    return None


# =====================================================================
# 1. bureau: 時間軸インターリーブ（生テーブルから計算し SK_ID_CURR 単位で返す）
# =====================================================================
def bureau_timeline_features(bureau: pd.DataFrame) -> pd.DataFrame:
    if bureau.empty or not {"SK_ID_CURR", "DAYS_CREDIT"}.issubset(bureau.columns):
        return pd.DataFrame()

    start = pd.to_numeric(bureau["DAYS_CREDIT"], errors="coerce")
    end_fact = _first(bureau, ["DAYS_ENDDATE_FACT"])
    end_plan = _first(bureau, ["DAYS_CREDIT_ENDDATE"])
    if end_fact is None:
        end_fact = pd.Series(np.nan, index=bureau.index)
    if end_plan is None:
        end_plan = pd.Series(np.nan, index=bureau.index)
    # 実終了日 > 予定終了日 > 申込日(=0, 継続中とみなす) の順で補完。未来の予定終了は0でクリップ
    end = end_fact.fillna(end_plan).fillna(0.0).clip(upper=0.0)
    end = np.maximum(end.values, start.values)  # end < start の汚れデータをガード

    b = pd.DataFrame({"SK_ID_CURR": bureau["SK_ID_CURR"].values,
                      "START": start.values, "END": end}).dropna(subset=["START"])
    if b.empty:
        return pd.DataFrame()
    feats = []

    # --- 同時進行ローンの最大件数（sweep line） ---
    starts = b[["SK_ID_CURR", "START"]].rename(columns={"START": "T"})
    starts["D"] = 1
    ends = b[["SK_ID_CURR", "END"]].rename(columns={"END": "T"})
    ends["D"] = -1
    ends["T"] = ends["T"] + 0.5  # 終了直後に開始した場合を重複と数えないためのオフセット
    ev = pd.concat([starts, ends], ignore_index=True).sort_values(
        ["SK_ID_CURR", "T"], kind="mergesort")
    ev["CUM"] = ev.groupby("SK_ID_CURR")["D"].cumsum()
    feats.append(ev.groupby("SK_ID_CURR")["CUM"].max()
                 .rename("GAP_TL_MAX_SIMULTANEOUS_LOANS").astype(np.float32))
    del starts, ends, ev

    # --- 最後の完済からの経過日数 ---
    if "CREDIT_ACTIVE" in bureau.columns:
        closed_mask = bureau["CREDIT_ACTIVE"].astype(str).eq("Closed").values
    else:
        closed_mask = b["END"].values < 0
    closed = b[closed_mask & (b["END"].values < 0)]
    if not closed.empty:
        feats.append((-closed.groupby("SK_ID_CURR")["END"].max())
                     .rename("GAP_TL_DAYS_SINCE_LAST_CLOSURE").astype(np.float32))

    # --- 借入開始間隔の最大値（借金空白期間の近似） ---
    b_sorted = b.sort_values(["SK_ID_CURR", "START"], kind="mergesort")
    gaps = b_sorted.groupby("SK_ID_CURR")["START"].diff()
    gap_max = gaps.groupby(b_sorted["SK_ID_CURR"]).max()
    feats.append(gap_max.rename("GAP_TL_LOAN_START_GAP_MAX").astype(np.float32))

    # --- アクティブローンの残存月数 ---
    if "CREDIT_ACTIVE" in bureau.columns:
        active_mask = bureau["CREDIT_ACTIVE"].astype(str).eq("Active")
    else:
        active_mask = end_plan > 0
    remain = end_plan.where(active_mask).clip(lower=0) / 30.44
    rem = pd.DataFrame({"SK_ID_CURR": bureau["SK_ID_CURR"], "R": remain}).dropna()
    if not rem.empty:
        g = rem.groupby("SK_ID_CURR")["R"]
        feats.append(g.max().rename("GAP_TL_ACTIVE_REMAIN_MONTHS_MAX").astype(np.float32))
        feats.append(g.mean().rename("GAP_TL_ACTIVE_REMAIN_MONTHS_MEAN").astype(np.float32))

    out = pd.concat(feats, axis=1).reset_index()
    print(f"  [gap/bureau] タイムライン特徴 {out.shape[1]-1}列を付与")
    gc.collect()
    return out


# =====================================================================
# 2. previous_application: 謝絶/承認のrecency
# =====================================================================
def previous_recency_features(prev: pd.DataFrame) -> pd.DataFrame:
    need = {"SK_ID_CURR", "DAYS_DECISION", "NAME_CONTRACT_STATUS"}
    if prev.empty or not need.issubset(prev.columns):
        return pd.DataFrame()
    p = prev[list(need)].copy()
    p["DAYS_DECISION"] = pd.to_numeric(p["DAYS_DECISION"], errors="coerce")
    st = p["NAME_CONTRACT_STATUS"].astype(str)
    feats = []

    refused = p[st == "Refused"].groupby("SK_ID_CURR")["DAYS_DECISION"].max()
    if not refused.empty:
        feats.append((-refused).rename("GAP_TL_DAYS_SINCE_LAST_REFUSAL").astype(np.float32))
    approved = p[st == "Approved"].groupby("SK_ID_CURR")["DAYS_DECISION"].max()
    if not approved.empty:
        feats.append((-approved).rename("GAP_TL_DAYS_SINCE_LAST_APPROVAL").astype(np.float32))

    last = p.sort_values("DAYS_DECISION", kind="mergesort").groupby("SK_ID_CURR").tail(1)
    feats.append(last.set_index("SK_ID_CURR")["NAME_CONTRACT_STATUS"].astype(str)
                 .eq("Refused").astype(np.int8).rename("GAP_TL_LAST_DECISION_REFUSED"))

    out = pd.concat(feats, axis=1)
    if {"GAP_TL_DAYS_SINCE_LAST_REFUSAL", "GAP_TL_DAYS_SINCE_LAST_APPROVAL"}.issubset(out.columns):
        # <1 なら「謝絶の方が承認より最近」= 直近で信用力が低下しているシグナル
        out["GAP_TL_REFUSAL_RECENCY_RATIO"] = (
            (out["GAP_TL_DAYS_SINCE_LAST_REFUSAL"] + 1.0)
            / (out["GAP_TL_DAYS_SINCE_LAST_APPROVAL"] + 1.0)
        ).astype(np.float32)
    print(f"  [gap/prev] recency特徴 {out.shape[1]}列を付与")
    return out.reset_index()


# =====================================================================
# 3. POS: 早期完済（優良客シグナル）
# =====================================================================
def pos_behavior_features(pos: pd.DataFrame) -> pd.DataFrame:
    need = {"SK_ID_CURR", "SK_ID_PREV", "MONTHS_BALANCE",
            "NAME_CONTRACT_STATUS", "CNT_INSTALMENT_FUTURE"}
    if pos.empty or not need.issubset(pos.columns):
        return pd.DataFrame()
    last = (pos.sort_values("MONTHS_BALANCE", kind="mergesort")
            .groupby("SK_ID_PREV").tail(1))
    early = (last["NAME_CONTRACT_STATUS"].astype(str).eq("Completed")
             & (pd.to_numeric(last["CNT_INSTALMENT_FUTURE"], errors="coerce") > 0))
    df = pd.DataFrame({"SK_ID_CURR": last["SK_ID_CURR"].values,
                       "EARLY": early.astype(np.int8).values})
    g = df.groupby("SK_ID_CURR")["EARLY"]
    out = pd.DataFrame({
        "GAP_BEH_POS_EARLY_COMPLETE_COUNT": g.sum().astype(np.float32),
        "GAP_BEH_POS_EARLY_COMPLETE_RATIO": g.mean().astype(np.float32),
    }).reset_index()
    print("  [gap/pos] 早期完済特徴 2列を付与")
    return out


# =====================================================================
# 4. credit_card: 最低額張り付き（リボ苦境シグナル）
# =====================================================================
def credit_card_behavior_features(cc: pd.DataFrame) -> pd.DataFrame:
    need = {"SK_ID_CURR", "MONTHS_BALANCE", "AMT_BALANCE",
            "AMT_PAYMENT_TOTAL_CURRENT", "AMT_INST_MIN_REGULARITY"}
    if cc.empty or not need.issubset(cc.columns):
        return pd.DataFrame()
    bal = pd.to_numeric(cc["AMT_BALANCE"], errors="coerce")
    pay = pd.to_numeric(cc["AMT_PAYMENT_TOTAL_CURRENT"], errors="coerce")
    minreg = pd.to_numeric(cc["AMT_INST_MIN_REGULARITY"], errors="coerce")
    minonly = ((bal > 0) & (minreg > 0) & (pay <= minreg * 1.05)).astype(np.int8)
    df = pd.DataFrame({"SK_ID_CURR": cc["SK_ID_CURR"].values,
                       "MONTHS_BALANCE": pd.to_numeric(cc["MONTHS_BALANCE"], errors="coerce").values,
                       "MINONLY": minonly.values})
    feats = [df.groupby("SK_ID_CURR")["MINONLY"].mean()
             .rename("GAP_BEH_CC_MINPAY_ONLY_RATIO").astype(np.float32)]
    recent = df[df["MONTHS_BALANCE"] >= -12]
    if not recent.empty:
        feats.append(recent.groupby("SK_ID_CURR")["MINONLY"].mean()
                     .rename("GAP_BEH_CC_MINPAY_ONLY_RATIO_12M").astype(np.float32))
    out = pd.concat(feats, axis=1).reset_index()
    print(f"  [gap/cc] 最低額張り付き特徴 {out.shape[1]-1}列を付与")
    return out


# =====================================================================
# 5. マージ後: 存在フラグ / NaNパターン / キリ番 / 整合性 / カテゴリ組合せ
# =====================================================================
def add_post_merge_gap_features(app_train: pd.DataFrame, app_test: pd.DataFrame) -> tuple:
    n_created = 0

    # --- テーブル存在フラグ（集約列のNaNパターンから逆算） ---
    for prefix, flag_name in PRESENCE_PREFIXES.items():
        cols = [c for c in app_train.columns
                if c.startswith(prefix) and c in app_test.columns
                and pd.api.types.is_numeric_dtype(app_train[c])]
        if not cols:
            continue
        probe = cols[:30]  # 全列は不要。先頭30列で十分に判定できる
        for df in (app_train, app_test):
            df[flag_name] = df[probe].notna().any(axis=1).astype(np.int8)
        n_created += 1
    flag_cols = [f for f in PRESENCE_PREFIXES.values() if f in app_train.columns]
    if flag_cols:
        for df in (app_train, app_test):
            df["GAP_PRES_MISSING_TABLE_COUNT"] = (len(flag_cols) - df[flag_cols].sum(axis=1)).astype(np.int8)
        n_created += 1

    # --- 行単位のNaN数・比率（train/testで同じ列集合を使う） ---
    num_cols = [c for c in app_train.columns
                if c not in ("SK_ID_CURR", "TARGET")
                and c in app_test.columns
                and pd.api.types.is_numeric_dtype(app_train[c])]
    if num_cols:
        for df in (app_train, app_test):
            nan_cnt = df[num_cols].isna().sum(axis=1)
            df["GAP_PRES_ROW_NAN_COUNT"] = nan_cnt.astype(np.int16)
            df["GAP_PRES_ROW_NAN_RATIO"] = (nan_cnt / len(num_cols)).astype(np.float32)
        n_created += 2

    for df in (app_train, app_test):
        # --- 収入・借入額のキリ番（自己申告の水増しシグナル） ---
        if "AMT_INCOME_TOTAL" in df.columns:
            inc = pd.to_numeric(df["AMT_INCOME_TOTAL"], errors="coerce").fillna(0)
            df["GAP_CON_INCOME_ROUND_50K"] = (inc.mod(50000) == 0).astype(np.int8)
            df["GAP_CON_INCOME_ROUND_100K"] = (inc.mod(100000) == 0).astype(np.int8)
            inc_str = inc.round().astype(np.int64).astype(str)
            df["GAP_CON_INCOME_TRAILING_ZEROS"] = (
                inc_str.str.len() - inc_str.str.rstrip("0").str.len()
            ).astype(np.int8)
        if "AMT_CREDIT" in df.columns:
            crd = pd.to_numeric(df["AMT_CREDIT"], errors="coerce").fillna(0)
            df["GAP_CON_CREDIT_ROUND_50K"] = (crd.mod(50000) == 0).astype(np.int8)

        # --- 全債務横断の月次返済負担 / 収入 ---
        if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"}.issubset(df.columns):
            bureau_ann = _first(df, ["BUREAU_ACT_AMT_ANNUITY_sum", "BUREAU_AMT_ANNUITY_sum"])
            if bureau_ann is not None:
                total = pd.to_numeric(df["AMT_ANNUITY"], errors="coerce").fillna(0) + bureau_ann.fillna(0)
                df["GAP_CON_TOTAL_ANNUITY_INCOME_RATIO"] = (
                    total / (pd.to_numeric(df["AMT_INCOME_TOTAL"], errors="coerce") + EPS)
                ).astype(np.float32)

        # --- bureau残債あり × 照会ゼロの不整合 ---
        debt = _first(df, ["BUREAU_AMT_CREDIT_SUM_DEBT_sum"])
        if debt is not None and "AMT_REQ_CREDIT_BUREAU_YEAR" in df.columns:
            inq = pd.to_numeric(df["AMT_REQ_CREDIT_BUREAU_YEAR"], errors="coerce")
            df["GAP_CON_DEBT_NO_INQUIRY"] = ((debt.fillna(0) > 0) & (inq.fillna(0) == 0)).astype(np.int8)

        # --- 新規ローン期間と既存債務残存期間のオーバーラップ ---
        if {"AMT_CREDIT", "AMT_ANNUITY"}.issubset(df.columns) and "GAP_TL_ACTIVE_REMAIN_MONTHS_MAX" in df.columns:
            new_term = (pd.to_numeric(df["AMT_CREDIT"], errors="coerce")
                        / (pd.to_numeric(df["AMT_ANNUITY"], errors="coerce") + EPS)).clip(0, 120)
            remain = pd.to_numeric(df["GAP_TL_ACTIVE_REMAIN_MONTHS_MAX"], errors="coerce").fillna(0)
            overlap = np.minimum(new_term, remain)
            df["GAP_CON_TERM_OVERLAP_MONTHS"] = overlap.astype(np.float32)
            df["GAP_CON_TERM_OVERLAP_RATIO"] = (overlap / (new_term + EPS)).astype(np.float32)

    # --- カテゴリ組合せ + frequency encoding（train+test結合、target非依存） ---
    combo_cols = []
    for a, b in COMBO_PAIRS:
        if not ({a, b}.issubset(app_train.columns) and {a, b}.issubset(app_test.columns)):
            continue
        name = f"COMBO_{a}_{b}"
        for df in (app_train, app_test):
            df[name] = df[a].astype(str).fillna("NA") + "__" + df[b].astype(str).fillna("NA")
        combo_cols.append(name)
    freq_targets = combo_cols + [c for c in ["ORGANIZATION_TYPE"]
                                 if c in app_train.columns and c in app_test.columns]
    for c in freq_targets:
        vc = pd.concat([app_train[c], app_test[c]], ignore_index=True).astype(str).value_counts()
        for df in (app_train, app_test):
            df[f"FREQ_{c}"] = df[c].astype(str).map(vc).astype(np.float32)

    print(f"  [gap/post-merge] 存在フラグ/NaN/キリ番/整合性 + combo {len(combo_cols)}列 "
          f"+ freq {len(freq_targets)}列を付与")
    gc.collect()
    return app_train, app_test


def get_combo_columns(df: pd.DataFrame) -> list:
    """OOF target encoding対象にするカテゴリ組合せ列名を返す。"""
    return [c for c in df.columns if c.startswith("COMBO_")]
