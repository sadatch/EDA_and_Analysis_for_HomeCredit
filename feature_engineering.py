"""
特徴量エンジニアリング パイプライン。

既存notebook (home-ccredit-re-03.ipynb) の実装をベースに、以下を追加で拡張：
  - bureau_balance.csv の集約をbureauにマージ（DPDステータスのトレンド）
  - credit_card_balance.csv のフル集約（既存はstreak特徴のみだった）
  - installments_payments.csv の全期間集約（既存は直近1年のみだった）
  - previous_application の「直近の申請」特徴（最新1件のスナップショット）
1位チームの知見（期間別集約、直近トレンド検知）は既存実装をそのまま継承。

出力:
  data/processed/train_features.parquet
  data/processed/test_features.parquet
  data/processed/categorical_features.json
  data/processed/feature_columns.json
"""
import gc
import os
import json
import re

import numpy as np
import pandas as pd
import lightgbm as lgb

import config
from utils import timer, reduce_mem_usage, build_agg_rules, flatten_agg_columns, safe_merge
import oof_features
import domain_features
import extra_features
import trend_velocity_features as trend_feats
import top_solution_features as top_feats
import gap_features
import final_features


# =====================================================================
# 1. bureau_balance + bureau
# =====================================================================
def get_aggregated_bureau() -> pd.DataFrame:
    print("Processing bureau.csv / bureau_balance.csv...")
    bureau = pd.read_csv(config.RAW_DIR / config.RAW_FILES["bureau"])

    # --- bureau_balance: SK_ID_BUREAU単位でステータスのトレンドを要約してからbureauにマージ ---
    bb_path = config.RAW_DIR / config.RAW_FILES["bureau_balance"]
    if bb_path.exists():
        bb = pd.read_csv(bb_path)
        # STATUSは 'C'(closed) 'X'(unknown) '0'..'5'(DPDバケット) の文字列。
        # 0=未払いなし、1-5は遅延度合いが大きいほど深刻。数値化してトレンドを取れるようにする。
        status_map = {"C": 0, "X": 0, "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}
        bb["STATUS_NUM"] = bb["STATUS"].map(status_map).fillna(0).astype(np.int8)
        bb["IS_DPD"] = (bb["STATUS_NUM"] > 0).astype(np.int8)

        bb_agg = bb.groupby("SK_ID_BUREAU").agg(
            BB_MONTHS_COUNT=("MONTHS_BALANCE", "count"),
            BB_MONTHS_MIN=("MONTHS_BALANCE", "min"),
            BB_STATUS_MAX=("STATUS_NUM", "max"),
            BB_STATUS_MEAN=("STATUS_NUM", "mean"),
            BB_DPD_RATIO=("IS_DPD", "mean"),
            BB_DPD_COUNT=("IS_DPD", "sum"),
        ).reset_index()

        # 直近6ヶ月分のDPDトレンド（多重債務の悪化を直近で検知）
        bb_6m = bb[bb["MONTHS_BALANCE"] >= -6]
        bb_6m_agg = bb_6m.groupby("SK_ID_BUREAU").agg(
            BB_6M_DPD_RATIO=("IS_DPD", "mean"),
            BB_6M_STATUS_MAX=("STATUS_NUM", "max"),
        ).reset_index()
        bb_agg = bb_agg.merge(bb_6m_agg, on="SK_ID_BUREAU", how="left")

        if config.FE_USE_TREND:
            # STATUS_NUMの線形回帰の傾き・加重平均（水準ではなく変化の速さ・方向を捉える）
            bb_trend = trend_feats.bureau_balance_trend(bb)
            if not bb_trend.empty:
                bb_agg = bb_agg.merge(bb_trend, on="SK_ID_BUREAU", how="left")
            del bb_trend
            # P3: 最後の延滞から何ヶ月経過したか（recency）。6ヶ月窓の水準特徴と違い
            # 「いつ悪化したか」を直接捉える正規化された指標。
            bb_recency = trend_feats.bureau_balance_recency(bb)
            if not bb_recency.empty:
                bb_agg = bb_agg.merge(bb_recency, on="SK_ID_BUREAU", how="left")
            del bb_recency

        bureau = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")
        del bb, bb_agg, bb_6m, bb_6m_agg
        gc.collect()
    else:
        print("  bureau_balance.csv が見つからないためスキップ")

    cat_cols = bureau.select_dtypes(include=["object", "string"]).columns.tolist()
    bureau_ohe = pd.get_dummies(bureau, columns=cat_cols, dummy_na=True)
    id_cols = {"SK_ID_CURR", "SK_ID_BUREAU"}
    agg_rules = build_agg_rules(bureau_ohe, id_cols)

    # ① 全期間の集約
    bureau_agg = bureau_ohe.groupby("SK_ID_CURR").agg(agg_rules)
    bureau_agg = flatten_agg_columns(bureau_agg, "BUREAU")

    # ② 直近6ヶ月（DAYS_CREDIT >= -180）：短期多重申込みの検知（1位チーム手法）
    bureau_6m = bureau_ohe[bureau_ohe["DAYS_CREDIT"] >= -180]
    if not bureau_6m.empty:
        bureau_6m_agg = bureau_6m.groupby("SK_ID_CURR").agg(agg_rules)
        bureau_6m_agg = flatten_agg_columns(bureau_6m_agg, "BUREAU_6M")
        bureau_agg = safe_merge(bureau_agg.reset_index(), bureau_6m_agg.reset_index(), on="SK_ID_CURR").set_index("SK_ID_CURR")
        del bureau_6m_agg

    # ③ 直近1年（DAYS_CREDIT >= -365）
    bureau_1yr = bureau_ohe[bureau_ohe["DAYS_CREDIT"] >= -365]
    if not bureau_1yr.empty:
        bureau_1yr_agg = bureau_1yr.groupby("SK_ID_CURR").agg(agg_rules)
        bureau_1yr_agg = flatten_agg_columns(bureau_1yr_agg, "BUREAU_1YR")
        bureau_agg = safe_merge(bureau_agg.reset_index(), bureau_1yr_agg.reset_index(), on="SK_ID_CURR").set_index("SK_ID_CURR")
        del bureau_1yr_agg

    if config.FE_USE_TREND:
        # 信用の種類の多様性（OHE+sum/meanでは失われるdistinct数。クレジットハンガーの別シグナル）
        diversity = trend_feats.bureau_credit_type_diversity(bureau)
        if not diversity.empty:
            bureau_agg = safe_merge(bureau_agg.reset_index(), diversity, on="SK_ID_CURR").set_index("SK_ID_CURR")
        del diversity

    if config.FE_USE_TOP_SOLUTION:
        # アクティブローンに絞った直近性・残債合計（1位解法discussion由来）
        last_active = top_feats.bureau_last_active_snapshot(bureau)
        if not last_active.empty:
            bureau_agg = safe_merge(bureau_agg.reset_index(), last_active, on="SK_ID_CURR").set_index("SK_ID_CURR")
        del last_active

    if config.FE_USE_GAP:
        # 同時進行ローン数・完済recency・借入空白期間・アクティブ残存月数（チャット提案分）
        timeline = gap_features.bureau_timeline_features(bureau)
        if not timeline.empty:
            bureau_agg = safe_merge(bureau_agg.reset_index(), timeline, on="SK_ID_CURR").set_index("SK_ID_CURR")
        del timeline

    del bureau, bureau_ohe, bureau_6m, bureau_1yr
    gc.collect()
    return reduce_mem_usage(bureau_agg.reset_index())


# =====================================================================
# 2. previous_application
# =====================================================================
def get_aggregated_previous() -> pd.DataFrame:
    print("Processing previous_application.csv...")
    prev = pd.read_csv(config.RAW_DIR / config.RAW_FILES["previous"])

    # 1位チーム手法：過去ローンスペックからの「金利」逆算
    prev["ESTIMATED_TOTAL_INTEREST"] = prev["AMT_ANNUITY"] * prev["CNT_PAYMENT"] - prev["AMT_CREDIT"]
    prev["ESTIMATED_INTEREST_RATE"] = prev["ESTIMATED_TOTAL_INTEREST"] / (prev["AMT_CREDIT"] + 1e-5)
    # 申請額と実際の承認額のギャップ（希望より少なく借りられた=リスク評価が厳しかった可能性）
    prev["APP_CREDIT_DIFF"] = prev["AMT_APPLICATION"] - prev["AMT_CREDIT"]
    prev["APP_CREDIT_RATIO"] = prev["AMT_CREDIT"] / (prev["AMT_APPLICATION"] + 1e-5)

    # --- 直近の申請1件のスナップショット（最新の意思決定状況を直接特徴化） ---
    prev_sorted = prev.sort_values(["SK_ID_CURR", "DAYS_DECISION"], ascending=[True, False])
    last_app = prev_sorted.groupby("SK_ID_CURR").first().reset_index()
    last_app_num_cols = [c for c in last_app.select_dtypes(include=[np.number]).columns
                          if c not in ("SK_ID_CURR", "SK_ID_PREV")]
    last_app_feats = last_app[["SK_ID_CURR"] + last_app_num_cols].copy()
    last_app_feats.columns = ["SK_ID_CURR"] + [f"PREV_LAST_{c}" for c in last_app_num_cols]
    del prev_sorted, last_app
    gc.collect()

    cat_cols = prev.select_dtypes(include=["object", "string"]).columns.tolist()
    prev_ohe = pd.get_dummies(prev, columns=cat_cols, dummy_na=True)
    id_cols = {"SK_ID_CURR", "SK_ID_PREV"}
    agg_rules = build_agg_rules(prev_ohe, id_cols)

    prev_agg = prev_ohe.groupby("SK_ID_CURR").agg(agg_rules)
    prev_agg = flatten_agg_columns(prev_agg, "PREV").reset_index()
    prev_agg = safe_merge(prev_agg, last_app_feats, on="SK_ID_CURR")

    if config.FE_USE_TREND:
        # 自社への申込間隔（ベロシティ）＋直近3件の謝絶率（bureauの他社照会件数とは別軸）
        velocity = trend_feats.previous_application_velocity(prev)
        if not velocity.empty:
            prev_agg = safe_merge(prev_agg, velocity, on="SK_ID_CURR")
        del velocity

    if config.FE_USE_TOP_SOLUTION:
        # 直近3/5件・最初2/4件のスライス集約 + 最新PRODUCT_COMBINATION（1位解法discussion由来）
        slices = top_feats.previous_application_slices(prev)
        if not slices.empty:
            prev_agg = safe_merge(prev_agg, slices, on="SK_ID_CURR")
        del slices

    if config.FE_USE_GAP:
        # 直近の謝絶/承認からの経過日数・最新決定が謝絶かフラグ（チャット提案分）
        recency = gap_features.previous_recency_features(prev)
        if not recency.empty:
            prev_agg = safe_merge(prev_agg, recency, on="SK_ID_CURR")
        del recency

    del prev, prev_ohe, last_app_feats
    gc.collect()
    return reduce_mem_usage(prev_agg)


# =====================================================================
# 3. POS_CASH_balance
# =====================================================================
def get_aggregated_pos_cash() -> pd.DataFrame:
    print("Processing POS_CASH_balance.csv...")
    pos = pd.read_csv(config.RAW_DIR / config.RAW_FILES["pos_cash"])
    cat_cols = pos.select_dtypes(include=["object", "string"]).columns.tolist()
    pos_ohe = pd.get_dummies(pos, columns=cat_cols, dummy_na=True)
    id_cols = {"SK_ID_CURR", "SK_ID_PREV"}
    agg_rules = build_agg_rules(pos_ohe, id_cols, numeric_aggs=("mean", "max", "min"))

    pos_agg = pos_ohe.groupby("SK_ID_CURR").agg(agg_rules)
    pos_agg = flatten_agg_columns(pos_agg, "POS")

    # 直近3ヶ月：急激な資金ショートの検知
    pos_3m = pos_ohe[pos_ohe["MONTHS_BALANCE"] >= -3]
    if not pos_3m.empty:
        pos_3m_agg = pos_3m.groupby("SK_ID_CURR").agg(agg_rules)
        pos_3m_agg = flatten_agg_columns(pos_3m_agg, "POS_3M")
        pos_agg = safe_merge(pos_agg.reset_index(), pos_3m_agg.reset_index(), on="SK_ID_CURR").set_index("SK_ID_CURR")
        del pos_3m_agg

    if config.FE_USE_TREND:
        # SK_DPDの線形回帰の傾き（ローン単位で悪化速度を捉えてから顧客単位に集約）
        pos_trend = trend_feats.pos_cash_trend_features(pos)
        if not pos_trend.empty:
            pos_agg = safe_merge(pos_agg.reset_index(), pos_trend, on="SK_ID_CURR").set_index("SK_ID_CURR")
        del pos_trend

    if config.FE_USE_GAP:
        # 予定より早く完済した契約数・比率（優良客シグナル、チャット提案分）
        pos_beh = gap_features.pos_behavior_features(pos)
        if not pos_beh.empty:
            pos_agg = safe_merge(pos_agg.reset_index(), pos_beh, on="SK_ID_CURR").set_index("SK_ID_CURR")
        del pos_beh

    del pos, pos_ohe, pos_3m
    gc.collect()
    return reduce_mem_usage(pos_agg.reset_index())


# =====================================================================
# 4. installments_payments（全期間 + 直近1年）
# =====================================================================
def get_aggregated_installments() -> pd.DataFrame:
    print("Processing installments_payments.csv...")
    ins = pd.read_csv(config.RAW_DIR / config.RAW_FILES["installments"])

    ins["PAYMENT_DEFICIT"] = ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]
    ins["PAYMENT_RATIO"] = ins["AMT_PAYMENT"] / (ins["AMT_INSTALMENT"] + 1e-5)
    ins["PAYMENT_DELAY"] = ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]
    ins["IS_LATE"] = (ins["PAYMENT_DELAY"] > 0).astype(np.int8)
    ins["IS_UNDERPAID"] = (ins["PAYMENT_DEFICIT"] > 0).astype(np.int8)

    base_rules = {
        "PAYMENT_DEFICIT": ["max", "mean", "sum", "std"],
        "PAYMENT_RATIO": ["min", "mean", "std"],
        "PAYMENT_DELAY": ["max", "min", "mean", "std"],
        "AMT_INSTALMENT": ["max", "min", "mean", "sum"],
        "AMT_PAYMENT": ["max", "min", "mean", "sum"],
        "IS_LATE": ["mean", "sum"],
        "IS_UNDERPAID": ["mean", "sum"],
        "NUM_INSTALMENT_NUMBER": ["max"],
    }

    # ① 全期間集約（既存notebookは直近1年のみだったのでフル期間を追加）
    ins_agg_all = ins.groupby("SK_ID_CURR").agg(base_rules)
    ins_agg_all = flatten_agg_columns(ins_agg_all, "INS_ALL").reset_index()

    # ② 直近1年（既存notebookの実装を継承）
    ins_1yr = ins[ins["DAYS_INSTALMENT"] >= -365]
    ins_1yr_agg = ins_1yr.groupby("SK_ID_CURR").agg(base_rules)
    ins_1yr_agg = flatten_agg_columns(ins_1yr_agg, "INS_1YR").reset_index()

    ins_agg = safe_merge(ins_agg_all, ins_1yr_agg, on="SK_ID_CURR")

    # ③ 直近3回の支払行動スナップショット（最新の挙動を直接特徴化）
    ins_sorted = ins.sort_values(["SK_ID_CURR", "DAYS_INSTALMENT"], ascending=[True, False])
    last3 = ins_sorted.groupby("SK_ID_CURR").head(3)
    last3_agg = last3.groupby("SK_ID_CURR").agg(
        INS_LAST3_LATE_MEAN=("IS_LATE", "mean"),
        INS_LAST3_DEFICIT_MEAN=("PAYMENT_DEFICIT", "mean"),
        INS_LAST3_PAYRATIO_MEAN=("PAYMENT_RATIO", "mean"),
        INS_LAST3_DELAY_MAX=("PAYMENT_DELAY", "max"),
    ).reset_index()
    ins_agg = safe_merge(ins_agg, last3_agg, on="SK_ID_CURR")

    if config.FE_USE_TREND:
        # PAYMENT_RATIOの線形回帰の傾き（ローン単位で悪化速度を捉えてから顧客単位に集約）
        ins_trend = trend_feats.installments_trend_features(ins)
        if not ins_trend.empty:
            ins_agg = safe_merge(ins_agg, ins_trend, on="SK_ID_CURR")
        del ins_trend

    if config.FE_USE_TOP_SOLUTION:
        # 期間別集約の細分化（60/90/180/1000日）+ 回次別集約（初回〜4回目）（1位解法discussion由来）
        ins_periods = top_feats.installments_period_slices(ins)
        if not ins_periods.empty:
            ins_agg = safe_merge(ins_agg, ins_periods, on="SK_ID_CURR")
        del ins_periods
        ins_by_num = top_feats.installments_by_number(ins)
        if not ins_by_num.empty:
            ins_agg = safe_merge(ins_agg, ins_by_num, on="SK_ID_CURR")
        del ins_by_num
        # P4: 指数減衰加重DPD合計 / 早期完済比率・日数（installments深掘り特徴）
        ins_adv = top_feats.installments_advanced_features(ins)
        if not ins_adv.empty:
            ins_agg = safe_merge(ins_agg, ins_adv, on="SK_ID_CURR")
        del ins_adv

    # ④ 悪化トレンド（直近1年 vs 全期間のデルタ。プラスなら直近で悪化）
    if "INS_1YR_IS_LATE_mean" in ins_agg and "INS_ALL_IS_LATE_mean" in ins_agg:
        ins_agg["INS_TREND_LATE"] = ins_agg["INS_1YR_IS_LATE_mean"] - ins_agg["INS_ALL_IS_LATE_mean"]
        # P4: 差分だけでなく比率でも捉える（悪化率の非線形な強調, 分母0近傍はEPSでガード）
        ins_agg["INS_TREND_LATE_RATIO"] = (
            ins_agg["INS_1YR_IS_LATE_mean"] / (ins_agg["INS_ALL_IS_LATE_mean"] + 1e-5)
        )
    if "INS_1YR_PAYMENT_DEFICIT_mean" in ins_agg and "INS_ALL_PAYMENT_DEFICIT_mean" in ins_agg:
        ins_agg["INS_TREND_DEFICIT"] = ins_agg["INS_1YR_PAYMENT_DEFICIT_mean"] - ins_agg["INS_ALL_PAYMENT_DEFICIT_mean"]

    del ins, ins_1yr, ins_agg_all, ins_1yr_agg, ins_sorted, last3, last3_agg
    gc.collect()
    return reduce_mem_usage(ins_agg)


# =====================================================================
# 5. credit_card_balance（フル集約 + streak特徴）
# =====================================================================
def get_aggregated_credit_card() -> pd.DataFrame:
    print("Processing credit_card_balance.csv...")
    cc = pd.read_csv(config.RAW_DIR / config.RAW_FILES["credit_card"])

    # 利用率（限度額に対する残高比率）：クレジットカードの危険度の核心指標
    cc["UTILIZATION"] = cc["AMT_BALANCE"] / (cc["AMT_CREDIT_LIMIT_ACTUAL"] + 1e-5)
    cc["IS_OVER_LIMIT"] = (cc["AMT_BALANCE"] > cc["AMT_CREDIT_LIMIT_ACTUAL"]).astype(np.int8)
    cc["MIN_PAYMENT_DEFICIT"] = cc["AMT_INST_MIN_REGULARITY"] - cc["AMT_PAYMENT_TOTAL_CURRENT"]

    cat_cols = cc.select_dtypes(include=["object", "string"]).columns.tolist()
    cc_ohe = pd.get_dummies(cc, columns=cat_cols, dummy_na=True)
    id_cols = {"SK_ID_CURR", "SK_ID_PREV"}
    agg_rules = build_agg_rules(cc_ohe, id_cols)

    cc_agg = cc_ohe.groupby("SK_ID_CURR").agg(agg_rules)
    cc_agg = flatten_agg_columns(cc_agg, "CC").reset_index()

    # 直近6ヶ月：直近の利用悪化を検知
    cc_6m = cc_ohe[cc_ohe["MONTHS_BALANCE"] >= -6]
    if not cc_6m.empty:
        cc_6m_agg = cc_6m.groupby("SK_ID_CURR").agg(agg_rules)
        cc_6m_agg = flatten_agg_columns(cc_6m_agg, "CC_6M").reset_index()
        cc_agg = safe_merge(cc_agg, cc_6m_agg, on="SK_ID_CURR")
        del cc_6m_agg

    if config.FE_USE_TREND:
        # 利用率(UTILIZATION)の線形回帰の傾き（ローン単位で悪化速度を捉えてから顧客単位に集約）
        cc_trend = trend_feats.credit_card_trend_features(cc)
        if not cc_trend.empty:
            cc_agg = safe_merge(cc_agg, cc_trend, on="SK_ID_CURR")
        del cc_trend

    if config.FE_USE_GAP:
        # 支払いが最低額に張り付いている月の比率（リボ苦境シグナル、チャット提案分）
        cc_beh = gap_features.credit_card_behavior_features(cc)
        if not cc_beh.empty:
            cc_agg = safe_merge(cc_agg, cc_beh, on="SK_ID_CURR")
        del cc_beh

    # --- streak特徴（既存notebook由来：残高が連続して増え続けた最大月数） ---
    cc_sorted = cc.sort_values(["SK_ID_CURR", "MONTHS_BALANCE"], ascending=[True, True])
    cc_sorted["BALANCE_DIFF"] = cc_sorted.groupby("SK_ID_CURR")["AMT_BALANCE"].diff()
    cc_sorted["IS_INCREASED"] = cc_sorted["BALANCE_DIFF"] > 0
    cc_sorted["STREAK_GROUP"] = (~cc_sorted["IS_INCREASED"]).groupby(cc_sorted["SK_ID_CURR"]).cumsum()
    streak_max = (cc_sorted[cc_sorted["IS_INCREASED"]]
                  .groupby(["SK_ID_CURR", "STREAK_GROUP"]).size()
                  .groupby("SK_ID_CURR").max())
    streak_df = pd.DataFrame({
        "SK_ID_CURR": streak_max.index,
        "CC_MAX_CONSECUTIVE_INCREASE": streak_max.values,
    })
    cc_agg = safe_merge(cc_agg, streak_df, on="SK_ID_CURR")

    del cc, cc_ohe, cc_6m, cc_sorted, streak_max, streak_df
    gc.collect()
    return reduce_mem_usage(cc_agg)


# =====================================================================
# 6. メイン処理：結合・ドメイン特徴・EXT_SOURCE欠損補完
# =====================================================================
def main():
    with timer("application_train/test 読み込み"):
        app_train = pd.read_csv(config.RAW_DIR / config.RAW_FILES["app_train"])
        app_test = pd.read_csv(config.RAW_DIR / config.RAW_FILES["app_test"])

    print("異常値処理 (DAYS_EMPLOYED=365243 -> NaN)...")
    for df in (app_train, app_test):
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    print("ドメイン特徴量を追加...")
    for df in (app_train, app_test):
        df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
        df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
        df["CREDIT_TERM"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]
        df["CREDIT_ANNUITY_RATIO"] = df["AMT_CREDIT"] / (df["AMT_ANNUITY"] + 1e-5)  # 1位の近傍特徴で使用
        df["DAYS_EMPLOYED_PERCENT"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
        df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]
        df["EXT_SOURCE_MEAN"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
        df["EXT_SOURCE_MAX"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].max(axis=1)
        df["EXT_SOURCE_MIN"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].min(axis=1)
        df["EXT_SOURCE_PROD"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
        df["EXT_SOURCE_STD"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].std(axis=1)
        # 補完前の欠損数（生の欠損パターンは情報）
        df["EXT_SOURCE_NAN_COUNT"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].isnull().sum(axis=1).astype(np.int8)

    print("算術交互作用特徴を追加 (1位チーム手法)...")
    app_train, app_test = oof_features.add_arithmetic_interactions(app_train, app_test)

    if config.FE_USE_TOP_SOLUTION:
        with timer("EXT_SOURCE_3除算特徴 / AGE_INT (1位解法discussion由来)"):
            app_train, app_test = top_feats.add_application_level_priority_features(app_train, app_test)

    table_builders = [
        ("bureau", get_aggregated_bureau),
        ("previous_application", get_aggregated_previous),
        ("POS_CASH", get_aggregated_pos_cash),
        ("installments", get_aggregated_installments),
        ("credit_card", get_aggregated_credit_card),
    ]
    for name, builder in table_builders:
        with timer(f"{name} 集約 & マージ"):
            agg_df = builder()
            app_train = safe_merge(app_train, agg_df, on="SK_ID_CURR")
            app_test = safe_merge(app_test, agg_df, on="SK_ID_CURR")
            del agg_df
            gc.collect()

    if config.FE_USE_TOP_SOLUTION:
        with timer("年利率(Newton法IRR近似) / 追加比率特徴 (1位解法discussion由来)"):
            app_train, app_test = top_feats.add_post_merge_priority_features(app_train, app_test)

    with timer("カラム名クリーニング"):
        # LightGBM/XGBoostが嫌う特殊文字を一括除去
        app_train = app_train.rename(columns=lambda x: re.sub(r"[^A-Za-z0-9_]+", "", x))
        app_test = app_test.rename(columns=lambda x: re.sub(r"[^A-Za-z0-9_]+", "", x))

    if config.FE_USE_GAP:
        with timer("ギャップ特徴 (存在フラグ/NaNパターン/キリ番/整合性/カテゴリ組合せ)"):
            # 注意: EXT_SOURCE補完・OOF特徴付与の前に呼ぶこと（生の欠損パターンを特徴化するため）
            app_train, app_test = gap_features.add_post_merge_gap_features(app_train, app_test)

    if config.FE_USE_FINAL:
        with timer("最終バッチ特徴 (定番比率/bureau後段/横断負担/交互作用/GRP2)"):
            app_train, app_test = final_features.add_final_features(app_train, app_test)

    with timer("EXT_SOURCE_1 欠損値のLightGBM予測補完"):
        features_for_imputation = [
            c for c in app_train.columns
            if app_train[c].dtype in ("int64", "float64", "int32", "float32")
            and c not in ("SK_ID_CURR", "TARGET", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
                          "EXT_SOURCE_MEAN", "EXT_SOURCE_MAX", "EXT_SOURCE_MIN", "EXT_SOURCE_PROD",
                          "EXT_SOURCE_STD")
        ]
        full_df = pd.concat([app_train, app_test], axis=0, ignore_index=True)
        known = full_df[full_df["EXT_SOURCE_1"].notnull()]
        unknown = full_df[full_df["EXT_SOURCE_1"].isnull()]
        if not unknown.empty:
            imp_model = lgb.LGBMRegressor(n_estimators=200, random_state=config.SEED, verbose=-1)
            imp_model.fit(known[features_for_imputation], known["EXT_SOURCE_1"])
            full_df.loc[full_df["EXT_SOURCE_1"].isnull(), "EXT_SOURCE_1"] = (
                imp_model.predict(unknown[features_for_imputation])
            )
        app_train = full_df[full_df["TARGET"].notnull()].copy()
        app_test = full_df[full_df["TARGET"].isnull()].drop(columns=["TARGET"]).copy()
        del full_df, known, unknown
        gc.collect()

    if config.FE_USE_DOMAIN:
        with timer("金融ドメイン特徴 (DTI/延滞/ベロシティ/利用率 等)"):
            app_train, app_test = domain_features.add_domain_features(app_train, app_test)

    if config.FE_USE_TREND:
        with timer("書類提出数/周期エンコーディング/IsolationForest異常度"):
            app_train, app_test = trend_feats.add_application_level_all(app_train, app_test)

    if config.FE_USE_NEIGHBORS:
        with timer("近傍TARGET平均 (1位の目玉特徴, OOFリーク制御)"):
            app_train, app_test = oof_features.add_neighbor_target_features(app_train, app_test)
        if config.FE_USE_NEIGHBORS_DIVERSITY:
            with timer("近傍特徴の多様化 (P2: 複数解像度k/別特徴空間/EXT_SOURCE差分)"):
                app_train, app_test = oof_features.add_neighbor_diversity_features(app_train, app_test)
        if config.FE_USE_FINAL:
            app_train, app_test = final_features.add_post_neighbor_interactions(app_train, app_test)

    if config.FE_USE_TARGET_ENC:
        with timer("OOF target encoding"):
            app_train, app_test = oof_features.add_target_encoding(app_train, app_test)
            if config.FE_USE_GAP:
                combo_cols = gap_features.get_combo_columns(app_train)
                if combo_cols:
                    app_train, app_test = oof_features.add_target_encoding(app_train, app_test, cols=combo_cols)

    # 追加特徴（EXT高次/相互作用・グループ相対z-score・k-meansクラスタ距離。target非依存）
    if os.environ.get("HC_FE_EXTRA", "1") == "1":
        with timer("追加特徴 (EXT poly / group相対 / k-means)"):
            app_train, app_test = extra_features.add_all(app_train, app_test)

    with timer("メモリ最適化 & 保存"):
        app_train = reduce_mem_usage(app_train)
        app_test = reduce_mem_usage(app_test)

        categorical_features = [c for c in app_train.columns
                                 if app_train[c].dtype == "object" or pd.api.types.is_string_dtype(app_train[c].dtype)]
        feature_columns = [c for c in app_train.columns if c not in ("SK_ID_CURR", "TARGET")]

        app_train.to_parquet(config.PROC_DIR / "train_features.parquet", index=False)
        app_test.to_parquet(config.PROC_DIR / "test_features.parquet", index=False)
        with open(config.PROC_DIR / "categorical_features.json", "w") as f:
            json.dump(categorical_features, f, ensure_ascii=False, indent=2)
        with open(config.PROC_DIR / "feature_columns.json", "w") as f:
            json.dump(feature_columns, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("特徴量エンジニアリング完了")
    print(f"train: {app_train.shape}  test: {app_test.shape}")
    print(f"カテゴリ特徴量: {len(categorical_features)}個")
    print("=" * 60)


if __name__ == "__main__":
    main()
