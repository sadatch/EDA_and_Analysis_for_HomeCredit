"""
最終バッチ用の追加特徴量（final_features.py, HC_FE_FINAL=0で無効化）。

公開kernel（upvote上位）・1位/2位解法writeup・AmEx上位解法から、既存パイプライン
（feature_engineering / domain / trend_velocity / top_solution / gap の各モジュール）に
まだ入っていないものだけを収集した「最後の取りこぼし回収」モジュール。

カテゴリ（プレフィックス FIN_）:
  FIN_APP_   application単独の定番比率・カウント
             - GOODS/収入比、子供1人あたり収入、車齢/年齢・車齢/勤続、書類日付/年齢の相対化
             - 連絡手段の充実度（FLAG_MOBIL〜FLAG_EMAILの合計）
             - 住所不一致カウント（REG_REGION_NOT_LIVE_REGION等6フラグの合計）
             - 社会的圏のデフォルト率（DEF/OBSの30日・60日）
             - 建物情報（*_AVG/_MODE/_MEDI）の行平均・非欠損数（情報の充実度そのもの）
  FIN_BUR_   bureau集約列からの後段比率（債務/与信、延滞/債務、延長回数フラグ）
  FIN_XT_    テーブル横断の負担比較（新規annuity vs 過去平均支払、新規与信 vs 過去与信/残債）
  FIN_INT_   上位重要度特徴同士の交互作用（1位解法「乗除の総当たりで効いたものを残す」の要点だけ）
             - EXT_SOURCE_MEAN × 金利、CREDIT_ANNUITY_RATIO vs 予測CNT_PAYMENTの差/比
               （後者は1位解法の「予測期間と実期間の乖離=早期返済確率」のシグナル）
  GRP2_      追加グループ相対特徴（OCCUPATION_TYPE / REGION_RATING_CLIENT / 年齢10歳刻み）
             既存extra_features(GRP_)と別キーで補完。train+test結合で算出（target非依存）

すべて「元になる列が存在するときだけ」作る（合成データ/実データのどちらでも落ちない）。
"""
import gc

import numpy as np
import pandas as pd

EPS = 1e-5

CONTACT_FLAGS = ["FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE",
                 "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL"]
MISMATCH_FLAGS = ["REG_REGION_NOT_LIVE_REGION", "REG_REGION_NOT_WORK_REGION",
                  "LIVE_REGION_NOT_WORK_REGION", "REG_CITY_NOT_LIVE_CITY",
                  "REG_CITY_NOT_WORK_CITY", "LIVE_CITY_NOT_WORK_CITY"]

# 追加グループ相対特徴のキーと値（存在するものだけ使う）
GRP2_KEYS = ["OCCUPATION_TYPE", "REGION_RATING_CLIENT"]
GRP2_VALUES = ["EXT_SOURCE_MEAN", "CREDIT_ANNUITY_RATIO", "AMT_INCOME_TOTAL",
               "YEARLY_INTEREST_RATE"]


def _num(df, col):
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else None


def _first(df, names):
    for n in names:
        if n in df.columns:
            return pd.to_numeric(df[n], errors="coerce")
    return None


def _add_application_ratios(df: pd.DataFrame) -> int:
    n = 0
    income = _num(df, "AMT_INCOME_TOTAL")
    goods = _num(df, "AMT_GOODS_PRICE")
    birth = _num(df, "DAYS_BIRTH")
    employed = _num(df, "DAYS_EMPLOYED")

    if income is not None and goods is not None:
        df["FIN_APP_GOODS_INCOME_RATIO"] = (goods / (income + EPS)).astype(np.float32)
        n += 1
    if income is not None and "CNT_CHILDREN" in df.columns:
        df["FIN_APP_INCOME_PER_CHILD"] = (
            income / (1.0 + pd.to_numeric(df["CNT_CHILDREN"], errors="coerce").fillna(0))
        ).astype(np.float32)
        n += 1
    if "OWN_CAR_AGE" in df.columns and birth is not None:
        car = _num(df, "OWN_CAR_AGE")
        df["FIN_APP_CAR_AGE_RATIO"] = (car / (birth.abs() / 365.0 + EPS)).astype(np.float32)
        n += 1
        if employed is not None:
            df["FIN_APP_CAR_EMPLOYED_RATIO"] = (
                car / (employed.abs() / 365.0 + 1.0)
            ).astype(np.float32)
            n += 1
    # 書類・登録日付の年齢相対化（「最近になって書類を更新した」行動シグナル）
    for col, name in [("DAYS_ID_PUBLISH", "ID_PUBLISH"), ("DAYS_REGISTRATION", "REGISTRATION"),
                       ("DAYS_LAST_PHONE_CHANGE", "PHONE_CHANGE")]:
        v = _num(df, col)
        if v is not None and birth is not None:
            df[f"FIN_APP_{name}_TO_BIRTH"] = (v / (birth + EPS)).astype(np.float32)
            n += 1
    # 連絡手段の充実度
    contact_cols = [c for c in CONTACT_FLAGS if c in df.columns]
    if contact_cols:
        df["FIN_APP_CONTACT_COUNT"] = df[contact_cols].sum(axis=1).astype(np.int8)
        n += 1
    # 住所不一致カウント（生活実態と登録情報の乖離）
    mm_cols = [c for c in MISMATCH_FLAGS if c in df.columns]
    if mm_cols:
        df["FIN_APP_REGION_MISMATCH_COUNT"] = df[mm_cols].sum(axis=1).astype(np.int8)
        n += 1
    # 社会的圏のデフォルト率
    for d_col, o_col, name in [("DEF_30_CNT_SOCIAL_CIRCLE", "OBS_30_CNT_SOCIAL_CIRCLE", "30"),
                                ("DEF_60_CNT_SOCIAL_CIRCLE", "OBS_60_CNT_SOCIAL_CIRCLE", "60")]:
        d = _num(df, d_col)
        o = _num(df, o_col)
        if d is not None and o is not None:
            df[f"FIN_APP_SOCIAL_DEF{name}_RATIO"] = (d / (o + 1.0)).astype(np.float32)
            n += 1
    # 建物情報の行平均・充実度
    avg_cols = [c for c in df.columns if c.endswith("_AVG")]
    info_cols = [c for c in df.columns if c.endswith(("_AVG", "_MODE", "_MEDI"))
                 and pd.api.types.is_numeric_dtype(df[c])]
    if avg_cols:
        num_avg = df[avg_cols].apply(pd.to_numeric, errors="coerce")
        df["FIN_APP_BUILDING_AVG_MEAN"] = num_avg.mean(axis=1).astype(np.float32)
        n += 1
    if info_cols:
        df["FIN_APP_BUILDING_INFO_COUNT"] = df[info_cols].notna().sum(axis=1).astype(np.int16)
        n += 1
    return n


def _add_bureau_post_ratios(df: pd.DataFrame) -> int:
    n = 0
    debt = _first(df, ["BUREAU_AMT_CREDIT_SUM_DEBT_sum"])
    credit = _first(df, ["BUREAU_AMT_CREDIT_SUM_sum"])
    overdue = _first(df, ["BUREAU_AMT_CREDIT_SUM_OVERDUE_sum"])
    if debt is not None and credit is not None:
        df["FIN_BUR_DEBT_CREDIT_RATIO"] = (debt / (credit + EPS)).astype(np.float32)
        n += 1
    if overdue is not None and debt is not None:
        df["FIN_BUR_OVERDUE_DEBT_RATIO"] = (overdue / (debt + EPS)).astype(np.float32)
        n += 1
    prolong = _first(df, ["BUREAU_CNT_CREDIT_PROLONG_sum"])
    if prolong is not None:
        df["FIN_BUR_HAS_PROLONG"] = (prolong.fillna(0) > 0).astype(np.int8)
        n += 1
    return n


def _add_cross_table_burden(df: pd.DataFrame) -> int:
    n = 0
    annuity = _num(df, "AMT_ANNUITY")
    credit = _num(df, "AMT_CREDIT")
    pay_mean = _first(df, ["INS_ALL_AMT_PAYMENT_mean"])
    if annuity is not None and pay_mean is not None:
        # 新規ローンの月次負担が「これまで実際に払えていた月額」の何倍か
        df["FIN_XT_ANNUITY_TO_PAST_PAYMENT"] = (annuity / (pay_mean + EPS)).astype(np.float32)
        n += 1
    prev_credit_mean = _first(df, ["PREV_AMT_CREDIT_mean"])
    if credit is not None and prev_credit_mean is not None:
        # 今回の借入が過去の平均与信の何倍か（分不相応な急拡大の検知）
        df["FIN_XT_CREDIT_TO_PREV_MEAN"] = (credit / (prev_credit_mean + EPS)).astype(np.float32)
        n += 1
    active_debt = _first(df, ["BUREAU_ACTIVE_DEBT_SUM", "BUREAU_AMT_CREDIT_SUM_DEBT_sum"])
    if credit is not None and active_debt is not None:
        df["FIN_XT_CREDIT_TO_BUREAU_DEBT"] = (credit / (active_debt.fillna(0) + 1.0)).astype(np.float32)
        n += 1
    return n


def _add_top_interactions(df: pd.DataFrame) -> int:
    n = 0
    ext = _num(df, "EXT_SOURCE_MEAN")
    rate = _num(df, "YEARLY_INTEREST_RATE")
    if ext is not None and rate is not None:
        df["FIN_INT_EXT_x_RATE"] = (ext * rate).astype(np.float32)
        df["FIN_INT_EXT_DIV_RATE"] = (ext / (rate + EPS)).replace(
            [np.inf, -np.inf], np.nan).astype(np.float32)
        n += 2
    # 1位解法の核心シグナル: CREDIT/ANNUITY（実効期間）とCNT_PAYMENT予測（名目期間）の乖離。
    # 名目より実効が短い顧客は「早期返済しがち=低リスク」の傾向（Olivierの記述）。
    car = _num(df, "CREDIT_ANNUITY_RATIO")
    pred_n = _num(df, "PRED_CNT_PAYMENT")
    if car is not None and pred_n is not None:
        df["FIN_INT_TERM_GAP"] = (pred_n - car).astype(np.float32)
        df["FIN_INT_TERM_GAP_RATIO"] = (car / (pred_n + EPS)).astype(np.float32)
        n += 2
    neigh = _first(df, [f"NEIGHBORS_TARGET_MEAN_{k}" for k in (500, 100)])
    if neigh is not None and ext is not None:
        # 近傍リスクが高いのに自分のEXTが良い（またはその逆）という乖離
        df["FIN_INT_NEIGH_x_EXT"] = (neigh * (1.0 - ext)).astype(np.float32)
        n += 1
    return n


def _add_group_relative_v2(app_train: pd.DataFrame, app_test: pd.DataFrame) -> int:
    """
    追加キーでのグループ相対特徴（既存GRP_と重複しないキーのみ）。
    統計は train+test 結合で算出する（target非依存なのでリークなし）。
    """
    n_tr = len(app_train)
    made = 0
    # 年齢10歳刻みキーを一時生成
    tmp_key = None
    if "AGE_INT" in app_train.columns and "AGE_INT" in app_test.columns:
        tmp_key = "_AGE_DECADE"
        for df in (app_train, app_test):
            df[tmp_key] = (pd.to_numeric(df["AGE_INT"], errors="coerce") // 10).fillna(-1).astype(np.int8)
    keys = [k for k in GRP2_KEYS if k in app_train.columns and k in app_test.columns]
    if tmp_key:
        keys.append(tmp_key)

    for key in keys:
        full_key = pd.concat([app_train[key], app_test[key]], ignore_index=True).astype(str)
        for val in GRP2_VALUES:
            if val not in app_train.columns or val not in app_test.columns:
                continue
            full_val = pd.to_numeric(
                pd.concat([app_train[val], app_test[val]], ignore_index=True), errors="coerce")
            g_mean = full_val.groupby(full_key).transform("mean")
            g_std = full_val.groupby(full_key).transform("std")
            dev = full_val - g_mean
            z = dev / (g_std + EPS)
            key_name = key.lstrip("_")
            app_train[f"GRP2_{key_name}_{val}_DEV"] = dev.iloc[:n_tr].values.astype(np.float32)
            app_test[f"GRP2_{key_name}_{val}_DEV"] = dev.iloc[n_tr:].values.astype(np.float32)
            app_train[f"GRP2_{key_name}_{val}_Z"] = z.iloc[:n_tr].values.astype(np.float32)
            app_test[f"GRP2_{key_name}_{val}_Z"] = z.iloc[n_tr:].values.astype(np.float32)
            made += 2
    if tmp_key:
        for df in (app_train, app_test):
            df.drop(columns=[tmp_key], inplace=True, errors="ignore")
    return made


def add_final_features(app_train: pd.DataFrame, app_test: pd.DataFrame) -> tuple:
    """table_builders・top_solution(金利)・gap特徴のマージ後に呼ぶこと。"""
    counts = {}
    for df in (app_train, app_test):
        counts["app"] = _add_application_ratios(df)
        counts["bureau"] = _add_bureau_post_ratios(df)
        counts["cross"] = _add_cross_table_burden(df)
        counts["inter"] = _add_top_interactions(df)
    counts["grp2"] = _add_group_relative_v2(app_train, app_test)
    print(f"  [final] 定番比率{counts['app']} / bureau後段{counts['bureau']} / "
          f"横断負担{counts['cross']} / 交互作用{counts['inter']} / GRP2 {counts['grp2']} 列を付与")
    gc.collect()
    return app_train, app_test


def add_post_neighbor_interactions(app_train: pd.DataFrame, app_test: pd.DataFrame) -> tuple:
    """近傍特徴(NEIGHBORS_*)生成後に呼ぶ交互作用のみを追加する。"""
    n = 0
    for df in (app_train, app_test):
        n = _add_top_interactions(df) if "FIN_INT_NEIGH_x_EXT" not in df.columns else 0
    if n:
        print(f"  [final/post-neighbors] 近傍交互作用 {n}列を付与")
    return app_train, app_test
