"""
スモークテスト用の合成データ生成スクリプト。

実データ無しでパイプラインの論理エラー（カラム名ミス、型不整合、マージ漏れ等）を
事前に検出するためのもの。本番実行の前に必ずこれで動作確認してから
本物のKaggleデータに切り替えることを推奨。

実行:
  python make_synthetic_data.py
  HC_RAW_DIR=./data/raw_synthetic python feature_engineering.py  # など環境変数で切り替え
"""
import numpy as np
import pandas as pd
from pathlib import Path

import config

RNG = np.random.RandomState(config.SEED)
N_TRAIN = 2000
N_TEST = 500
N_APPLICANTS = N_TRAIN + N_TEST


def _ids():
    return np.arange(100000, 100000 + N_APPLICANTS)


def make_application(ids):
    n = len(ids)
    df = pd.DataFrame({
        "SK_ID_CURR": ids,
        "NAME_CONTRACT_TYPE": RNG.choice(["Cash loans", "Revolving loans"], n),
        "CODE_GENDER": RNG.choice(["M", "F"], n),
        "FLAG_OWN_CAR": RNG.choice(["Y", "N"], n),
        "FLAG_OWN_REALTY": RNG.choice(["Y", "N"], n),
        "CNT_CHILDREN": RNG.randint(0, 4, n),
        "AMT_INCOME_TOTAL": RNG.uniform(50000, 500000, n),
        "AMT_CREDIT": RNG.uniform(50000, 1000000, n),
        "AMT_ANNUITY": RNG.uniform(5000, 60000, n),
        "AMT_GOODS_PRICE": RNG.uniform(40000, 900000, n),
        "NAME_TYPE_SUITE": RNG.choice(["Unaccompanied", "Family", None], n),
        "NAME_INCOME_TYPE": RNG.choice(["Working", "Pensioner", "Commercial associate"], n),
        "NAME_EDUCATION_TYPE": RNG.choice(["Secondary", "Higher education"], n),
        "NAME_FAMILY_STATUS": RNG.choice(["Married", "Single", "Civil marriage"], n),
        "NAME_HOUSING_TYPE": RNG.choice(["House / apartment", "With parents"], n),
        "DAYS_BIRTH": -RNG.randint(7000, 25000, n),
        "DAYS_EMPLOYED": np.where(RNG.rand(n) < 0.1, 365243, -RNG.randint(0, 15000, n)),
        "CNT_FAM_MEMBERS": RNG.randint(1, 6, n).astype(float),
        "EXT_SOURCE_1": np.where(RNG.rand(n) < 0.3, np.nan, RNG.uniform(0, 1, n)),
        "EXT_SOURCE_2": RNG.uniform(0, 1, n),
        "EXT_SOURCE_3": np.where(RNG.rand(n) < 0.2, np.nan, RNG.uniform(0, 1, n)),
    })
    return df


def make_bureau(ids):
    rows = []
    for sk_id in ids:
        for _ in range(RNG.randint(0, 6)):
            rows.append({
                "SK_ID_CURR": sk_id,
                "SK_ID_BUREAU": RNG.randint(1, 10_000_000),
                "CREDIT_ACTIVE": RNG.choice(["Active", "Closed"]),
                "CREDIT_CURRENCY": "currency 1",
                "DAYS_CREDIT": -RNG.randint(0, 2000),
                "CREDIT_DAY_OVERDUE": RNG.randint(0, 30),
                "DAYS_CREDIT_ENDDATE": RNG.randint(-500, 1000),
                "AMT_CREDIT_SUM": RNG.uniform(1000, 500000),
                "AMT_CREDIT_SUM_DEBT": RNG.uniform(0, 400000),
                "CREDIT_TYPE": RNG.choice(["Consumer credit", "Credit card"]),
            })
    return pd.DataFrame(rows)


def make_bureau_balance(bureau_df):
    rows = []
    for sk_bureau in bureau_df["SK_ID_BUREAU"].unique():
        for m in range(-RNG.randint(1, 24), 1):
            rows.append({
                "SK_ID_BUREAU": sk_bureau,
                "MONTHS_BALANCE": m,
                "STATUS": RNG.choice(["C", "0", "1", "X"]),
            })
    return pd.DataFrame(rows)


def make_previous(ids):
    rows = []
    for sk_id in ids:
        for _ in range(RNG.randint(0, 4)):
            rows.append({
                "SK_ID_CURR": sk_id,
                "SK_ID_PREV": RNG.randint(1, 10_000_000),
                "NAME_CONTRACT_TYPE": RNG.choice(["Cash loans", "Consumer loans"]),
                "AMT_ANNUITY": RNG.uniform(2000, 50000),
                "AMT_APPLICATION": RNG.uniform(10000, 800000),
                "AMT_CREDIT": RNG.uniform(10000, 800000),
                "CNT_PAYMENT": RNG.choice([6, 12, 24, 36, np.nan]),
                "DAYS_DECISION": -RNG.randint(0, 3000),
                "NAME_CONTRACT_STATUS": RNG.choice(["Approved", "Refused", "Canceled"]),
            })
    return pd.DataFrame(rows)


def make_pos_cash(ids):
    rows = []
    for sk_id in ids:
        for _ in range(RNG.randint(0, 8)):
            rows.append({
                "SK_ID_CURR": sk_id,
                "SK_ID_PREV": RNG.randint(1, 10_000_000),
                "MONTHS_BALANCE": -RNG.randint(0, 24),
                "CNT_INSTALMENT": RNG.randint(1, 36),
                "CNT_INSTALMENT_FUTURE": RNG.randint(0, 36),
                "NAME_CONTRACT_STATUS": RNG.choice(["Active", "Completed"]),
                "SK_DPD": RNG.randint(0, 30),
                "SK_DPD_DEF": RNG.randint(0, 10),
            })
    return pd.DataFrame(rows)


def make_installments(ids):
    rows = []
    for sk_id in ids:
        for _ in range(RNG.randint(0, 10)):
            rows.append({
                "SK_ID_CURR": sk_id,
                "SK_ID_PREV": RNG.randint(1, 10_000_000),
                "NUM_INSTALMENT_VERSION": RNG.randint(0, 3),
                "NUM_INSTALMENT_NUMBER": RNG.randint(1, 36),
                "DAYS_INSTALMENT": -RNG.randint(0, 1000),
                "DAYS_ENTRY_PAYMENT": -RNG.randint(0, 1000),
                "AMT_INSTALMENT": RNG.uniform(1000, 30000),
                "AMT_PAYMENT": RNG.uniform(0, 30000),
            })
    return pd.DataFrame(rows)


def make_credit_card(ids):
    rows = []
    for sk_id in ids:
        for m in range(-RNG.randint(0, 24), 1):
            rows.append({
                "SK_ID_CURR": sk_id,
                "SK_ID_PREV": RNG.randint(1, 10_000_000),
                "MONTHS_BALANCE": m,
                "AMT_BALANCE": RNG.uniform(0, 200000),
                "AMT_CREDIT_LIMIT_ACTUAL": RNG.choice([50000, 100000, 200000, 300000]),
                "AMT_PAYMENT_TOTAL_CURRENT": RNG.uniform(0, 20000),
                "AMT_INST_MIN_REGULARITY": RNG.uniform(0, 10000),
                "NAME_CONTRACT_STATUS": RNG.choice(["Active", "Completed"]),
                "SK_DPD": RNG.randint(0, 30),
            })
    return pd.DataFrame(rows)


def main():
    out_dir = Path(config.RAW_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = _ids()
    train_ids, test_ids = ids[:N_TRAIN], ids[N_TRAIN:]

    app = make_application(ids)
    app_train = app[app["SK_ID_CURR"].isin(train_ids)].copy()
    app_train["TARGET"] = RNG.choice([0, 1], len(app_train), p=[0.92, 0.08])
    app_test = app[app["SK_ID_CURR"].isin(test_ids)].copy()

    app_train.to_csv(out_dir / config.RAW_FILES["app_train"], index=False)
    app_test.to_csv(out_dir / config.RAW_FILES["app_test"], index=False)

    bureau = make_bureau(ids)
    bureau.to_csv(out_dir / config.RAW_FILES["bureau"], index=False)
    make_bureau_balance(bureau).to_csv(out_dir / config.RAW_FILES["bureau_balance"], index=False)

    make_previous(ids).to_csv(out_dir / config.RAW_FILES["previous"], index=False)
    make_pos_cash(ids).to_csv(out_dir / config.RAW_FILES["pos_cash"], index=False)
    make_installments(ids).to_csv(out_dir / config.RAW_FILES["installments"], index=False)
    make_credit_card(ids).to_csv(out_dir / config.RAW_FILES["credit_card"], index=False)

    print(f"合成データ生成完了: {out_dir}")
    print(f"  train: {len(app_train)}件 / test: {len(app_test)}件")


if __name__ == "__main__":
    main()
