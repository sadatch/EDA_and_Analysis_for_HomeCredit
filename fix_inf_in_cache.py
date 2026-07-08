"""
既に生成済みの cache/features_train.parquet / features_test.parquet に
残っている inf/-inf を NaN に置換して上書き保存する。

hc_campaign.py の features ステージ(数時間かかる)をやり直さずに、
XGBoost/CatBoost/MLPが "Input data contains `inf`" で落ちる問題を解消する。

使い方:
  python3 fix_inf_in_cache.py
"""
import numpy as np
import pandas as pd

for name in ("features_train.parquet", "features_test.parquet"):
    path = f"cache/{name}"
    print(f"[load] {path}")
    df = pd.read_parquet(path)

    num_cols = df.select_dtypes(include=[np.number]).columns
    inf_mask = np.isinf(df[num_cols].to_numpy())
    n_inf = int(inf_mask.sum())

    if n_inf > 0:
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
        df.to_parquet(path)
        print(f"[fixed] {path}: {n_inf}個のinfセルをNaNに置換して上書き保存")
    else:
        print(f"[ok] {path}: infなし（修正不要）")

print("done. 続けて train xgb / train cat / train mlp を実行してください。")
