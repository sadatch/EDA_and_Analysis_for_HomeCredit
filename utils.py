"""
共通ユーティリティ。メモリ削減・集約ルール生成・タイマーなど。
"""
import time
import gc
import numpy as np
import pandas as pd
from contextlib import contextmanager


@contextmanager
def timer(name: str):
    """処理時間を計測して出力するだけのシンプルなコンテキストマネージャ。"""
    t0 = time.time()
    print(f"[START] {name}")
    yield
    print(f"[DONE ] {name} ({time.time() - t0:.1f}s)")


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    数値列のdtypeを値域に応じてダウンキャストし、メモリを削減する。
    大量の集約特徴量を作る本タスクではメモリ不足対策として必須。
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtype
        # object / category / pandas2+の string dtype は対象外（pandas4のArrowバックエンドstringを含む）
        if col_type == object or str(col_type) in ("category",) or pd.api.types.is_string_dtype(col_type):
            continue
        if not pd.api.types.is_numeric_dtype(col_type):
            continue
        c_min, c_max = df[col].min(), df[col].max()
        if pd.isna(c_min) or pd.isna(c_max):
            # 全NaN列はそのまま（float32にしておく）
            df[col] = df[col].astype(np.float32)
            continue
        if str(col_type)[:3] == "int":
            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.int64)
        else:
            # float64 -> float32（精度はAUC評価には十分）
            df[col] = df[col].astype(np.float32)
    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    if verbose:
        print(f"  メモリ使用量: {start_mem:.1f}MB -> {end_mem:.1f}MB "
              f"({100 * (start_mem - end_mem) / max(start_mem, 1e-9):.1f}% 削減)")
    return df


def build_agg_rules(df: pd.DataFrame, id_cols, bool_aggs=("mean", "sum"),
                     numeric_aggs=("mean", "max", "min", "sum")) -> dict:
    """
    one-hot化済みのサブテーブルに対し、列のdtypeに応じてagg_rulesを自動生成する。
    bool/uint8 (one-hotフラグ) -> mean, sum
    その他数値          -> mean, max, min, sum
    id_cols は除外する。
    """
    agg_rules = {}
    for col in df.columns:
        if col in id_cols:
            continue
        dtype = df[col].dtype
        if dtype == "uint8" or dtype == "bool" or str(dtype).startswith("uint"):
            agg_rules[col] = list(bool_aggs)
        elif pd.api.types.is_numeric_dtype(dtype):
            agg_rules[col] = list(numeric_aggs)
    return agg_rules


def flatten_agg_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """groupby().agg()後のMultiIndex列名を 'PREFIX_COL_STAT' 形式にフラット化する。"""
    df.columns = [f"{prefix}_" + "_".join(col).strip("_") if isinstance(col, tuple) else f"{prefix}_{col}"
                  for col in df.columns.values]
    return df


def safe_merge(left: pd.DataFrame, right: pd.DataFrame, on: str, how: str = "left") -> pd.DataFrame:
    """merge後すぐにgcを挟む薄いラッパー。大規模結合でのメモリスパイク対策。"""
    out = left.merge(right, on=on, how=how)
    gc.collect()
    return out
