"""
共通ユーティリティ。メモリ削減・集約ルール生成・タイマー・デバイス設定など。
"""
import os
import time
import gc
import random
import numpy as np
import pandas as pd
from contextlib import contextmanager

import config


def seed_everything(seed: int) -> None:
    """numpy / random / (あれば)torch のシードをまとめて固定する。"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def fast_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    sklearn非依存のAUC（順位ベース）。hill climbingで何千回も呼ぶため軽量実装にしておく。
    NaN/定数入力でも落ちないようにガードする。
    """
    y_true = np.asarray(y_true).astype(np.float64)
    y_score = np.asarray(y_score).astype(np.float64)
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=np.float64)
    sorted_scores = y_score[order]
    # tie（同値）はaverage rankにする
    ranks_sorted = np.arange(1, len(y_score) + 1, dtype=np.float64)
    i = 0
    n = len(sorted_scores)
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks_sorted[i:j + 1] = (i + 1 + j + 1) / 2.0
        i = j + 1
    ranks[order] = ranks_sorted
    auc = (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


# ===== GBDTのデバイス/スレッド設定ヘルパ =====
def lgb_device_params() -> dict:
    """LightGBMのデバイス/並列パラメータ。GPU指定でも失敗時はtrainer側でCPUへ落とす想定。"""
    p = {"num_threads": config.N_THREADS}
    if config.LGB_DEVICE == "gpu":
        # OpenCLビルド済みのLightGBMが必要。未対応ならtrainerがCPUへフォールバック。
        p.update({"device_type": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0,
                  "max_bin": 255})
    else:
        p.update({"device_type": "cpu"})
    return p


def xgb_device_params() -> dict:
    """XGBoost 2.x のデバイス指定。GPUなら device='cuda'。"""
    if config.XGB_DEVICE == "cuda":
        return {"device": "cuda", "tree_method": "hist", "nthread": config.N_THREADS}
    return {"device": "cpu", "tree_method": "hist", "nthread": config.N_THREADS}


def cat_device_params() -> dict:
    """CatBoostのデバイス指定。"""
    if config.CAT_TASK_TYPE == "GPU":
        return {"task_type": "GPU", "devices": "0"}
    return {"task_type": "CPU", "thread_count": config.N_THREADS}


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
