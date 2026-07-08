"""
リーク制御付き（Out-Of-Fold）の高度特徴量。

1. neighbors_target_mean  — Home Credit 1位チームの目玉特徴。
   EXT_SOURCE 1/2/3 と CREDIT_ANNUITY_RATIO の空間で K近傍を取り、
   その近傍のTARGET平均を特徴量にする。trainはOOF（自分が属さないfoldで学習した近傍器で予測）、
   testは全trainで学習した近傍器で予測することでリークを防ぐ。

2. target encoding — カテゴリ列のTARGET平均をスムージング付きでエンコード。
   trainはOOF、testは全trainの統計で付与する（同じくリーク制御）。

どちらも application レベルの少数の列だけを使うため、本番(30万行)でも数分で終わる。
"""
import gc

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import config


NEIGHBOR_BASE_COLS = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "CREDIT_ANNUITY_RATIO"]


def _prep_neighbor_matrix(df: pd.DataFrame, cols, scaler: StandardScaler = None):
    """近傍計算用に欠損をmedian埋め+標準化した行列を返す。"""
    sub = df[cols].copy()
    for c in cols:
        med = sub[c].median()
        sub[c] = sub[c].fillna(med if not np.isnan(med) else 0.0)
    sub = sub.replace([np.inf, -np.inf], 0.0)
    if scaler is None:
        scaler = StandardScaler()
        mat = scaler.fit_transform(sub.values.astype(np.float64))
        return mat, scaler
    mat = scaler.transform(sub.values.astype(np.float64))
    return mat, scaler


def add_neighbor_target_features(app_train: pd.DataFrame, app_test: pd.DataFrame,
                                 k: int = None, cols=None, col_name: str = None) -> tuple:
    """
    neighbors_target_mean_{k} 列を train/test に付与して返す。
    1位チーム手法（EXT_SOURCE × CREDIT_ANNUITY_RATIO 空間の近傍TARGET平均）。
    cols/col_nameを指定すると別の特徴空間・別の列名で同じロジックを使い回せる
    （P2: neighbors_target_meanの多様化、add_neighbor_diversity_features参照）。
    """
    k = k or config.NEIGHBORS_K
    cols = [c for c in (cols or NEIGHBOR_BASE_COLS) if c in app_train.columns]
    if len(cols) < 2:
        print("  [neighbors] 必要な列が不足しているためスキップ")
        return app_train, app_test
    col_name = col_name or f"NEIGHBORS_TARGET_MEAN_{k}"
    y = app_train["TARGET"].values.astype(np.float64)

    # --- train: OOF ---
    oof = np.full(len(app_train), np.nan, dtype=np.float64)
    folds = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
    train_mat_full, scaler = _prep_neighbor_matrix(app_train, cols)
    for trn_idx, val_idx in folds.split(app_train, y):
        k_eff = min(k, len(trn_idx) - 1)
        nn = NearestNeighbors(n_neighbors=k_eff, algorithm="auto", n_jobs=config.N_THREADS)
        nn.fit(train_mat_full[trn_idx])
        _, neigh = nn.kneighbors(train_mat_full[val_idx])
        # neigh は trn_idx内の位置。対応するTARGETを引いて平均
        y_trn = y[trn_idx]
        oof[val_idx] = y_trn[neigh].mean(axis=1)
    app_train[col_name] = oof.astype(np.float32)

    # --- test: 全trainで学習 ---
    k_eff = min(k, len(app_train) - 1)
    nn_full = NearestNeighbors(n_neighbors=k_eff, algorithm="auto", n_jobs=config.N_THREADS)
    nn_full.fit(train_mat_full)
    test_mat, _ = _prep_neighbor_matrix(app_test, cols, scaler=scaler)
    _, neigh_test = nn_full.kneighbors(test_mat)
    app_test[col_name] = y[neigh_test].mean(axis=1).astype(np.float32)

    print(f"  [neighbors] {col_name} を付与 (k={k}, base={cols})")
    del train_mat_full, nn_full
    gc.collect()
    return app_train, app_test


def add_neighbor_avg_feature(app_train: pd.DataFrame, app_test: pd.DataFrame,
                              avg_col: str, cols=None, k: int = None) -> tuple:
    """
    近傍(k件)の avg_col 平均と、自分の値との差分(自分が近傍より良い/悪いか)を付与する。
    TARGETではなくEXT_SOURCE_MEAN等の非target列を平均するため本質的にリークは無いが、
    既存の近傍探索インフラ(fold分割込み)を再利用してtrain/testの扱いを一貫させる
    （P2: neighbors_target_meanの多様化、「近傍のEXT_SOURCE平均との差分」）。
    """
    k = k or config.NEIGHBORS_K
    cols = [c for c in (cols or NEIGHBOR_BASE_COLS) if c in app_train.columns]
    if len(cols) < 2 or avg_col not in app_train.columns or avg_col not in app_test.columns:
        print(f"  [neighbors_avg] {avg_col}: 必要な列が不足しているためスキップ")
        return app_train, app_test

    avg_col_name = f"NEIGHBORS_{avg_col}_MEAN"
    dev_col_name = f"NEIGHBORS_{avg_col}_DEV"
    v_train = pd.to_numeric(app_train[avg_col], errors="coerce").values.astype(np.float64)
    v_test = pd.to_numeric(app_test[avg_col], errors="coerce").values.astype(np.float64)

    oof = np.full(len(app_train), np.nan, dtype=np.float64)
    folds = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
    y_strat = app_train["TARGET"].values
    train_mat_full, scaler = _prep_neighbor_matrix(app_train, cols)
    for trn_idx, val_idx in folds.split(app_train, y_strat):
        k_eff = min(k, len(trn_idx) - 1)
        nn = NearestNeighbors(n_neighbors=k_eff, algorithm="auto", n_jobs=config.N_THREADS)
        nn.fit(train_mat_full[trn_idx])
        _, neigh = nn.kneighbors(train_mat_full[val_idx])
        v_trn = v_train[trn_idx]
        med = np.nanmedian(v_trn)
        med = med if np.isfinite(med) else 0.0
        v_trn_filled = np.where(np.isnan(v_trn), med, v_trn)
        oof[val_idx] = v_trn_filled[neigh].mean(axis=1)
    app_train[avg_col_name] = oof.astype(np.float32)

    k_eff = min(k, len(app_train) - 1)
    nn_full = NearestNeighbors(n_neighbors=k_eff, algorithm="auto", n_jobs=config.N_THREADS)
    nn_full.fit(train_mat_full)
    test_mat, _ = _prep_neighbor_matrix(app_test, cols, scaler=scaler)
    _, neigh_test = nn_full.kneighbors(test_mat)
    med_all = np.nanmedian(v_train)
    med_all = med_all if np.isfinite(med_all) else 0.0
    v_train_filled = np.where(np.isnan(v_train), med_all, v_train)
    app_test[avg_col_name] = v_train_filled[neigh_test].mean(axis=1).astype(np.float32)

    app_train[dev_col_name] = (v_train - app_train[avg_col_name].values).astype(np.float32)
    app_test[dev_col_name] = (v_test - app_test[avg_col_name].values).astype(np.float32)

    print(f"  [neighbors_avg] {avg_col_name}/{dev_col_name} を付与 (k={k}, base={cols})")
    del train_mat_full, nn_full
    gc.collect()
    return app_train, app_test


def add_neighbor_diversity_features(app_train: pd.DataFrame, app_test: pd.DataFrame) -> tuple:
    """
    neighbors_target_mean(1位チームの目玉特徴)の派生を増やす(P2):
      - 複数解像度: k=100/1000/2000 (既存default k=500はadd_neighbor_target_featuresでそのまま維持)
      - 特徴空間バリエーション: (EXT_SOURCE+予測金利), (EXT_SOURCE+DAYS_BIRTH+DAYS_EMPLOYED)
      - 近傍のEXT_SOURCE_MEANとの差分(自分が近傍より良い/悪いか)
    既存のadd_neighbor_target_features呼び出しの後に呼ぶこと(既存k=500列と共存)。
    """
    for k in (100, 1000, 2000):
        app_train, app_test = add_neighbor_target_features(
            app_train, app_test, k=k, cols=NEIGHBOR_BASE_COLS,
            col_name=f"NEIGHBORS_TARGET_MEAN_MULTIRES_{k}")

    ext_rate_cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "YEARLY_INTEREST_RATE"]
                     if c in app_train.columns]
    if len(ext_rate_cols) >= 3:
        app_train, app_test = add_neighbor_target_features(
            app_train, app_test, k=config.NEIGHBORS_K, cols=ext_rate_cols,
            col_name="NEIGHBORS_TARGET_MEAN_EXTRATE")

    age_emp_cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "DAYS_BIRTH", "DAYS_EMPLOYED"]
                    if c in app_train.columns]
    if len(age_emp_cols) >= 3:
        app_train, app_test = add_neighbor_target_features(
            app_train, app_test, k=config.NEIGHBORS_K, cols=age_emp_cols,
            col_name="NEIGHBORS_TARGET_MEAN_AGEEMP")

    if "EXT_SOURCE_MEAN" in app_train.columns:
        app_train, app_test = add_neighbor_avg_feature(
            app_train, app_test, avg_col="EXT_SOURCE_MEAN", cols=NEIGHBOR_BASE_COLS,
            k=config.NEIGHBORS_K)

    return app_train, app_test


def add_target_encoding(app_train: pd.DataFrame, app_test: pd.DataFrame,
                        cols=None, smoothing: float = None) -> tuple:
    """
    カテゴリ列のOOF target encoding（スムージング付き）。
    trainはStratifiedKFoldのOOF、testは全trainの統計を使う。
    """
    cols = cols if cols is not None else config.TARGET_ENC_COLS
    cols = [c for c in cols if c in app_train.columns and c in app_test.columns]
    if not cols:
        print("  [target_enc] 対象カテゴリ列が無いためスキップ")
        return app_train, app_test
    smoothing = smoothing if smoothing is not None else config.TARGET_ENC_SMOOTHING

    y = app_train["TARGET"].values.astype(np.float64)
    global_mean = y.mean()
    folds = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)

    for c in cols:
        te_col = f"TE_{c}"
        oof = np.full(len(app_train), np.nan, dtype=np.float64)
        s = app_train[c].astype("object").fillna("__NA__").values
        for trn_idx, val_idx in folds.split(app_train, y):
            df_trn = pd.DataFrame({"cat": s[trn_idx], "y": y[trn_idx]})
            stats = df_trn.groupby("cat")["y"].agg(["mean", "count"])
            smooth = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
            mapping = smooth.to_dict()
            oof[val_idx] = pd.Series(s[val_idx]).map(mapping).fillna(global_mean).values
        app_train[te_col] = oof.astype(np.float32)

        # test: 全trainの統計
        df_all = pd.DataFrame({"cat": s, "y": y})
        stats_all = df_all.groupby("cat")["y"].agg(["mean", "count"])
        smooth_all = (stats_all["mean"] * stats_all["count"] + global_mean * smoothing) / (stats_all["count"] + smoothing)
        mapping_all = smooth_all.to_dict()
        s_test = app_test[c].astype("object").fillna("__NA__").values
        app_test[te_col] = pd.Series(s_test).map(mapping_all).fillna(global_mean).astype(np.float32).values

    print(f"  [target_enc] {len(cols)}列をエンコード: {cols}")
    return app_train, app_test


def add_arithmetic_interactions(app_train: pd.DataFrame, app_test: pd.DataFrame) -> tuple:
    """
    1位チーム手法の算術交互作用特徴（乗除）。EXT_SOURCEと主要金額/日数の組合せを中心に。
    """
    for df in (app_train, app_test):
        ext_mean = df.get("EXT_SOURCE_MEAN")
        # CREDIT_ANNUITY_RATIO は近傍特徴でも使うので確実に作る
        if "CREDIT_ANNUITY_RATIO" not in df.columns:
            df["CREDIT_ANNUITY_RATIO"] = df["AMT_CREDIT"] / (df["AMT_ANNUITY"] + 1e-5)
        if ext_mean is not None:
            df["EXT_x_CREDIT_ANNUITY"] = df["EXT_SOURCE_MEAN"] * df["CREDIT_ANNUITY_RATIO"]
            df["EXT_DIV_DAYS_BIRTH"] = df["EXT_SOURCE_MEAN"] / (df["DAYS_BIRTH"].abs() + 1.0)
            df["EXT_x_DAYS_EMPLOYED"] = df["EXT_SOURCE_MEAN"] * df["DAYS_EMPLOYED"].abs()
        df["ANNUITY_x_AGE"] = df["AMT_ANNUITY"] / (df["DAYS_BIRTH"].abs() + 1.0)
        df["CREDIT_GOODS_DIFF"] = df["AMT_CREDIT"] - df.get("AMT_GOODS_PRICE", 0)
        df["CREDIT_GOODS_RATIO"] = df["AMT_CREDIT"] / (df.get("AMT_GOODS_PRICE", np.nan) + 1e-5)
        df["INCOME_CREDIT_RATIO"] = df["AMT_INCOME_TOTAL"] / (df["AMT_CREDIT"] + 1e-5)
    print("  [arithmetic] 交互作用特徴を付与")
    return app_train, app_test
