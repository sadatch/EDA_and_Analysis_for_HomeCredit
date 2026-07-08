"""
特徴量選択（Null Importance drop list）適用の高速A/Bテスト（チューニング施策）。

feature_selection.json の drop リストは前回計測で1588本と多く、適用の是非は
データで決めるべき。本スクリプトは軽量LightGBM（1シード・5fold・lr高め）で
「全特徴」vs「drop適用」のOOF AUCを比較し、artifacts/fs_ab.json に推奨を書き出す。

使い方（feature_engineering.py と feature_selection.py 実行後）:
  python3 fs_quick_ab.py
  # -> "recommend_apply_fs": true なら HC_APPLY_FS=1 を付けて本学習を回す

軽量設定（lr=0.05, 早期停止100）なので本学習の代理指標。差が +0.0003 未満なら
「差なし」とみなし、安全側（適用しない）を推奨する。
"""
import json
import gc

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import config

MARGIN = 0.0003  # これ以上改善しないなら適用を推奨しない（安全側）


def _load_data():
    train = pd.read_parquet(config.PROC_DIR / "train_features.parquet")
    y = train["TARGET"].values.astype(np.int8)
    feat_cols = [c for c in train.columns if c not in ("SK_ID_CURR", "TARGET")]
    X = train[feat_cols].copy()
    # object列は簡易label encode（A/B比較用途なので train内factorizeで十分）
    for c in X.columns:
        if X[c].dtype == "object" or pd.api.types.is_string_dtype(X[c].dtype):
            X[c] = pd.factorize(X[c].astype(str))[0].astype(np.int32)
    del train
    gc.collect()
    return X, y


def _quick_cv_auc(X, y, label):
    params = {
        "objective": "binary", "metric": "auc", "learning_rate": 0.05,
        "num_leaves": 48, "min_child_samples": 40, "feature_fraction": 0.8,
        "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
        "verbosity": -1, "num_threads": config.N_THREADS, "seed": config.SEED,
    }
    folds = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
    oof = np.zeros(len(y))
    for i, (trn, val) in enumerate(folds.split(X, y)):
        dtr = lgb.Dataset(X.iloc[trn], label=y[trn])
        dva = lgb.Dataset(X.iloc[val], label=y[val], reference=dtr)
        model = lgb.train(params, dtr, valid_sets=[dva], num_boost_round=4000,
                          callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[val] = model.predict(X.iloc[val], num_iteration=model.best_iteration)
        print(f"  [{label}] fold{i} AUC={roc_auc_score(y[val], oof[val]):.6f} "
              f"(iter={model.best_iteration})")
    auc = roc_auc_score(y, oof)
    print(f"  [{label}] OOF AUC={auc:.6f}")
    return auc


def main():
    fs_path = config.ARTIFACT_DIR / "feature_selection.json"
    if not fs_path.exists():
        raise RuntimeError("feature_selection.json がありません。feature_selection.py を先に実行してください。")
    with open(fs_path) as f:
        fs = json.load(f)
    drop = set(fs.get("drop", []))

    X, y = _load_data()
    print(f"全特徴: {X.shape[1]}本 / drop候補: {len(drop)}本")

    auc_full = _quick_cv_auc(X, y, "full")
    keep_cols = [c for c in X.columns if c not in drop]
    auc_fs = _quick_cv_auc(X[keep_cols], y, "fs")

    recommend = bool(auc_fs > auc_full + MARGIN)
    report = {
        "auc_full": float(auc_full),
        "auc_fs_applied": float(auc_fs),
        "delta": float(auc_fs - auc_full),
        "margin": MARGIN,
        "n_features_full": int(X.shape[1]),
        "n_features_fs": len(keep_cols),
        "recommend_apply_fs": recommend,
    }
    with open(config.ARTIFACT_DIR / "fs_ab.json", "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"full={auc_full:.6f}  fs={auc_fs:.6f}  delta={auc_fs - auc_full:+.6f}")
    print(f"推奨: HC_APPLY_FS={'1' if recommend else '0'}  (artifacts/fs_ab.json に保存)")
    print("=" * 60)


if __name__ == "__main__":
    main()
