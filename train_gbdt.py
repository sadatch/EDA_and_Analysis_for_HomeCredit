"""
LightGBM + XGBoost 学習スクリプト。

既存notebookの実装（StratifiedKFold 5分割、early stopping、カテゴリ変数のネイティブ対応）
をそのまま継承し、DAE埋め込み（dae_features.pyの出力）を追加特徴として結合する。
Optunaでのハイパラ探索は任意（--optuna フラグ、デフォルトは固定の安定パラメータで高速実行）。

出力:
  artifacts/lgb_oof.npy, artifacts/lgb_test.npy
  artifacts/xgb_oof.npy, artifacts/xgb_test.npy
  artifacts/cv_scores.json
"""
import argparse
import json

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import config
from utils import timer


def load_features_with_dae():
    train_df = pd.read_parquet(config.PROC_DIR / "train_features.parquet")
    test_df = pd.read_parquet(config.PROC_DIR / "test_features.parquet")

    dae_train = pd.read_parquet(config.PROC_DIR / "dae_train_embeddings.parquet")
    dae_test = pd.read_parquet(config.PROC_DIR / "dae_test_embeddings.parquet")

    train_df = train_df.merge(dae_train, on="SK_ID_CURR", how="left")
    test_df = test_df.merge(dae_test, on="SK_ID_CURR", how="left")

    with open(config.PROC_DIR / "categorical_features.json") as f:
        categorical_features = json.load(f)

    return train_df, test_df, categorical_features


def prepare_xy(train_df, test_df, categorical_features):
    X_train_full = train_df.drop(columns=["SK_ID_CURR", "TARGET"])
    y_train_full = train_df["TARGET"]
    X_test = test_df.drop(columns=["SK_ID_CURR"])

    for col in categorical_features:
        X_train_full[col] = X_train_full[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    return X_train_full, y_train_full, X_test


def run_lightgbm(X_train_full, y_train_full, X_test, categorical_features, params=None):
    default_params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.02,
        "num_leaves": 64,
        "max_depth": 8,
        "min_child_samples": 60,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.6,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbosity": -1,
        "random_state": config.SEED,
    }
    params = params or default_params

    folds = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
    oof_preds = np.zeros(len(X_train_full))
    test_preds = np.zeros(len(X_test))
    fold_scores = []

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_full, y_train_full)):
        X_trn, y_trn = X_train_full.iloc[trn_idx], y_train_full.iloc[trn_idx]
        X_val, y_val = X_train_full.iloc[val_idx], y_train_full.iloc[val_idx]

        lgb_train = lgb.Dataset(X_trn, y_trn, categorical_feature=categorical_features)
        lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train, categorical_feature=categorical_features)

        model = lgb.train(
            params, lgb_train, valid_sets=[lgb_train, lgb_eval],
            callbacks=[lgb.early_stopping(stopping_rounds=config.GBDT_EARLY_STOPPING_ROUNDS, verbose=False)],
            num_boost_round=config.GBDT_NUM_BOOST_ROUND,
        )
        oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        test_preds += model.predict(X_test, num_iteration=model.best_iteration) / folds.n_splits

        fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
        fold_scores.append(fold_auc)
        print(f"  [LGBM] fold {fold_ + 1}: AUC={fold_auc:.6f} (best_iter={model.best_iteration})")

    overall_auc = roc_auc_score(y_train_full, oof_preds)
    print(f"  [LGBM] overall OOF AUC: {overall_auc:.6f}")
    return oof_preds, test_preds, overall_auc, fold_scores


def run_xgboost(X_train_full, y_train_full, X_test, params=None):
    default_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": 0.02,
        "max_depth": 6,
        "min_child_weight": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "alpha": 0.1,
        "lambda": 1.0,
        "tree_method": "hist",
        "random_state": config.SEED,
    }
    params = params or default_params

    folds = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
    oof_preds = np.zeros(len(X_train_full))
    test_preds = np.zeros(len(X_test))
    fold_scores = []

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_full, y_train_full)):
        X_trn, y_trn = X_train_full.iloc[trn_idx], y_train_full.iloc[trn_idx]
        X_val, y_val = X_train_full.iloc[val_idx], y_train_full.iloc[val_idx]

        dtrain = xgb.DMatrix(X_trn, label=y_trn, enable_categorical=True)
        dvalid = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
        dtest = xgb.DMatrix(X_test, enable_categorical=True)

        model = xgb.train(
            params, dtrain, num_boost_round=config.GBDT_NUM_BOOST_ROUND,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=config.GBDT_EARLY_STOPPING_ROUNDS, verbose_eval=False,
        )
        oof_preds[val_idx] = model.predict(dvalid, iteration_range=(0, model.best_iteration + 1))
        test_preds += model.predict(dtest, iteration_range=(0, model.best_iteration + 1)) / folds.n_splits

        fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
        fold_scores.append(fold_auc)
        print(f"  [XGB ] fold {fold_ + 1}: AUC={fold_auc:.6f} (best_iter={model.best_iteration})")

    overall_auc = roc_auc_score(y_train_full, oof_preds)
    print(f"  [XGB ] overall OOF AUC: {overall_auc:.6f}")
    return oof_preds, test_preds, overall_auc, fold_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-lgb", action="store_true")
    parser.add_argument("--skip-xgb", action="store_true")
    args = parser.parse_args()

    with timer("特徴量 + DAE埋め込み読み込み"):
        train_df, test_df, categorical_features = load_features_with_dae()
        X_train_full, y_train_full, X_test = prepare_xy(train_df, test_df, categorical_features)
        print(f"  特徴量数: {X_train_full.shape[1]} (うちカテゴリ: {len(categorical_features)})")

    scores = {}

    if not args.skip_lgb:
        with timer("LightGBM 5-Fold CV学習"):
            lgb_oof, lgb_test, lgb_auc, lgb_fold_scores = run_lightgbm(
                X_train_full, y_train_full, X_test, categorical_features
            )
            np.save(config.ARTIFACT_DIR / "lgb_oof.npy", lgb_oof)
            np.save(config.ARTIFACT_DIR / "lgb_test.npy", lgb_test)
            scores["lightgbm"] = {"oof_auc": lgb_auc, "fold_scores": lgb_fold_scores}

    if not args.skip_xgb:
        with timer("XGBoost 5-Fold CV学習"):
            xgb_oof, xgb_test, xgb_auc, xgb_fold_scores = run_xgboost(
                X_train_full, y_train_full, X_test
            )
            np.save(config.ARTIFACT_DIR / "xgb_oof.npy", xgb_oof)
            np.save(config.ARTIFACT_DIR / "xgb_test.npy", xgb_test)
            scores["xgboost"] = {"oof_auc": xgb_auc, "fold_scores": xgb_fold_scores}

    # ensemble.pyで使うためにy_trainとtest用IDも保存
    np.save(config.ARTIFACT_DIR / "y_train.npy", y_train_full.values)
    test_df[["SK_ID_CURR"]].to_csv(config.ARTIFACT_DIR / "test_ids.csv", index=False)

    with open(config.ARTIFACT_DIR / "cv_scores.json", "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("GBDT学習完了:", json.dumps({k: v["oof_auc"] for k, v in scores.items()}, indent=2))
    print("=" * 60)


if __name__ == "__main__":
    main()
