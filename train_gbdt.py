"""
LightGBM + XGBoost 学習スクリプト（GPU対応 + シード平均）。

既存notebookの StratifiedKFold 5分割 / early stopping / カテゴリ変数ネイティブ対応 を継承しつつ:
  - GPU実行（LightGBMはOpenCLビルドが無ければ自動でCPUへフォールバック、XGBoostは device=cuda）
  - シード平均（config.SEED_LIST 本数。fold分割とモデルseedの両方を変えてOOF/testを平均）
  - Optunaで見つけた best params を artifacts から自動ロード（あれば）
  - DAE埋め込み（dae_features.pyの出力）を追加特徴として結合（存在すれば）

pseudo_label.py からも run_lightgbm / run_xgboost を import して再利用する。

出力:
  artifacts/lgb_oof.npy, artifacts/lgb_test.npy
  artifacts/xgb_oof.npy, artifacts/xgb_test.npy
  artifacts/cv_scores.json (追記)
"""
import argparse
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import config
from utils import timer, lgb_device_params, xgb_device_params

warnings.filterwarnings("ignore")

# LightGBM GPUが使えなかった場合に一度だけCPUへ切替えるためのフラグ
_LGB_GPU_DISABLED = False


# =====================================================================
# データ読み込み
# =====================================================================
def load_features_with_dae(train_path=None, test_path=None):
    train_path = train_path or (config.PROC_DIR / "train_features.parquet")
    test_path = test_path or (config.PROC_DIR / "test_features.parquet")
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    dae_train_path = config.PROC_DIR / "dae_train_embeddings.parquet"
    dae_test_path = config.PROC_DIR / "dae_test_embeddings.parquet"
    if dae_train_path.exists() and dae_test_path.exists():
        dae_train = pd.read_parquet(dae_train_path)
        dae_test = pd.read_parquet(dae_test_path)
        train_df = train_df.merge(dae_train, on="SK_ID_CURR", how="left")
        test_df = test_df.merge(dae_test, on="SK_ID_CURR", how="left")
    else:
        print("  注: DAE埋め込みが見つからないためGBDTは生特徴のみで学習")

    with open(config.PROC_DIR / "categorical_features.json") as f:
        categorical_features = json.load(f)

    # 特徴量選択の結果を反映（HC_APPLY_FS=1 のときのみ）
    fs_path = config.ARTIFACT_DIR / "feature_selection.json"
    if config.FE_SELECTION_APPLY and fs_path.exists():
        with open(fs_path) as f:
            drop = set(json.load(f).get("drop", []))
        drop = [c for c in drop if c in train_df.columns and c not in ("SK_ID_CURR", "TARGET")]
        if drop:
            train_df = train_df.drop(columns=drop)
            test_df = test_df.drop(columns=[c for c in drop if c in test_df.columns])
            print(f"  特徴量選択を適用: {len(drop)}列をdrop")

    return train_df, test_df, categorical_features


def prepare_xy(train_df, test_df, categorical_features):
    X_train_full = train_df.drop(columns=["SK_ID_CURR", "TARGET"])
    y_train_full = train_df["TARGET"]
    X_test = test_df.drop(columns=["SK_ID_CURR"])
    for col in categorical_features:
        if col in X_train_full.columns:
            X_train_full[col] = X_train_full[col].astype("category")
            X_test[col] = X_test[col].astype("category")
    return X_train_full, y_train_full, X_test


def _load_best_params(name: str):
    """artifacts/{name}_best_params.json があれば読み込む（Optunaチューニング結果）。"""
    p = config.ARTIFACT_DIR / f"{name}_best_params.json"
    if p.exists():
        with open(p) as f:
            params = json.load(f)
        print(f"  [{name}] Optuna best params をロード: {p.name}")
        return params
    return None


# =====================================================================
# LightGBM
# =====================================================================
def _lgb_base_params():
    p = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": config.GBDT_LEARNING_RATE,
        "num_leaves": 64,
        "max_depth": 8,
        "min_child_samples": 60,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.6,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbosity": -1,
    }
    p.update(lgb_device_params())
    return p


def _lgb_train_one(params, X_trn, y_trn, X_val, y_val, cats):
    """LightGBM 1モデル学習。GPUが使えなければ自動でCPUへ落として再試行。"""
    global _LGB_GPU_DISABLED
    import lightgbm as lgb
    if _LGB_GPU_DISABLED:
        params = {**params, "device_type": "cpu"}
        params.pop("gpu_platform_id", None)
        params.pop("gpu_device_id", None)
    lgb_train = lgb.Dataset(X_trn, y_trn, categorical_feature=cats)
    lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train, categorical_feature=cats)
    try:
        model = lgb.train(
            params, lgb_train, valid_sets=[lgb_eval],
            callbacks=[lgb.early_stopping(config.GBDT_EARLY_STOPPING_ROUNDS, verbose=False)],
            num_boost_round=config.GBDT_NUM_BOOST_ROUND,
        )
    except Exception as e:
        if not _LGB_GPU_DISABLED and params.get("device_type") == "gpu":
            print(f"  [LGBM] GPU学習に失敗({type(e).__name__})。CPUへフォールバックします。")
            _LGB_GPU_DISABLED = True
            cpu_params = {**params, "device_type": "cpu"}
            cpu_params.pop("gpu_platform_id", None)
            cpu_params.pop("gpu_device_id", None)
            model = lgb.train(
                cpu_params, lgb_train, valid_sets=[lgb_eval],
                callbacks=[lgb.early_stopping(config.GBDT_EARLY_STOPPING_ROUNDS, verbose=False)],
                num_boost_round=config.GBDT_NUM_BOOST_ROUND,
            )
        else:
            raise
    return model


def _run_lightgbm_single(X, y, X_test, cats, params, seed):
    folds = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    test = np.zeros(len(X_test))
    best_iters = []
    params = {**params, "random_state": seed, "seed": seed}
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
        model = _lgb_train_one(params, X.iloc[trn_idx], y.iloc[trn_idx],
                               X.iloc[val_idx], y.iloc[val_idx], cats)
        oof[val_idx] = model.predict(X.iloc[val_idx], num_iteration=model.best_iteration)
        test += model.predict(X_test, num_iteration=model.best_iteration) / folds.n_splits
        best_iters.append(model.best_iteration)
    return oof, test, best_iters


def _lgb_full_refit(X, y, X_test, cats, params, seed, n_rounds):
    """全データ(100%)で再学習し、test予測を返す（early stop不可なのでCVの平均best_iterを使う）。"""
    import lightgbm as lgb
    p = {**params, "random_state": seed, "seed": seed}
    if _LGB_GPU_DISABLED:
        p = {**p, "device_type": "cpu"}
        p.pop("gpu_platform_id", None); p.pop("gpu_device_id", None)
    dall = lgb.Dataset(X, y, categorical_feature=cats)
    model = lgb.train(p, dall, num_boost_round=max(n_rounds, 1))
    return model.predict(X_test, num_iteration=max(n_rounds, 1))


def run_lightgbm(X, y, X_test, cats, params=None, seeds=None):
    """シード平均版 LightGBM。各seedで5-fold CVを回しOOF/testを平均。さらに全データ再学習をブレンド。"""
    params = params or _load_best_params("lgb") or _lgb_base_params()
    # tuned paramsにはobjective等が無いことがあるのでベースで補完
    params = {**_lgb_base_params(), **params}
    seeds = seeds or config.SEED_LIST

    oof_acc = np.zeros(len(X))
    test_acc = np.zeros(len(X_test))
    per_seed = []
    all_iters = []
    for s in seeds:
        oof, test, best_iters = _run_lightgbm_single(X, y, X_test, cats, params, s)
        oof_acc += oof / len(seeds)
        test_acc += test / len(seeds)
        all_iters += best_iters
        auc_s = roc_auc_score(y, oof)
        per_seed.append(auc_s)
        print(f"  [LGBM] seed={s}: OOF AUC={auc_s:.6f}")

    if config.FULL_REFIT and all_iters:
        n_rounds = int(np.mean(all_iters) * 1.1)
        print(f"  [LGBM] 全データ再学習 (n_rounds={n_rounds}, seed平均{len(seeds)}本)...")
        full_test = np.zeros(len(X_test))
        for s in seeds:
            full_test += _lgb_full_refit(X, y, X_test, cats, params, s, n_rounds) / len(seeds)
        w = config.FULL_REFIT_WEIGHT
        test_acc = (1 - w) * test_acc + w * full_test

    overall = roc_auc_score(y, oof_acc)
    print(f"  [LGBM] seed平均後 OOF AUC: {overall:.6f}  (device={'cpu' if _LGB_GPU_DISABLED else config.LGB_DEVICE})")
    return oof_acc, test_acc, overall, per_seed


# =====================================================================
# XGBoost
# =====================================================================
def _xgb_base_params():
    p = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": config.GBDT_LEARNING_RATE,
        "max_depth": 6,
        "min_child_weight": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "alpha": 0.1,
        "lambda": 1.0,
    }
    p.update(xgb_device_params())
    return p


def _run_xgboost_single(X, y, X_test, params, seed):
    import xgboost as xgb
    folds = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    test = np.zeros(len(X_test))
    best_iters = []
    params = {**params, "random_state": seed, "seed": seed}
    dtest = xgb.DMatrix(X_test, enable_categorical=True)
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
        dtrain = xgb.DMatrix(X.iloc[trn_idx], label=y.iloc[trn_idx], enable_categorical=True)
        dvalid = xgb.DMatrix(X.iloc[val_idx], label=y.iloc[val_idx], enable_categorical=True)
        model = xgb.train(
            params, dtrain, num_boost_round=config.GBDT_NUM_BOOST_ROUND,
            evals=[(dvalid, "valid")],
            early_stopping_rounds=config.GBDT_EARLY_STOPPING_ROUNDS, verbose_eval=False,
        )
        best = model.best_iteration + 1
        oof[val_idx] = model.predict(dvalid, iteration_range=(0, best))
        test += model.predict(dtest, iteration_range=(0, best)) / folds.n_splits
        best_iters.append(best)
    return oof, test, best_iters


def _xgb_full_refit(X, y, X_test, params, seed, n_rounds):
    import xgboost as xgb
    p = {**params, "random_state": seed, "seed": seed}
    dall = xgb.DMatrix(X, label=y, enable_categorical=True)
    dtest = xgb.DMatrix(X_test, enable_categorical=True)
    model = xgb.train(p, dall, num_boost_round=max(n_rounds, 1), verbose_eval=False)
    return model.predict(dtest, iteration_range=(0, max(n_rounds, 1)))


def run_xgboost(X, y, X_test, params=None, seeds=None):
    """シード平均版 XGBoost（全データ再学習ブレンド付き）。"""
    params = params or _load_best_params("xgb") or _xgb_base_params()
    params = {**_xgb_base_params(), **params}
    seeds = seeds or config.SEED_LIST

    oof_acc = np.zeros(len(X))
    test_acc = np.zeros(len(X_test))
    per_seed = []
    all_iters = []
    for s in seeds:
        oof, test, best_iters = _run_xgboost_single(X, y, X_test, params, s)
        oof_acc += oof / len(seeds)
        test_acc += test / len(seeds)
        all_iters += best_iters
        auc_s = roc_auc_score(y, oof)
        per_seed.append(auc_s)
        print(f"  [XGB ] seed={s}: OOF AUC={auc_s:.6f}")

    if config.FULL_REFIT and all_iters:
        n_rounds = int(np.mean(all_iters) * 1.1)
        print(f"  [XGB ] 全データ再学習 (n_rounds={n_rounds})...")
        full_test = np.zeros(len(X_test))
        for s in seeds:
            full_test += _xgb_full_refit(X, y, X_test, params, s, n_rounds) / len(seeds)
        w = config.FULL_REFIT_WEIGHT
        test_acc = (1 - w) * test_acc + w * full_test

    overall = roc_auc_score(y, oof_acc)
    print(f"  [XGB ] seed平均後 OOF AUC: {overall:.6f}  (device={config.XGB_DEVICE})")
    return oof_acc, test_acc, overall, per_seed


def _update_scores(key, oof_auc, fold_scores):
    scores_path = config.ARTIFACT_DIR / "cv_scores.json"
    scores = json.load(open(scores_path)) if scores_path.exists() else {}
    scores[key] = {"oof_auc": oof_auc, "seed_scores": fold_scores}
    with open(scores_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-lgb", action="store_true")
    parser.add_argument("--skip-xgb", action="store_true")
    args = parser.parse_args()

    print(config.describe())
    with timer("特徴量 + DAE埋め込み読み込み"):
        train_df, test_df, categorical_features = load_features_with_dae()
        X, y, X_test = prepare_xy(train_df, test_df, categorical_features)
        cats = [c for c in categorical_features if c in X.columns]
        print(f"  特徴量数: {X.shape[1]} (うちカテゴリ: {len(cats)})  train={len(X)} test={len(X_test)}")

    np.save(config.ARTIFACT_DIR / "y_train.npy", y.values)
    test_df[["SK_ID_CURR"]].to_csv(config.ARTIFACT_DIR / "test_ids.csv", index=False)

    if not args.skip_lgb:
        try:
            import lightgbm  # noqa
            with timer("LightGBM 学習 (seed平均)"):
                oof, test, auc, seed_scores = run_lightgbm(X, y, X_test, cats)
                np.save(config.ARTIFACT_DIR / "lgb_oof.npy", oof)
                np.save(config.ARTIFACT_DIR / "lgb_test.npy", test)
                _update_scores("lightgbm", auc, seed_scores)
        except ImportError:
            print("LightGBM未インストールのためスキップ")

    if not args.skip_xgb:
        try:
            import xgboost  # noqa
            with timer("XGBoost 学習 (seed平均)"):
                oof, test, auc, seed_scores = run_xgboost(X, y, X_test)
                np.save(config.ARTIFACT_DIR / "xgb_oof.npy", oof)
                np.save(config.ARTIFACT_DIR / "xgb_test.npy", test)
                _update_scores("xgboost", auc, seed_scores)
        except ImportError:
            print("XGBoost未インストールのためスキップ")

    print("=" * 60)
    print("GBDT(LGBM/XGB)学習完了")
    print("=" * 60)


if __name__ == "__main__":
    main()
