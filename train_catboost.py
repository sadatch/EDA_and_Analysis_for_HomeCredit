"""
CatBoost 学習スクリプト（GPU対応 + シード平均）。

LGBM/XGBとは欠損・カテゴリの扱い（ordered target statistics）が異なり、
誤りの系統が変わるためアンサンブル多様性が増す（最近の上位解法では定番の3本柱）。

出力:
  artifacts/cat_oof.npy, artifacts/cat_test.npy
  artifacts/cv_scores.json (追記)
"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import config
from utils import timer, cat_device_params
from train_gbdt import load_features_with_dae

warnings.filterwarnings("ignore")


def _prepare_catboost(train_df, test_df, categorical_features):
    """CatBoost用にX/yを準備。カテゴリ列はNaNを文字列化して欠損もカテゴリの一種として扱う。"""
    X = train_df.drop(columns=["SK_ID_CURR", "TARGET"])
    y = train_df["TARGET"]
    X_test = test_df.drop(columns=["SK_ID_CURR"])
    cats = [c for c in categorical_features if c in X.columns]
    for c in cats:
        X[c] = X[c].astype("object").where(X[c].notna(), "__NA__").astype(str)
        X_test[c] = X_test[c].astype("object").where(X_test[c].notna(), "__NA__").astype(str)
    return X, y, X_test, cats


def _cat_base_params():
    p = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "learning_rate": config.GBDT_LEARNING_RATE,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "iterations": config.GBDT_NUM_BOOST_ROUND,
        "od_type": "Iter",
        "od_wait": config.GBDT_EARLY_STOPPING_ROUNDS,
        "verbose": False,
        "allow_writing_files": False,
    }
    p.update(cat_device_params())
    return p


def _load_best_params():
    p = config.ARTIFACT_DIR / "cat_best_params.json"
    if p.exists():
        with open(p) as f:
            params = json.load(f)
        print(f"  [CAT ] Optuna best params をロード: {p.name}")
        return params
    return None


def _run_single(X, y, X_test, cats, params, seed):
    from catboost import CatBoostClassifier, Pool
    folds = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    test = np.zeros(len(X_test))
    params = {**params, "random_seed": seed}
    test_pool = Pool(X_test, cat_features=cats)
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
        trn_pool = Pool(X.iloc[trn_idx], y.iloc[trn_idx], cat_features=cats)
        val_pool = Pool(X.iloc[val_idx], y.iloc[val_idx], cat_features=cats)
        model = CatBoostClassifier(**params)
        try:
            model.fit(trn_pool, eval_set=val_pool, use_best_model=True)
        except Exception as e:
            # GPUが使えない等の場合はCPUへフォールバック
            if params.get("task_type") == "GPU":
                print(f"  [CAT ] GPU学習に失敗({type(e).__name__})。CPUへフォールバック。")
                cpu_params = {**params, "task_type": "CPU", "thread_count": config.N_THREADS}
                cpu_params.pop("devices", None)
                model = CatBoostClassifier(**cpu_params)
                model.fit(trn_pool, eval_set=val_pool, use_best_model=True)
                params = cpu_params
            else:
                raise
        oof[val_idx] = model.predict_proba(val_pool)[:, 1]
        test += model.predict_proba(test_pool)[:, 1] / folds.n_splits
    return oof, test


def run_catboost(X, y, X_test, cats, params=None, seeds=None):
    params = params or _load_best_params() or _cat_base_params()
    params = {**_cat_base_params(), **params}
    seeds = seeds or config.SEED_LIST
    oof_acc = np.zeros(len(X))
    test_acc = np.zeros(len(X_test))
    per_seed = []
    for s in seeds:
        oof, test = _run_single(X, y, X_test, cats, params, s)
        oof_acc += oof / len(seeds)
        test_acc += test / len(seeds)
        auc_s = roc_auc_score(y, oof)
        per_seed.append(auc_s)
        print(f"  [CAT ] seed={s}: OOF AUC={auc_s:.6f}")
    overall = roc_auc_score(y, oof_acc)
    print(f"  [CAT ] seed平均後 OOF AUC: {overall:.6f}  (task_type={config.CAT_TASK_TYPE})")
    return oof_acc, test_acc, overall, per_seed


def main():
    print(config.describe())
    try:
        import catboost  # noqa
    except ImportError:
        print("CatBoost未インストールのためスキップ (pip install catboost)")
        return

    with timer("特徴量 + DAE埋め込み読み込み"):
        train_df, test_df, categorical_features = load_features_with_dae()
        X, y, X_test, cats = _prepare_catboost(train_df, test_df, categorical_features)
        print(f"  特徴量数: {X.shape[1]} (うちカテゴリ: {len(cats)})")

    with timer("CatBoost 学習 (seed平均)"):
        oof, test, auc, seed_scores = run_catboost(X, y, X_test, cats)
        np.save(config.ARTIFACT_DIR / "cat_oof.npy", oof)
        np.save(config.ARTIFACT_DIR / "cat_test.npy", test)
        scores_path = config.ARTIFACT_DIR / "cv_scores.json"
        scores = json.load(open(scores_path)) if scores_path.exists() else {}
        scores["catboost"] = {"oof_auc": auc, "seed_scores": seed_scores}
        with open(scores_path, "w") as f:
            json.dump(scores, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"CatBoost学習完了: OOF AUC = {auc:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
