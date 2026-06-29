"""
Optuna ハイパーパラメータ探索（中断・再開可能）。

SQLiteに探索履歴を保存するため、寝バッチが途中で止まっても同じコマンドで再開できる
（既存trial数を引き継ぎ、不足分だけ追加探索）。見つかった best params は
artifacts/{lgb,xgb,cat}_best_params.json に保存され、train_*.py が自動でロードする。

使い方:
  python tune.py --model lgb         # LightGBMを探索
  python tune.py --model xgb
  python tune.py --model cat
  python tune.py --model all         # 3つ順に
  HC_OPTUNA_TRIALS=80 python tune.py --model all

探索を速くするため、ここでは 3-fold・単一seed・やや高めのlearning rateで評価する
（最終学習は train_*.py が 5-fold・seed平均・小さいlrで行う）。
"""
import argparse
import json
import warnings

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import config
from utils import timer, lgb_device_params, xgb_device_params, cat_device_params
from train_gbdt import load_features_with_dae, prepare_xy
from train_catboost import _prepare_catboost

warnings.filterwarnings("ignore")

TUNE_FOLDS = 3
TUNE_ROUNDS = 150 if config.SMOKE else 3000
TUNE_LR = 0.03
TUNE_ES = 50


def _cv_auc_lgb(params, X, y, cats):
    import lightgbm as lgb
    folds = StratifiedKFold(n_splits=TUNE_FOLDS, shuffle=True, random_state=config.SEED)
    oof = np.zeros(len(X))
    for trn, val in folds.split(X, y):
        dtr = lgb.Dataset(X.iloc[trn], y.iloc[trn], categorical_feature=cats)
        dva = lgb.Dataset(X.iloc[val], y.iloc[val], reference=dtr, categorical_feature=cats)
        m = lgb.train(params, dtr, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(TUNE_ES, verbose=False)],
                      num_boost_round=TUNE_ROUNDS)
        oof[val] = m.predict(X.iloc[val], num_iteration=m.best_iteration)
    return roc_auc_score(y, oof)


def _objective_lgb(trial, X, y, cats):
    max_depth = trial.suggest_int("max_depth", 4, 12)
    params = {
        "objective": "binary", "metric": "auc", "learning_rate": TUNE_LR, "verbosity": -1,
        "max_depth": max_depth,
        "num_leaves": trial.suggest_int("num_leaves", 16, min(2 ** max_depth, 256)),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }
    params.update(lgb_device_params())
    return _cv_auc_lgb(params, X, y, cats)


def _objective_xgb(trial, X, y):
    import xgboost as xgb
    params = {
        "objective": "binary:logistic", "eval_metric": "auc", "learning_rate": TUNE_LR,
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
        "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
    }
    params.update(xgb_device_params())
    folds = StratifiedKFold(n_splits=TUNE_FOLDS, shuffle=True, random_state=config.SEED)
    oof = np.zeros(len(X))
    for trn, val in folds.split(X, y):
        dtr = xgb.DMatrix(X.iloc[trn], label=y.iloc[trn], enable_categorical=True)
        dva = xgb.DMatrix(X.iloc[val], label=y.iloc[val], enable_categorical=True)
        m = xgb.train(params, dtr, num_boost_round=TUNE_ROUNDS, evals=[(dva, "v")],
                      early_stopping_rounds=TUNE_ES, verbose_eval=False)
        oof[val] = m.predict(dva, iteration_range=(0, m.best_iteration + 1))
    return roc_auc_score(y, oof)


def _objective_cat(trial, X, y, cats):
    from catboost import CatBoostClassifier, Pool
    params = {
        "loss_function": "Logloss", "eval_metric": "AUC", "learning_rate": TUNE_LR,
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "iterations": TUNE_ROUNDS, "od_type": "Iter", "od_wait": TUNE_ES,
        "verbose": False, "allow_writing_files": False,
    }
    params.update(cat_device_params())
    folds = StratifiedKFold(n_splits=TUNE_FOLDS, shuffle=True, random_state=config.SEED)
    oof = np.zeros(len(X))
    for trn, val in folds.split(X, y):
        trn_pool = Pool(X.iloc[trn], y.iloc[trn], cat_features=cats)
        val_pool = Pool(X.iloc[val], y.iloc[val], cat_features=cats)
        m = CatBoostClassifier(**params)
        m.fit(trn_pool, eval_set=val_pool, use_best_model=True)
        oof[val] = m.predict_proba(val_pool)[:, 1]
    return roc_auc_score(y, oof)


def _tune_one(model: str, X, y, cats):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        study_name=f"hc_{model}",
        storage=config.OPTUNA_STORAGE,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=config.SEED),
    )
    done = len([t for t in study.trials if t.state.is_finished()])
    remaining = max(config.OPTUNA_TRIALS - done, 0)
    print(f"  [{model}] 既存trial={done} / 目標={config.OPTUNA_TRIALS} -> 追加{remaining}回")

    if model == "lgb":
        func = lambda t: _objective_lgb(t, X, y, cats)
    elif model == "xgb":
        func = lambda t: _objective_xgb(t, X, y)
    else:
        func = lambda t: _objective_cat(t, X, y, cats)

    if remaining > 0:
        study.optimize(func, n_trials=remaining,
                       timeout=(config.OPTUNA_TIMEOUT or None),
                       show_progress_bar=False)

    print(f"  [{model}] best AUC(3-fold)={study.best_value:.6f}")
    out_path = config.ARTIFACT_DIR / f"{model}_best_params.json"
    with open(out_path, "w") as f:
        json.dump(study.best_params, f, ensure_ascii=False, indent=2)
    print(f"  [{model}] best params 保存: {out_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lgb", "xgb", "cat", "all"], default="all")
    args = parser.parse_args()

    try:
        import optuna  # noqa
    except ImportError:
        print("optuna未インストールのためチューニングをスキップ")
        return

    print(config.describe())
    with timer("特徴量読み込み"):
        train_df, test_df, categorical_features = load_features_with_dae()

    targets = ["lgb", "xgb", "cat"] if args.model == "all" else [args.model]
    for model in targets:
        try:
            if model == "cat":
                import catboost  # noqa
                X, y, _, cats = _prepare_catboost(train_df, test_df, categorical_features)
            else:
                if model == "lgb":
                    import lightgbm  # noqa
                else:
                    import xgboost  # noqa
                X, y, _ = prepare_xy(train_df, test_df, categorical_features)
                cats = [c for c in categorical_features if c in X.columns]
            with timer(f"Optuna探索: {model}"):
                _tune_one(model, X, y, cats)
        except ImportError:
            print(f"  {model}: ライブラリ未インストールのためスキップ")

    print("=" * 60)
    print("Optunaチューニング完了。train_*.pyが best_params を自動ロードします。")
    print("=" * 60)


if __name__ == "__main__":
    main()
