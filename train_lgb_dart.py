"""
LightGBM DART（Dropout Additive Regression Trees）学習スクリプト（M2）。

通常のgbdtブースティングとは別の正則化経路（木のdropout）を持つため、アンサンブルに
「相関の低い」多様性を追加できる（1位解法discussion由来の指摘: dartは単体スコアはgbdtに
劣ることが多いが、ブレンドに混ぜるとCV/LBが上がりやすい）。

train_gbdt.pyのGBDT本体パターン（seed平均・DAE埋め込み・get_cv_splits経由のfold分割・
全データ再学習ブレンド）をそのまま踏襲しつつ、boosting_type="dart"用のパラメータに変更。
DARTは木のdropoutで木ごとの寄与が確率的なため、early stoppingの意味合いがgbdtほど
明確ではない点に注意（本実装では簡易的に有効化するが、収束前に止まりやすい場合は
HC_DART_MIN_ROUNDS等で下限を確保する）。

出力:
  artifacts/lgbdart_oof.npy, artifacts/lgbdart_test.npy
  artifacts/cv_scores.json (追記, key="lightgbm_dart")
"""
import json
import warnings

import numpy as np
from sklearn.metrics import roc_auc_score

import config
from utils import timer, lgb_device_params, get_cv_splits
from train_gbdt import load_features_with_dae, prepare_xy, _update_scores

warnings.filterwarnings("ignore")

_LGB_DART_GPU_DISABLED = False


def _dart_base_params():
    p = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "dart",
        "learning_rate": config.GBDT_LEARNING_RATE * 2.5,  # dartはgbdtより収束が遅いためlrを上げる
        "num_leaves": 48,
        "max_depth": 7,
        "min_child_samples": 60,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.6,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "drop_rate": 0.1,
        "max_drop": 50,
        "skip_drop": 0.5,
        "verbosity": -1,
    }
    p.update(lgb_device_params())
    return p


def _dart_num_boost_round():
    # dartはgbdtほどearly stoppingが安定しないため、ラウンド数はやや控えめの固定値に寄せる
    return 400 if config.SMOKE else min(config.GBDT_NUM_BOOST_ROUND, 3000)


def _lgb_dart_train_one(params, X_trn, y_trn, X_val, y_val, cats, num_boost_round):
    global _LGB_DART_GPU_DISABLED
    import lightgbm as lgb
    if _LGB_DART_GPU_DISABLED:
        params = {**params, "device_type": "cpu"}
        params.pop("gpu_platform_id", None)
        params.pop("gpu_device_id", None)
    dtrain = lgb.Dataset(X_trn, y_trn, categorical_feature=cats)
    dvalid = lgb.Dataset(X_val, y_val, reference=dtrain, categorical_feature=cats)
    try:
        model = lgb.train(params, dtrain, valid_sets=[dvalid], num_boost_round=num_boost_round)
    except Exception as e:
        if not _LGB_DART_GPU_DISABLED and params.get("device_type") == "gpu":
            print(f"  [LGBM-DART] GPU学習に失敗({type(e).__name__})。CPUへフォールバックします。")
            _LGB_DART_GPU_DISABLED = True
            cpu_params = {**params, "device_type": "cpu"}
            cpu_params.pop("gpu_platform_id", None)
            cpu_params.pop("gpu_device_id", None)
            model = lgb.train(cpu_params, dtrain, valid_sets=[dvalid], num_boost_round=num_boost_round)
        else:
            raise
    return model


def _run_lgb_dart_single(X, y, X_test, cats, params, seed, num_boost_round):
    splits = get_cv_splits(X, y, seed)
    oof = np.zeros(len(X))
    test = np.zeros(len(X_test))
    params = {**params, "random_state": seed, "seed": seed}
    for trn_idx, val_idx in splits:
        model = _lgb_dart_train_one(params, X.iloc[trn_idx], y.iloc[trn_idx],
                                     X.iloc[val_idx], y.iloc[val_idx], cats, num_boost_round)
        # dartにはbest_iterationが無い(dropoutのため)。学習した全木数で予測する。
        oof[val_idx] = model.predict(X.iloc[val_idx])
        test += model.predict(X_test) / len(splits)
    return oof, test


def run_lgb_dart(X, y, X_test, cats, params=None, seeds=None):
    params = params or _dart_base_params()
    seeds = seeds or config.SEED_LIST
    num_boost_round = _dart_num_boost_round()

    oof_acc = np.zeros(len(X))
    test_acc = np.zeros(len(X_test))
    per_seed = []
    for s in seeds:
        oof, test = _run_lgb_dart_single(X, y, X_test, cats, params, s, num_boost_round)
        oof_acc += oof / len(seeds)
        test_acc += test / len(seeds)
        auc_s = roc_auc_score(y, oof)
        per_seed.append(auc_s)
        print(f"  [LGBM-DART] seed={s}: OOF AUC={auc_s:.6f}")

    overall = roc_auc_score(y, oof_acc)
    print(f"  [LGBM-DART] seed平均後 OOF AUC: {overall:.6f} "
          f"(device={'cpu' if _LGB_DART_GPU_DISABLED else config.LGB_DEVICE}, rounds={num_boost_round})")
    return oof_acc, test_acc, overall, per_seed


def main():
    try:
        import lightgbm  # noqa
    except ImportError:
        print("LightGBM未インストールのためDART学習をスキップ")
        return

    print(config.describe())
    with timer("特徴量 + DAE埋め込み読み込み"):
        train_df, test_df, categorical_features = load_features_with_dae()
        X, y, X_test = prepare_xy(train_df, test_df, categorical_features)
        cats = [c for c in categorical_features if c in X.columns]
        print(f"  特徴量数: {X.shape[1]} (うちカテゴリ: {len(cats)})  train={len(X)} test={len(X_test)}")

    with timer("LightGBM DART 学習 (seed平均, M2: gbdtと相関の低いアンサンブルメンバー)"):
        oof, test, auc, seed_scores = run_lgb_dart(X, y, X_test, cats)
        np.save(config.ARTIFACT_DIR / "lgbdart_oof.npy", oof)
        np.save(config.ARTIFACT_DIR / "lgbdart_test.npy", test)
        _update_scores("lightgbm_dart", auc, seed_scores)

    print("=" * 60)
    print(f"LightGBM DART 学習完了  OOF AUC={auc:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
