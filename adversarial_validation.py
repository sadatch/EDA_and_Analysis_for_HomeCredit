"""
Adversarial Validation（Grandmaster Playbook「1. smarter EDA」）。

train/test を見分ける分類器（target = is_test）を作り、そのAUCで分布シフトの大きさを測る。
AUCが0.5付近ならtrain/testは同分布で安心。0.5から大きく外れる場合、
重要度上位の列が「test特有の偏り」を持っているため、特徴から外すか扱いに注意する。

出力:
  artifacts/adversarial_report.json   (AUC + 重要度上位の列)
"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import config
from utils import timer
from train_gbdt import load_features_with_dae, prepare_xy

warnings.filterwarnings("ignore")


def main():
    try:
        import lightgbm as lgb
    except ImportError:
        print("LightGBM未インストールのためadversarial validationをスキップ")
        return

    print(config.describe())
    with timer("特徴量読み込み"):
        train_df, test_df, categorical_features = load_features_with_dae()
        X_tr, _, X_te = prepare_xy(train_df, test_df, categorical_features)
        cats = [c for c in categorical_features if c in X_tr.columns]

    # train=0, test=1 のラベルで結合
    X_tr = X_tr.copy(); X_te = X_te.copy()
    # OOF由来の特徴(target encoding / 近傍TARGET平均)は構造上train(OOF)とtest(full-fit)で
    # 分布が必ず少し異なるため、adversarialが常にそこを拾って本来の分布シフト診断を覆い隠す。
    # これらは除外して「生の特徴」での分布シフトを測る。
    excluded = [c for c in X_tr.columns if c.startswith("TE_") or c.startswith("NEIGHBORS_TARGET_MEAN")]
    common = [c for c in X_tr.columns if c in X_te.columns and c not in excluded]
    if excluded:
        print(f"  adversarialから除外したOOF特徴: {len(excluded)}列")
    X = pd.concat([X_tr[common], X_te[common]], axis=0, ignore_index=True)
    y = np.concatenate([np.zeros(len(X_tr)), np.ones(len(X_te))])
    cats = [c for c in cats if c in common]

    folds = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
    oof = np.zeros(len(X))
    importances = np.zeros(len(common))
    params = {
        "objective": "binary", "metric": "auc", "learning_rate": 0.05,
        "num_leaves": 64, "verbosity": -1, "num_threads": config.N_THREADS,
    }

    with timer("adversarial 分類器の学習"):
        for trn_idx, val_idx in folds.split(X, y):
            dtr = lgb.Dataset(X.iloc[trn_idx], y[trn_idx], categorical_feature=cats)
            dva = lgb.Dataset(X.iloc[val_idx], y[val_idx], reference=dtr, categorical_feature=cats)
            model = lgb.train(params, dtr, valid_sets=[dva],
                              callbacks=[lgb.early_stopping(30, verbose=False)],
                              num_boost_round=500)
            oof[val_idx] = model.predict(X.iloc[val_idx], num_iteration=model.best_iteration)
            importances += model.feature_importance(importance_type="gain") / folds.n_splits

    auc = roc_auc_score(y, oof)
    imp = pd.Series(importances, index=common).sort_values(ascending=False)
    top = imp.head(30)

    print("=" * 60)
    print(f"Adversarial AUC = {auc:.4f}  (0.5付近=同分布で安心 / 高いほど分布シフト大)")
    print("train/testを最も見分けている列 TOP15:")
    for name, val in top.head(15).items():
        print(f"  {name:40s} {val:12.1f}")
    print("=" * 60)

    report = {
        "adversarial_auc": float(auc),
        "interpretation": ("train/testはほぼ同分布。リーク列の心配は小さい。" if auc < 0.6
                            else "分布シフトあり。重要度上位の列は値の意味を要確認（IDライクな列やtest特有の偏りに注意）。"),
        "top_shift_features": {k: float(v) for k, v in top.items()},
    }
    with open(config.ARTIFACT_DIR / "adversarial_report.json", "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"レポート保存: {config.ARTIFACT_DIR / 'adversarial_report.json'}")


if __name__ == "__main__":
    main()
