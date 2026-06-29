"""
LightGBM 特徴量重要度の分析ツール（金融ドメイン特徴の効き目検証用）。

5-fold CVでLightGBMを学習し、
  - 特徴ごとの gain / split 重要度（CSV）
  - プレフィックス・グループ別のロールアップ（どのカテゴリが全体gainの何%を占めるか）
  - DOM_* 金融ドメイン特徴のランキングと寄与
を出力する。「この指標は要るのか？」をLightGBM視点で一覧確認できる。

使い方:
  python feature_importance.py            # DAE埋め込みも含めて評価
  python feature_importance.py --no-dae   # DOM等の手作り特徴に集中して見る

出力:
  artifacts/feature_importance.csv
  artifacts/feature_importance_groups.csv
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import config
from utils import timer, lgb_device_params
from train_gbdt import load_features_with_dae, prepare_xy, _lgb_train_one

warnings.filterwarnings("ignore")

try:
    from domain_features import DOMAIN_GROUPS
except Exception:
    DOMAIN_GROUPS = {}


def group_of(name: str) -> str:
    """特徴名をカテゴリへ割り当てる。"""
    if name.startswith("DOM_"):
        return "_".join(name.split("_")[:2])          # DOM_CAP 等
    if name.startswith("DAE_"):
        return "DAE(2位)"
    if name.startswith("TE_"):
        return "TargetEnc"
    if name.startswith("NEIGHBORS_TARGET_MEAN"):
        return "Neighbors(1位)"
    for pref, g in [("BUREAU", "BUREAU"), ("PREV", "PREV"), ("POS", "POS"),
                    ("INS", "INSTALLMENTS"), ("CC", "CREDIT_CARD")]:
        if name.startswith(pref):
            return g
    if name.startswith("EXT_SOURCE"):
        return "EXT_SOURCE"
    return "APPLICATION(他)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-dae", action="store_true", help="DAE埋め込みを除外して評価")
    args = parser.parse_args()

    try:
        import lightgbm  # noqa
    except ImportError:
        print("LightGBM未インストールのため重要度分析をスキップ")
        return

    print(config.describe())
    with timer("特徴量読み込み"):
        train_df, test_df, categorical_features = load_features_with_dae()
        X, y, X_test = prepare_xy(train_df, test_df, categorical_features)
        if args.no_dae:
            drop = [c for c in X.columns if c.startswith("DAE_")]
            X = X.drop(columns=drop); X_test = X_test.drop(columns=drop)
            print(f"  DAE列 {len(drop)} を除外")
        cats = [c for c in categorical_features if c in X.columns]
        print(f"  特徴量数: {X.shape[1]} (うちカテゴリ {len(cats)})")

    params = {
        "objective": "binary", "metric": "auc",
        "learning_rate": config.GBDT_LEARNING_RATE, "num_leaves": 64, "max_depth": 8,
        "min_child_samples": 60, "subsample": 0.85, "subsample_freq": 1,
        "colsample_bytree": 0.6, "reg_alpha": 0.1, "reg_lambda": 1.0, "verbosity": -1,
        "seed": config.SEED,
    }
    params.update(lgb_device_params())

    folds = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
    oof = np.zeros(len(X))
    gain = np.zeros(X.shape[1])
    split = np.zeros(X.shape[1])
    with timer("LightGBM 5-fold（重要度集計）"):
        for trn, val in folds.split(X, y):
            model = _lgb_train_one(params, X.iloc[trn], y.iloc[trn], X.iloc[val], y.iloc[val], cats)
            oof[val] = model.predict(X.iloc[val], num_iteration=model.best_iteration)
            gain += model.feature_importance("gain") / folds.n_splits
            split += model.feature_importance("split") / folds.n_splits

    auc = roc_auc_score(y, oof)
    imp = pd.DataFrame({"feature": X.columns, "gain": gain, "split": split})
    imp["group"] = imp["feature"].map(group_of)
    imp["gain_share_%"] = 100 * imp["gain"] / imp["gain"].sum()
    imp = imp.sort_values("gain", ascending=False).reset_index(drop=True)
    imp["rank"] = imp.index + 1
    imp.to_csv(config.ARTIFACT_DIR / "feature_importance.csv", index=False)

    grp = (imp.groupby("group")
           .agg(n_features=("feature", "size"), gain_sum=("gain", "sum"),
                gain_share_pct=("gain_share_%", "sum"), top_rank=("rank", "min"))
           .sort_values("gain_sum", ascending=False))
    grp.to_csv(config.ARTIFACT_DIR / "feature_importance_groups.csv")

    print("=" * 70)
    print(f"OOF AUC = {auc:.6f}   （この特徴セットでのLightGBM単体の水準）")
    print("\n■ カテゴリ別 gain シェア:")
    for g, row in grp.iterrows():
        desc = DOMAIN_GROUPS.get(g, "")
        print(f"  {g:16s} {row['gain_share_pct']:5.1f}%  "
              f"(特徴{int(row['n_features']):3d}個, 最高位{int(row['top_rank'])}位) {desc}")

    dom = imp[imp["feature"].str.startswith("DOM_")]
    if not dom.empty:
        print(f"\n■ 金融ドメイン特徴 DOM_* の効き目（全{len(dom)}個, 合計gainシェア {dom['gain_share_%'].sum():.1f}%）:")
        for _, r in dom.sort_values("gain", ascending=False).iterrows():
            print(f"  {int(r['rank']):4d}位  {r['feature']:34s} gain={r['gain']:12.1f}  ({r['gain_share_%']:.2f}%)")
        weak = dom[dom["gain"] <= imp["gain"].median()]
        if not weak.empty:
            print(f"\n  ※ gainが中央値以下＝効きが弱い候補（落とす検討）: "
                  + ", ".join(weak["feature"].tolist()))
    print("\n■ 全体 TOP20:")
    for _, r in imp.head(20).iterrows():
        print(f"  {int(r['rank']):4d}  {r['feature']:34s} [{r['group']:14s}] gain={r['gain']:12.1f}")
    print("=" * 70)
    print(f"CSV: {config.ARTIFACT_DIR/'feature_importance.csv'} / {config.ARTIFACT_DIR/'feature_importance_groups.csv'}")


if __name__ == "__main__":
    main()
