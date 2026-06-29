"""
Null Importance による特徴量選択（Olivier法）。

本物のTARGETで学習した時の重要度と、TARGETをシャッフルして学習した時の重要度（=偶然の重要度＝null）
を比較し、nullの分布を有意に上回る特徴だけを「効いている特徴」として残す。
大量に生成した集約特徴の中からノイズ列を落とし、GBDTの過学習・学習時間を減らせる。

出力:
  artifacts/feature_selection.json  (keep / drop リスト + スコア)

注: この結果を実際の学習に反映するには HC_APPLY_FS=1 を設定する（train_gbdt等が drop リストを読む）。
    デフォルトではレポート生成のみ（まず人間が中身を確認できるように安全側）。
"""
import json
import warnings

import numpy as np
import pandas as pd

import config
from utils import timer
from train_gbdt import load_features_with_dae, prepare_xy

warnings.filterwarnings("ignore")

N_NULL_RUNS = 3 if config.SMOKE else 30


def _get_importances(X, y, cats, shuffle: bool, seed: int):
    import lightgbm as lgb
    rng = np.random.RandomState(seed)
    y_use = rng.permutation(y) if shuffle else y
    params = {
        "objective": "binary", "metric": "auc", "learning_rate": 0.05,
        "num_leaves": 96, "min_child_samples": 60, "subsample": 0.8,
        "colsample_bytree": 0.7, "verbosity": -1, "num_threads": config.N_THREADS,
        "seed": seed,
    }
    dtrain = lgb.Dataset(X, y_use, categorical_feature=cats, free_raw_data=False)
    n_rounds = 80 if config.SMOKE else 350
    model = lgb.train(params, dtrain, num_boost_round=n_rounds)
    return model.feature_importance(importance_type="gain")


def main():
    try:
        import lightgbm  # noqa
    except ImportError:
        print("LightGBM未インストールのため特徴量選択をスキップ")
        return

    print(config.describe())
    with timer("特徴量読み込み"):
        train_df, test_df, categorical_features = load_features_with_dae()
        X, y, _ = prepare_xy(train_df, test_df, categorical_features)
        y = y.values
        cats = [c for c in categorical_features if c in X.columns]
        cols = list(X.columns)

    with timer("実際のTARGETでの重要度（複数seed平均）"):
        n_actual = 1 if config.SMOKE else 3
        actual_imp = np.mean([_get_importances(X, y, cats, False, config.SEED + i)
                              for i in range(n_actual)], axis=0)

    with timer(f"null重要度の分布を作成 ({N_NULL_RUNS}回シャッフル学習)"):
        null_imp = np.zeros((N_NULL_RUNS, len(cols)))
        for i in range(N_NULL_RUNS):
            null_imp[i] = _get_importances(X, y, cats, True, config.SEED + 100 + i)

    # スコア: 実際の重要度が null分布の何パーセンタイルに相当するか
    null_p75 = np.percentile(null_imp, 75, axis=0)
    scores = np.log1p(actual_imp) - np.log1p(null_p75)   # 正なら本物がnullを上回る
    score_s = pd.Series(scores, index=cols).sort_values(ascending=False)

    # keep: スコア>0（nullの75%点を上回る）。ただしカテゴリ列は安全のため常にkeep。
    keep = [c for c in cols if (score_s[c] > 0) or (c in cats)]
    drop = [c for c in cols if c not in keep]

    print("=" * 60)
    print(f"特徴量選択: 全{len(cols)}列 -> keep {len(keep)} / drop {len(drop)}")
    print("落とす候補（nullを超えられなかった）上位例:")
    for c in score_s.tail(10).index[::-1]:
        print(f"  {c:45s} score={score_s[c]:+.3f}")
    print("=" * 60)

    out = {
        "n_total": len(cols),
        "n_keep": len(keep),
        "n_drop": len(drop),
        "keep": keep,
        "drop": drop,
        "note": "学習に反映するには HC_APPLY_FS=1 を設定（dropリストを除外して学習）",
    }
    with open(config.ARTIFACT_DIR / "feature_selection.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"レポート保存: {config.ARTIFACT_DIR / 'feature_selection.json'}")


if __name__ == "__main__":
    main()
