"""
アンサンブル & submission作成スクリプト。

LightGBM / XGBoost / MLP(DAE) の OOF予測を使い、scipy.optimizeでAUCを最大化する
ブレンド重みを探索（単純な0.5:0.5ではなく実測ベースで決定）。
見つかった重みでtest予測をブレンドし、最終submission.csvを作成する。
"""
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

import config


def load_available_models():
    """artifacts/にOOF/testが存在するモデルだけを読み込む。"""
    candidates = ["lgb", "xgb", "mlp"]
    oofs, tests, names = [], [], []
    for name in candidates:
        oof_path = config.ARTIFACT_DIR / f"{name}_oof.npy"
        test_path = config.ARTIFACT_DIR / f"{name}_test.npy"
        if oof_path.exists() and test_path.exists():
            oofs.append(np.load(oof_path))
            tests.append(np.load(test_path))
            names.append(name)
        else:
            print(f"  {name}: 見つからないためスキップ")
    return oofs, tests, names


def optimize_weights(oofs: list, y: np.ndarray) -> np.ndarray:
    """Nelder-Meadで -AUC(weighted_blend) を最小化する重みを探索。重みはsoftmaxで正規化して非負和1制約を満たす。"""
    n = len(oofs)
    oof_matrix = np.stack(oofs, axis=1)  # (n_samples, n_models)

    def neg_auc(raw_weights):
        w = np.exp(raw_weights)
        w = w / w.sum()
        blend = oof_matrix @ w
        return -roc_auc_score(y, blend)

    x0 = np.zeros(n)  # softmax(0,...,0) = 等重み からスタート
    res = minimize(neg_auc, x0, method="Nelder-Mead",
                    options={"xatol": 1e-5, "fatol": 1e-7, "maxiter": 2000})
    w = np.exp(res.x)
    w = w / w.sum()
    return w, -res.fun


def main():
    print("モデル読み込み...")
    oofs, tests, names = load_available_models()
    if len(oofs) == 0:
        raise RuntimeError("OOF予測が1つも見つかりません。train_gbdt.py / train_nn.py を先に実行してください。")

    y = np.load(config.ARTIFACT_DIR / "y_train.npy")
    test_ids = pd.read_csv(config.ARTIFACT_DIR / "test_ids.csv")["SK_ID_CURR"].values

    print("\n単体モデルのOOF AUC:")
    for name, oof in zip(names, oofs):
        print(f"  {name}: {roc_auc_score(y, oof):.6f}")

    if len(oofs) == 1:
        print("\nモデルが1つのみのため重み最適化はスキップ、そのまま使用します。")
        weights = np.array([1.0])
        blended_auc = roc_auc_score(y, oofs[0])
    else:
        print("\nブレンド重みを最適化中...")
        weights, blended_auc = optimize_weights(oofs, y)

    print("\n最適ブレンド重み:")
    for name, w in zip(names, weights):
        print(f"  {name}: {w:.4f}")
    print(f"ブレンド後 OOF AUC: {blended_auc:.6f}")

    test_blend = np.zeros(len(tests[0]))
    for w, t in zip(weights, tests):
        test_blend += w * t

    submission = pd.DataFrame({"SK_ID_CURR": test_ids, "TARGET": test_blend})
    sub_path = config.SUB_DIR / "submission_ensemble.csv"
    submission.to_csv(sub_path, index=False)

    with open(config.ARTIFACT_DIR / "ensemble_weights.json", "w") as f:
        json.dump({
            "names": names,
            "weights": weights.tolist(),
            "blended_oof_auc": blended_auc,
        }, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"submission保存: {sub_path}")
    print(f"最終ブレンドOOF AUC: {blended_auc:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
