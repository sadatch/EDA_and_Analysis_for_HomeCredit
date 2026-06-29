"""
アンサンブル & submission作成（hill climbing + 2段スタッキング + 重み最適化を比較）。

利用可能な全baseモデル (lgb / xgb / cat / mlp / lgbpl) のOOF・test予測を読み込み、
以下3手法を実行してOOF AUCが最も高い手法のtest予測を最終submissionにする:

  1. weighted   : softmax重みをNelder-MeadでAUC最大化（順位ブレンド）
  2. hillclimb  : Caruanaのアンサンブル選択（強いモデルから貪欲に加重、最近の上位解法の定番）
  3. stacking   : OOFを特徴にした2段目メタモデル（LogisticRegression, OOF評価でリーク制御）

出力:
  submissions/submission_ensemble.csv         （最良手法）
  submissions/submission_{weighted,hillclimb,stacking}.csv （各手法）
  artifacts/ensemble_report.json
"""
import json
from collections import Counter

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import config

CANDIDATES = ["lgb", "xgb", "cat", "mlp", "lgbpl"]


def load_available_models():
    oofs, tests, names = [], [], []
    for name in CANDIDATES:
        oof_p = config.ARTIFACT_DIR / f"{name}_oof.npy"
        test_p = config.ARTIFACT_DIR / f"{name}_test.npy"
        if oof_p.exists() and test_p.exists():
            oofs.append(np.load(oof_p))
            tests.append(np.load(test_p))
            names.append(name)
        else:
            print(f"  {name}: 見つからないためスキップ")
    return oofs, tests, names


def _rank01(a):
    """順位を[0,1]に正規化（AUCは順位のみで決まるためキャリブレーション差を吸収）。"""
    return (rankdata(a) - 1) / (len(a) - 1)


# ---------- 1. weighted (Nelder-Mead) ----------
def optimize_weights(oof_matrix, y):
    n = oof_matrix.shape[1]

    def neg_auc(raw):
        w = np.exp(raw); w /= w.sum()
        return -roc_auc_score(y, oof_matrix @ w)

    res = minimize(neg_auc, np.zeros(n), method="Nelder-Mead",
                   options={"xatol": 1e-5, "fatol": 1e-7, "maxiter": 3000})
    w = np.exp(res.x); w /= w.sum()
    return w, -res.fun


# ---------- 2. hill climbing (Caruana) ----------
def hill_climb(oof_matrix, y, max_steps=None):
    max_steps = max_steps or config.ENSEMBLE_HILLCLIMB_STEPS
    n_models = oof_matrix.shape[1]
    aucs = [roc_auc_score(y, oof_matrix[:, i]) for i in range(n_models)]
    start = int(np.argmax(aucs))
    selected = [start]
    current = oof_matrix[:, start].copy()
    cur_auc = aucs[start]
    for _ in range(max_steps):
        best_auc, best_i = cur_auc, None
        m = len(selected)
        for i in range(n_models):
            cand = (current * m + oof_matrix[:, i]) / (m + 1)
            a = roc_auc_score(y, cand)
            if a > best_auc + 1e-8:
                best_auc, best_i = a, i
        if best_i is None:
            break
        selected.append(best_i)
        current = (current * m + oof_matrix[:, best_i]) / (m + 1)
        cur_auc = best_auc
    counts = Counter(selected)
    weights = np.array([counts.get(i, 0) for i in range(n_models)], dtype=float)
    weights /= weights.sum()
    return weights, cur_auc


# ---------- 3. stacking (2-level meta model) ----------
def stacking(oof_matrix, test_matrix, y):
    folds = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
    meta_oof = np.zeros(len(y))
    test_fold = np.zeros(test_matrix.shape[0])
    scaler = StandardScaler()
    Xs = scaler.fit_transform(oof_matrix)
    Ts = scaler.transform(test_matrix)
    for trn, val in folds.split(Xs, y):
        lr = LogisticRegression(max_iter=1000, C=1.0)
        lr.fit(Xs[trn], y[trn])
        meta_oof[val] = lr.predict_proba(Xs[val])[:, 1]
        test_fold += lr.predict_proba(Ts)[:, 1] / folds.n_splits
    auc = roc_auc_score(y, meta_oof)
    return test_fold, auc


def main():
    print("モデル読み込み...")
    oofs, tests, names = load_available_models()
    if not oofs:
        raise RuntimeError("OOF予測が1つも見つかりません。train_*.py を先に実行してください。")

    y = np.load(config.ARTIFACT_DIR / "y_train.npy")
    test_ids = pd.read_csv(config.ARTIFACT_DIR / "test_ids.csv")["SK_ID_CURR"].values

    print("\n単体モデルのOOF AUC:")
    for name, oof in zip(names, oofs):
        print(f"  {name:7s}: {roc_auc_score(y, oof):.6f}")

    # 順位正規化版（weighted / hillclimb 用）
    oof_rank = np.stack([_rank01(o) for o in oofs], axis=1)
    test_rank = np.stack([_rank01(t) for t in tests], axis=1)
    # 生確率版（stacking 用）
    oof_raw = np.stack(oofs, axis=1)
    test_raw = np.stack(tests, axis=1)

    results = {}

    if len(names) == 1:
        print("\nモデルが1つのみ。そのまま使用します。")
        final_test = tests[0]
        best_method, best_auc = "single", roc_auc_score(y, oofs[0])
        results[best_method] = best_auc
        method_tests = {"single": tests[0]}
    else:
        # 1. weighted
        w, auc_w = optimize_weights(oof_rank, y)
        test_w = test_rank @ w
        results["weighted"] = auc_w
        print(f"\n[weighted ] OOF AUC={auc_w:.6f}  weights=" +
              ", ".join(f"{n}:{wi:.3f}" for n, wi in zip(names, w)))

        # 2. hill climbing
        w_h, auc_h = hill_climb(oof_rank, y)
        test_h = test_rank @ w_h
        results["hillclimb"] = auc_h
        print(f"[hillclimb] OOF AUC={auc_h:.6f}  weights=" +
              ", ".join(f"{n}:{wi:.3f}" for n, wi in zip(names, w_h)))

        # 3. stacking
        test_s, auc_s = stacking(oof_raw, test_raw, y)
        results["stacking"] = auc_s
        print(f"[stacking ] OOF AUC={auc_s:.6f}  (meta=LogisticRegression)")

        method_tests = {"weighted": test_w, "hillclimb": test_h, "stacking": test_s}
        best_method = max(results, key=results.get)
        best_auc = results[best_method]
        final_test = method_tests[best_method]

    # 各手法のsubmissionを保存
    for method, t in method_tests.items():
        pd.DataFrame({"SK_ID_CURR": test_ids, "TARGET": t}).to_csv(
            config.SUB_DIR / f"submission_{method}.csv", index=False)

    # 最良手法を最終submissionに
    pd.DataFrame({"SK_ID_CURR": test_ids, "TARGET": final_test}).to_csv(
        config.SUB_DIR / "submission_ensemble.csv", index=False)

    report = {
        "models": names,
        "single_oof_auc": {n: float(roc_auc_score(y, o)) for n, o in zip(names, oofs)},
        "method_oof_auc": {k: float(v) for k, v in results.items()},
        "best_method": best_method,
        "best_oof_auc": float(best_auc),
    }
    with open(config.ARTIFACT_DIR / "ensemble_report.json", "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"最良手法: {best_method}  (OOF AUC={best_auc:.6f})")
    print(f"submission: {config.SUB_DIR / 'submission_ensemble.csv'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
