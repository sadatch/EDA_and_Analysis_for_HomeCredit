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

CANDIDATES = ["lgb", "xgb", "cat", "mlp", "tabm", "tabpfn", "lgbpl", "lgbdart", "gru"]


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


# ---------- 3. rank-average blending (M5) ----------
def rank_average_blend(oof_rank_matrix, test_rank_matrix, y):
    """
    等重み・順位平均ブレンド。weighted/hillclimbが「AUCを最大化する重み」を探すのに対し、
    こちらは重み最適化を一切せず単純平均するため、少数モデル・小サンプルでの重み過学習に
    強く、多様性の高いモデル集合ではシンプルに効くことが多い（1位解法discussion M5由来）。
    """
    oof_blend = oof_rank_matrix.mean(axis=1)
    test_blend = test_rank_matrix.mean(axis=1)
    auc = roc_auc_score(y, oof_blend)
    return test_blend, auc


# ---------- 4. stacking (2-level meta model) ----------
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


# ---------- 5. stacking on logits (M4修理版) ----------
def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def stacking_logit(oof_matrix, test_matrix, y):
    """
    生確率ではなくlogit空間でメタLRを学習する（AmEx等の上位解法定石）。
    確率の端(0/1付近)の情報が線形空間に引き延ばされるためLRとの相性が良く、
    C=0.1の強めのL2でメタ過学習（前回stackingがweightedに大敗した原因の候補）を抑える。
    """
    Xl = _logit(oof_matrix)
    Tl = _logit(test_matrix)
    folds = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
    meta_oof = np.zeros(len(y))
    test_fold = np.zeros(test_matrix.shape[0])
    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xl)
    Ts = scaler.transform(Tl)
    for trn, val in folds.split(Xs, y):
        lr = LogisticRegression(max_iter=1000, C=0.1)
        lr.fit(Xs[trn], y[trn])
        meta_oof[val] = lr.predict_proba(Xs[val])[:, 1]
        test_fold += lr.predict_proba(Ts)[:, 1] / folds.n_splits
    auc = roc_auc_score(y, meta_oof)
    return test_fold, auc


# ---------- 6. top-k rank average ----------
def topk_rank_average(oof_rank_matrix, test_rank_matrix, y, names, k=3):
    """
    単体OOF AUC上位k本だけの等重み順位平均。弱いメンバー（tabpfn等）を除外した
    rankavgで、全員平均が弱メンバーに引きずられるケースの保険。
    """
    aucs = [roc_auc_score(y, oof_rank_matrix[:, i]) for i in range(oof_rank_matrix.shape[1])]
    top_idx = np.argsort(aucs)[::-1][:min(k, len(aucs))]
    oof_blend = oof_rank_matrix[:, top_idx].mean(axis=1)
    test_blend = test_rank_matrix[:, top_idx].mean(axis=1)
    auc = roc_auc_score(y, oof_blend)
    used = [names[i] for i in top_idx]
    return test_blend, auc, used


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

        # 3. rank-average blend (等重み、重み最適化なし)
        test_r, auc_r = rank_average_blend(oof_rank, test_rank, y)
        results["rankavg"] = auc_r
        print(f"[rankavg  ] OOF AUC={auc_r:.6f}  (等重み順位平均, 重み最適化なし)")

        # 4. stacking
        test_s, auc_s = stacking(oof_raw, test_raw, y)
        results["stacking"] = auc_s
        print(f"[stacking ] OOF AUC={auc_s:.6f}  (meta=LogisticRegression)")

        # 5. stacking on logits (M4修理版: logit変換 + 強めのL2)
        test_sl, auc_sl = stacking_logit(oof_raw, test_raw, y)
        results["stacking_logit"] = auc_sl
        print(f"[stack_lgt] OOF AUC={auc_sl:.6f}  (meta=LR on logits, C=0.1)")

        # 6. 上位3本のみのrank平均（弱メンバー除外の保険）
        test_t3, auc_t3, top3_used = topk_rank_average(oof_rank, test_rank, y, names, k=3)
        results["top3rankavg"] = auc_t3
        print(f"[top3rank ] OOF AUC={auc_t3:.6f}  (使用: {top3_used})")

        method_tests = {"weighted": test_w, "hillclimb": test_h, "rankavg": test_r,
                        "stacking": test_s, "stacking_logit": test_sl, "top3rankavg": test_t3}
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
