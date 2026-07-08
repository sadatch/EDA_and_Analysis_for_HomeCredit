"""
TabPFN v2 / TabICL（2025 tabular foundation model）をアンサンブル多様性要員として利用。

foundation modelは文脈長に上限がある（TabPFNは~10k, 8GB VRAMだとさらに制約）。
そこで「サブサンプルした文脈で複数回推論して平均する」bagging方式で 307k 行に対応する。
fold安全に:
  - train OOF : 各foldのvalを、他foldからサンプリングした文脈で推論
  - test      : 全trainからサンプリングした文脈で推論
特徴はyとの絶対相関が高い上位 TABPFN_MAXFEAT に絞る（基盤モデルは少特徴前提）。

fold: 既存モデルと同じ StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)。
出力: artifacts/tabpfn_oof.npy, artifacts/tabpfn_test.npy  （ensemble.py が自動収集）

tabpfn / tabicl が未導入なら自動skip（パイプラインは止めない）。
環境変数: HC_USE_TABPFN(0で無効) / HC_TABPFN_BAGS / HC_TABPFN_CTX / HC_TABPFN_MAXFEAT
"""
import os
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import config
from utils import timer, seed_everything


def _flag(name, default):
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


USE_TABPFN = _flag("HC_USE_TABPFN", True)
TABPFN_BAGS = int(os.environ.get("HC_TABPFN_BAGS", 2 if config.SMOKE else 8))
TABPFN_CTX = int(os.environ.get("HC_TABPFN_CTX", 1000 if config.SMOKE else 10000))
TABPFN_MAXFEAT = int(os.environ.get("HC_TABPFN_MAXFEAT", 100))


def _get_classifier():
    """TabPFN > TabICL の順で探し、(name, constructor) を返す。無ければ None。"""
    try:
        from tabpfn import TabPFNClassifier
        return ("tabpfn", TabPFNClassifier)
    except Exception:
        pass
    try:
        from tabicl import TabICLClassifier
        return ("tabicl", TabICLClassifier)
    except Exception:
        pass
    return None


def build_input():
    """数値特徴を結合し、欠損median補完 + 標準化。DAEは高次元なので使わない。"""
    train_df = pd.read_parquet(config.PROC_DIR / "train_features.parquet")
    test_df = pd.read_parquet(config.PROC_DIR / "test_features.parquet")
    try:
        with open(config.PROC_DIR / "categorical_features.json") as f:
            cat_cols = set(json.load(f))
    except Exception:
        cat_cols = set()
    feats = [c for c in train_df.columns
             if c not in ("SK_ID_CURR", "TARGET") and c not in cat_cols
             and c in test_df.columns and train_df[c].dtype.kind in "biufc"]
    y = train_df["TARGET"].values.astype(np.float32)
    Xtr = train_df[feats].astype(np.float32)
    Xte = test_df[feats].astype(np.float32)
    med = Xtr.median()
    Xtr = Xtr.fillna(med).replace([np.inf, -np.inf], 0.0).values
    Xte = Xte.fillna(med).replace([np.inf, -np.inf], 0.0).values
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr).astype(np.float32)
    Xte = sc.transform(Xte).astype(np.float32)
    return Xtr, Xte, y, test_df["SK_ID_CURR"].values, feats


def select_topk(Xtr, y, k):
    """yとの絶対相関が高い上位k特徴のインデックス。"""
    Xc = Xtr - Xtr.mean(axis=0)
    yc = y - y.mean()
    denom = np.sqrt((Xc ** 2).sum(axis=0)) * np.sqrt((yc ** 2).sum()) + 1e-9
    corr = np.abs((Xc * yc[:, None]).sum(axis=0) / denom)
    return np.argsort(-corr)[:min(k, Xtr.shape[1])]


def bagged_predict(make_clf, X_ctx, y_ctx, X_query, device, rng):
    """サブサンプル文脈でbagging推論し平均確率を返す。陽性を一定割合確保して文脈を作る。"""
    n_ctx = min(TABPFN_CTX, len(X_ctx))
    pos = np.where(y_ctx == 1)[0]
    neg = np.where(y_ctx == 0)[0]
    preds = np.zeros(len(X_query))
    for _ in range(TABPFN_BAGS):
        n_pos = min(len(pos), max(1, n_ctx // 4))
        n_neg = min(len(neg), n_ctx - n_pos)
        samp = np.concatenate([
            rng.choice(pos, size=n_pos, replace=len(pos) < n_pos),
            rng.choice(neg, size=n_neg, replace=len(neg) < n_neg),
        ])
        rng.shuffle(samp)
        try:
            clf = make_clf(device=device)
        except TypeError:
            clf = make_clf()
        clf.fit(X_ctx[samp], y_ctx[samp])
        preds += clf.predict_proba(X_query)[:, 1] / TABPFN_BAGS
    return preds


def main():
    if not USE_TABPFN:
        print("USE_TABPFN=0 のためskip")
        return
    found = _get_classifier()
    if found is None:
        print("tabpfn / tabicl が未導入のためskip（pip install tabpfn で有効化）")
        return
    name, make_clf = found
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"
    print(f"foundation model: {name}  device={device}")
    seed_everything(config.SEED)
    rng = np.random.RandomState(config.SEED)

    with timer("TabPFN入力構築"):
        Xtr, Xte, y, test_ids, feats = build_input()
        idx = select_topk(Xtr, y, TABPFN_MAXFEAT)
        Xtr, Xte = Xtr[:, idx], Xte[:, idx]
        print(f"  使用特徴 {Xtr.shape[1]} (相関上位) / bags={TABPFN_BAGS} ctx={TABPFN_CTX}")

    folds = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
    oof = np.zeros(len(Xtr))
    test = np.zeros(len(Xte))
    try:
        with timer("TabPFN OOF (fold安全)"):
            for fold_, (trn_idx, val_idx) in enumerate(folds.split(Xtr, y)):
                oof[val_idx] = bagged_predict(make_clf, Xtr[trn_idx], y[trn_idx], Xtr[val_idx], device, rng)
                print(f"  fold {fold_ + 1}: AUC={roc_auc_score(y[val_idx], oof[val_idx]):.6f}")
        with timer("TabPFN test推論"):
            test = bagged_predict(make_clf, Xtr, y, Xte, device, rng)
    except Exception as e:
        print(f"TabPFN推論に失敗したためskip: {e}")
        return

    overall = roc_auc_score(y, oof)
    np.save(config.ARTIFACT_DIR / "tabpfn_oof.npy", oof)
    np.save(config.ARTIFACT_DIR / "tabpfn_test.npy", test)
    sp = config.ARTIFACT_DIR / "cv_scores.json"
    scores = json.load(open(sp)) if sp.exists() else {}
    scores["tabpfn"] = {"oof_auc": overall, "backend": name}
    with open(sp, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)
    print("=" * 60)
    print(f"TabPFN({name})学習完了: OOF AUC = {overall:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
