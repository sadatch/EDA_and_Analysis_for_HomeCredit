"""
TabM-lite: 効率的なMLPアンサンブル（Gorishniy et al. 2025 "TabM" の発想を簡略実装）。

2025年のテーブル系トレンドのひとつ。1本のモジュール内に k 個の並列MLPメンバ
（重みは独立、計算はバッチ化テンソルで並列）を持ち、出力ロジットを平均する。
deep ensembleの利得を1回の学習で得られ、GBDT/DAE-MLPとは誤り方が異なるため
アンサンブル多様性に寄与する。

入力: train_features/test_features の数値特徴 + (あれば)DAE埋め込み を標準化。
fold: 既存モデルと同じ StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)。
出力: artifacts/tabm_oof.npy, artifacts/tabm_test.npy  （ensemble.py が自動収集）

torch未導入の環境では自動スキップ（パイプラインは止めない）。
環境変数: HC_TABM_K / HC_TABM_EPOCHS / HC_TABM_BATCH / HC_TABM_HIDDEN / HC_USE_TABM(0で無効)
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


USE_TABM = _flag("HC_USE_TABM", True)
TABM_K = int(os.environ.get("HC_TABM_K", 8))
TABM_HIDDEN = [int(x) for x in os.environ.get("HC_TABM_HIDDEN", "512,256").split(",")]
TABM_DROPOUT = float(os.environ.get("HC_TABM_DROPOUT", 0.2))
TABM_EPOCHS = int(os.environ.get("HC_TABM_EPOCHS", 3 if config.SMOKE else 50))
TABM_BATCH = int(os.environ.get("HC_TABM_BATCH", 1024))
TABM_LR = float(os.environ.get("HC_TABM_LR", 1e-3))
TABM_WD = float(os.environ.get("HC_TABM_WD", 1e-5))
TABM_PATIENCE = int(os.environ.get("HC_TABM_PATIENCE", 8))


def build_numeric_input():
    """数値特徴 + DAE埋め込み を結合し、欠損median補完 + 標準化した行列を返す。"""
    train_df = pd.read_parquet(config.PROC_DIR / "train_features.parquet")
    test_df = pd.read_parquet(config.PROC_DIR / "test_features.parquet")

    # DAE埋め込みがあれば結合（無くても可）
    dae_tr = config.PROC_DIR / "dae_train_embeddings.parquet"
    dae_te = config.PROC_DIR / "dae_test_embeddings.parquet"
    if dae_tr.exists() and dae_te.exists():
        train_df = train_df.merge(pd.read_parquet(dae_tr), on="SK_ID_CURR", how="left")
        test_df = test_df.merge(pd.read_parquet(dae_te), on="SK_ID_CURR", how="left")

    # カテゴリ列を除外して数値のみ
    try:
        with open(config.PROC_DIR / "categorical_features.json") as f:
            cat_cols = set(json.load(f))
    except Exception:
        cat_cols = set()
    feats = []
    for c in train_df.columns:
        if c in ("SK_ID_CURR", "TARGET") or c in cat_cols:
            continue
        if c not in test_df.columns:
            continue
        if train_df[c].dtype.kind in "biufc":
            feats.append(c)

    y = train_df["TARGET"].values.astype(np.float32)
    Xtr = train_df[feats].astype(np.float32)
    Xte = test_df[feats].astype(np.float32)
    med = Xtr.median()
    Xtr = Xtr.fillna(med).replace([np.inf, -np.inf], 0.0).values
    Xte = Xte.fillna(med).replace([np.inf, -np.inf], 0.0).values
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr).astype(np.float32)
    Xte = scaler.transform(Xte).astype(np.float32)
    return Xtr, Xte, y, test_df["SK_ID_CURR"].values, feats


def _build_model(input_dim):
    import torch
    import torch.nn as nn

    class EnsembleLinear(nn.Module):
        """k本のLinearを並列に持つ層。入力(k,batch,in)->出力(k,batch,out)。"""
        def __init__(self, k, in_dim, out_dim):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(k, in_dim, out_dim))
            self.bias = nn.Parameter(torch.zeros(k, 1, out_dim))
            for i in range(k):
                nn.init.kaiming_uniform_(self.weight[i], a=5 ** 0.5)

        def forward(self, x):
            return torch.einsum("kbi,kio->kbo", x, self.weight) + self.bias

    class TabMLite(nn.Module):
        def __init__(self, in_dim, k, hidden, dropout):
            super().__init__()
            self.k = k
            self.layers = nn.ModuleList()
            self.bns = nn.ModuleList()
            prev = in_dim
            for h in hidden:
                self.layers.append(EnsembleLinear(k, prev, h))
                self.bns.append(nn.BatchNorm1d(h))
                prev = h
            self.dropout = nn.Dropout(dropout)
            self.head = EnsembleLinear(k, prev, 1)

        def forward(self, x):
            h = x.unsqueeze(0).expand(self.k, -1, -1).contiguous()
            for layer, bn in zip(self.layers, self.bns):
                h = layer(h)
                kk, b, f = h.shape
                h = bn(h.reshape(kk * b, f)).reshape(kk, b, f)
                h = torch.relu(h)
                h = self.dropout(h)
            return self.head(h).squeeze(-1).mean(dim=0)   # メンバ平均ロジット

    return TabMLite(input_dim, TABM_K, TABM_HIDDEN, TABM_DROPOUT)


def train_one_fold(X_trn, y_trn, X_val, y_val, device):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    model = _build_model(X_trn.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=TABM_LR, weight_decay=TABM_WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(
        TensorDataset(torch.tensor(X_trn), torch.tensor(y_trn)),
        batch_size=TABM_BATCH, shuffle=True, drop_last=True)
    X_val_t = torch.tensor(X_val, device=device)

    best_auc, best_state, patience = -1.0, None, 0
    for _ in range(TABM_EPOCHS):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vp = torch.sigmoid(model(X_val_t)).cpu().numpy()
        auc = roc_auc_score(y_val, vp)
        sched.step(auc)
        if auc > best_auc:
            best_auc, best_state, patience = auc, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= TABM_PATIENCE:
                break
    model.load_state_dict(best_state)
    return model


def main():
    if not USE_TABM:
        print("USE_TABM=0 のためskip")
        return
    try:
        import torch
    except Exception:
        print("torch未導入のためTabMをskip（pip install torch で有効化）")
        return

    seed_everything(config.SEED)
    device = config.DEVICE
    print(f"TabM device: {device}  (k={TABM_K}, hidden={TABM_HIDDEN})")

    with timer("TabM入力構築"):
        Xtr, Xte, y, test_ids, feats = build_numeric_input()
        print(f"  入力次元: {Xtr.shape[1]}")

    folds = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
    oof = np.zeros(len(Xtr))
    test = np.zeros(len(Xte))
    fold_scores = []
    X_test_t = torch.tensor(Xte, device=device)

    with timer("TabM 5-Fold CV学習"):
        for fold_, (trn_idx, val_idx) in enumerate(folds.split(Xtr, y)):
            model = train_one_fold(Xtr[trn_idx], y[trn_idx], Xtr[val_idx], y[val_idx], device)
            model.eval()
            with torch.no_grad():
                oof[val_idx] = torch.sigmoid(
                    model(torch.tensor(Xtr[val_idx], device=device))).cpu().numpy()
                test += torch.sigmoid(model(X_test_t)).cpu().numpy() / config.N_FOLDS
            auc = roc_auc_score(y[val_idx], oof[val_idx])
            fold_scores.append(auc)
            print(f"  [TabM] fold {fold_ + 1}: AUC={auc:.6f}")
            if device == "cuda":
                torch.cuda.empty_cache()

    overall = roc_auc_score(y, oof)
    np.save(config.ARTIFACT_DIR / "tabm_oof.npy", oof)
    np.save(config.ARTIFACT_DIR / "tabm_test.npy", test)
    sp = config.ARTIFACT_DIR / "cv_scores.json"
    scores = json.load(open(sp)) if sp.exists() else {}
    scores["tabm"] = {"oof_auc": overall, "fold_scores": fold_scores}
    with open(sp, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)
    print("=" * 60)
    print(f"TabM学習完了: OOF AUC = {overall:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
