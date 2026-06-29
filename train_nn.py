"""
DAE埋め込み + 数値特徴を入力とするMLP学習スクリプト。
2位チームの "DAE -> MLP" (toshNN) 相当。LightGBM/XGBoostとは誤り方の系統が異なるため、
アンサンブルに加えると多様性によるブレンド効果が期待できる。

出力:
  artifacts/mlp_oof.npy, artifacts/mlp_test.npy
"""
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import config
from utils import timer


class MLPHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(512, 128), dropout: float = 0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_mlp_input():
    """DAE埋め込み + EXT_SOURCE系の主要数値特徴を結合してMLP入力を作る（高次元すぎる生特徴は除外）。"""
    train_df = pd.read_parquet(config.PROC_DIR / "train_features.parquet")
    test_df = pd.read_parquet(config.PROC_DIR / "test_features.parquet")
    dae_train = pd.read_parquet(config.PROC_DIR / "dae_train_embeddings.parquet")
    dae_test = pd.read_parquet(config.PROC_DIR / "dae_test_embeddings.parquet")

    core_num_cols = [c for c in train_df.columns if c.startswith("EXT_SOURCE")
                      or c in ("CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_TERM",
                                "DAYS_EMPLOYED_PERCENT", "INCOME_PER_PERSON", "AMT_CREDIT", "AMT_ANNUITY",
                                "AMT_INCOME_TOTAL", "DAYS_BIRTH", "DAYS_EMPLOYED")]
    core_num_cols = [c for c in core_num_cols if c in train_df.columns]

    train_core = train_df[["SK_ID_CURR"] + core_num_cols].copy()
    test_core = test_df[["SK_ID_CURR"] + core_num_cols].copy()
    for c in core_num_cols:
        med = train_core[c].median()
        train_core[c] = train_core[c].fillna(med)
        test_core[c] = test_core[c].fillna(med)

    train_merged = dae_train.merge(train_core, on="SK_ID_CURR", how="left")
    test_merged = dae_test.merge(test_core, on="SK_ID_CURR", how="left")

    y = train_df["TARGET"].values
    feat_cols = [c for c in train_merged.columns if c != "SK_ID_CURR"]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_merged[feat_cols].values.astype(np.float32))
    X_test = scaler.transform(test_merged[feat_cols].values.astype(np.float32))

    return X_train, X_test, y, test_df["SK_ID_CURR"].values


def train_one_fold(X_trn, y_trn, X_val, y_val, device, input_dim):
    model = MLPHead(input_dim, hidden_dims=config.MLP_HIDDEN_DIMS, dropout=config.MLP_DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.MLP_LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    loss_fn = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_trn, dtype=torch.float32), torch.tensor(y_trn, dtype=torch.float32)),
        batch_size=config.MLP_BATCH_SIZE, shuffle=True
    )
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)

    best_auc = -1.0
    best_state = None
    patience_counter = 0

    for epoch in range(config.MLP_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = torch.sigmoid(model(X_val_t)).cpu().numpy()
        val_auc = roc_auc_score(y_val, val_pred)
        scheduler.step(val_auc)

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.MLP_EARLY_STOP_PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, best_auc


def main():
    device = config.DEVICE
    print(f"MLP device: {device}")

    with timer("MLP入力データ構築"):
        X_train, X_test, y, test_ids = build_mlp_input()
        print(f"  入力次元: {X_train.shape[1]}")

    folds = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    fold_scores = []

    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)

    with timer("MLP 5-Fold CV学習"):
        for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y)):
            model, val_auc = train_one_fold(
                X_train[trn_idx], y[trn_idx], X_train[val_idx], y[val_idx], device, X_train.shape[1]
            )
            model.eval()
            with torch.no_grad():
                oof_preds[val_idx] = torch.sigmoid(
                    model(torch.tensor(X_train[val_idx], dtype=torch.float32, device=device))
                ).cpu().numpy()
                test_preds += torch.sigmoid(model(X_test_t)).cpu().numpy() / folds.n_splits
            fold_scores.append(val_auc)
            print(f"  [MLP ] fold {fold_ + 1}: AUC={val_auc:.6f}")

    overall_auc = roc_auc_score(y, oof_preds)
    print(f"  [MLP ] overall OOF AUC: {overall_auc:.6f}")

    np.save(config.ARTIFACT_DIR / "mlp_oof.npy", oof_preds)
    np.save(config.ARTIFACT_DIR / "mlp_test.npy", test_preds)

    scores_path = config.ARTIFACT_DIR / "cv_scores.json"
    scores = json.load(open(scores_path)) if scores_path.exists() else {}
    scores["mlp_dae"] = {"oof_auc": overall_auc, "fold_scores": fold_scores}
    with open(scores_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"MLP(DAE特徴)学習完了: OOF AUC = {overall_auc:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
