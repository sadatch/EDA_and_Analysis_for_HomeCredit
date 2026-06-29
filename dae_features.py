"""
DAE学習 & 特徴抽出スクリプト。

1. train_features/test_features を読み込み、DAE用の完全数値行列を作る
   （メインテーブルのカテゴリ列はone-hot化、欠損はmedian埋め+欠損フラグ、RankGaussでスケーリング）
2. train+testを合わせた行列でswap noise DAEを教師なし学習
3. 学習済みEncoderの各層出力をconcatし、train/testそれぞれの埋め込みをparquetで保存

出力:
  data/processed/dae_train_embeddings.parquet
  data/processed/dae_test_embeddings.parquet
  artifacts/dae_model.pt
  artifacts/dae_scaler.pkl
"""
import json
import pickle
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import QuantileTransformer

import config
from utils import timer
from dae_model import StackedDAE, swap_noise


def build_dae_matrix(train_df: pd.DataFrame, test_df: pd.DataFrame, categorical_features: list):
    """
    DAE入力用に完全数値の行列を作る。
    - カテゴリ列: one-hot化（train/test共通のカテゴリ集合で）
    - 数値列: 欠損は列medianで埋め、別途欠損フラグ列を追加。RankGauss変換でスケーリング。
    """
    id_train = train_df["SK_ID_CURR"].values
    id_test = test_df["SK_ID_CURR"].values

    drop_cols = {"SK_ID_CURR", "TARGET"}
    feat_cols = [c for c in train_df.columns if c not in drop_cols]

    full = pd.concat([train_df[feat_cols], test_df[feat_cols]], axis=0, ignore_index=True)

    cat_cols = [c for c in categorical_features if c in full.columns]
    num_cols = [c for c in feat_cols if c not in cat_cols]

    # --- 数値列: 欠損フラグ + median埋め ---
    print(f"  数値列 {len(num_cols)}個 / カテゴリ列 {len(cat_cols)}個")
    missing_flag_data = {}
    for c in num_cols:
        if full[c].isnull().any():
            missing_flag_data[f"{c}_ISNA"] = full[c].isnull().astype(np.int8)
    missing_flags = pd.DataFrame(missing_flag_data, index=full.index)
    for c in num_cols:
        full[c] = full[c].astype(np.float32)
        med = full[c].median()
        full[c] = full[c].fillna(med if not np.isnan(med) else 0.0)

    # --- RankGauss変換（QuantileTransformer + normal output）。ニューラル系学習の安定化に有効 ---
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=min(1000, len(full)),
                              random_state=config.SEED, subsample=int(2e5))
    num_scaled = qt.fit_transform(full[num_cols].values).astype(np.float32)
    num_scaled_df = pd.DataFrame(num_scaled, columns=num_cols)

    # --- カテゴリ列: one-hot ---
    if cat_cols:
        cat_ohe = pd.get_dummies(full[cat_cols].astype("category"), dummy_na=True)
        cat_ohe = cat_ohe.astype(np.float32)
    else:
        cat_ohe = pd.DataFrame(index=full.index)

    full_matrix = pd.concat([num_scaled_df, cat_ohe, missing_flags.astype(np.float32)], axis=1)
    full_matrix = full_matrix.fillna(0.0)

    n_train = len(train_df)
    X_train = full_matrix.iloc[:n_train].values.astype(np.float32)
    X_test = full_matrix.iloc[n_train:].values.astype(np.float32)

    scaler_artifacts = {
        "quantile_transformer": qt,
        "num_cols": num_cols,
        "feature_names": list(full_matrix.columns),
    }
    return X_train, X_test, id_train, id_test, scaler_artifacts


def train_dae(X: np.ndarray, device: str):
    """train+test全体（ラベル不要）でDAEを学習。再構成lossをhold-outで監視してEarlyStopping。"""
    n = X.shape[0]
    rng = np.random.RandomState(config.SEED)
    idx = rng.permutation(n)
    n_val = max(int(n * config.DAE_VAL_RATIO), 1)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_train_t = torch.tensor(X[train_idx], dtype=torch.float32)
    X_val_t = torch.tensor(X[val_idx], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t), batch_size=config.DAE_BATCH_SIZE, shuffle=True,
                               drop_last=True, num_workers=0)
    val_loader = DataLoader(TensorDataset(X_val_t), batch_size=config.DAE_BATCH_SIZE, shuffle=False)

    model = StackedDAE(input_dim=X.shape[1], hidden_dim=config.DAE_HIDDEN_DIM,
                        n_layers=config.DAE_N_LAYERS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.DAE_LR, weight_decay=config.DAE_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=config.DAE_PLATEAU_PATIENCE
    )
    loss_fn = torch.nn.MSELoss()

    best_val = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(config.DAE_EPOCHS):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            noisy = swap_noise(batch, config.DAE_SWAP_RATE)
            recon, _ = model(noisy)
            loss = loss_fn(recon, batch)  # ノイズなし元データへの再構成を学習させる

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
        train_loss = total_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                recon, _ = model(batch)
                val_loss += loss_fn(recon, batch).item() * batch.size(0)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)
        elapsed = time.time() - t0
        print(f"  epoch {epoch + 1:3d}/{config.DAE_EPOCHS}  train_loss={train_loss:.5f}  "
              f"val_loss={val_loss:.5f}  ({elapsed:.1f}s)")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.DAE_EARLY_STOP_PATIENCE:
                print(f"  EarlyStopping (patience={config.DAE_EARLY_STOP_PATIENCE}) at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def extract_embeddings(model: StackedDAE, X: np.ndarray, device: str, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    outs = []
    for i in range(0, len(X), batch_size):
        batch = torch.tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
        feat = model.extract_features(batch)
        outs.append(feat.cpu().numpy())
    return np.concatenate(outs, axis=0)


def main():
    device = config.DEVICE
    print(f"DAE device: {device}")

    with timer("特徴量読み込み"):
        train_df = pd.read_parquet(config.PROC_DIR / "train_features.parquet")
        test_df = pd.read_parquet(config.PROC_DIR / "test_features.parquet")
        with open(config.PROC_DIR / "categorical_features.json") as f:
            categorical_features = json.load(f)

    with timer("DAE入力行列の構築（one-hot + RankGauss + 欠損フラグ）"):
        X_train, X_test, id_train, id_test, scaler_artifacts = build_dae_matrix(
            train_df, test_df, categorical_features
        )
        print(f"  DAE入力次元: {X_train.shape[1]}")

    X_all = np.concatenate([X_train, X_test], axis=0)

    with timer(f"DAE学習 (hidden_dim={config.DAE_HIDDEN_DIM} x {config.DAE_N_LAYERS}層, swap_rate={config.DAE_SWAP_RATE})"):
        model = train_dae(X_all, device)

    with timer("埋め込み抽出"):
        emb_train = extract_embeddings(model, X_train, device)
        emb_test = extract_embeddings(model, X_test, device)
        emb_cols = [f"DAE_{i}" for i in range(emb_train.shape[1])]
        print(f"  埋め込み次元（concat）: {emb_train.shape[1]}")

    with timer("保存"):
        emb_train_df = pd.DataFrame(emb_train, columns=emb_cols)
        emb_train_df.insert(0, "SK_ID_CURR", id_train)
        emb_test_df = pd.DataFrame(emb_test, columns=emb_cols)
        emb_test_df.insert(0, "SK_ID_CURR", id_test)

        emb_train_df.to_parquet(config.PROC_DIR / "dae_train_embeddings.parquet", index=False)
        emb_test_df.to_parquet(config.PROC_DIR / "dae_test_embeddings.parquet", index=False)

        torch.save(model.state_dict(), config.ARTIFACT_DIR / "dae_model.pt")
        with open(config.ARTIFACT_DIR / "dae_input_dim.json", "w") as f:
            json.dump({"input_dim": int(X_train.shape[1]),
                       "hidden_dim": config.DAE_HIDDEN_DIM,
                       "n_layers": config.DAE_N_LAYERS}, f)
        with open(config.ARTIFACT_DIR / "dae_scaler.pkl", "wb") as f:
            pickle.dump(scaler_artifacts, f)

    print("=" * 60)
    print("DAE特徴抽出完了")
    print(f"train embeddings: {emb_train_df.shape}  test embeddings: {emb_test_df.shape}")
    print("=" * 60)


if __name__ == "__main__":
    main()
