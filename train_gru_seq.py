"""
月次系列を直接読むGRUモデル（M3）— スコープ縮小版。

1位解法discussionのM3「POS/installments/credit_card/bureau_balanceの月次系列を集約特徴
(mean/max/slope等)に潰さず、系列のままRNN/Transformerに読ませてGBDTと系統の違う予測を作る」
という発想の第一歩として、POS_CASH_balance.csv（最もシンプルで欠損の少ない月次テーブル）
だけを対象にしたGRU分類器を実装する。

スコープ縮小の理由（M3タスクで「スコープ検討」が明記されていたため）:
  - bureau_balance/installments/credit_cardまで含めた完全な多系列融合モデルは、
    テーブルごとに異なる欠損パターン・サンプリング頻度・SK_ID_PREV粒度の扱いが必要で、
    実装・デバッグ・VRAM設計のコストが本セッションの他タスクと比べて著しく大きい。
  - まずPOS_CASH単体で「系列を潰さない予測器」がアンサンブルに寄与するかを安価に検証し、
    寄与が確認できれば installments/credit_card を同様のパターンで追加するのが安全な拡張順序。
  - 拡張時のTODO: 各テーブルを同じ(SK_ID_CURR, MONTHS_BALANCE)グリッドに正規化し、
    チャンネル方向にstackしてマルチチャンネル系列にする。bureau_balanceはSK_ID_BUREAU粒度
    なので事前に顧客単位へ集約(例: 月ごとにアクティブローン数で重み付き平均)する必要がある。

前処理:
  POS_CASH_balance.csv を (SK_ID_CURR, MONTHS_BALANCE) 単位に集約(複数ローンがある月は
  SK_DPDの最大値・CNT_INSTALMENT_FUTUREの合計)し、直近 config.GRU_MAX_LEN ヶ月分の
  固定長系列（不足月はゼロ埋め+マスク、それより古い月は切り捨て）にする。
  POS履歴が無い顧客は全ゼロ+全マスクの系列になる（GRUは自然に「情報無し」として扱う）。

出力:
  artifacts/gru_oof.npy, artifacts/gru_test.npy   (ensemble.pyが追加メンバーとして拾う)
  artifacts/cv_scores.json (追記, key="gru_pos_seq")
"""
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

import config
from utils import timer, get_cv_splits

N_CHANNELS = 3  # SK_DPD(正規化), CNT_INSTALMENT_FUTURE(正規化), マスク


def _build_sequences(pos: pd.DataFrame, id_list: np.ndarray, max_len: int):
    """
    (SK_ID_CURR, MONTHS_BALANCE)単位に集約後、id_listの順序で
    [n_ids, max_len, N_CHANNELS] のfloat32テンソルを構築する。
    MONTHS_BALANCEは0が最新、負値が過去。直近max_len ヶ月分だけを使う
    （-(max_len-1) 〜 0 の範囲。それより古い月は切り捨て）。
    """
    needed = {"SK_ID_CURR", "MONTHS_BALANCE", "SK_DPD"}
    if not needed.issubset(pos.columns):
        return None

    p = pos[pos["MONTHS_BALANCE"] >= -(max_len - 1)].copy()
    agg_rules = {"SK_DPD": "max"}
    if "CNT_INSTALMENT_FUTURE" in p.columns:
        agg_rules["CNT_INSTALMENT_FUTURE"] = "sum"
    monthly = p.groupby(["SK_ID_CURR", "MONTHS_BALANCE"]).agg(agg_rules).reset_index()

    # 正規化（外れ値に強いようclipしてからスケール）
    monthly["SK_DPD"] = monthly["SK_DPD"].clip(0, 365) / 365.0
    if "CNT_INSTALMENT_FUTURE" in monthly.columns:
        monthly["CNT_INSTALMENT_FUTURE"] = monthly["CNT_INSTALMENT_FUTURE"].clip(0, 200) / 200.0
    else:
        monthly["CNT_INSTALMENT_FUTURE"] = 0.0

    id_to_row = {sk: i for i, sk in enumerate(id_list)}
    n = len(id_list)
    seq = np.zeros((n, max_len, N_CHANNELS), dtype=np.float32)

    rows = monthly["SK_ID_CURR"].map(id_to_row)
    valid = rows.notna()
    if valid.any():
        r = rows[valid].astype(int).values
        # MONTHS_BALANCE(負値, 0が最新) -> 系列内インデックス(0が最古側, max_len-1が最新側)
        t = (monthly.loc[valid, "MONTHS_BALANCE"].values + (max_len - 1)).astype(int)
        t = np.clip(t, 0, max_len - 1)
        dpd_v = monthly.loc[valid, "SK_DPD"].values.astype(np.float32)
        cnt_v = monthly.loc[valid, "CNT_INSTALMENT_FUTURE"].values.astype(np.float32)
        seq[r, t, 0] = dpd_v
        seq[r, t, 1] = cnt_v
        seq[r, t, 2] = 1.0  # マスク: この月にデータがある
    return seq


class GRUClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(inplace=True),
            nn.Dropout(0.2), nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1]).squeeze(-1)


def _train_one_fold(X_trn, y_trn, X_val, y_val, device):
    model = GRUClassifier(N_CHANNELS, config.GRU_HIDDEN).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.GRU_LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    loss_fn = nn.BCEWithLogitsLoss()

    loader = DataLoader(
        TensorDataset(torch.tensor(X_trn, dtype=torch.float32), torch.tensor(y_trn, dtype=torch.float32)),
        batch_size=config.GRU_BATCH_SIZE, shuffle=True,
    )
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)

    best_auc, best_state, patience_ctr = -1.0, None, 0
    for epoch in range(config.GRU_EPOCHS):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
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
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= config.GRU_EARLY_STOP_PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, best_auc


def main():
    if not config.GRU_ENABLE:
        print("GRU月次系列モデルは無効化されています (HC_USE_GRU=0)")
        return
    pos_path = config.RAW_DIR / config.RAW_FILES["pos_cash"]
    if not pos_path.exists():
        print(f"  {pos_path} が見つからないためGRU学習をスキップ")
        return

    device = config.DEVICE
    print(f"GRU(POS_CASH系列)device: {device}")

    with timer("TARGET/ID読み込み + POS_CASH系列構築"):
        train_df = pd.read_parquet(config.PROC_DIR / "train_features.parquet", columns=["SK_ID_CURR", "TARGET"])
        test_df = pd.read_parquet(config.PROC_DIR / "test_features.parquet", columns=["SK_ID_CURR"])
        y = train_df["TARGET"].values.astype(np.float32)
        train_ids = train_df["SK_ID_CURR"].values
        test_ids = test_df["SK_ID_CURR"].values

        pos = pd.read_csv(pos_path, usecols=lambda c: c in
                          {"SK_ID_CURR", "MONTHS_BALANCE", "SK_DPD", "CNT_INSTALMENT_FUTURE"})
        X_train = _build_sequences(pos, train_ids, config.GRU_MAX_LEN)
        X_test = _build_sequences(pos, test_ids, config.GRU_MAX_LEN)
        del pos
        if X_train is None:
            print("  POS_CASH_balanceに必要な列が無いためGRU学習をスキップ")
            return
        print(f"  系列shape: train={X_train.shape} test={X_test.shape}")

    splits = get_cv_splits(pd.DataFrame(index=np.arange(len(train_ids))), y, config.SEED)
    oof_preds = np.zeros(len(train_ids))
    test_preds = np.zeros(len(test_ids))
    fold_scores = []
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)

    with timer(f"GRU {len(splits)}-Fold CV学習 (POS_CASH月次系列, M3スコープ縮小版)"):
        for fold_, (trn_idx, val_idx) in enumerate(splits):
            model, val_auc = _train_one_fold(
                X_train[trn_idx], y[trn_idx], X_train[val_idx], y[val_idx], device)
            model.eval()
            with torch.no_grad():
                oof_preds[val_idx] = torch.sigmoid(
                    model(torch.tensor(X_train[val_idx], dtype=torch.float32, device=device))
                ).cpu().numpy()
                test_preds += torch.sigmoid(model(X_test_t)).cpu().numpy() / len(splits)
            fold_scores.append(val_auc)
            print(f"  [GRU ] fold {fold_ + 1}: AUC={val_auc:.6f}")

    overall_auc = roc_auc_score(y, oof_preds)
    print(f"  [GRU ] overall OOF AUC: {overall_auc:.6f}")

    np.save(config.ARTIFACT_DIR / "gru_oof.npy", oof_preds)
    np.save(config.ARTIFACT_DIR / "gru_test.npy", test_preds)

    scores_path = config.ARTIFACT_DIR / "cv_scores.json"
    scores = json.load(open(scores_path)) if scores_path.exists() else {}
    scores["gru_pos_seq"] = {"oof_auc": float(overall_auc), "fold_scores": [float(s) for s in fold_scores]}
    with open(scores_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"GRU(POS_CASH月次系列)学習完了: OOF AUC = {overall_auc:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
