"""
共通設定。パス・シード・各モデルのハイパラを一元管理する。
home server (WSL2 + CUDA 12.6 + PyTorch) での実行を想定。

環境変数で上書き可能:
  HC_RAW_DIR       生CSVの配置場所 (デフォルト ./data/raw)
  HC_PROC_DIR      前処理済みparquetの出力先
  HC_ARTIFACT_DIR  モデル・埋め込みの保存先
  HC_SUB_DIR       submission csvの出力先
  HC_DEVICE        "cuda" or "cpu"。DAE/MLP両方のNN学習で共通利用 (デフォルト: cudaが使えればcuda)
  HC_DAE_HIDDEN    DAEの隠れ層サイズ (デフォルト 1024、VRAM 8GB級向け)
  HC_DAE_BATCH     DAEのバッチサイズ (デフォルト 1024)
"""
import os
from pathlib import Path

import torch

# ===== パス設定 =====
RAW_DIR = Path(os.environ.get("HC_RAW_DIR", "./data/raw"))
PROC_DIR = Path(os.environ.get("HC_PROC_DIR", "./data/processed"))
ARTIFACT_DIR = Path(os.environ.get("HC_ARTIFACT_DIR", "./artifacts"))
SUB_DIR = Path(os.environ.get("HC_SUB_DIR", "./submissions"))

for _d in [PROC_DIR, ARTIFACT_DIR, SUB_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# 生CSVファイル名（Kaggle配布のまま）
RAW_FILES = {
    "app_train": "application_train.csv",
    "app_test": "application_test.csv",
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",
    "previous": "previous_application.csv",
    "pos_cash": "POS_CASH_balance.csv",
    "installments": "installments_payments.csv",
    "credit_card": "credit_card_balance.csv",
}

SEED = 42
N_FOLDS = 5

# ===== デバイス設定（DAE学習・MLP学習で共通利用） =====
# HC_DEVICE未指定時はCUDAが使えればcuda、なければcpuに自動フォールバック。
# "cuda"を明示指定したがCUDAが使えない場合も警告の上cpuにフォールバックする。
_requested_device = os.environ.get("HC_DEVICE", "auto").lower()
if _requested_device == "auto":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
elif _requested_device == "cuda" and not torch.cuda.is_available():
    print("警告: HC_DEVICE=cuda が指定されましたがCUDAが利用できないため cpu にフォールバックします")
    DEVICE = "cpu"
else:
    DEVICE = _requested_device

# ===== DAE設定 (ikiri_DS 2位解法を参考) =====
# 元実装は隠れ層4096×3層、concat ~3300次元。
# VRAM 8GB級なら HIDDEN_DIM=1024 (concat ~3072) が安全。
# 24GB+ あるなら 4096 まで上げて元実装に近づけられる。
DAE_HIDDEN_DIM = int(os.environ.get("HC_DAE_HIDDEN", 1024))
DAE_N_LAYERS = 3
DAE_SWAP_RATE = 0.15
DAE_EPOCHS = 80
DAE_BATCH_SIZE = int(os.environ.get("HC_DAE_BATCH", 1024))
DAE_LR = 1e-3
DAE_WEIGHT_DECAY = 1e-6
DAE_EARLY_STOP_PATIENCE = 10      # epoch単位
DAE_PLATEAU_PATIENCE = 5          # ReduceLROnPlateau
DAE_VAL_RATIO = 0.05              # 再構成lossモニタ用のhold-out（ラベル不要）

# 後方互換のため残置（新規コードはDEVICEを参照すること）
DAE_DEVICE = DEVICE

# ===== LightGBM/XGBoost共通 =====
GBDT_EARLY_STOPPING_ROUNDS = 50
GBDT_NUM_BOOST_ROUND = 1500

# ===== MLP (DAE特徴用ヘッド, toshNN相当) =====
MLP_HIDDEN_DIMS = [512, 128]
MLP_DROPOUT = 0.3
MLP_EPOCHS = 40
MLP_BATCH_SIZE = 2048
MLP_LR = 1e-3
MLP_EARLY_STOP_PATIENCE = 8
