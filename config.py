"""
共通設定。パス・シード・各モデルのハイパラ・デバイス方針を一元管理する。
home server (WSL2 + RTX 3070Ti 8GB + Ryzen 7 3800XT 16T + 48GB RAM) でのバッチ実行を想定。

ほぼ全ての挙動は環境変数で上書きできる。代表的なもの:
  HC_RAW_DIR        生CSVの配置場所 (デフォルト ./data/raw)
  HC_PROC_DIR       前処理済みparquetの出力先
  HC_ARTIFACT_DIR   モデル・埋め込み・OOFの保存先
  HC_SUB_DIR        submission csvの出力先
  HC_DEVICE         "cuda" / "cpu"。DAE/MLP等のNN学習デバイス (デフォルト: GPUがあればcuda)
  HC_GPU            "1"=GBDTもGPUを使う / "0"=CPU強制 (デフォルト: GPUが見えれば1)
  HC_N_THREADS      CPU並列数 (デフォルト 16, Ryzen 7 3800XT想定)
  HC_N_SEEDS        シード平均の本数 (デフォルト 5)
  HC_OPTUNA_TRIALS  Optuna試行回数 (デフォルト 60)
  HC_SMOKE          "1"で合成データ用の超軽量設定 (epochs/trials/seedsを最小化)

寝バッチ(一晩〜丸一日)向けデフォルト:
  シード5本平均 / Optuna 60試行 / 2段スタッキング+hill climbing / 擬似ラベル1ラウンド
"""
import os
from pathlib import Path


# ===== 小さなヘルパ =====
def _env_flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# ===== GPU検出（torchが無い環境でも壊れないようにする） =====
# torchはDAE/MLPでしか使わないため、GBDTのみ回したい環境ではtorch未インストールでも動くようにする。
def _detect_cuda() -> bool:
    try:
        import torch  # noqa: WPS433
        return bool(torch.cuda.is_available())
    except Exception:
        # torchが無い場合はnvidia-smiの有無でGPUを推定（GBDT GPUの判断に使う）
        import shutil
        return shutil.which("nvidia-smi") is not None


HAS_CUDA = _detect_cuda()
SMOKE = _env_flag("HC_SMOKE", False)

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
N_THREADS = _env_int("HC_N_THREADS", 16)        # Ryzen 7 3800XT = 8C/16T

# シード平均（Grandmaster Playbook「7. extra training」: シードを変えて平均すると安定）
N_SEEDS = _env_int("HC_N_SEEDS", 1 if SMOKE else 5)
SEED_LIST = [SEED + 1000 * i for i in range(N_SEEDS)]

# ===== デバイス設定 =====
# NN(DAE/MLP)用のデバイス
_requested_device = os.environ.get("HC_DEVICE", "auto").lower()
if _requested_device == "auto":
    DEVICE = "cuda" if HAS_CUDA else "cpu"
elif _requested_device == "cuda" and not HAS_CUDA:
    print("警告: HC_DEVICE=cuda 指定だがCUDA未検出のため cpu にフォールバック")
    DEVICE = "cpu"
else:
    DEVICE = _requested_device
DAE_DEVICE = DEVICE  # 後方互換

# GBDTでGPUを使うか（ユーザ方針: GPU前提でフル最適化）。
# LightGBMのGPUはOpenCLビルドが必要なため、trainer側でtry/exceptしCPUへ安全にフォールバックする。
USE_GPU_GBDT = _env_flag("HC_GPU", HAS_CUDA)
# 個別デバイス文字列（trainerが参照）。環境変数で個別上書きも可能。
LGB_DEVICE = os.environ.get("HC_LGB_DEVICE", "gpu" if USE_GPU_GBDT else "cpu")   # "gpu"/"cpu"
XGB_DEVICE = os.environ.get("HC_XGB_DEVICE", "cuda" if USE_GPU_GBDT else "cpu")  # "cuda"/"cpu"
CAT_TASK_TYPE = os.environ.get("HC_CAT_TASK", "GPU" if USE_GPU_GBDT else "CPU")  # "GPU"/"CPU"

# ===== 特徴量エンジニアリングのトグル =====
FE_USE_NEIGHBORS = _env_flag("HC_FE_NEIGHBORS", True)   # 1位の目玉: neighbors_target_mean
FE_USE_TARGET_ENC = _env_flag("HC_FE_TARGET_ENC", True)  # CV安全なOOF target encoding
FE_USE_DOMAIN = _env_flag("HC_FE_DOMAIN", True)          # 金融ドメイン特徴(DOM_*)
FE_SELECTION_APPLY = _env_flag("HC_APPLY_FS", False)     # feature_selection.jsonのdropを学習に反映するか
NEIGHBORS_K = _env_int("HC_NEIGHBORS_K", 100 if SMOKE else 500)
# target encodingをかけるカテゴリ列（存在する列のみ使用）
TARGET_ENC_COLS = [
    "ORGANIZATION_TYPE", "OCCUPATION_TYPE", "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "CODE_GENDER",
    "NAME_HOUSING_TYPE", "NAME_CONTRACT_TYPE",
]
TARGET_ENC_SMOOTHING = 20.0

# ===== DAE設定 (ikiri_DS 2位解法を参考) =====
# 元実装は隠れ層4096×3層。VRAM 8GBなら 1024〜2048 が安全。
DAE_HIDDEN_DIM = _env_int("HC_DAE_HIDDEN", 256 if SMOKE else 1024)
DAE_N_LAYERS = 3
DAE_SWAP_RATE = _env_float("HC_DAE_SWAP_RATE", 0.15)
DAE_EPOCHS = _env_int("HC_DAE_EPOCHS", 2 if SMOKE else 80)
DAE_BATCH_SIZE = _env_int("HC_DAE_BATCH", 256 if SMOKE else 1024)
DAE_LR = 1e-3
DAE_WEIGHT_DECAY = 1e-6
DAE_EARLY_STOP_PATIENCE = 10
DAE_PLATEAU_PATIENCE = 5
DAE_VAL_RATIO = 0.05

# ===== LightGBM/XGBoost/CatBoost共通 =====
GBDT_EARLY_STOPPING_ROUNDS = 50 if not SMOKE else 10
GBDT_NUM_BOOST_ROUND = 200 if SMOKE else 8000   # 寝バッチはlr小さめ+多ラウンド+early stop前提
GBDT_LEARNING_RATE = 0.02

# ===== Optuna =====
OPTUNA_TRIALS = _env_int("HC_OPTUNA_TRIALS", 5 if SMOKE else 60)
OPTUNA_TIMEOUT = _env_int("HC_OPTUNA_TIMEOUT", 0)   # 秒。0で無制限（試行回数で制御）
OPTUNA_STORAGE = os.environ.get("HC_OPTUNA_STORAGE", f"sqlite:///{(ARTIFACT_DIR / 'optuna.db').as_posix()}")

# ===== MLP (DAE特徴用ヘッド, toshNN相当) =====
MLP_HIDDEN_DIMS = [512, 128]
MLP_DROPOUT = 0.3
MLP_EPOCHS = _env_int("HC_MLP_EPOCHS", 3 if SMOKE else 40)
MLP_BATCH_SIZE = 2048
MLP_LR = 1e-3
MLP_EARLY_STOP_PATIENCE = 8

# ===== 擬似ラベル (Grandmaster Playbook「6. pseudo-labeling」) =====
PSEUDO_ENABLE = _env_flag("HC_PSEUDO", True)
PSEUDO_ROUNDS = _env_int("HC_PSEUDO_ROUNDS", 1)
# testの予測確率がこのしきい値より両極端な行だけをsoftラベルとして学習に追加
PSEUDO_LOW = _env_float("HC_PSEUDO_LOW", 0.02)
PSEUDO_HIGH = _env_float("HC_PSEUDO_HIGH", 0.30)

# ===== 全データ再学習 (Grandmaster Playbook「7. extra training」) =====
# CV後、best_iterationの平均で全データ(100%)再学習し、その予測をtest予測にブレンドする。
FULL_REFIT = _env_flag("HC_FULL_REFIT", True)
FULL_REFIT_WEIGHT = _env_float("HC_FULL_REFIT_WEIGHT", 0.5)  # final = (1-w)*cv_test + w*full_test

# ===== アンサンブル =====
ENSEMBLE_HILLCLIMB_STEPS = _env_int("HC_HILLCLIMB_STEPS", 2000)


def describe() -> str:
    """現在の実行設定を1ブロックで返す（ログ冒頭に出すと再現性確認に便利）。"""
    lines = [
        "================ HC config ================",
        f"SMOKE={SMOKE}  HAS_CUDA={HAS_CUDA}",
        f"DEVICE(NN)={DEVICE}  USE_GPU_GBDT={USE_GPU_GBDT}",
        f"LGB_DEVICE={LGB_DEVICE}  XGB_DEVICE={XGB_DEVICE}  CAT_TASK_TYPE={CAT_TASK_TYPE}",
        f"N_THREADS={N_THREADS}  N_FOLDS={N_FOLDS}  SEED_LIST={SEED_LIST}",
        f"OPTUNA_TRIALS={OPTUNA_TRIALS}  NUM_BOOST_ROUND={GBDT_NUM_BOOST_ROUND}  LR={GBDT_LEARNING_RATE}",
        f"FE_USE_NEIGHBORS={FE_USE_NEIGHBORS}(K={NEIGHBORS_K})  FE_USE_TARGET_ENC={FE_USE_TARGET_ENC}",
        f"DAE_HIDDEN_DIM={DAE_HIDDEN_DIM}  DAE_EPOCHS={DAE_EPOCHS}  PSEUDO={PSEUDO_ENABLE}(rounds={PSEUDO_ROUNDS})",
        "===========================================",
    ]
    return "\n".join(lines)
