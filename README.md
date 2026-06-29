# Home Credit Default Risk — フル寝バッチ・パイプライン

既存notebook (`home-ccredit-re-03.ipynb`) の手法（期間別集約、EXT_SOURCE予測補完、LGBM+XGBアンサンブル）を
ベースに、**Home Credit 1位/2位解法**と**最近のテーブルコンペ上位の定石**（NVIDIA Kaggle Grandmasters Playbook）を
全部入りにして、`Ryzen 7 3800XT (16T) / RTX 3070Ti 8GB / 48GB RAM` で一晩〜半日回す前提のバッチに仕立てたもの。

## このパイプラインで入っているもの（取り込んだトレンド）

| 区分 | 手法 | 出典 |
|---|---|---|
| 特徴量 | 期間別時系列集約（全期間/直近6M/1Y/3M）、bureau_balance DPDトレンド、credit_cardフル集約 | 既存notebook + HC 1位 |
| 特徴量 | **neighbors_target_mean**（EXT_SOURCE×CREDIT_ANNUITY_RATIO空間のK近傍TARGET平均, OOFリーク制御） | HC **1位**の目玉特徴 |
| 特徴量 | 算術交互作用（EXT×金額/日数の乗除）、CV安全な**OOF target encoding** | HC 1位 / 定石 |
| 表現学習 | **Swap Noise DAE**（教師なし, Encoder各層concat埋め込み） | HC **2位** ikiri_DS |
| モデル | **LightGBM / XGBoost / CatBoost** 3本柱 + DAE特徴MLP | 定石（3 GBDT + NN） |
| 学習 | **GPU実行**（XGB=cuda, CatBoost=GPU, LightGBMはGPUビルドがあればGPU/無ければ自動CPU） | Playbook #fast-exp |
| 学習 | **シード平均**（fold分割+モデルseedを変えて平均）＋**全データ100%再学習**ブレンド | Playbook 7 |
| 探索 | **Optuna**（中断・再開可能なSQLite、lgb/xgb/cat個別） | 定石 |
| 診断 | **Adversarial Validation**（train/test分布シフト）、**Null Importance特徴量選択** | Playbook 1 |
| 半教師 | **擬似ラベル**（確信test行をsoft追加, k-fold安全） | Playbook 6 |
| 統合 | **hill climbing（Caruana選択）/ 2段スタッキング / 重み最適化** を比較して最良を採用 | Playbook 4,5 |

## ファイル構成

```
config.py                設定一元管理（環境変数で全部上書き可。GPU/シード数/Optuna試行数など）
utils.py                 メモリ削減・集約ヘルパ・GPUデバイス設定・fast_auc・seed固定
make_synthetic_data.py   スモークテスト用の合成データ生成（本番データ不要）

feature_engineering.py   テーブル集約＋ドメイン特徴＋EXT_SOURCE補完
oof_features.py          1位の近傍TARGET平均 / OOF target encoding / 算術交互作用
dae_model.py             Swap Noise DAE本体（PyTorch）
dae_features.py          DAE学習 & 埋め込み抽出

tune.py                  Optunaチューニング（resume可能, --model lgb/xgb/cat/all）
train_gbdt.py            LightGBM + XGBoost（GPU/seed平均/全データ再学習）
train_catboost.py        CatBoost（GPU/seed平均）
train_nn.py              DAE特徴MLP（アンサンブル多様性用, toshNN相当）
pseudo_label.py          擬似ラベルLightGBM（追加メンバー lgbpl）

adversarial_validation.py  train/test分布シフト診断（レポートのみ）
feature_selection.py       Null Importanceで効かない特徴を抽出（レポートのみ）
ensemble.py                hill climbing / stacking / weighted を比較し最終submission作成

run_pipeline.sh          全部を順に回す寝バッチ（再開可能・ログ付き）
```

## 使い方

### 0. セットアップ

```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

GPUについて（requirements.txt にも記載）:
- **XGBoost / CatBoost** … pip版そのままで `device="cuda"` / `task_type="GPU"` が効く（追加ビルド不要）
- **LightGBM** … GPU(OpenCL)版は別ビルドが必要。無くても**自動でCPUにフォールバック**する
  （`pip install lightgbm --config-settings=cmake.define.USE_GPU=ON`）
- **torch** … https://pytorch.org の指示でCUDA版を入れる（DAE/MLPのGPU学習用）

### 1. データ配置

```bash
kaggle competitions download -c home-credit-default-risk -p data/raw
cd data/raw && unzip home-credit-default-risk.zip
```

パスを変えたい場合は `HC_RAW_DIR` で上書き。

### 2. スモークテスト（本番前に必ず一度）

合成データで全パイプラインの論理エラーを数分で検出。専用ディレクトリ（`*_smoke`）に隔離されるので本番を汚さない。

```bash
./run_pipeline.sh smoke
```

### 3. 本番・寝バッチ実行

```bash
nohup ./run_pipeline.sh full > logs/run_$(date +%Y%m%d_%H%M%S).log 2>&1 &
tail -f logs/run_*.log     # Tailscale経由で外からSSHして進捗確認
```

各ステップは出力ファイルがあれば**自動スキップ（再開可能）**。最初からやり直したいときは `HC_FORCE=1`。

### 4. 個別ステップ再実行

```bash
python3 feature_engineering.py
python3 tune.py --model all          # Optunaだけ追い足し（途中再開OK）
python3 train_gbdt.py --skip-xgb     # LightGBMだけ
python3 train_catboost.py
python3 train_nn.py
python3 pseudo_label.py
python3 ensemble.py
```

## 想定スペックでのチューニングの勘所（12〜24h想定のデフォルト）

環境変数で実行規模を調整できる（カッコ内がデフォルト）。

| 変数 | 意味 | 12〜24h想定 | 一晩(6-10h)に縮める例 |
|---|---|---|---|
| `HC_N_SEEDS` | シード平均の本数 | 5 | 3 |
| `HC_OPTUNA_TRIALS` | Optuna試行数（lgb/xgb/cat各） | 60 | 25 |
| `HC_DO_TUNE` | Optunaを回すか | 1 | 0（固定パラメータで高速化） |
| `HC_DAE_HIDDEN` | DAE隠れ層（VRAM 8GBなら1024〜2048） | 1024 | 1024 |
| `HC_GPU` | GBDTでGPUを使うか | 1 | 1 |
| `HC_FULL_REFIT` | 全データ100%再学習ブレンド | 1 | 0 |
| `HC_PSEUDO` | 擬似ラベルを回すか | 1 | 0 |

例:
```bash
# 一晩で終わらせたい
HC_OPTUNA_TRIALS=25 HC_N_SEEDS=3 ./run_pipeline.sh full
# 数日かけて最大火力
HC_OPTUNA_TRIALS=120 HC_N_SEEDS=10 HC_DAE_HIDDEN=2048 ./run_pipeline.sh full
# VRAMが厳しい時はDAEを下げる
HC_DAE_HIDDEN=512 HC_DAE_BATCH=512 ./run_pipeline.sh full
```

### 特徴量選択を実際に反映する
`feature_selection.py` はデフォルトでは**レポートのみ**（まず中身を確認できるように安全側）。
反映するには `HC_APPLY_FS=1` を付けて学習する（`feature_selection.json` の drop 列を除外）。

```bash
python3 feature_selection.py          # まずレポート生成
HC_APPLY_FS=1 python3 train_gbdt.py   # 効かない特徴を落として学習
```

## 出力物

```
submissions/submission_ensemble.csv            最終提出（最良手法を自動選択）
submissions/submission_{hillclimb,stacking,weighted}.csv  各手法
artifacts/cv_scores.json          各モデルのOOF AUC
artifacts/ensemble_report.json    採用手法・各手法AUC・ブレンド重み
artifacts/adversarial_report.json train/test分布シフト診断
artifacts/feature_selection.json  keep/dropリスト
artifacts/optuna.db               Optuna探索履歴（resume用）
```

## 設計上のポイント / 既知の制約

- **リーク制御**: 近傍TARGET平均とtarget encodingは train=OOF / test=全train で算出。
  これらは構造上trainとtestで分布が少し異なるため、`adversarial_validation.py` は
  これらOOF特徴を除外して「生の特徴」の分布シフトのみを測る（除外しないとAUCが常に1.0付近に張り付く）。
- **GPUフォールバック**: LightGBM GPUが使えない環境では1回だけ警告を出してCPUに切替えて継続する。
  CatBoost/XGBもGPU失敗時はCPUへフォールバック。どの環境でも止まらない。
- **欠損ライブラリ耐性**: catboost/xgboost/torch/optunaが未インストールでも、
  該当ステップだけスキップしてアンサンブルまで到達する（`run_pipeline.sh` のopt/must制御）。
- **検証状況**: 本コードは合成データでパイプライン全体（特徴量→各モデル→擬似ラベル→
  hill climbing/stacking→submission）の動作を確認済み。**本番のKaggle実データでの数値検証は未実施**なので、
  実行後に `cv_scores.json` / `ensemble_report.json` のOOF AUCが
  既存notebook単体（LGBM+XGB）を下回っていないか必ず確認すること。
- **bureau_balance**は実データで2,700万行規模。48GBあれば一括で載るが、メモリ不足が出たら
  `feature_engineering.py` の該当集約を `chunksize` 読みに変更するのが先に手を付ける箇所。

## 参考にした解法・記事

- Home Credit Default Risk **1位** 解法（neighbors_target_mean、算術特徴、期間別集約、weighted moving average）
  https://www.kaggle.com/c/home-credit-default-risk/discussion/64821
- Home Credit Default Risk **2位** ikiri_DS（Swap Noise DAE）
  https://github.com/KazukiOnodera/Home-Credit-Default-Risk / https://speakerdeck.com/hoxomaxwell/home-credit-default-risk-2nd-place-solutions
- NVIDIA Kaggle Grandmasters Playbook: 7 Battle-Tested Techniques（adversarial validation / 多様なベースライン /
  大量特徴生成 / hill climbing / stacking / pseudo-labeling / seed平均・全データ再学習）
  https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/
```
