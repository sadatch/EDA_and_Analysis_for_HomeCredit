# Home Credit Default Risk — DAE強化パイプライン

既存notebook (`home-ccredit-re-03.ipynb`) で実装済みの1位チーム手法（期間別集約、
EXT_SOURCE予測補完、LGBM+XGBアンサンブル）をベースに、2位チーム(ikiri_DS)の
**Swap Noise DAE**を追加し、バッチ実行可能な形にスクリプト化したもの。

## 構成

```
config.py              パス・ハイパラ設定（環境変数で上書き可）
utils.py                メモリ削減・集約ヘルパ
feature_engineering.py  テーブル集約（bureau/previous/POS/installments/credit_card）
dae_model.py             PyTorch DAE本体（Swap Noise + 3層スタックEncoder）
dae_features.py          DAE学習 & 埋め込み抽出
train_gbdt.py            LightGBM + XGBoost 5-Fold CV学習
train_nn.py              DAE埋め込み + MLP学習（アンサンブル多様性用、toshNN相当）
ensemble.py              OOFベースの最適ブレンド + submission作成
make_synthetic_data.py   スモークテスト用の合成データ生成（本番データ不要）
run_pipeline.sh          一括実行スクリプト
```

## 既存notebookからの拡張点

| テーブル | 既存notebook | 本パイプラインでの追加 |
|---|---|---|
| bureau | 全期間+6ヶ月+1年集約 | bureau_balanceのDPDトレンドをマージ |
| previous_application | 全期間集約+金利逆算 | 直近1件のスナップショット特徴を追加 |
| POS_CASH | 全期間+3ヶ月集約 | （変更なし、既存実装を継承） |
| installments_payments | 直近1年のみ | 全期間集約を追加 |
| credit_card_balance | streak特徴のみ | フル集約（利用率・限度超過等）+ 直近6ヶ月を追加 |
| — | — | **DAE (Swap Noise) 埋め込みを新規追加** |
| — | — | **DAE特徴ベースのMLP（3本目のモデル）を新規追加** |

## 使い方

### 0. セットアップ

```bash
uv venv && source .venv/bin/activate   # あるいは普段のuv環境
uv pip install -r requirements.txt
```

GPU(CUDA 12.6)があれば`torch`はCUDA対応版を、`lightgbm`はGPU版（`--config-settings=cmake.define.USE_GPU=ON`等）に
差し替えると高速化できる。デフォルトはCPU LightGBM / GPU PyTorch(DAE,MLP)想定。

### 1. データ配置

Kaggle APIで取得した生CSVを `data/raw/` に配置（`application_train.csv`等、Kaggle配布のファイル名そのまま）。
パスを変えたい場合は環境変数 `HC_RAW_DIR` で上書き可能。

```bash
kaggle competitions download -c home-credit-default-risk -p data/raw
cd data/raw && unzip home-credit-default-risk.zip
```

### 2. スモークテスト（推奨：本番実行前に必ず一度）

実データなしで合成データを使い、全パイプラインのロジックエラーを数分で検出できる。

```bash
./run_pipeline.sh smoke
```

### 3. 本番実行

```bash
nohup ./run_pipeline.sh full > logs/run_$(date +%Y%m%d_%H%M%S).log 2>&1 &
tail -f logs/run_*.log
```

Tailscale経由で外出先からSSHして`tail -f`で進捗確認、という運用を想定。

### 4. 個別ステップだけ再実行したい場合

```bash
python3 feature_engineering.py     # 特徴量だけ作り直す
python3 dae_features.py            # DAEだけ再学習（特徴量はキャッシュ済みparquetを再利用）
python3 train_gbdt.py --skip-xgb   # LightGBMだけ
python3 train_nn.py
python3 ensemble.py
```

## チューニングの勘所

- デバイス指定：`HC_DEVICE=cuda` / `cpu` でDAE・MLP両方の学習デバイスを一括指定できる（未指定時はCUDAが使えれば自動でcuda）
  ```bash
  HC_DEVICE=cpu ./run_pipeline.sh full          # 強制CPU
  HC_DAE_HIDDEN=2048 ./run_pipeline.sh full      # 隠れ層を2048に（VRAM 8GBなら余裕あり）
  HC_DAE_HIDDEN=512 HC_DAE_BATCH=512 ./run_pipeline.sh full  # VRAMが厳しい場合に下げる
  ```
- `HC_DAE_HIDDEN`：VRAM 8GB級なら1024（デフォルト）〜2048、24GB+なら4096まで上げて2位解法に近づけられる
- `HC_DAE_BATCH`：DAEのバッチサイズ。デフォルト1024。CUDA OOM時はまずここを512に下げる
- `config.py`の`DAE_SWAP_RATE`：0.15が2位解法のデフォルト値。0.1〜0.2の範囲で試す価値あり
- `train_gbdt.py`内のLightGBM/XGBoostパラメータは固定値。元notebookのOptuna探索（25 trials）を
  この特徴量セットに対してかけ直すと数値が動く可能性が高い（DAE特徴追加で最適パラメータも変わるため）
- `ensemble.py`はOOF AUCに基づいてNelder-Meadで重みを最適化するため、手動の0.5:0.5よりは確実に改善するはず
- bureau_balanceは現状の合成データ生成でもカバーしているが、実データでは行数が非常に多い
  （2億行規模）ため、メモリに余裕がない場合は`feature_engineering.py`の該当部分をchunk処理に変更すること

## 既知の制約

- 本コードはこの場（Claude.aiのサンドボックス環境）では合成データでのみ動作確認済み。
  本番のKaggle実データでの数値的な検証はできていないため、実行後にCV AUCが
  既存notebook単体（LGBM+XGBアンサンブル）の結果を下回っていないか必ず確認すること
- bureau_balanceは2億行規模なので、メモリ不足が出た場合は`pd.read_csv(..., chunksize=...)`で
  チャンク読み込みに変更するのが先に手を付けるべき箇所
