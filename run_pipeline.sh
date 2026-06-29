#!/bin/bash
# Home Credit Default Risk パイプライン 一括実行スクリプト。
# home server (WSL2 + CUDA 12.6) でのバッチ実行を想定。
# nohup ./run_pipeline.sh > logs/run_$(date +%Y%m%d_%H%M%S).log 2>&1 & で背景実行推奨。
#
# 使い方:
#   ./run_pipeline.sh smoke   # 合成データでロジック確認のみ（数分）
#   ./run_pipeline.sh full    # 本番データで全パイプライン実行（数時間〜）

set -e  # いずれかのステップが失敗したら即停止

MODE="${1:-full}"
mkdir -p logs

echo "=================================================="
echo "Home Credit Default Risk Pipeline (mode=$MODE)"
echo "開始時刻: $(date)"
echo "=================================================="

if [ "$MODE" = "smoke" ]; then
    echo "[0/5] スモークテスト用合成データ生成"
    python3 make_synthetic_data.py
fi

echo "[1/5] 特徴量エンジニアリング（テーブル集約、bureau_balance/credit_card/installments拡張）"
python3 feature_engineering.py

echo "[2/5] DAE学習 & 特徴抽出（Swap Noise, ikiri_DS 2位解法参考）"
python3 dae_features.py

echo "[3/5] LightGBM + XGBoost 学習"
python3 train_gbdt.py

echo "[4/5] MLP(DAE特徴) 学習（アンサンブル多様性のため）"
python3 train_nn.py

echo "[5/5] アンサンブル & submission作成"
python3 ensemble.py

echo "=================================================="
echo "完了時刻: $(date)"
echo "submission: ./submissions/submission_ensemble.csv"
echo "CVスコア: ./artifacts/cv_scores.json"
echo "ブレンド重み: ./artifacts/ensemble_weights.json"
echo "=================================================="
