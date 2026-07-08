#!/bin/bash
# =====================================================================
# Home Credit Default Risk フルパイプライン（寝バッチ用）
# 想定マシン: Ryzen 7 3800XT (16T) / RTX 3070Ti 8GB / 48GB RAM
#
# 使い方:
#   ./run_pipeline.sh smoke        # 合成データでロジック確認（数分, GPU不要）
#   ./run_pipeline.sh full         # 本番データでフル実行（数時間〜半日）
#   HC_DO_TUNE=0 ./run_pipeline.sh full   # Optunaチューニングを省略して高速化
#   HC_FORCE=1 ./run_pipeline.sh full     # キャッシュを無視して全ステップ再実行
#
# 推奨運用（外出先からTailscale経由で進捗確認）:
#   nohup ./run_pipeline.sh full > logs/run_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#   tail -f logs/run_*.log
#
# 各ステップは出力ファイルがあればスキップ（再開可能）。HC_FORCE=1で強制再実行。
# =====================================================================
set -uo pipefail

MODE="${1:-full}"
mkdir -p logs

# ---- smokeモードは専用ディレクトリに隔離（本番のdata/artifactsを汚さない） ----
if [ "$MODE" = "smoke" ]; then
    export HC_SMOKE=1
    export HC_RAW_DIR="${HC_RAW_DIR:-./data/raw_smoke}"
    export HC_PROC_DIR="${HC_PROC_DIR:-./data/proc_smoke}"
    export HC_ARTIFACT_DIR="${HC_ARTIFACT_DIR:-./artifacts_smoke}"
    export HC_SUB_DIR="${HC_SUB_DIR:-./sub_smoke}"
    export HC_DO_TUNE="${HC_DO_TUNE:-1}"   # smokeでも軽く回して動作確認
else
    export HC_DO_TUNE="${HC_DO_TUNE:-1}"
fi

PROC_DIR="${HC_PROC_DIR:-./data/processed}"
ART_DIR="${HC_ARTIFACT_DIR:-./artifacts}"
FORCE="${HC_FORCE:-0}"

echo "=================================================="
echo "Home Credit Pipeline (mode=$MODE)  開始: $(date)"
echo "PROC_DIR=$PROC_DIR  ART_DIR=$ART_DIR  DO_TUNE=$HC_DO_TUNE  FORCE=$FORCE"
echo "=================================================="

# must: 失敗したらパイプライン全体を停止（前提となる重要ステップ用）
must() {
    echo ">>> [$(date +%H:%M:%S)] $1"
    shift
    if ! "$@"; then echo "!!! 致命的エラー: 上記ステップが失敗しました。停止します。"; exit 1; fi
}
# opt: 失敗しても警告だけ出して継続（個別モデル等、欠けてもアンサンブルは進められる）
opt() {
    echo ">>> [$(date +%H:%M:%S)] $1"
    shift
    if ! "$@"; then echo "### 警告: 上記ステップは失敗しましたが継続します。"; fi
}
# 出力ファイルが既にあればスキップ（FORCE=1で無効化）
skip_if() {
    [ "$FORCE" = "0" ] && [ -e "$1" ]
}

# ----- 0. smoke: 合成データ生成 -----
if [ "$MODE" = "smoke" ]; then
    must "[0] 合成データ生成" python3 make_synthetic_data.py
fi

# ----- 1. 特徴量エンジニアリング -----
if skip_if "$PROC_DIR/train_features.parquet"; then
    echo ">>> [1] 特徴量: キャッシュ済みスキップ ($PROC_DIR/train_features.parquet)"
else
    must "[1] 特徴量エンジニアリング (集約+1位特徴+target enc)" python3 feature_engineering.py
fi

# ----- 2. Adversarial Validation（分布シフト診断, レポートのみ） -----
opt "[2] Adversarial Validation" python3 adversarial_validation.py

# ----- 3. Null Importance 特徴量選択（レポートのみ。反映は HC_APPLY_FS=1） -----
opt "[3] 特徴量選択 (null importance)" python3 feature_selection.py

# ----- 3b. 特徴量重要度の分析（ドメイン特徴の効き目をカテゴリ別に確認） -----
opt "[3b] 特徴量重要度の分析 (gain/グループ別ロールアップ)" python3 feature_importance.py

# ----- 3c. 特徴量選択の高速A/B（軽量lgbでdrop適用の是非をデータで判定） -----
# HC_AUTO_FS=1(デフォルト)なら、fs_ab.jsonの推奨に従い以降の学習にHC_APPLY_FSを自動設定。
# 手動で固定したい場合は HC_AUTO_FS=0 HC_APPLY_FS=0/1 を指定。
if [ "${HC_AUTO_FS:-1}" = "1" ] && [ -z "${HC_APPLY_FS:-}" ]; then
    opt "[3c] 特徴量選択A/B (fs_quick_ab)" python3 fs_quick_ab.py
    if [ -e "$ART_DIR/fs_ab.json" ]; then
        export HC_APPLY_FS=$(python3 -c "import json;print(1 if json.load(open('$ART_DIR/fs_ab.json')).get('recommend_apply_fs') else 0)")
        echo ">>> [3c] fs_ab.json の推奨により HC_APPLY_FS=$HC_APPLY_FS で以降を実行"
    fi
fi

# ----- 4. DAE 学習 & 埋め込み抽出 -----
if skip_if "$PROC_DIR/dae_train_embeddings.parquet"; then
    echo ">>> [4] DAE: キャッシュ済みスキップ"
else
    opt "[4] DAE学習 & 埋め込み抽出 (Swap Noise)" python3 dae_features.py
fi

# ----- 5. Optuna チューニング（resume可能） -----
if [ "$HC_DO_TUNE" = "1" ]; then
    opt "[5] Optuna チューニング (lgb/xgb/cat)" python3 tune.py --model all
else
    echo ">>> [5] Optunaチューニング: スキップ (HC_DO_TUNE=0)"
fi

# ----- 6. GBDT学習 (LightGBM + XGBoost, seed平均, 全データ再学習) -----
must "[6] LightGBM + XGBoost 学習" python3 train_gbdt.py

# ----- 7. CatBoost学習 -----
opt "[7] CatBoost 学習" python3 train_catboost.py

# ----- 7b〜8d: デフォルト無効化（2026-07-02時点の実データ結果で判断） -----
# artifacts/ensemble_report.json の実測: mlp=0.7659 / tabm=0.7735 / tabpfn=0.7642 は
# lgb=0.7963 / xgb=0.7968 / cat=0.7917 に対し大幅に劣り、ブレンド後(weighted 0.79694)は
# 単体最強xgb(0.79677)から+0.00017しか改善していない＝この3モデルの学習時間はほぼ無駄。
# DART/GRUも同種の「効果未検証のモデル多様化」なので、まずは単体で軽く試してから
# 効果が確認できた場合にのみ以下のコメントを外してフル実行に組み込むこと。
# opt "[7b] LightGBM DART 学習" python3 train_lgb_dart.py
# opt "[8] MLP(DAE特徴) 学習" python3 train_nn.py
# opt "[8b] TabM (efficient MLP ensemble)" python3 train_tabm.py
# opt "[8c] TabPFN/TabICL (foundation model)" python3 train_tabpfn.py
# opt "[8d] GRU(POS_CASH月次系列)" python3 train_gru_seq.py

# ----- 9. 擬似ラベル (LightGBM変種を追加) -----
opt "[9] 擬似ラベル LightGBM" python3 pseudo_label.py

# ----- 10. アンサンブル (hill climbing + stacking + weighted) -----
must "[10] アンサンブル & submission作成" python3 ensemble.py

echo "=================================================="
echo "完了: $(date)"
echo "submission : $HC_SUB_DIR/submission_ensemble.csv (未設定なら ./submissions/)"
echo "CVスコア   : $ART_DIR/cv_scores.json"
echo "アンサンブル: $ART_DIR/ensemble_report.json"
echo "分布診断   : $ART_DIR/adversarial_report.json"
echo "特徴量選択 : $ART_DIR/feature_selection.json"
echo "=================================================="
