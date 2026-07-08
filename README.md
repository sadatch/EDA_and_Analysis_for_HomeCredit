# Home Credit Default Risk — フル寝バッチ・パイプライン

既存notebook (`home-ccredit-re-03.ipynb`) の手法（期間別集約、EXT_SOURCE予測補完、LGBM+XGBアンサンブル）を
ベースに、**Home Credit 1位/2位解法**と**最近のテーブルコンペ上位の定石**（NVIDIA Kaggle Grandmasters Playbook）を
全部入りにして、`Ryzen 7 3800XT (16T) / RTX 3070Ti 8GB / 48GB RAM` で一晩〜半日回す前提のバッチに仕立てたもの。

途中から、独立に構築した**セカンドパイプライン（`hc_ensemble_optuna`系 / `hc_campaign.py`）**を
別トラックとして走らせ、両パイプラインのOOF予測をアンサンブル統合するに至った（詳細は
[セカンドパイプラインの章](#セカンドパイプライン-hc_campaign--独立特徴量セットとのアンサンブル統合)参照）。
実際の取り組みの流れ・全特徴量・遭遇したバグを含む完全な記録は
[`Home_Credit_取り組みまとめ.md`](./Home_Credit_取り組みまとめ.md) にまとめてある。

## このパイプラインで入っているもの（取り込んだトレンド）

| 区分 | 手法 | 出典 |
|---|---|---|
| 特徴量 | 期間別時系列集約（全期間/直近6M/1Y/3M）、bureau_balance DPDトレンド、credit_cardフル集約 | 既存notebook + HC 1位 |
| 特徴量 | **neighbors_target_mean**（EXT_SOURCE×CREDIT_ANNUITY_RATIO空間のK近傍TARGET平均, OOFリーク制御） | HC **1位**の目玉特徴 |
| 特徴量 | 算術交互作用（EXT×金額/日数の乗除）、CV安全な**OOF target encoding** | HC 1位 / 定石 |
| 特徴量 | **金融ドメイン指標**（全債務横断DTI/延滞トレンド/申込ベロシティ/利用率, DOM_*）＋重要度分析ツール | クレジットリスク実務 |
| 表現学習 | **Swap Noise DAE**（教師なし, Encoder各層concat埋め込み） | HC **2位** ikiri_DS |
| モデル | **LightGBM / XGBoost / CatBoost** 3本柱 + DAE特徴MLP | 定石（3 GBDT + NN） |
| 学習 | **GPU実行**（XGB=cuda, CatBoost=GPU, LightGBMはGPUビルドがあればGPU/無ければ自動CPU） | Playbook #fast-exp |
| 学習 | **シード平均**（fold分割+モデルseedを変えて平均）＋**全データ100%再学習**ブレンド | Playbook 7 |
| 探索 | **Optuna**（中断・再開可能なSQLite、lgb/xgb/cat個別） | 定石 |
| 診断 | **Adversarial Validation**（train/test分布シフト）、**Null Importance特徴量選択** | Playbook 1 |
| 半教師 | **擬似ラベル**（確信test行をsoft追加, k-fold安全） | Playbook 6 |
| 統合 | **hill climbing（Caruana選択）/ 2段スタッキング / 重み最適化** を比較して最良を採用 | Playbook 4,5 |
| 特徴量 | **時系列の傾き(TREND_SLOPE)・加重移動平均(WMA)**（bureau_balance/POS/CC/installmentsの水準ではなく変化速度） | HC 1位(WMA) / AmEx上位解法(slope) |
| 特徴量 | **信用種類の多様性**（BUREAU_CREDIT_TYPE_NUNIQUE）、**自社申込ベロシティ**（PREV_APP_INTERVAL）、**書類提出数**（DOC_SUBMIT_COUNT）、**申込時刻の周期エンコーディング**（HOUR_SIN/COS） | 公開kernel定石の横展開 |
| 特徴量 | **IsolationForest異常度スコア**（k-meansクラスタ距離とは別の分割ベース教師なし異常検知） | Grandmaster定石（異常検知の多様化） |
| 特徴量 | **年利率**(Newton法IRR近似)・**EXT_SOURCE_3除算特徴**・**AGE_INT**・**previous_applicationの直近/最初N件スライス**・**installments期間細分化(60/90/180/1000d)+回次別集約** | HC 1位解法discussion（Bojan/Olivier/Ryan/Phil/Yang/Michael Jahrer） |
| 検証 | **Adversarial Fold Split**（testらしさスコアで層化したCV。CV/LB相関の改善） | HC 1位解法discussion |
| 特徴量 | **CNT_PAYMENT予測モデル**（LightGBM回帰、AMT_CREDIT/ANNUITY/GOODS_PRICEから分割回数を推定）で年利率の精度向上 | ギャップ分析 P1 |
| 特徴量 | **neighbors_target_mean の多様化**（k=100/1000/2000の複数解像度、EXT+金利/EXT+年齢雇用の別特徴空間、近傍EXT_SOURCE差分） | ギャップ分析 P2 |
| 特徴量 | **bureau_balance recency**（最後の延滞から何ヶ月経過したか, BB_MONTHS_SINCE_LAST_DPD） | ギャップ分析 P3 |
| 特徴量 | **installments深掘り**（指数減衰加重DPD合計、早期完済比率・日数、直近/全期間の延滞率「比」） | ギャップ分析 P4 |
| モデル | **LightGBM DART**（木のdropoutで正則化経路が異なる追加アンサンブルメンバー, lgbdart） | ギャップ分析 M2 |
| モデル | **GRU月次系列モデル**（POS_CASH_balanceの月次系列を直接読む, スコープ縮小版, gru） | ギャップ分析 M3 |
| 統合 | **rank-average blending**（等重み順位平均。weighted/hillclimbの重み最適化と対照的な選択肢） | ギャップ分析 M5 |
| 半教師 | **擬似ラベル閾値見直し**（PSEUDO_HIGH: 0.30→0.75、誤ラベル混入リスクを下げるためより確信度の高い行に限定） | ギャップ分析 M6 |
| 特徴量 | **ギャップ特徴（GAP_*/COMBO_*/FREQ_*, `HC_FE_GAP=0`で無効化）**: テーブル間整合性（収入キリ番/残債×照会ゼロ不整合/返済期間オーバーラップ/全債務annuity負担）、時間軸インターリーブ（同時進行ローン最大数/完済・謝絶recency/借入空白期間）、テーブル存在フラグ+行NaNパターン、行動の質（POS早期完済/CC最低額張り付き）、カテゴリ組合せ+OOF TE+frequency encoding | チャット提案（テーブル間の矛盾・整合性系ほか） |
| 特徴量 | **最終バッチ特徴（FIN_*/GRP2_*, `HC_FE_FINAL=0`で無効化）**: 公開kernel定番比率（GOODS/収入、車齢/年齢、連絡手段数、住所不一致数、社会的圏デフォルト率、建物情報の充実度）、bureau後段比率（債務/与信・延滞/債務・延長フラグ）、横断負担（新規annuity/過去実払い月額、新規与信/過去平均与信・bureau残債）、上位交互作用（EXT×金利、**実効期間vs予測CNT_PAYMENTの乖離=早期返済シグナル**、近傍リスク×EXT乖離）、追加グループ相対（OCCUPATION/REGION_RATING/年齢10歳刻み） | 公開kernel上位 + 1位解法writeup（Olivier: 期間乖離） |
| 探索 | **特徴量選択の高速A/B（fs_quick_ab.py）**: 軽量lgbで「全特徴 vs Null Importance drop適用」を実測比較し、run_pipeline.shが推奨(`fs_ab.json`)に従い `HC_APPLY_FS` を自動設定（`HC_AUTO_FS=0`で無効化） | チューニング施策（M1の自動化） |
| 統合 | **stacking_logit**（logit変換+強L2のメタLR、生確率stackingの過学習対策）と **top3rankavg**（上位3本のみの順位平均）をアンサンブル比較に追加 | AmEx上位定石 / M4修理 |

## ファイル構成

```
config.py                設定一元管理（環境変数で全部上書き可。GPU/シード数/Optuna試行数など）
utils.py                 メモリ削減・集約ヘルパ・GPUデバイス設定・fast_auc・seed固定
make_synthetic_data.py   スモークテスト用の合成データ生成（本番データ不要）

feature_engineering.py   テーブル集約＋ドメイン特徴＋EXT_SOURCE補完
oof_features.py          1位の近傍TARGET平均 / OOF target encoding / 算術交互作用
domain_features.py       金融ドメイン特徴(DTI/延滞トレンド/申込ベロシティ/利用率 等, DOM_*)
trend_velocity_features.py 時系列傾き(TREND_SLOPE)/WMA・信用種類多様性・申込ベロシティ・書類数・周期特徴・IsolationForest異常度
top_solution_features.py 1位解法discussion由来（年利率/EXT3除算/AGE_INT/PREVスライス/INS期間細分化・回次別集約）
gap_features.py          ギャップ特徴（整合性/インターリーブ/存在フラグ/行動の質/カテゴリ組合せ, GAP_*/COMBO_*/FREQ_*）
final_features.py        最終バッチ特徴（定番比率/bureau後段比率/横断負担/上位交互作用/追加グループ相対, FIN_*/GRP2_*）
fs_quick_ab.py           特徴量選択適用の高速A/B（軽量lgbで実測→fs_ab.jsonに推奨を出力、run_pipelineが自動反映）
feature_importance.py    LightGBM重要度をカテゴリ別に集計（ドメイン特徴の効き目検証）
dae_model.py             Swap Noise DAE本体（PyTorch）
dae_features.py          DAE学習 & 埋め込み抽出

tune.py                  Optunaチューニング（resume可能, --model lgb/xgb/cat/all）
train_gbdt.py            LightGBM + XGBoost（GPU/seed平均/全データ再学習）
train_catboost.py        CatBoost（GPU/seed平均）
train_lgb_dart.py        LightGBM DART（木のdropoutで正則化経路が異なる追加メンバー lgbdart, M2）
train_gru_seq.py         GRU月次系列モデル（POS_CASH_balanceのみのスコープ縮小版, 追加メンバー gru, M3）
train_nn.py              DAE特徴MLP（アンサンブル多様性用, toshNN相当）
pseudo_label.py          擬似ラベルLightGBM（追加メンバー lgbpl）

adversarial_validation.py  train/test分布シフト診断（レポートのみ）
feature_selection.py       Null Importanceで効かない特徴を抽出（レポートのみ）
ensemble.py                hill climbing / stacking / weighted を比較し最終submission作成

run_pipeline.sh          全部を順に回す寝バッチ（再開可能・ログ付き）

# --- セカンドパイプライン（独立に構築、詳細は下の専用章を参照） ---
hc_ensemble_optuna.py    セカンドパイプライン本体（前処理・GBDT共通学習フレーム・Optuna）
hc_ensemble_optuna_v2.py 時間窓/last-k集約・トレンド・グループ統計・Null Importance選択
hc_ensemble_optuna_v3.py KNNターゲット特徴・行レベル補助モデル(prev/installments)・EMA/lag特徴
hc_ensemble_optuna_v4.py 行レベル補助モデル(bureau_balance)・モデル動物園・pseudo label・2層stack
hc_campaign.py           段階実行ランナー（features/tune/train/register/ensemble/status）
hc_v5_final_push.py      最終施策: 現申込金利推定特徴 + restackアンサンブル
hc_v6_lastday.py         最終施策: POS行レベル特徴・KNNバリエーション・CatBoostネイティブカテゴリ・10-fold低lr LGBM
export_old_oof_for_registration.py  メインパイプラインのOOF/predをセカンド側に登録するための変換
blend_final.py / blend_d4d6.py      提出ファイル同士のrank平均ブレンド（最後の保険）
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
python3 pseudo_label.py
python3 ensemble.py

# 以下はデフォルトでrun_pipeline.shから外している（下の「効果が薄いモデルを外した」節を参照）。
# 個別に効果を検証したい場合だけ単体実行する。
python3 train_nn.py        # MLP(DAE特徴)
python3 train_tabm.py      # TabM
python3 train_tabpfn.py    # TabPFN
python3 train_lgb_dart.py  # LightGBM DART（M2）
python3 train_gru_seq.py   # GRU月次系列（M3, torch必須）
```

## 想定スペックでのチューニングの勘所（12〜24h想定のデフォルト）

環境変数で実行規模を調整できる（カッコ内がデフォルト）。

| 変数 | 意味 | 12〜24h想定 | 一晩(6-10h)に縮める例 |
|---|---|---|---|
| `HC_N_SEEDS` | シード平均の本数 | 5 | 3 |
| `HC_OPTUNA_TRIALS` | Optuna試行数（lgb/xgb/cat各） | 60 | 25 |
| `HC_DO_TUNE` | Optunaを回すか | 1 | 0（固定パラメータで高速化） |
| `HC_DAE_HIDDEN` | DAE隠れ層 | **256**（※下記の注記参照） | 256 |
| `HC_GPU` | GBDTでGPUを使うか | 1 | 1 |
| `HC_FULL_REFIT` | 全データ100%再学習ブレンド | 1 | 0 |
| `HC_PSEUDO` | 擬似ラベルを回すか | 1 | 0 |

hc_campaign 系（hc_ensemble_optuna v1〜v4）専用の環境変数:

| 変数 | 意味 | デフォルト |
|---|---|---|
| `HC_DATA_DIR` | 生CSVの場所（hc_ensemble_optuna系のDATA_DIR上書き。スモークテスト用） | ./data/raw |
| `HC_NULLIMP_RUNS` | null importanceのシャッフル回数 | 30 |
| `HC_NULLIMP_SAMPLE` | null importance時の行サブサンプル率（actual/null両方に同一適用なので選択の公平性は保たれる。0.5でほぼ半分の時間） | 1.0 |

※ null importance は lgb.Dataset を1回だけ構築して `set_label()` でラベル差し替えする
高速化実装に変更済み（ビニング再計算を排除、X はfloat32 numpy化。従来実測3.9h→大幅短縮）。

例:
```bash
# 一晩で終わらせたい
HC_OPTUNA_TRIALS=25 HC_N_SEEDS=3 ./run_pipeline.sh full
# 数日かけて最大火力（ただしDAE_HIDDENは下記の注記を読んでから上げること）
HC_OPTUNA_TRIALS=120 HC_N_SEEDS=10 ./run_pipeline.sh full
```

> **⚠ DAE_HIDDENを上げる際の注意（実際に発生したVRAM事故）**
> 特徴量数を増やした状態で`HC_DAE_HIDDEN=1024`（3層concatで埋め込み3072次元、
> 総特徴量4788本）にしたところ、RTX 3070 Ti (8GB VRAM) 上でXGBoost GPUの
> `QuantileDMatrix`構築がVRAM容量を超えて17時間以上ハングした（nvidia-smi上は
> 動いているように見えるが進捗ゼロ）。デフォルトを`256`（768次元embedding）に
> 戻すことで解消済み。特徴量数を大きく増やす変更をした後にDAE_HIDDENを上げる場合は、
> 必ずnvidia-smiで最初の数分だけ監視し、進捗が止まっていたら即座にCPUへ切り替えること。

### 金融ドメイン特徴（DOM_*）の検証ワークフロー

`domain_features.py` がクレジットリスク実務の指標を7カテゴリで付与する（列名プレフィックスで識別）:

| プレフィックス | 内容 | 主な指標例 |
|---|---|---|
| `DOM_CAP_` | 返済能力 / DTI（全債務横断） | 他社残債/収入、総エクスポージャ/収入、可処分残/人 |
| `DOM_LEV_` | レバレッジ / 与信妥当性 | 延滞債務比率、頭金率、申請/承認ギャップ |
| `DOM_DLQ_` | 延滞の深刻度・直近トレンド | 最大DPD、直近1年の延滞/過小払いトレンド |
| `DOM_VEL_` | 申込ベロシティ | 信用照会の直近集中度、アクティブ口座比率 |
| `DOM_UTL_` | カード利用・キャッシング苦境 | 利用率、ATM現金引出依存、最低返済不足 |
| `DOM_PAY_` | 返済行動（installments） | 生涯支払充足率、最大遅延、延滞率 |
| `DOM_STB_` | 安定性・外部スコア交互作用 | EXT×地域、社会的圏のデフォルト率、電話番号変更/年齢 |

各特徴は「元の列が存在するときだけ」作るので、合成データでも実データでも落ちない（実データの方が
照会・社会的圏・キャッシング系の列がある分、生成数は増える）。

**「この指標は要るか？」をLightGBM視点で確認**するのが `feature_importance.py`:

```bash
python3 feature_importance.py            # カテゴリ別gainシェア + DOM_*のランキングを表示
python3 feature_importance.py --no-dae   # DAEを除いて手作り特徴に集中して見る
```

出力 `artifacts/feature_importance.csv`（特徴別 gain/split/順位）と
`artifacts/feature_importance_groups.csv`（カテゴリ別シェア）を見て、
gainが極端に低いDOM_列は外す、効くカテゴリは派生を増やす、という形で**一個ずつ検証**できる。
カテゴリ単位でON/OFFしてA/Bしたいときは `HC_FE_DOMAIN=0`（ドメイン特徴を丸ごと無効化）も使える。

同様に `trend_velocity_features.py`（時系列傾き/WMA・信用種類多様性・申込ベロシティ・書類数・
周期特徴・IsolationForest異常度）は `HC_FE_TREND=0`、`top_solution_features.py`
（年利率/EXT3除算/AGE_INT/PREVスライス/INS期間細分化）は `HC_FE_TOP_SOLUTION=0` で
それぞれ丸ごと無効化してA/Bできる。`feature_importance.py` のロールアップでは
`TREND_VEL(新規)` / `TOP_SOLUTION(新規)` カテゴリとして表示される（傾き・スライス系は
BUREAU_/POS_/CC_/INS_/PREVプレフィックスに紛れるため、既存カテゴリのgainが底上げされて
いないか確認したい場合は該当列名で個別に絞り込む）。

### Adversarial Fold Split（CV/LB相関の改善）

HC 1位解法discussionの指摘（train/testの分布差が大きい場合、CVをtest分布に近づけると
LBとの相関が上がる）に対応した、任意で使えるCV分割の改善策。

```bash
python3 adversarial_validation.py   # trainの「testらしさ」OOFスコアを artifacts/adversarial_oof_score.npy に保存
HC_ADV_FOLD=1 python3 train_gbdt.py       # このスコアで層化したfoldで学習（TARGET層化も維持）
HC_ADV_FOLD=1 python3 train_catboost.py
```

`utils.get_cv_splits()` が実体で、スコアファイルが無い/行数不一致なら自動的に通常の
TARGET層化StratifiedKFoldにフォールバックするため、`adversarial_validation.py` を
実行し忘れても安全に動く。doc記載の「testに近い上位20%を常にvalidationに固定」方式とは異なり、
（TARGET, testらしさ分位）の複合ラベルで層化するため全foldが均等にOOFカバレッジを持つ
（test寄りの行だけが検証から漏れ続ける、ということがない）。

### 特徴量選択を実際に反映する
`feature_selection.py` はデフォルトでは**レポートのみ**（まず中身を確認できるように安全側）。
反映するには `HC_APPLY_FS=1` を付けて学習する（`feature_selection.json` の drop 列を除外）。

```bash
python3 feature_selection.py          # まずレポート生成
HC_APPLY_FS=1 python3 train_gbdt.py   # 効かない特徴を落として学習
```

## 2026-07-02 効果が薄いモデルをデフォルトから除外

実データ完走結果（`artifacts/ensemble_report.json`）を見ると、mlp(DAE特徴)/tabm/tabpfnの
単体OOF AUCはそれぞれ0.7659/0.7735/0.7642で、lgb(0.7963)/xgb(0.7968)/cat(0.7917)より
大幅に劣る。ブレンド後（weighted 0.79694）は単体最強xgb(0.79677)から**+0.00017しか
改善しておらず**、この3モデルの学習コストは実質無駄になっている（アンサンブルの重み最適化が
自動的にこの3つをほぼ0重みに絞っているだけ）。

そのため `run_pipeline.sh` ではステップ7b/8/8b/8c/8d（DART/MLP/TabM/TabPFN/GRU）を
**デフォルトでコメントアウト**した。標準構成は「LightGBM + XGBoost + CatBoost + 擬似ラベル +
アンサンブル」のみ。DAE埋め込み自体は`feature_importance_groups.csv`でgain 27%を占め
GBDT側に効いているため`dae_features.py`は引き続き実行する（MLPヘッド単体モデルだけを外す）。

DART/GRUのような「効果未検証の新規モデル」は同じ轍を踏まないよう、まず単体で軽く実行して
`cv_scores.json`のoof_aucとlgb/xgbの差を見てから、効果が確認できた場合のみ
`run_pipeline.sh`のコメントを外してフル実行に組み込むこと。

## 2026-07-02 追加分（データに基づくギャップ分析への対応）

実データでの完走結果（weighted OOF AUC 0.79695）と `feature_importance.csv` / `adversarial_report.json`
の診断を踏まえたギャップ分析ドキュメントを反映。すべて既存のトグル機構（`config.py`のフラグ）を
踏襲しており、無効化しても他の処理には影響しない。

- **P1: CNT_PAYMENT予測モデル** (`top_solution_features.add_cnt_payment_prediction`)
  previous_application.csv を軽量に読み込み、AMT_CREDIT/AMT_ANNUITY/AMT_GOODS_PRICE（申込・過去ローン
  どちらにも同じ意味で存在）からLightGBM回帰でCNT_PAYMENTを学習し、現在の申込に適用（`PRED_CNT_PAYMENT`）。
  `add_yearly_interest_rate`はこれを`PREV_CNT_PAYMENT_mean`より優先して使う。
- **P2: neighbors_target_mean の多様化** (`oof_features.add_neighbor_diversity_features`)
  既存k=500に加えてk=100/1000/2000の複数解像度、(EXT+予測金利)・(EXT+DAYS_BIRTH+DAYS_EMPLOYED)の
  別特徴空間、近傍のEXT_SOURCE_MEANとの差分（自分が近傍より良いか悪いか）を追加。
  `HC_FE_NEIGHBORS_DIV=0` で丸ごと無効化可能。
- **P3: bureau_balance recency** (`trend_velocity_features.bureau_balance_recency`)
  最後の延滞から何ヶ月経過したか（`BB_MONTHS_SINCE_LAST_DPD`）。既存の6ヶ月窓水準特徴
  （adversarial診断でtop shift特徴に多数挙がっていたBUREAU_BB_6M_*系）を置き換えるのではなく補完する
  形。6ヶ月窓特徴自体の要否は `feature_selection.py` のnull importanceスコアで個別に判断すること
  （`HC_APPLY_FS=1`で反映）。
- **P4: installments深掘り** (`top_solution_features.installments_advanced_features`)
  指数減衰加重DPD合計（`INS_DPD_EWM_SUM`）、早期完済比率・平均日数（`INS_EARLY_PAY_RATIO`/
  `INS_EARLY_PAY_DAYS_mean`）、直近1年/全期間の延滞率「比」（`INS_TREND_LATE_RATIO`、既存の
  `INS_TREND_LATE`は差分）。
- **M2: LightGBM DART** (`train_lgb_dart.py`)
  boosting_type="dart"の独立トレーナー。dartは木のdropoutでearly stoppingの意味合いがgbdtほど
  明確でないため、固定ラウンド数（本番最大3000）で学習する。`artifacts/lgbdart_{oof,test}.npy`を
  ensemble.pyが自動で拾う。
- **M3: GRU月次系列モデル（スコープ縮小版）** (`train_gru_seq.py`)
  POS_CASH_balanceの(SK_ID_CURR, MONTHS_BALANCE)月次系列（直近48ヶ月, 不足月はゼロ埋め+マスク）を
  GRUに直接読ませる。bureau_balance/installments/credit_cardまで含めた完全な多系列融合は
  テーブルごとの粒度・欠損パターンの違いを揃える必要があり実装コストが大きいため、まずPOS_CASH単体で
  「系列を潰さない予測器」がアンサンブルに寄与するかを検証する位置づけ。寄与が確認できれば
  同じパターンで他テーブルを追加するのが安全な拡張順序（`train_gru_seq.py`冒頭のdocstring参照）。
  torch必須、`HC_USE_GRU=0`で無効化。
- **M5: rank-average blending** (`ensemble.py:rank_average_blend`)
  重み最適化を一切行わない等重み順位平均。weighted/hillclimbは少数モデル・小サンプルで重みが
  過学習しやすいため、比較対象として追加（`ensemble_report.json`のmethod_oof_aucで他手法と横並び比較できる）。
- **M6: 擬似ラベル閾値見直し** (`config.PSEUDO_HIGH`)
  0.30 → 0.75。旧値は「やや自信がある」程度まで含んでしまい誤ラベル混入リスクが高いとの指摘に対応。
  `HC_PSEUDO_HIGH`で個別調整可能。
- **M1: Null Importance適用のA/Bテスト**（コード変更なし、既存`HC_APPLY_FS`フラグの手順を明文化）
  ```bash
  python3 feature_selection.py             # レポート生成（drop候補確認）
  HC_APPLY_FS=1 python3 train_gbdt.py       # dropを反映して学習
  # cv_scores.jsonのoof_aucを HC_APPLY_FS=0 のときと比較
  ```

新規列は `feature_importance.py` のロールアップで確認できる（`Neighbors(1位+P2多様化)` /
`TOP_SOLUTION(新規)` カテゴリに集約される）。全て `HC_FE_TREND=1 HC_FE_TOP_SOLUTION=1
HC_FE_NEIGHBORS_DIV=1`（すべてデフォルトON）を前提にビルド・スモークテスト済み（合成データ
train (2000,752) / test (500,751) で完走、`PRED_CNT_PAYMENT`/`BB_MONTHS_SINCE_LAST_DPD`/
`INS_DPD_EWM_SUM`/`NEIGHBORS_TARGET_MEAN_MULTIRES_*`等の新規列生成を確認）。

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
- **検証状況**: 合成データでの動作確認に加え、本番のKaggle実データで完走・実際に複数回submitし、
  Private LB 0.79996（上記「最終結果」参照）まで確認済み。CVとPrivateの乖離は実測で
  ±0.002程度に収まっており、`cv_scores.json` / `ensemble_report.json` のOOF AUCは
  実際のLBと相関することを確認している。
- **bureau_balance**は実データで2,700万行規模。48GBあれば一括で載るが、メモリ不足が出たら
  `feature_engineering.py` の該当集約を `chunksize` 読みに変更するのが先に手を付ける箇所。

## セカンドパイプライン (hc_campaign) — 独立特徴量セットとのアンサンブル統合

上記のメインパイプラインとは別に、独立した特徴量セット・モデル構成を持つ「セカンドパイプライン」
（`hc_ensemble_optuna.py` + `_v2/_v3/_v4` + `hc_campaign.py`）を並行して構築し、
最終的に**両パイプラインのOOF予測を混ぜる**ことで単独では届かなかったスコアを達成した。
2つの独立した特徴量セットのOOFを混ぜること自体が強い多様性（アンサンブル効果）の源になる、
という考え方（Kaggleの「チームマージ」に近い発想）。

### 使い方

```bash
# 1. 特徴量構築（時間窓集約・KNN特徴・行レベル補助モデル・Null Importance選択、数時間）
python3 hc_campaign.py features

# 2. ハイパーパラメータ探索（Optuna、35%サンプル・3-fold・50試行）
python3 hc_campaign.py tune

# 3. モデル学習（lgb_gbdt_full/top600/nometa, lgb_dart/goss/rf, xgb, cat, mlp）
python3 hc_campaign.py train all
python3 hc_campaign.py train pseudo   # 最良モデルをベースに擬似ラベル

# 4. メインパイプライン側のOOFを登録して多様性を追加
python3 export_old_oof_for_registration.py
python3 hc_campaign.py register old_lgb old_lgb_oof.csv old_lgb_pred.csv
python3 hc_campaign.py register old_xgb old_xgb_oof.csv old_xgb_pred.csv
python3 hc_campaign.py register old_cat old_cat_oof.csv old_cat_pred.csv

# 5. アンサンブル（hillclimb + 2層stacking、良い方を自動選択）
python3 hc_campaign.py ensemble my_tag

# 状況確認（OOFストアの一覧・experiments.csvの履歴）
python3 hc_campaign.py status
```

`hc_campaign.py train`は既存の`oof_store/{name}_oof.npy`があれば自動スキップするため、
PCがクラッシュしても再実行するだけで途中から再開できる。

### 最終追加施策（hc_v5 / hc_v6）

CVが頭打ちになった段階で追加した最後の施策。

```bash
# 現申込の金利推定特徴を追加（previous_applicationでCNT_PAYMENT回帰モデル学習→適用）
python3 hc_v5_final_push.py augment
rm oof_store/lgb_gbdt_full_* oof_store/xgb_*
python3 hc_campaign.py train lgb_gbdt_full
python3 hc_campaign.py train xgb

# POS_CASH行レベル特徴・KNNバリエーション（k=100/1000, 代替特徴空間k=500）
python3 hc_v6_lastday.py posrow
python3 hc_v6_lastday.py knnv

# CatBoostネイティブカテゴリモデル（one-hotと異なる表現の多様性源）
python3 hc_v6_lastday.py catnative

# LightGBM 10-fold・低学習率(0.005)版
python3 hc_v6_lastday.py f10

# restackアンサンブル（OOFのrank + 強い生特徴量を混ぜたL2メタモデル）
python3 hc_v5_final_push.py ensemble2
```

### 両パイプラインの整合性

`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`をメイン・セカンド両方で
統一しているため、OOFを`register`で相互に取り込んでもfold境界のズレによるリークが起きない。

## 最終結果

`experiments.csv`に記録した実際のKaggle提出結果（late submission）の推移:

| 施策 | CV AUC | Public | Private |
|---|---|---|---|
| メインパイプライン単独（stacking_logit） | 0.79812 | — | — |
| セカンドパイプライン単独（hillclimb） | 0.79942 | 0.79973 | 0.79634 |
| 両パイプラインのOOFを統合（13モデルstack） | 0.80203 | 0.80065 | 0.79964 |
| + NN/TabPFN系の多様性追加 | 0.80217 | — | — |
| + Optunaチューニング + 全部入りstack | 0.80211 | 0.79989 | **0.79996** |

Home Creditコンペは Public LB が Private LB より低く出る既知の傾向があり、CVとPrivateが
ほぼ一致していることが「CVが信頼できている（過学習していない）」健全さのサイン。
特徴量・バグ修正・デバッグの詳細な経緯は[`Home_Credit_取り組みまとめ.md`](./Home_Credit_取り組みまとめ.md)を参照。

## 参考にした解法・記事

- Home Credit Default Risk **1位** 解法（neighbors_target_mean、算術特徴、期間別集約、weighted moving average）
  https://www.kaggle.com/c/home-credit-default-risk/discussion/64821
- Home Credit Default Risk **2位** ikiri_DS（Swap Noise DAE）
  https://github.com/KazukiOnodera/Home-Credit-Default-Risk / https://speakerdeck.com/hoxomaxwell/home-credit-default-risk-2nd-place-solutions
- NVIDIA Kaggle Grandmasters Playbook: 7 Battle-Tested Techniques（adversarial validation / 多様なベースライン /
  大量特徴生成 / hill climbing / stacking / pseudo-labeling / seed平均・全データ再学習）
  https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/
```
