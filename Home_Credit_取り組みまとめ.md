# Home Credit Default Risk 取り組みまとめ

大学の授業コンペ（Kaggle "Home Credit Default Risk"、二値分類・評価指標AUC）に向けて行った特徴量エンジニアリング・モデリング・アンサンブル・デバッグの全記録。

---

## 1. コンペ概要とデータ

- **タスク**: 申込者がローンを返済できるか（`TARGET`: 0=正常、1=デフォルト）を予測する二値分類
- **評価指標**: ROC-AUC
- **データ**: `application_train/test`（メインテーブル）+ 関連6テーブル
  - `bureau.csv` / `bureau_balance.csv`（他の金融機関への信用情報）
  - `previous_application.csv`（Home Creditへの過去申込）
  - `POS_CASH_balance.csv`（POS/キャッシュローンの月次残高）
  - `installments_payments.csv`（分割返済の実績）
  - `credit_card_balance.csv`（クレジットカード月次残高）
- **検証方法**: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` に統一（後述する2本のパイプライン間でOOFを混ぜ合わせるための前提条件）

## 2. 全体アーキテクチャ：2本立てパイプライン

途中から「メインパイプライン」（12ファイル構成、自分たちで最初から構築）と、「セカンドパイプライン」（`hc_ensemble_optuna`系、別のAIエージェントが構築したものを引き継いでデバッグ・統合）の2系統を最終的にアンサンブルで統合する構成になった。

```
メインパイプライン                    セカンドパイプライン (hc_campaign)
feature_engineering.py 他11ファイル    hc_ensemble_optuna.py / v2 / v3 / v4
  → train_features.parquet             → cache/features_train.parquet
  → DAE埋め込み                        → Null Importance選択済み特徴(208〜220本)
  → lgb/xgb/cat/mlp/tabm/tabpfn/dart    → lgb系5variant/xgb/cat/mlp
  → ensemble.py (weighted/hillclimb/    → hc_campaign.py (hillclimb/stack)
     stacking)                         → hc_v5/v6 (restack, 追加特徴)
        └───────────── OOFを相互登録 (register) して統合 ─────────────┘
```

2つのパイプラインは**独立に構築された特徴量セット**を持つため、双方のOOF予測を混ぜること自体が強力な多様性（アンサンブルの精度向上）の源になった。

---

## 3. ベース集約特徴量（`feature_engineering.py`）

関連6テーブルを`SK_ID_CURR`単位に集約し、メインの`application`テーブルへ結合する土台部分。

| 対象テーブル | 集約内容 |
|---|---|
| `bureau_balance` + `bureau` | ステータスの数値化(`STATUS_NUM`)・延滞フラグ(`IS_DPD`)、信用情報の期間別（全期間/直近1年等）集約 |
| `previous_application` | 推定金利(`ESTIMATED_TOTAL_INTEREST`, `ESTIMATED_INTEREST_RATE`)、申請額と承認額の差・比(`APP_CREDIT_DIFF`, `APP_CREDIT_RATIO`) |
| `POS_CASH_balance` | 残債・延滞の期間別集約 |
| `installments_payments` | 支払不足額(`PAYMENT_DEFICIT`)、支払比率(`PAYMENT_RATIO`)、支払遅延日数(`PAYMENT_DELAY`)、遅延/不足フラグ(`IS_LATE`, `IS_UNDERPAID`)、直近1年 vs 全期間のトレンド差(`INS_TREND_LATE`, `INS_TREND_LATE_RATIO`, `INS_TREND_DEFICIT`) |
| `credit_card_balance` | 利用率(`UTILIZATION`)、限度超過フラグ(`IS_OVER_LIMIT`)、最低支払不足額(`MIN_PAYMENT_DEFICIT`)、残高増減のストリーク検知(`BALANCE_DIFF`, `IS_INCREASED`, `STREAK_GROUP`) |

メインテーブル側の基本派生特徴：

- `DAYS_EMPLOYED`の異常値(365243)をNaN化
- `CREDIT_INCOME_RATIO` / `ANNUITY_INCOME_RATIO` / `CREDIT_TERM` / `CREDIT_ANNUITY_RATIO`
- `DAYS_EMPLOYED_PERCENT`（勤続日数/年齢）
- `INCOME_PER_PERSON`（世帯人数あたり収入）
- `EXT_SOURCE_MEAN` / `_MAX` / `_MIN` / `_PROD` / `_STD` / `_NAN_COUNT`（外部信用スコア3種の集約）

## 4. 金融ドメイン特徴量（`domain_features.py`）

クレジットリスク実務の指標をHome Creditデータに落とし込んだもの。7カテゴリ・プレフィックスで管理し、あとで重要度を分解して「どのカテゴリが効くか」を検証できるようにしてある。

| プレフィックス | カテゴリ | 内容 |
|---|---|---|
| `DOM_CAP_*` | 返済能力/DTI | `DOM_CAP_BUREAU_DEBT_TO_INCOME`, `DOM_CAP_BUREAU_DEBT_TO_CREDIT`, `DOM_CAP_TOTAL_EXPOSURE_TO_INCOME`, `DOM_CAP_TOTAL_ANNUITY_TO_INCOME`, `DOM_CAP_RESIDUAL_INCOME`（収入-年間返済額）, `DOM_CAP_RESIDUAL_PER_PERSON` |
| `DOM_LEV_*` | レバレッジ/与信妥当性 | `DOM_LEV_OVERDUE_DEBT_RATIO`, `DOM_LEV_DOWNPAY`（頭金相当）, `DOM_LEV_PREV_APP_CREDIT_RATIO` |
| `DOM_DLQ_*` | 延滞の深刻度・トレンド | `DOM_DLQ_MAX_DPD`, `DOM_DLQ_HAS_BUREAU_OVERDUE`, `DOM_DLQ_LATE_TREND_1YR`, `DOM_DLQ_DEFICIT_TREND_1YR` |
| `DOM_VEL_*` | 申込ベロシティ/クレジットハンガー | `DOM_VEL_INQ_SHORT`（短期照会件数）, `DOM_VEL_INQ_RECENT_RATIO`（直近四半期への集中度）, `DOM_VEL_ACTIVE_RATIO` |
| `DOM_UTL_*` | カード利用/キャッシング苦境 | `DOM_UTL_MEAN`, `DOM_UTL_MAX`, `DOM_UTL_CASH_ADVANCE_RATIO`（ATM引出依存度）, `DOM_UTL_MINPAY_DEFICIT` |
| `DOM_PAY_*` | 返済行動 | `DOM_PAY_COVERAGE`（生涯の支払充足率）, `DOM_PAY_DELAY_MAX`, `DOM_PAY_LATE_RATE` |
| `DOM_STB_*` | 安定性/外部スコア交互作用 | `DOM_STB_EXT_x_REGION`, `DOM_STB_PHONE_TO_AGE`, `DOM_STB_SOCIAL_DEF_RATIO`（社会的圏のデフォルト率） |

## 5. EXT_SOURCE高次特徴・グループ相対・クラスタリング（`extra_features.py`）

target非依存（リークなし）でtrain+test合算算出。

- **`EXT_POLY_*`**: EXT_SOURCE 1/2/3のpairwise積(`_12_PROD`等)・比・差、二乗(`_1_SQ`等)、加重和（EXT_SOURCE_2を重み2）
- **`GRP_*`**: `ORGANIZATION_TYPE`/`OCCUPATION_TYPE`等のグループ平均からの乖離・z-score
- **`KMEANS_*`**: EXT+主要数値空間でのk-meansクラスタ距離・クラスタID（教師なしクラスタリング）

## 6. リーク制御付き高度特徴（`oof_features.py`）

Home Credit 1位チームの目玉特徴を含む、Out-Of-Foldで安全にリークを防ぐ特徴群。

- **`neighbors_target_mean`**: `EXT_SOURCE_1/2/3` + `CREDIT_ANNUITY_RATIO`の空間でK近傍を取り、近傍のTARGET平均を特徴量化。trainはOOF（自分の属さないfoldで学習した近傍器）、testは全train学習の近傍器で算出しリークを防止
- **Target Encoding**: カテゴリ列のTARGET平均をスムージング付きでOOFエンコード

## 7. 時系列トレンド・ベロシティ・周期性（`trend_velocity_features.py`）

- **`TREND_*`**（`groupby_slope`）: bureau_balance/POS_CASH/credit_card/installmentsの月次系列に対する線形回帰の傾き（解析的に高速算出）。水準ではなく"変化の速さ・方向"を捉える
- **`WMA_*`**（`groupby_wma`）: 直近ほど指数的に重みが大きい加重移動平均（Home Credit 1位解法相当）
- **`BUREAU_CREDIT_TYPE_*`**: 信用情報に登録された債務の種類の多様性
- **`PREV_APP_INTERVAL_*` / `PREV_LAST3_REFUSED_RATIO`**: Home Creditへの過去申込の間隔（申込ベロシティ）・直近謝絶率
- **`DOC_*`**: 本人確認書類の提出数
- **`HOUR_*` / `WEEKDAY_*`**: 申込時刻・曜日の周期エンコーディング（sin/cos）+ オフタイム申込×地域リスク交互作用
- **`ISO_ANOMALY_SCORE`**: IsolationForestによる教師なし異常度スコア

## 8. Kaggle 1位解法由来の追加特徴（`top_solution_features.py`）

Home Credit Default Risk 1位ソリューションのdiscussion（Bojan/Olivier/Ryan/Phil/Yang/Michael Jahrer, 2018）で明記された効く特徴のうち、既存パイプラインに無かったもの。

- **`YEARLY_INTEREST_RATE` / `MONTHLY_INTEREST_RATE`**: 現在の申込の年利率をNewton法でIRR近似（Olivierが「最もスコアに効いた特徴の一つ」と明言）。CNT_PAYMENT(分割回数)が申込時点では無いため、同一顧客の過去ローン平均回数を代理変数に使用
- **`EXT3_DIV_*`**: EXT_SOURCE_3による除算特徴（AMT_CREDIT/AMT_ANNUITY/DAYS_BIRTH/AMT_INCOME_TOTAL）
- **`AGE_INT`**: 年齢の離散化
- **`INCOME_ANNUITY_RATIO` / `ANNUITY_TO_MAX_INSTALLMENT_RATIO`**
- **`BUREAU_LAST_ACTIVE_DAYS_CREDIT` / `BUREAU_ACTIVE_DEBT_SUM`**: アクティブローンに絞った直近性・残債合計
- **`PREV_LAST3S_*` / `PREV_LAST5S_*` / `PREV_FIRST2S_*` / `PREV_FIRST4S_*` / `PREV_LAST_PRODUCT_COMBINATION`**: previous_applicationの直近/最初N件スライス集約
- **`INS_60D_*` / `INS_90D_*` / `INS_180D_*` / `INS_1000D_*` / `INS_NUM{1,2,3,4}_*`**: installmentsの期間別集約細分化・支払回次別集約

## 9. ギャップ特徴量（`gap_features.py`）

チャット提案分。時間軸の"隙間"・整合性・欠損パターンそのものを情報として使う。

| プレフィックス | 内容 |
|---|---|
| `GAP_TL_*` | 同時進行ローン最大件数（sweep line法）、最後の完済からの経過日数、借入開始間隔の最大値、アクティブローン残存月数、直近謝絶/承認からの経過日数・最新決定が謝絶かフラグ |
| `GAP_BEH_*` | POS: 予定より早く完済した契約数・比率、CC: 支払いが最低額に張り付いている月の比率（全期間/直近12M） |
| `GAP_PRES_*` | bureau/prev/POS/INS/CC履歴の有無フラグ+欠けているテーブル数、行単位のNaN数・比率 |
| `GAP_CON_*` | 収入のキリ番フラグ・末尾ゼロ数（自己申告水増しシグナル）、全債務横断の月次返済負担/収入、bureau残債があるのに直近照会0の不整合フラグ、新規ローン期間と既存債務残存期間のオーバーラップ月数・比率 |

## 10. 最終バッチ回収特徴（`final_features.py`）

公開kernel・1位/2位解法writeup・AmEx上位解法から、既存パイプラインの取りこぼしを最後に回収。

| プレフィックス | 内容 |
|---|---|
| `FIN_APP_*` | GOODS/収入比、子供1人あたり収入、車齢/年齢・車齢/勤続、連絡手段充実度、住所不一致カウント、建物情報の行平均・非欠損数 |
| `FIN_BUR_*` | bureau集約列からの後段比率（債務/与信、延滞/債務、延長回数フラグ） |
| `FIN_XT_*` | テーブル横断の負担比較（新規annuity vs 過去平均支払、新規与信 vs 過去与信/残債） |
| `FIN_INT_*` | 上位重要度特徴同士の交互作用（EXT_SOURCE_MEAN×金利、CREDIT_ANNUITY_RATIO vs 予測CNT_PAYMENTの差/比） |
| `GRP2_*` | 追加グループ相対特徴（OCCUPATION_TYPE/REGION_RATING_CLIENT/年齢10歳刻み） |

## 11. DAE（Denoising Autoencoder）埋め込み（`dae_features.py` / `dae_model.py`）

2位解法（ikiri_DS）構成を参考にしたPyTorch実装。

1. カテゴリ列one-hot化、数値列median埋め+欠損フラグ、RankGaussスケーリング
2. train+testを合わせた行列で **Swap Noise DAE** を教師なし学習（各列を確率的にバッチ内の別行の値へ入れ替えるノイズ。ガウスノイズと違いカテゴリ変数にも自然に効く）
3. Encoder 3層（各256、以前は1024で試して後述のVRAM問題により縮小）→ bottleneck → Decoder
4. 学習後、Encoder各層の出力をconcatして特徴量として抽出（3層×256=768次元）

## 12. 特徴量選択・重要度分析・分布シフト検証

- **`feature_selection.py`**: Null Importance法（Olivier法）。本物のTARGETでの重要度と、シャッフルしたTARGETでの重要度（偶然の重要度）を比較し、有意に上回る特徴だけを残す
- **`fs_quick_ab.py`**: 上記drop適用の是非を軽量LightGBMでA/Bテスト（差+0.0003未満なら「適用しない」を推奨する安全側判定）
- **`feature_importance.py`**: 5-fold LightGBMでgain/split重要度、プレフィックス・グループ別ロールアップ集計
- **`adversarial_validation.py`**: train/testを見分ける分類器のAUCで分布シフトを測定。`HC_ADV_FOLD=1`でCV foldをtest分布に近づける層化を実施（1位解法の「CVをtest分布に近づけるとLBとの相関が上がる」という指摘に対応）

---

## 13. モデルカタログ

### メインパイプライン

| モデル | ファイル | 特徴 |
|---|---|---|
| LightGBM (GBDT) | `train_gbdt.py` | GPU対応、シード平均、Optunaベストパラメータ自動ロード、DAE埋め込み結合 |
| XGBoost | `train_gbdt.py` | 同上（`device=cuda`） |
| CatBoost | `train_catboost.py` | Ordered Target Statisticsで欠損・カテゴリの扱いがLGBM/XGBと異なり、誤りの系統が変わるため多様性に寄与 |
| LightGBM DART | `train_lgb_dart.py` | 木のdropoutで別の正則化経路。単体スコアはgbdtに劣るがブレンドで効く（1位解法由来） |
| MLP (DAE→MLP) | `train_nn.py` | 2位チームの"toshNN"相当。DAE埋め込み+数値特徴を入力 |
| TabM-lite | `train_tabm.py` | Gorishniy et al. 2025 "TabM"の簡略実装。1モジュール内に複数の並列MLPメンバーを持つdeep ensemble |
| TabPFN v2 / TabICL | `train_tabpfn.py` | 2025年のtabular基盤モデル。文脈長上限対策でサブサンプルbagging方式を採用 |
| GRU（月次系列） | `train_gru_seq.py` | POS_CASH_balanceの月次系列を直接読むRNN。1位解法M3の第一歩（スコープ縮小版） |
| 擬似ラベリング LightGBM | `pseudo_label.py` | 確信度の高いtest行を擬似ラベル付きでtrainに追加し再学習 |

### セカンドパイプライン（`hc_campaign.py`経由）

- LightGBM: gbdt(full/top600/no_meta), dart, goss, rf の5バリエーション
- XGBoost, CatBoost, MLP
- KNN ターゲット特徴（`knn_target_feature`）
- 行レベル補助モデル（previous_application / installments_payments / bureau_balance / POS_CASH の**行単位**でLightGBMを学習し、顧客単位に集約してリークなく特徴化）
- 擬似ラベリング（最良OOFモデルをベースに再構成）

### 最終追加分（`hc_v5_final_push.py` / `hc_v6_lastday.py`）

- 現申込金利推定モデル（`LGBMRegressor`でCNT_PAYMENTを予測→推定金利特徴を生成）
- CatBoostネイティブカテゴリモデル（one-hotと別表現の多様性源）
- LightGBM 10-fold・低学習率(0.005)版

---

## 14. アンサンブル手法

両パイプラインで共通して以下の手法を比較し、OOF AUCが最良のものを採用：

1. **weighted**: softmax重みをNelder-MeadでAUC最大化（順位ブレンド）
2. **hillclimb**: Caruanaのアンサンブル選択。最強モデルから貪欲に加重（重複選択可＝実質重み付け）していく
3. **stacking**: OOFをrank変換した特徴として、LogisticRegression + 浅いLightGBMの2層メタモデルで学習
4. **restacking**（`hc_v5_final_push.py`）: stackingのL2入力に、全モデルのOOF(rank)に加えて強い生特徴量（`NEW_EXTSOURCE_MEAN`, `NEW_DAYS_BIRTH`, `AMT_CREDIT`等）も混ぜ、「どの領域でどのモデルを信じるか」をメタモデルに学習させる
5. **rank平均ブレンド**: 複数submissionファイル自体をrank変換して単純平均（最後の保険）

---

## 15. 直面した技術的課題とデバッグ

開発中に実際に遭遇し、原因究明・修正した主なバグ・問題。

### (1) XGBoost GPUのVRAMハング
DAE埋め込みを256→1024次元(3層で768→3072次元)に拡大した際、特徴量合計が2436→4788まで増加。RTX 3070 Ti（8GB VRAM）の容量を超え、XGBoostのGPU `QuantileDMatrix`構築が17時間以上ハング（nvidia-smi上は「動いている」ように見えるが実際は進捗ゼロ）。**対策**: DAEを256次元(768次元embedding)に戻すことで解決。以前の実行ログでXGBoostが正常完走していた実績と突き合わせ、「GPU自体が不安定」ではなく「VRAM容量の閾値超え」が真因と特定。

### (2) pandas Copy-on-Writeによる無音no-opバグ
`df[col].replace(365243, np.nan, inplace=True)`のような連鎖代入は、新しいpandasのCopy-on-Write環境下では**警告は出るが実際には元のDataFrameを変更しない**。Home Creditの定番前処理「365243という異常値をNaN化」が複数箇所で実際には効いていなかった。**修正**: `df[col] = df[col].replace(365243, np.nan)`の代入形に統一。

### (3) pandas Arrow-backed string dtype の見逃し
`dtype == "object"`だけでカテゴリ列を判定するコードが、Arrow-backed `string`型の列を見逃す（同一バグが計4箇所で発生）。**修正**: `pd.api.types.is_string_dtype(...)`を追加、または`select_dtypes(include=["object", "string"])`に変更。

### (4) 除算によるinf値でXGBoostがクラッシュ
`AMT_CREDIT / AMT_ANNUITY`等、epsilonガードのない除算が`AMT_ANNUITY==0`（リボ払い等の実データの癖）でinfを生成。LightGBM/CatBoostはinfを許容するが、XGBoostの`QuantileDMatrix`は`Input data contains inf`で例外を送出。**修正**: 該当箇所に`+1e-5`等のepsilonガードを追加し、さらに特徴量保存直前に一括でinf→NaN変換する安全策を各所に追加（多層防御）。

### (5) セカンドパイプラインの統合
`hc_ensemble_optuna.py`本体ファイルが当初欠落しておりModuleNotFoundError、`DATA_DIR`のパス誤りでFileNotFoundError等、複数の実行環境不整合を解消。両パイプラインの`StratifiedKFold(5, shuffle=True, seed=42)`が一致することを確認し、OOFの相互登録（`hc_campaign.py register`）でリークなくアンサンブルできる土台を整備。

### (6) Kaggle Public/Private スコアの解釈
Home Creditコンペは Public LB が Private LB より低く出る既知の傾向がある。CVとPrivateがほぼ一致していることこそが「健全（CVが信頼できる・過学習していない）」のサインであり、Publicが低いこと自体は問題ではないと確認。

---

## 16. スコア推移

| 段階 | 施策 | CV AUC | 備考 |
|---|---|---|---|
| メインパイプライン最終 | 12ファイル構成（DAE 768次元・lgb/xgb/cat/pseudo等） | best: stacking_logit 0.79812 | lgb 0.79522, xgb 0.79774, cat 0.79320 |
| セカンドパイプライン day3 | hc_campaignのhillclimb（新パイプライン単独） | 0.79942 | Public 0.79973 / Private 0.79634 |
| day4 | 旧パイプラインのlgb/xgb/catをOOF登録し13モデルstack | 0.80203 | Public 0.80065 / Private 0.79964（**この時点でPrivateが初めて0.8台**） |
| day5 | 旧パイプラインのmlp/tabm/tabpfn（NN/基盤モデル系）も追加登録 | 0.80217 | 弱モデルだが多様性源として試行 |
| day6 | Optunaチューニング(35%サンプル・3-fold・50試行)後、全7 LGBM系+xgb再学習、16モデルstack | 0.80211 | Public 0.79989 / Private 0.79996（**Private最高値**） |
| 最終追加 | 金利推定特徴(augment)後: lgb_gbdt_full単体 | 0.80005 | augment前0.798937から改善 |
| 最終追加 | 同、xgb単体 | 0.79976→0.79992 | POS行レベル+KNN追加後さらに微増 |
| 最終追加 | LightGBM 10-fold・低学習率版(f10) | **0.80070** | 現時点の単体モデル最高値 |

※ `ensemble2`（restack版の最終アンサンブル）は実行中・結果確定前の状態でこの資料を作成している。

---

## 17. 運用面の工夫

- `experiments.csv`に「日付・タグ・変更内容・CV/Public/Private」を毎回記録し、CVを判断基準としてLBは検証用に限定する規律を徹底
- 重い学習は`nohup`でバックグラウンド実行しログファイルに残す。PCがクラッシュしても`oof_store/`の`.npy`ファイルの有無・タイムスタンプで進捗を復元できる設計（`hc_campaign.py train`は既存OOFがあれば自動スキップ）
- GPU使用は「安全側（CPU）をデフォルトにし、VRAM超過リスクが低いと判断できた範囲だけ明示的に有効化する」という一貫した方針

---

## 18. まとめ

- ベースの集約特徴（bureau/prev/POS/installments/credit_card）に加え、金融ドメイン指標・時系列トレンド・Kaggle上位解法由来の特徴・DAE埋め込み・KNNターゲット特徴など、多層的な特徴量エンジニアリングを積み上げた
- LightGBM/XGBoost/CatBoost/DART/MLP/TabM/TabPFN/GRUという多様なアーキテクチャのモデルを揃え、hillclimb・stacking・restackingで統合
- 独立に構築された2本のパイプラインをOOF単位で統合するという「チームマージ」的な発想により、単独では届かなかったスコア向上を達成
- 開発過程では、VRAM容量設計・pandasの仕様変化・浮動小数点例外など、実務でも頻出する典型的なバグに複数回遭遇し、その都度原因を切り分けて修正した
