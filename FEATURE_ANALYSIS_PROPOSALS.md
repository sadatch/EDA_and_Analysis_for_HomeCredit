# 特徴量ギャップ分析と次の一手（2026-07-02）

現状: weighted ensemble OOF AUC **0.79694**（lgb 0.7963 / xgb 0.7968 / cat 0.7917 / tabm 0.7735 / mlp_dae 0.7659 / tabpfn 0.7642）。
上位解法のCVは0.80前後（1位 private 0.80570）なので、残りギャップは **+0.003〜0.006** 程度。
以下、`artifacts/` の実測値に基づく診断と、優先度付きの施策。

---

## 1. 現状診断（feature_importance / adversarial_report から読めること）

### 1-1. 効いているもの
- `NEIGHBORS_TARGET_MEAN_500` が単独で **gain share 14.7%（1位）**。1位解法の目玉が期待通り機能。
- `EXT_SOURCE_MEAN`（4.7%）、`PREV_ESTIMATED_INTEREST_RATE_max`（rank 5）→ **金利系シグナルが強い**。
- INSTALLMENTS系は100特徴で7.0%と効率が良い（`INS_1000D_IS_LATE_mean` rank 10 等）。

### 1-2. 効いていないもの（ノイズ源）
| 問題 | 実測 | 対応 |
|---|---|---|
| ゼロgain特徴 | **350本/2359本** | 削除して学習高速化＋ノイズ減 |
| Null Importance判定 | **drop候補1588本（keep 771）** | `HC_APPLY_FS=1` が未適用。A/B必須 |
| DOM_DLQ/LEV/UTL/VEL | 4グループ合計 gain share **0.18%** | ほぼ死んでいる。剪定 or 再設計 |
| TREND_VEL系 | 0.30%、BUREAU_6M_BB系はゼロgain多数 | 6M窓のBB傾き系は観測期間不足で機能せず |

### 1-3. 構造的リスク: adversarial AUC = 0.984
train/testの分布シフトが極大。シフト上位は `BUREAU_BB_MONTHS_COUNT_*`、`BUREAU_BB_6M_STATUS_MAX_max`、`DOM_VEL_INQ_SHORT`、`CREDIT_TERM` 等。
これは既知の性質（**testの申込は時期が新しく、bureau_balanceの観測窓カバレッジが異なる**）で、
BB系の「観測月数そのもの」に依存する特徴はCV楽観化の主因になりうる。

- `BUREAU_BB_*` のカウント系は **観測窓長で正規化**（例: DPD件数/観測月数）した比率版に置き換える
- `HC_ADV_FOLD=1`（実装済み・デフォルトOFF）でCVを回し、通常CVとのスコア差を確認する
- シフト最上位の数本（MONTHS_COUNT系）は drop してCV/LB相関を見る

---

## 2. スコアに貢献しそうな追加特徴量（優先度順）

### P1: CNT_PAYMENT予測モデル → 申込ごとの金利推定（1位解法の"magic"の完全版）
現状の `add_yearly_interest_rate` は `PREV_CNT_PAYMENT_mean`（過去申込の平均、欠損はfallback 12ヶ月）を返済回数として使っており、**申込ごとの精度が粗い**。
1位解法の核心は「previous_applicationで `CNT_PAYMENT` を回帰モデルで学習し、**現在の申込のCNT_PAYMENTを予測**→そこから金利を逆算」する点。

```
学習データ: previous_application (AMT_CREDIT, AMT_ANNUITY, AMT_GOODS_PRICE, ...) → CNT_PAYMENT
適用: application_train/test に予測CNT_PAYMENT → Newton法IRRへ入力
派生: 予測金利, 予測期間, AMT_CREDIT×予測金利, 金利の対PREV平均比
```
`PREV_ESTIMATED_INTEREST_RATE_max` が既にrank 5であることが、この方向の伸びしろの傍証。
**期待効果: 単体で+0.001〜0.002**（1位チーム報告ベース）。

### P2: neighbors_target_mean の多様化
現状 k=500 の1本のみで gain 14.7%。これだけ効くなら派生を増やす価値が高い。
- k = 100 / 1000 / 2000 の複数解像度
- 特徴空間の変更: (EXT_SOURCE群 + 予測金利)、(DAE埋め込み空間でのkNN)、(EXT + DAYS_BIRTH + DAYS_EMPLOYED)
- 近傍target平均だけでなく **近傍のEXT_SOURCE平均との差**（自分が近傍より良い/悪い）
- リーク制御は既存のOOF機構をそのまま流用

### P3: bureau_balance 正規化リメイク
1-3のシフト対応と同時に、ゼロgainだった6M窓を廃止し:
- DPD率 = DPD月数 / 観測月数（全期間・直近12M）
- STATUSの単調悪化フラグ（直近3値が悪化方向）
- 「最後に延滞してからの月数」（recency。カウントより頑健）

### P4: installments の深掘り（効率最良グループの横展開)
- 支払額/予定額の**分散・トレンド**（回を追うごとに遅れが拡大しているか）
- `DPD` の指数減衰加重和（直近重視、AmEx定石）
- 前倒し返済（早払い日数）の平均・比率 — 良質客シグナル
- 直近3回 vs 全期間の遅延率比（既存 `INS_LAST3_DELAY_MAX` の比率版）

### P5: 死んでいるDOM_*の再設計（剪定が先）
DOM_DLQ/LEV/UTL/VELは一旦drop（`HC_APPLY_FS=1`で自動的に落ちるはず）。
再設計するなら「絶対値→グループ内相対値」（例: 同ORGANIZATION_TYPE内での利用率Z）だけ試す。
既に `GRP_ORGANIZATION_TYPE_EXT_SOURCE_MEAN_DEV` がrank 6に入っており、**グループ内偏差の形は効く**ことが実証済み。

---

## 3. 追加でやるべき手法（モデル・学習側）

### M1: 特徴量選択の適用A/B（コスト最小・期待値大）
`HC_APPLY_FS=1 python3 train_gbdt.py` は未実施。2359→771本でノイズ減＋高速化。
1 seedでlgbだけ回して差分確認 → 良ければ全パイプライン再実行。**最初にやるべき**。

### M2: LightGBM DART モード
AmEx上位解法のほぼ全員が使用（`boosting="dart"`）。収束は遅いがGBDTブースト間の多様性が増え、
単体+0.0005〜0.001、アンサンブル寄与も別物になる。既存lgbとは**別メンバー**として追加。

### M3: 月次シーケンスのGRU/Transformer（アンサンブル多様性の本命）
installments / POS / credit_card / bureau_balance は月次系列なのに、現在は全メンバーが集約特徴ベース
（DAEも集約後の行を復元しているだけ）。系列を直接food するNNは**誤差の相関が低い**メンバーになる。
- 入力: SK_ID_CURRごとに直近36ヶ月の (支払遅延, 支払額比, 利用率, DPD) 系列 + application静的特徴
- 小さいGRU(64-128)で十分。OOF出力を ensemble.py に `gru` として追加
- HC当時の上位（17位等）とAmEx上位（2位はLSTM/GRU混成）で実証済みの型

### M4: スタッキングの修理（現状0.7913で weighted 0.7969 に大敗）
メタ学習器が過学習している兆候。直すなら:
- メタ入力を**logit変換**したOOF 6本+生特徴少数（EXT_SOURCE_MEAN等5本以内）に制限
- メタはL2強めのLogisticRegression、fold構造は必ずbaseと同一
- それでもweightedに勝てなければ捨ててよい（hillclimb/weightedで十分）

### M5: 確率のrank平均ブレンド
現在のweightedは確率空間の加重平均。**rank空間**（各モデルの予測を順位に変換して平均）は
AUC評価と整合的で、モデル間のキャリブレーション差を無視できる。ensemble.pyに1手法追加するだけ。

### M6: 擬似ラベルの閾値見直し
`PSEUDO_LOW=0.02 / HIGH=0.30` は上側が緩い（正例率8%のデータでp>0.30を正例扱いはノイズ源）。
`HIGH=0.7〜0.8`（soft labelなら現行でも可）でA/B。

---

## 4. 推奨実行順（コスト/期待値バランス）

| # | 施策 | コスト | 期待値 |
|---|---|---|---|
| 1 | M1: `HC_APPLY_FS=1` A/B | 1-2h | +0.0005〜0.001（＋学習2倍速） |
| 2 | P1: CNT_PAYMENT予測→金利 | 半日 | +0.001〜0.002 |
| 3 | P2: neighbors多様化 | 半日 | +0.0005〜0.001 |
| 4 | 1-3: BB正規化＋ADV_FOLD検証 | 半日 | CV/LB相関改善（LB事故防止） |
| 5 | M2: lgb-dart追加 | 実装1h+学習数h | +0.0005 |
| 6 | M3: GRUメンバー追加 | 1-2日 | +0.001〜0.002（アンサンブル経由） |
| 7 | P4: installments深掘り | 数h | +0.0005 |
| 8 | M5/M6: rank平均・擬似ラベル閾値 | 各1h | +0.0002〜0.0005 |

合計で **OOF 0.800前後**が現実的な射程。

## 5. 検証時の注意
- 各A/Bは `HC_N_SEEDS=1`・Optunaスキップ（`HC_DO_TUNE=0`相当）で高速に回し、採用時のみフル実行
- neighbors系・TE系のOOF特徴はadversarial_validation.pyから除外済みである点を維持
- DAE 768本はGBDT側でgain 27%を占めるが、**DAE無しlgb単体のA/B**は一度も取っていないなら取る価値あり
  （NN専用に留めた方がGBDTが締まる可能性。2位解法でもDAEはNN入力）

## 参考
- [1st Place Solution (discussion/64821)](https://www.kaggle.com/c/home-credit-default-risk/discussion/64821) — neighbors_target_mean_500 / CNT_PAYMENT予測→金利
- [2nd Place ikiri_DS DAE (GitHub)](https://github.com/ireko8/home-credit) / [Speaker Deck](https://speakerdeck.com/hoxomaxwell/home-credit-default-risk-2nd-place-solutions)
- [AmEx Default Prediction まとめ（lag/last特徴・dart・NN混成）](https://bullettech.github.io/BulletTech/Main_Course/Machine_Learning/2022-08-30-AMEX-Kaggle-Summary/)
- [Home Credit Credit Risk Model Stability 2024（時間安定性・時系列CV）](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability)
