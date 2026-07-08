# =============================================================================
# Home Credit Default Risk — v4: 0.8057（優勝スコア）に肉薄するための総力戦
#
# 方針: 単一モデルの改善は v3 でほぼ限界。ここからは
#       「多様性 × 多層スタッキング」で 0.001 単位を削るフェーズ。
#       9th place チームは 6 層スタッキングでローカル CV 0.806 に到達した。
#
# v3 に対する追加:
#   [O] モデル動物園 — LGBM(gbdt/dart/goss/rf-mode) + XGB + CatBoost +
#       RankGauss-MLP(PyTorch)。dart は Home Credit で単体最強の報告多数。
#   [P] 特徴量サブセットによる多様性 — full / top-600 / KNNなし の 3 セット
#   [Q] bureau_balance 行レベルモデル（STATUS 系列 → 顧客スコア）
#   [R] 疑似ラベリング — 確信度の高い test 予測を train に混ぜて再学習
#   [S] 2 層スタッキング — L1: 全モデル OOF → L2: LGBM + LogReg → L3: 重み平均
#   [T] Adversarial Validation — train/test 分布シフトの診断ユーティリティ
#
# 実行: v1〜v3 と同じディレクトリで python hc_ensemble_optuna_v4.py
#   ※ 全モデル実行は 3070 Ti で半日級。ZOO_MODELS で間引き可能。
# =============================================================================

import gc
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import QuantileTransformer

import lightgbm as lgb

from hc_ensemble_optuna import (
    DATA_DIR, N_FOLDS, SEED, timer, kfold_train,
    make_xgb_fn, make_cat_fn, tune_lgb,
)
from hc_ensemble_optuna_v2 import null_importance_selection
from hc_ensemble_optuna_v3 import (
    knn_target_feature, row_level_prev_score, row_level_installments_score,
    impute_ext_sources, ema_and_lag_features,
)
from hc_ensemble_optuna_v2 import build_dataset_v2

warnings.simplefilter(action="ignore", category=FutureWarning)

# 実行するモデルの選択（時間がないときはここを間引く）
ZOO_MODELS = ["lgb_gbdt", "lgb_dart", "lgb_goss", "lgb_rf",
              "xgb", "cat", "mlp"]
USE_PSEUDO_LABELING = True
PSEUDO_THRESHOLDS = (0.001, 0.75)  # (負例の上限, 正例の下限)


# =============================================================================
# [Q] bureau_balance 行レベルモデル
#     STATUS 系列（月次の延滞ステータス）を SK_ID_BUREAU 単位で特徴量化し、
#     顧客 TARGET を教師に補助モデルを学習 → 顧客スコアに集計。
# =============================================================================
def row_level_bureau_score(train_ids_target):
    # ★ メモリ節約: bureau_balance は約 2700 万行あるため dtype を明示指定
    bb = pd.read_csv(f"{DATA_DIR}/bureau_balance.csv",
                     dtype={"SK_ID_BUREAU": "int32",
                            "MONTHS_BALANCE": "int16",
                            "STATUS": "object"})
    link = pd.read_csv(f"{DATA_DIR}/bureau.csv",
                       usecols=["SK_ID_BUREAU", "SK_ID_CURR"],
                       dtype={"SK_ID_BUREAU": "int32", "SK_ID_CURR": "int32"})

    status_map = {"C": 0, "X": 0, "0": 0, "1": 1, "2": 2,
                  "3": 3, "4": 4, "5": 5}
    bb["ST"] = bb["STATUS"].map(status_map)
    bb = bb.sort_values(["SK_ID_BUREAU", "MONTHS_BALANCE"])
    g = bb.groupby("SK_ID_BUREAU")

    loan = pd.DataFrame({
        "ST_MAX": g["ST"].max(),
        "ST_MEAN": g["ST"].mean(),
        "ST_LAST6_MAX": g["ST"].apply(lambda s: s.tail(6).max()),
        "ST_LAST6_MEAN": g["ST"].apply(lambda s: s.tail(6).mean()),
        "ST_TREND": g["ST"].apply(
            lambda s: s.tail(6).mean() - s.mean()),
        "N_MONTHS": g.size(),
        "MB_MIN": g["MONTHS_BALANCE"].min(),
        "MB_MAX": g["MONTHS_BALANCE"].max(),
    }).reset_index()
    loan = loan.merge(link, on="SK_ID_BUREAU", how="left")
    loan = loan.dropna(subset=["SK_ID_CURR"])
    del bb, g
    gc.collect()

    feats = ["ST_MAX", "ST_MEAN", "ST_LAST6_MAX", "ST_LAST6_MEAN",
             "ST_TREND", "N_MONTHS", "MB_MIN", "MB_MAX"]
    loan = loan.merge(train_ids_target, on="SK_ID_CURR", how="left")
    tr_rows = loan[loan["TARGET"].notnull()]
    y_rows = tr_rows["TARGET"].values
    groups = tr_rows["SK_ID_CURR"].values

    params = {"objective": "binary", "metric": "auc", "verbosity": -1,
              "n_estimators": 1500, "learning_rate": 0.05, "num_leaves": 31,
              "colsample_bytree": 0.8, "subsample": 0.8, "subsample_freq": 1,
              "random_state": SEED, "n_jobs": -1}

    score = np.zeros(len(loan))
    gkf = GroupKFold(n_splits=N_FOLDS)
    with timer("row-level bureau_balance model"):
        for tr_idx, va_idx in gkf.split(tr_rows[feats], y_rows, groups):
            m = lgb.LGBMClassifier(**params)
            m.fit(tr_rows[feats].iloc[tr_idx], y_rows[tr_idx],
                  eval_set=[(tr_rows[feats].iloc[va_idx], y_rows[va_idx])],
                  eval_metric="auc",
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            score[tr_rows.index[va_idx]] = \
                m.predict_proba(tr_rows[feats].iloc[va_idx])[:, 1]
            te_mask = loan["TARGET"].isnull()
            score[te_mask.values] += \
                m.predict_proba(loan.loc[te_mask, feats])[:, 1] / N_FOLDS

    loan["ROW_SCORE"] = score
    agg = loan.groupby("SK_ID_CURR").agg(
        BBROW_SCORE_MEAN=("ROW_SCORE", "mean"),
        BBROW_SCORE_MAX=("ROW_SCORE", "max"),
        BBROW_SCORE_STD=("ROW_SCORE", "std"))
    del loan, tr_rows
    gc.collect()
    return agg


# =============================================================================
# [O] モデル動物園
# =============================================================================
def make_lgb_variant_fn(best_params, boosting="gbdt"):
    """gbdt / dart / goss / rf の 4 変種。dart は early stopping が効かない
    ため木の本数を固定する。"""
    base = {"objective": "binary", "metric": "auc", "verbosity": -1,
            "subsample_freq": 1, "random_state": SEED, "n_jobs": -1,
            **best_params}
    if boosting == "dart":
        base.update({"boosting_type": "dart", "n_estimators": 3000,
                     "learning_rate": 0.02, "drop_rate": 0.1,
                     "skip_drop": 0.5})
    elif boosting == "goss":
        base.update({"boosting_type": "goss", "n_estimators": 20000,
                     "learning_rate": 0.01})
        base.pop("subsample", None)       # goss は subsample 不可
        base["subsample_freq"] = 0
    elif boosting == "rf":
        base.update({"boosting_type": "rf", "n_estimators": 800,
                     "subsample": 0.7, "subsample_freq": 1,
                     "colsample_bytree": 0.3})
    else:
        base.update({"n_estimators": 20000, "learning_rate": 0.01})

    def fn(tr_x, tr_y, va_x, va_y):
        m = lgb.LGBMClassifier(**base)
        if boosting in ("gbdt", "goss"):
            m.fit(tr_x, tr_y, eval_set=[(va_x, va_y)], eval_metric="auc",
                  callbacks=[lgb.early_stopping(200, verbose=False)])
        else:  # dart / rf は固定本数
            m.fit(tr_x, tr_y)
        return m
    return fn


def make_mlp_fn(feats):
    """RankGauss 前処理 + PyTorch MLP（9th place の RankGauss NN の簡易版）。
    GPU があれば自動使用。"""
    import torch
    import torch.nn as nn

    device = "cuda" if torch.cuda.is_available() else "cpu"

    class MLP(nn.Module):
        def __init__(self, d_in):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_in, 512), nn.BatchNorm1d(512), nn.SiLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256), nn.BatchNorm1d(256), nn.SiLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 1))

        def forward(self, x):
            return self.net(x).squeeze(-1)

    class MLPWrapper:
        """predict_proba インターフェースを合わせる"""
        def __init__(self, model, qt, med):
            self.model, self.qt, self.med = model, qt, med

        def _prep(self, X):
            X = X.fillna(self.med)
            return self.qt.transform(X.values).astype(np.float32)

        def predict_proba(self, X):
            import torch
            self.model.eval()
            Xt = torch.tensor(self._prep(X)).to(device)
            with torch.no_grad():
                out = []
                for i in range(0, len(Xt), 8192):
                    out.append(torch.sigmoid(
                        self.model(Xt[i:i + 8192])).cpu().numpy())
            p = np.concatenate(out)
            return np.stack([1 - p, p], axis=1)

    def fn(tr_x, tr_y, va_x, va_y):
        med = tr_x.median()
        qt = QuantileTransformer(output_distribution="normal",
                                 n_quantiles=1000, random_state=SEED)
        qt.fit(tr_x.fillna(med).values)

        Xtr = torch.tensor(qt.transform(tr_x.fillna(med).values)
                           .astype(np.float32)).to(device)
        ytr = torch.tensor(tr_y.values.astype(np.float32)).to(device)
        Xva = torch.tensor(qt.transform(va_x.fillna(med).values)
                           .astype(np.float32)).to(device)
        yva = va_y.values

        model = MLP(Xtr.shape[1]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
        lossf = nn.BCEWithLogitsLoss()

        best_auc, best_state, patience = 0.0, None, 0
        idx = torch.randperm(len(Xtr))
        for epoch in range(30):
            model.train()
            for i in range(0, len(Xtr), 2048):
                b = idx[i:i + 2048]
                opt.zero_grad()
                loss = lossf(model(Xtr[b]), ytr[b])
                loss.backward()
                opt.step()
            sched.step()
            # validation
            model.eval()
            with torch.no_grad():
                pv = []
                for i in range(0, len(Xva), 8192):
                    pv.append(torch.sigmoid(model(Xva[i:i + 8192]))
                              .cpu().numpy())
            auc = roc_auc_score(yva, np.concatenate(pv))
            if auc > best_auc:
                best_auc, patience = auc, 0
                best_state = {k: v.clone() for k, v
                              in model.state_dict().items()}
            else:
                patience += 1
                if patience >= 5:
                    break
        model.load_state_dict(best_state)
        return MLPWrapper(model, qt, med)
    return fn


# =============================================================================
# [P] 特徴量サブセット
# =============================================================================
def feature_subsets(train, feats, best_params):
    """importance を一度計測して top-600 サブセットを作る +
    KNN/2段モデル系（強すぎて他を食う特徴量）を除いたサブセット"""
    params = {"objective": "binary", "verbosity": -1, "n_estimators": 1000,
              "learning_rate": 0.05, "random_state": SEED, "n_jobs": -1,
              **{k: v for k, v in best_params.items()
                 if k not in ("n_estimators", "learning_rate")}}
    m = lgb.LGBMClassifier(**params)
    m.fit(train[feats], train["TARGET"])
    imp = pd.Series(m.feature_importances_, index=feats) \
            .sort_values(ascending=False)

    top600 = imp.head(600).index.tolist()
    no_meta = [f for f in feats if not f.startswith(
        ("NEW_TARGET_NEIGHBORS", "PREVROW_", "INSROW_", "BBROW_"))]
    return {"full": feats, "top600": top600, "no_meta": no_meta}


# =============================================================================
# [R] 疑似ラベリング
# =============================================================================
def pseudo_label_train(train, test, feats, test_pred, best_params):
    lo, hi = PSEUDO_THRESHOLDS
    conf_neg = test[test_pred < lo].copy()
    conf_pos = test[test_pred > hi].copy()
    conf_neg["TARGET"] = 0
    conf_pos["TARGET"] = 1
    pseudo = pd.concat([conf_neg, conf_pos])
    print(f"[pseudo] adding {len(conf_neg)} neg / {len(conf_pos)} pos")

    aug = pd.concat([train, pseudo], ignore_index=True)

    # 疑似ラベル込みで LGBM を学習し直す（OOF は元 train 部分のみで評価）
    oof = np.zeros(len(train))
    pred = np.zeros(len(test))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    params = {"objective": "binary", "metric": "auc", "verbosity": -1,
              "n_estimators": 20000, "learning_rate": 0.01,
              "subsample_freq": 1, "random_state": SEED, "n_jobs": -1,
              **best_params}
    for tr_idx, va_idx in skf.split(train[feats], train["TARGET"]):
        tr_df = pd.concat([train.iloc[tr_idx], pseudo], ignore_index=True)
        m = lgb.LGBMClassifier(**params)
        m.fit(tr_df[feats], tr_df["TARGET"],
              eval_set=[(train[feats].iloc[va_idx],
                         train["TARGET"].iloc[va_idx])],
              eval_metric="auc",
              callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[va_idx] = m.predict_proba(train[feats].iloc[va_idx])[:, 1]
        pred += m.predict_proba(test[feats])[:, 1] / N_FOLDS
    print(f"[pseudo LGB] OOF AUC = "
          f"{roc_auc_score(train['TARGET'], oof):.5f}")
    return oof, pred


# =============================================================================
# [T] Adversarial Validation（診断用）
# =============================================================================
def adversarial_validation(train, test, feats, top_n=20):
    """AUC が 0.5 に近いほど train/test が同分布。特定特徴量の AUC 寄与が
    大きい場合、その特徴量は分布シフトを持つ（=CVとLBの乖離の原因候補）。"""
    X = pd.concat([train[feats], test[feats]], ignore_index=True)
    y = np.r_[np.zeros(len(train)), np.ones(len(test))]
    m = lgb.LGBMClassifier(objective="binary", n_estimators=300,
                           learning_rate=0.05, num_leaves=31,
                           colsample_bytree=0.5, random_state=SEED,
                           n_jobs=-1, verbosity=-1)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    aucs = []
    for tr_idx, va_idx in skf.split(X, y):
        m.fit(X.iloc[tr_idx], y[tr_idx])
        aucs.append(roc_auc_score(y[va_idx],
                                  m.predict_proba(X.iloc[va_idx])[:, 1]))
    m.fit(X, y)
    imp = pd.Series(m.feature_importances_, index=feats) \
            .sort_values(ascending=False)
    print(f"[adversarial] AUC = {np.mean(aucs):.4f} "
          f"(0.5 に近いほど良い)")
    print("[adversarial] シフトの大きい特徴量 top:")
    print(imp.head(top_n))
    return np.mean(aucs), imp


# =============================================================================
# [S] 2 層スタッキング
# =============================================================================
def two_layer_stack(oof_dict, pred_dict, y):
    names = list(oof_dict.keys())
    L1_oof = np.column_stack([pd.Series(oof_dict[n]).rank(pct=True)
                              for n in names])
    L1_pred = np.column_stack([pd.Series(pred_dict[n]).rank(pct=True)
                               for n in names])
    for n in names:
        print(f"  L1 {n}: OOF AUC = {roc_auc_score(y, oof_dict[n]):.5f}")

    # L2: LogReg と 浅い LGBM の 2 種類のメタモデル
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                          random_state=SEED + 7)
    l2_oof = {"logreg": np.zeros(len(y)), "lgbm": np.zeros(len(y))}
    l2_pred = {"logreg": np.zeros(L1_pred.shape[0]),
               "lgbm": np.zeros(L1_pred.shape[0])}

    for tr_idx, va_idx in skf.split(L1_oof, y):
        lr = LogisticRegression(C=0.5, max_iter=1000)
        lr.fit(L1_oof[tr_idx], y[tr_idx])
        l2_oof["logreg"][va_idx] = lr.predict_proba(L1_oof[va_idx])[:, 1]
        l2_pred["logreg"] += lr.predict_proba(L1_pred)[:, 1] / N_FOLDS

        gm = lgb.LGBMClassifier(objective="binary", n_estimators=500,
                                learning_rate=0.03, num_leaves=7,
                                min_child_samples=500, colsample_bytree=0.8,
                                random_state=SEED, n_jobs=-1, verbosity=-1)
        gm.fit(L1_oof[tr_idx], y[tr_idx])
        l2_oof["lgbm"][va_idx] = gm.predict_proba(L1_oof[va_idx])[:, 1]
        l2_pred["lgbm"] += gm.predict_proba(L1_pred)[:, 1] / N_FOLDS

    for k in l2_oof:
        print(f"  L2 {k}: OOF AUC = {roc_auc_score(y, l2_oof[k]):.5f}")

    # L3: L2 二つの rank 平均（単純だが頑健）
    l3_oof = (pd.Series(l2_oof["logreg"]).rank(pct=True) +
              pd.Series(l2_oof["lgbm"]).rank(pct=True)).values / 2
    l3_pred = (pd.Series(l2_pred["logreg"]).rank(pct=True) +
               pd.Series(l2_pred["lgbm"]).rank(pct=True)).values / 2
    l3_auc = roc_auc_score(y, l3_oof)
    print(f"  L3 blend: OOF AUC = {l3_auc:.5f}")

    # L2 単体が L3 を上回るならそちらを採用
    cands = {"L3": (l3_auc, l3_pred),
             "L2_logreg": (roc_auc_score(y, l2_oof["logreg"]),
                           l2_pred["logreg"]),
             "L2_lgbm": (roc_auc_score(y, l2_oof["lgbm"]),
                         l2_pred["lgbm"])}
    best = max(cands, key=lambda k: cands[k][0])
    print(f"[stack] final = {best} ({cands[best][0]:.5f})")
    return cands[best][1]


# =============================================================================
# main
# =============================================================================
def main():
    # ---------- 特徴量構築（v2 + v3 の全部入り + [Q]）----------
    train, test, feats = build_dataset_v2()
    y_target = train[["SK_ID_CURR", "TARGET"]]

    with timer("ext_source imputation"):
        train, test = impute_ext_sources(train, test)
        feats += [c for c in train.columns
                  if c.endswith("_IMPUTED") and c not in feats]
    with timer("knn target feature"):
        train, test = knn_target_feature(train, test, k=500)
        feats.append("NEW_TARGET_NEIGHBORS_500_MEAN")
    for builder in (lambda: row_level_prev_score(y_target,
                                                 test["SK_ID_CURR"]),
                    lambda: row_level_installments_score(y_target),
                    lambda: row_level_bureau_score(y_target)):
        block = builder()
        train = train.merge(block, on="SK_ID_CURR", how="left")
        test = test.merge(block, on="SK_ID_CURR", how="left")
        feats += list(block.columns)
    with timer("ema and lag features"):
        el = ema_and_lag_features()
        train = train.merge(el, on="SK_ID_CURR", how="left")
        test = test.merge(el, on="SK_ID_CURR", how="left")
        feats += list(el.columns)

    train = train.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x))
    test = test.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x))
    feats = list(dict.fromkeys(
        re.sub("[^A-Za-z0-9_]+", "", f) for f in feats))
    print(f"n_features: {len(feats)}")

    # ---------- [T] 診断 ----------
    adversarial_validation(train, test, feats)

    # ---------- 特徴量選択 ----------
    feats = null_importance_selection(train, feats)

    # ---------- Optuna ----------
    with timer("optuna_lgb"):
        best_lgb = tune_lgb(train, feats)

    # ---------- [P] 特徴量サブセット ----------
    subsets = feature_subsets(train, feats, best_lgb)

    # ---------- [O] モデル動物園を回す ----------
    oof_dict, pred_dict = {}, {}

    runs = []
    if "lgb_gbdt" in ZOO_MODELS:
        runs += [("lgb_gbdt_full", subsets["full"],
                  make_lgb_variant_fn(best_lgb, "gbdt")),
                 ("lgb_gbdt_top600", subsets["top600"],
                  make_lgb_variant_fn(best_lgb, "gbdt")),
                 ("lgb_gbdt_nometa", subsets["no_meta"],
                  make_lgb_variant_fn(best_lgb, "gbdt"))]
    if "lgb_dart" in ZOO_MODELS:
        runs.append(("lgb_dart", subsets["top600"],
                     make_lgb_variant_fn(best_lgb, "dart")))
    if "lgb_goss" in ZOO_MODELS:
        runs.append(("lgb_goss", subsets["full"],
                     make_lgb_variant_fn(best_lgb, "goss")))
    if "lgb_rf" in ZOO_MODELS:
        runs.append(("lgb_rf", subsets["top600"],
                     make_lgb_variant_fn(best_lgb, "rf")))
    if "xgb" in ZOO_MODELS:
        runs.append(("xgb", subsets["full"], make_xgb_fn(best_lgb)))
    if "cat" in ZOO_MODELS:
        runs.append(("cat", subsets["full"], make_cat_fn()))
    if "mlp" in ZOO_MODELS:
        runs.append(("mlp", subsets["top600"],
                     make_mlp_fn(subsets["top600"])))

    for name, fs, fn in runs:
        with timer(name):
            o, p, _ = kfold_train(train, test, fs, fn, name)
            oof_dict[name], pred_dict[name] = o, p

    # ---------- [R] 疑似ラベリング（最良 L1 の予測で）----------
    y = train["TARGET"].values
    if USE_PSEUDO_LABELING:
        best_l1 = max(oof_dict, key=lambda k: roc_auc_score(y, oof_dict[k]))
        o, p = pseudo_label_train(train, test, subsets["full"],
                                  pred_dict[best_l1], best_lgb)
        oof_dict["lgb_pseudo"], pred_dict["lgb_pseudo"] = o, p

    # ---------- [S] 2 層スタッキング ----------
    final_pred = two_layer_stack(oof_dict, pred_dict, y)

    submission = test[["SK_ID_CURR"]].copy()
    submission["TARGET"] = final_pred
    submission.to_csv("submission.csv", index=False)

    oof_df = pd.DataFrame({"SK_ID_CURR": train["SK_ID_CURR"], "target": y})
    for n in oof_dict:
        oof_df[f"oof_{n}"] = oof_dict[n]
    oof_df.to_csv("oof_predictions_v4.csv", index=False)
    print("done: submission.csv / oof_predictions_v4.csv")


if __name__ == "__main__":
    main()
