# =============================================================================
# hc_campaign.py — Private 0.803+ (歴代トップ10圏) 攻略キャンペーン用ランナー
#
# v4 を「一発半日ジョブ」から「段階実行 + キャッシュ + 提出管理」に分解する。
#
# 使い方:
#   python hc_campaign.py features            # 特徴量構築 → parquet キャッシュ
#   python hc_campaign.py tune                # Optuna → best_params.json
#   python hc_campaign.py train lgb_gbdt_full # モデル 1 本学習 → OOF 保存
#   python hc_campaign.py train all           # 動物園全部（時間あるとき）
#   python hc_campaign.py register my_tabm oof.csv pred.csv
#                                             # 既存パイプラインの OOF を登録
#   python hc_campaign.py ensemble            # hillclimb + stack → submission
#   python hc_campaign.py status              # OOF ストアと成績の一覧
#
# ディレクトリ構成（自動生成）:
#   cache/features_train.parquet / features_test.parquet / feats.json
#   cache/best_params.json
#   oof_store/{name}_oof.npy / {name}_pred.npy   ← モデルごとに追記式
#   submissions/submission_{tag}.csv
#   experiments.csv                               ← CV/LB の手動記録台帳
#
# 外部 OOF の取り込み（旧 12 ファイルパイプラインの TabM / DAE / MLP など）:
#   - SK_ID_CURR で並び順を照合して register コマンドで登録する
#   - fold 分割が異なる OOF はわずかに楽観バイアスを持つ。可能なら旧側を
#     StratifiedKFold(5, shuffle=True, seed=42) で再生成して揃えるのが理想。
# =============================================================================

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

import lightgbm as lgb

warnings.simplefilter(action="ignore", category=FutureWarning)

ROOT = Path(".")
CACHE = ROOT / "cache"
STORE = ROOT / "oof_store"
SUBS = ROOT / "submissions"
for d in (CACHE, STORE, SUBS):
    d.mkdir(exist_ok=True)

N_FOLDS = 5
SEED = 42


# -----------------------------------------------------------------------------
# stage: features
# -----------------------------------------------------------------------------
def stage_features():
    from hc_ensemble_optuna import timer
    from hc_ensemble_optuna_v2 import build_dataset_v2, null_importance_selection
    from hc_ensemble_optuna_v3 import (
        knn_target_feature, row_level_prev_score,
        row_level_installments_score, impute_ext_sources,
        ema_and_lag_features)
    from hc_ensemble_optuna_v4 import row_level_bureau_score
    import re
    import gc

    train, test, feats = build_dataset_v2()
    y_target = train[["SK_ID_CURR", "TARGET"]]

    with timer("ext_source imputation"):
        train, test = impute_ext_sources(train, test)
        feats += [c for c in train.columns
                  if c.endswith("_IMPUTED") and c not in feats]
    with timer("knn target feature"):
        train, test = knn_target_feature(train, test, k=500)
        feats.append("NEW_TARGET_NEIGHBORS_500_MEAN")
    for builder in (lambda: row_level_prev_score(y_target, test["SK_ID_CURR"]),
                    lambda: row_level_installments_score(y_target),
                    lambda: row_level_bureau_score(y_target)):
        block = builder()
        train = train.merge(block, on="SK_ID_CURR", how="left")
        test = test.merge(block, on="SK_ID_CURR", how="left")
        feats += list(block.columns)
        del block
        gc.collect()
    with timer("ema and lag features"):
        el = ema_and_lag_features()
        train = train.merge(el, on="SK_ID_CURR", how="left")
        test = test.merge(el, on="SK_ID_CURR", how="left")
        feats += list(el.columns)

    train = train.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x))
    test = test.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x))
    feats = list(dict.fromkeys(re.sub("[^A-Za-z0-9_]+", "", f) for f in feats))

    with timer("null importance selection"):
        feats = null_importance_selection(train, feats)

    # 安全策: 個々の比率計算にepsilonガードを入れ忘れた箇所が他にもあった場合の保険として、
    # 保存直前に一括でinf/-infをNaNに変換する。LightGBMはinfを許容するため気づかずにいたが、
    # XGBoostのQuantileDMatrixはinfで例外を投げる(実際にNEW_CREDIT_TERM等で発生した)。
    num_cols_tr = train.select_dtypes(include=[np.number]).columns
    num_cols_te = test.select_dtypes(include=[np.number]).columns
    train[num_cols_tr] = train[num_cols_tr].replace([np.inf, -np.inf], np.nan)
    test[num_cols_te] = test[num_cols_te].replace([np.inf, -np.inf], np.nan)

    train.to_parquet(CACHE / "features_train.parquet")
    test.to_parquet(CACHE / "features_test.parquet")
    (CACHE / "feats.json").write_text(json.dumps(feats))
    print(f"cached: {len(feats)} features, "
          f"train {train.shape}, test {test.shape}")


def load_features():
    train = pd.read_parquet(CACHE / "features_train.parquet")
    test = pd.read_parquet(CACHE / "features_test.parquet")
    feats = json.loads((CACHE / "feats.json").read_text())
    return train, test, feats


# -----------------------------------------------------------------------------
# stage: tune
# -----------------------------------------------------------------------------
def stage_tune():
    from hc_ensemble_optuna import tune_lgb
    train, _, feats = load_features()
    best = tune_lgb(train, feats)
    (CACHE / "best_params.json").write_text(json.dumps(best))
    print("saved: cache/best_params.json")


def load_params():
    p = CACHE / "best_params.json"
    if p.exists():
        return json.loads(p.read_text())
    print("[warn] best_params.json なし。既知の良好パラメータで代用")
    return {"num_leaves": 34, "max_depth": 8, "min_child_samples": 60,
            "min_child_weight": 40, "subsample": 0.87,
            "colsample_bytree": 0.35, "reg_alpha": 0.04, "reg_lambda": 0.07,
            "min_split_gain": 0.02}


# -----------------------------------------------------------------------------
# stage: train <model>
# -----------------------------------------------------------------------------
def stage_train(model_name):
    from hc_ensemble_optuna import kfold_train, make_xgb_fn, make_cat_fn
    from hc_ensemble_optuna_v4 import (
        make_lgb_variant_fn, make_mlp_fn, feature_subsets,
        pseudo_label_train)

    train, test, feats = load_features()
    best = load_params()
    subsets = feature_subsets(train, feats, best)

    registry = {
        "lgb_gbdt_full":  (subsets["full"],   lambda: make_lgb_variant_fn(best, "gbdt")),
        "lgb_gbdt_top600": (subsets["top600"], lambda: make_lgb_variant_fn(best, "gbdt")),
        "lgb_gbdt_nometa": (subsets["no_meta"], lambda: make_lgb_variant_fn(best, "gbdt")),
        "lgb_dart":       (subsets["top600"], lambda: make_lgb_variant_fn(best, "dart")),
        "lgb_goss":       (subsets["full"],   lambda: make_lgb_variant_fn(best, "goss")),
        "lgb_rf":         (subsets["top600"], lambda: make_lgb_variant_fn(best, "rf")),
        "xgb":            (subsets["full"],   lambda: make_xgb_fn(best)),
        "cat":            (subsets["full"],   lambda: make_cat_fn()),
        "mlp":            (subsets["top600"], lambda: make_mlp_fn(subsets["top600"])),
    }

    # 疑似ラベリングは最良モデルが揃った後に一度だけ
    # 修正: 従来は下の registry ループに "pseudo" が流れて KeyError で即死していた
    if model_name == "pseudo":
        y = train["TARGET"].values
        stored = {p.stem[:-4]: np.load(p) for p in STORE.glob("*_oof.npy")}
        if not stored:
            print("OOF がありません。先に train を実行してください")
            return
        best_l1 = max(stored, key=lambda k: roc_auc_score(y, stored[k]))
        print(f"[pseudo] base model = {best_l1}")
        base_pred = np.load(STORE / f"{best_l1}_pred.npy")
        oof, pred = pseudo_label_train(train, test, subsets["full"],
                                       base_pred, best)
        np.save(STORE / "lgb_pseudo_oof.npy", oof)
        np.save(STORE / "lgb_pseudo_pred.npy", pred)
        return

    if model_name != "all" and model_name not in registry:
        print(f"[error] 未知のモデル名: {model_name} "
              f"(候補: {', '.join(registry)}, pseudo, all)")
        return

    names = list(registry) if model_name == "all" else [model_name]
    for name in names:
        if (STORE / f"{name}_oof.npy").exists():
            print(f"[skip] {name} は学習済み（削除すれば再学習）")
            continue
        fs, fn_factory = registry[name]
        oof, pred, _ = kfold_train(train, test, fs, fn_factory(), name)
        np.save(STORE / f"{name}_oof.npy", oof)
        np.save(STORE / f"{name}_pred.npy", pred)


# -----------------------------------------------------------------------------
# stage: register（外部 OOF の取り込み）
# -----------------------------------------------------------------------------
def stage_register(name, oof_csv, pred_csv):
    """旧パイプラインの OOF/pred を登録する。
    oof_csv: SK_ID_CURR, oof 列を持つ CSV
    pred_csv: SK_ID_CURR, pred 列を持つ CSV"""
    train, test, _ = load_features()

    o = pd.read_csv(oof_csv)
    oof = train[["SK_ID_CURR"]].merge(o, on="SK_ID_CURR", how="left")
    col = [c for c in oof.columns if c != "SK_ID_CURR"][0]
    if oof[col].isnull().any():
        raise ValueError("OOF に SK_ID_CURR の欠けがあります")

    p = pd.read_csv(pred_csv)
    pred = test[["SK_ID_CURR"]].merge(p, on="SK_ID_CURR", how="left")
    pcol = [c for c in pred.columns if c != "SK_ID_CURR"][0]
    # 修正: pred 側の欠損チェックが漏れていた（欠損があると NaN の TARGET を提出してしまう）
    if pred[pcol].isnull().any():
        raise ValueError("pred に SK_ID_CURR の欠けがあります")

    np.save(STORE / f"{name}_oof.npy", oof[col].values)
    np.save(STORE / f"{name}_pred.npy", pred[pcol].values)
    auc = roc_auc_score(train["TARGET"], oof[col])
    print(f"registered: {name} (OOF AUC = {auc:.5f})")


# -----------------------------------------------------------------------------
# stage: ensemble（hillclimb + 2 層スタック → 提出ファイル）
# -----------------------------------------------------------------------------
def hillclimb(oofs, y, n_iter=100, init_best=True):
    """rank 変換済み OOF 群を貪欲に足し込む（重複選択可 = 実質重み付け）"""
    names = list(oofs)
    ranked = {n: pd.Series(oofs[n]).rank(pct=True).values for n in names}

    order = sorted(names, key=lambda n: -roc_auc_score(y, oofs[n]))
    ens = ranked[order[0]].copy() if init_best else np.zeros(len(y))
    picks = [order[0]] if init_best else []
    best_auc = roc_auc_score(y, ens) if init_best else 0.0

    for _ in range(n_iter):
        cand_best, cand_name = best_auc, None
        for n in names:
            trial = (ens * len(picks) + ranked[n]) / (len(picks) + 1)
            auc = roc_auc_score(y, trial)
            if auc > cand_best:
                cand_best, cand_name = auc, n
        if cand_name is None:
            break
        ens = (ens * len(picks) + ranked[cand_name]) / (len(picks) + 1)
        picks.append(cand_name)
        best_auc = cand_best

    weights = pd.Series(picks).value_counts(normalize=True)
    print(f"[hillclimb] OOF AUC = {best_auc:.5f}")
    print(f"[hillclimb] weights:\n{weights}")
    return weights, best_auc


def stage_ensemble(tag="latest"):
    train, test, _ = load_features()
    y = train["TARGET"].values

    oofs = {p.stem[:-4]: np.load(p) for p in STORE.glob("*_oof.npy")}
    preds = {p.stem[:-5]: np.load(p) for p in STORE.glob("*_pred.npy")}
    if not oofs:
        print("OOF がありません。先に train を実行してください")
        return
    print(f"models in store: {sorted(oofs)}")

    # --- 経路 1: hillclimb ---
    weights, hc_auc = hillclimb(oofs, y)
    hc_pred = sum(w * pd.Series(preds[n]).rank(pct=True).values
                  for n, w in weights.items())

    # --- 経路 2: 2 層スタック（LogReg + 浅い LGBM）---
    names = sorted(oofs)
    L1o = np.column_stack([pd.Series(oofs[n]).rank(pct=True) for n in names])
    L1p = np.column_stack([pd.Series(preds[n]).rank(pct=True) for n in names])
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + 7)
    st_oof = np.zeros(len(y))
    st_pred = np.zeros(L1p.shape[0])
    for tr_idx, va_idx in skf.split(L1o, y):
        lr = LogisticRegression(C=0.5, max_iter=1000)
        lr.fit(L1o[tr_idx], y[tr_idx])
        gm = lgb.LGBMClassifier(objective="binary", n_estimators=400,
                                learning_rate=0.03, num_leaves=7,
                                min_child_samples=500, random_state=SEED,
                                n_jobs=-1, verbosity=-1)
        gm.fit(L1o[tr_idx], y[tr_idx])
        st_oof[va_idx] = (lr.predict_proba(L1o[va_idx])[:, 1] +
                          gm.predict_proba(L1o[va_idx])[:, 1]) / 2
        st_pred += (lr.predict_proba(L1p)[:, 1] +
                    gm.predict_proba(L1p)[:, 1]) / 2 / N_FOLDS
    st_auc = roc_auc_score(y, st_oof)
    print(f"[stack] OOF AUC = {st_auc:.5f}")

    # --- 良い方を採用（差が僅少なら両方の rank 平均も検討）---
    if hc_auc >= st_auc:
        final, mode, cv = hc_pred, "hillclimb", hc_auc
    else:
        final, mode, cv = st_pred, "stack", st_auc
    sub = test[["SK_ID_CURR"]].copy()
    sub["TARGET"] = final
    path = SUBS / f"submission_{tag}_{mode}_cv{cv:.5f}.csv"
    sub.to_csv(path, index=False)
    print(f"saved: {path}")
    print("\n次: kaggle competitions submit -c home-credit-default-risk "
          f"-f {path} -m '{tag} {mode} CV={cv:.5f}'")
    print("提出後、experiments.csv に public/private を記録してください")


# -----------------------------------------------------------------------------
# stage: status
# -----------------------------------------------------------------------------
def stage_status():
    if not (CACHE / "features_train.parquet").exists():
        print("features: 未構築")
        return
    train, _, feats = load_features()
    y = train["TARGET"].values
    print(f"features: {len(feats)}")
    rows = []
    for p in sorted(STORE.glob("*_oof.npy")):
        name = p.stem[:-4]
        rows.append((name, roc_auc_score(y, np.load(p))))
    if rows:
        df = pd.DataFrame(rows, columns=["model", "oof_auc"]) \
               .sort_values("oof_auc", ascending=False)
        print(df.to_string(index=False))
    log = ROOT / "experiments.csv"
    if log.exists():
        print("\n--- experiments.csv ---")
        print(pd.read_csv(log).to_string(index=False))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "features":
        stage_features()
    elif cmd == "tune":
        stage_tune()
    elif cmd == "train":
        stage_train(sys.argv[2] if len(sys.argv) > 2 else "all")
    elif cmd == "register":
        stage_register(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "ensemble":
        stage_ensemble(sys.argv[2] if len(sys.argv) > 2 else "latest")
    else:
        stage_status()
