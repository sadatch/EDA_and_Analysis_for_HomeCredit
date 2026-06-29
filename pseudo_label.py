"""
擬似ラベル (Pseudo-Labeling) — Grandmaster Playbook「6. pseudo-labeling」。

学習済みモデルのtest予測のうち、確信度が高い行（予測確率が非常に低い/高い行）だけを
擬似ラベル付きで学習データに加え、LightGBMを再学習する。
これによりtest分布の情報を学習に取り込み、汎化を上げる（知識蒸留に近い効果）。

リーク対策:
  - 擬似ラベルは「test行」にのみ付与する。OOFは元のtrain行に対してのみ計算する。
  - 各foldの学習に擬似行を加え、検証は常に本物のtrain行だけで行う（test行は検証に入らない）。

入力（先にbaseモデルを学習しておくこと）:
  artifacts/{lgb,xgb,cat}_test.npy のいずれか（teacherとして平均を使う）
出力:
  artifacts/lgbpl_oof.npy, artifacts/lgbpl_test.npy   (ensemble.pyが追加メンバーとして拾う)
"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import config
from utils import timer
from train_gbdt import load_features_with_dae, prepare_xy, _lgb_train_one, _lgb_base_params, _load_best_params

warnings.filterwarnings("ignore")


def _teacher_test_pred(n_test: int):
    """既存のbaseモデルtest予測の平均をteacherにする。"""
    preds = []
    for name in ["lgb", "xgb", "cat"]:
        p = config.ARTIFACT_DIR / f"{name}_test.npy"
        if p.exists():
            arr = np.load(p)
            if len(arr) == n_test:
                preds.append(arr)
    if not preds:
        return None
    return np.mean(preds, axis=0)


def main():
    if not config.PSEUDO_ENABLE:
        print("擬似ラベルは無効化されています (HC_PSEUDO=0)")
        return
    try:
        import lightgbm  # noqa
    except ImportError:
        print("LightGBM未インストールのため擬似ラベルをスキップ")
        return

    print(config.describe())
    with timer("特徴量読み込み"):
        train_df, test_df, categorical_features = load_features_with_dae()
        X, y, X_test = prepare_xy(train_df, test_df, categorical_features)
        cats = [c for c in categorical_features if c in X.columns]

    teacher = _teacher_test_pred(len(X_test))
    if teacher is None:
        print("teacher予測が無いため擬似ラベルをスキップ（先にtrain_gbdt.py等を実行してください）")
        return

    # 確信度が高い行を抽出（低リスク/高リスクの両端）
    conf_neg = np.where(teacher < config.PSEUDO_LOW)[0]
    conf_pos = np.where(teacher > config.PSEUDO_HIGH)[0]
    conf_idx = np.concatenate([conf_neg, conf_pos])
    pseudo_y = np.concatenate([np.zeros(len(conf_neg)), np.ones(len(conf_pos))])
    print(f"  擬似ラベル対象: 負例{len(conf_neg)} + 正例{len(conf_pos)} = {len(conf_idx)}行 "
          f"(test {len(X_test)}行中)")
    if len(conf_idx) < 10:
        print("  確信行が少なすぎるため擬似ラベルをスキップ")
        return

    X_conf = X_test.iloc[conf_idx].copy()
    params = {**_lgb_base_params(), **(_load_best_params("lgb") or {})}

    oof_acc = np.zeros(len(X))
    test_acc = np.zeros(len(X_test))
    seeds = config.SEED_LIST
    with timer("擬似ラベル付きLightGBM再学習 (seed平均)"):
        for s in seeds:
            p = {**params, "random_state": s, "seed": s}
            folds = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=s)
            oof = np.zeros(len(X))
            test = np.zeros(len(X_test))
            for trn_idx, val_idx in folds.split(X, y):
                X_trn = pd.concat([X.iloc[trn_idx], X_conf], axis=0)
                y_trn = np.concatenate([y.iloc[trn_idx].values, pseudo_y])
                model = _lgb_train_one(p, X_trn, y_trn, X.iloc[val_idx], y.iloc[val_idx].values, cats)
                oof[val_idx] = model.predict(X.iloc[val_idx], num_iteration=model.best_iteration)
                test += model.predict(X_test, num_iteration=model.best_iteration) / folds.n_splits
            oof_acc += oof / len(seeds)
            test_acc += test / len(seeds)
            print(f"  [LGBpl] seed={s}: OOF AUC={roc_auc_score(y, oof):.6f}")

    auc = roc_auc_score(y, oof_acc)
    np.save(config.ARTIFACT_DIR / "lgbpl_oof.npy", oof_acc)
    np.save(config.ARTIFACT_DIR / "lgbpl_test.npy", test_acc)
    scores_path = config.ARTIFACT_DIR / "cv_scores.json"
    scores = json.load(open(scores_path)) if scores_path.exists() else {}
    scores["lgb_pseudo"] = {"oof_auc": auc, "n_pseudo_rows": int(len(conf_idx))}
    with open(scores_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"擬似ラベルLightGBM完了: OOF AUC = {auc:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
