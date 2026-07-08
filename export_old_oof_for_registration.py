"""
旧12ファイルパイプライン(train_gbdt.py / train_catboost.py)が吐いた
artifacts/*_oof.npy, *_test.npy を hc_campaign.py register が読める
CSV形式(SK_ID_CURR, 値)に変換する。

register()はSK_ID_CURRでmergeするだけなので行順は気にしなくてよいが、
train側のIDはtrain_features.parquetから、test側のIDはtest_ids.csvから
それぞれ「npyと同じ行順」で取得する必要がある。
train_gbdt.py/train_catboost.pyはどちらもtrain_features.parquetを
そのままの行順でX_train_full化しているので、SK_ID_CURR列を
そのまま使えば.npyの行と1対1で対応する。

使い方:
  python3 export_old_oof_for_registration.py
  # -> old_lgb_oof.csv / old_lgb_pred.csv
  #    old_xgb_oof.csv / old_xgb_pred.csv
  #    old_cat_oof.csv / old_cat_pred.csv
  # を EDA_and_Analysis_for_HomeCredit/ 直下に出力

その後:
  python3 hc_campaign.py register old_lgb old_lgb_oof.csv old_lgb_pred.csv
  python3 hc_campaign.py register old_xgb old_xgb_oof.csv old_xgb_pred.csv
  python3 hc_campaign.py register old_cat old_cat_oof.csv old_cat_pred.csv
  python3 hc_campaign.py ensemble day4
"""
import numpy as np
import pandas as pd
from pathlib import Path

import config

ART = config.ARTIFACT_DIR
PROC = config.PROC_DIR

# train側のSK_ID_CURR（train_gbdt.py/train_catboost.pyがX_train_full化する
# 直前のtrain_dfと同じ行順 = train_features.parquetそのまま）
train_ids = pd.read_parquet(PROC / "train_features.parquet", columns=["SK_ID_CURR"])
# test側のSK_ID_CURR（train_gbdt.pyがtest_ids.csvとして保存済み、test_dfと同じ行順）
test_ids = pd.read_csv(ART / "test_ids.csv")

y = np.load(ART / "y_train.npy")
assert len(train_ids) == len(y), (
    f"train_features.parquetの行数({len(train_ids)})とy_train.npyの行数({len(y)})が"
    f"一致しません。train_gbdt.py実行時と特徴量が変わっていないか確認してください。"
)

MODELS = {
    "lgb": ("lgb_oof.npy", "lgb_test.npy"),
    "xgb": ("xgb_oof.npy", "xgb_test.npy"),
    "cat": ("cat_oof.npy", "cat_test.npy"),
    # 単体AUCは弱い(0.76-0.77台)がNN/TabPFN系でツリー系とアーキテクチャが
    # 全く異なるため、hillclimb/stackの多様性ソースとして試す価値がある
    "mlp": ("mlp_oof.npy", "mlp_test.npy"),
    "tabm": ("tabm_oof.npy", "tabm_test.npy"),
    "tabpfn": ("tabpfn_oof.npy", "tabpfn_test.npy"),
}

for name, (oof_file, test_file) in MODELS.items():
    oof_path = ART / oof_file
    test_path = ART / test_file
    if not oof_path.exists() or not test_path.exists():
        print(f"[skip] {name}: {oof_path} または {test_path} が見つかりません")
        continue

    oof = np.load(oof_path)
    pred = np.load(test_path)
    assert len(oof) == len(train_ids), f"{name}: oof長({len(oof)}) != train行数({len(train_ids)})"
    assert len(pred) == len(test_ids), f"{name}: pred長({len(pred)}) != test行数({len(test_ids)})"

    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y, oof)

    oof_df = train_ids.copy()
    oof_df[f"old_{name}"] = oof
    oof_df.to_csv(f"old_{name}_oof.csv", index=False)

    pred_df = test_ids.copy()
    pred_df[f"old_{name}"] = pred
    pred_df.to_csv(f"old_{name}_pred.csv", index=False)

    print(f"[export] old_{name}: OOF AUC={auc:.5f}  "
          f"-> old_{name}_oof.csv / old_{name}_pred.csv")

print("\n次のコマンドで登録:")
for name in MODELS:
    if (ART / MODELS[name][0]).exists():
        print(f"  python3 hc_campaign.py register old_{name} old_{name}_oof.csv old_{name}_pred.csv")
