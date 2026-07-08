from sklearn.model_selection import cross_val_score
import lightgbm as lgb
import pandas as pd

train = pd.read_parquet("cache/features_train.parquet")
check_feats = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, verbosity=-1)
print(cross_val_score(m, train[check_feats], train["TARGET"], cv=5, scoring="roc_auc").mean())
