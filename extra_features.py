"""
追加特徴量（target非依存＝リーク無し、train+testを合わせて算出）。

  EXT_POLY_*   EXT_SOURCEの高次/相互作用（pairwise積・比、二乗、加重、×金額）
  GRP_*        グループ相対（ORGANIZATION_TYPE/OCCUPATION_TYPE等の平均からの乖離・z-score）
  KMEANS_*     k-meansクラスタ距離 + クラスタID（EXT+主要数値空間の教師なしクラスタリング）

いずれも「元の列が存在するときだけ」作る（合成/実データのどちらでも落ちない）。
feature_engineering.py の EXT_SOURCE補完後・保存前に呼ぶ想定。
"""
import numpy as np
import pandas as pd

EPS = 1e-5


# =====================================================================
# 1. EXT_SOURCE 高次 / 相互作用
# =====================================================================
def add_ext_poly_features(app_train: pd.DataFrame, app_test: pd.DataFrame) -> tuple:
    n_made = 0
    for df in (app_train, app_test):
        cols = [c for c in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3") if c in df.columns]
        if len(cols) < 2:
            continue
        e = {c: pd.to_numeric(df[c], errors="coerce") for c in
             ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3") if c in df.columns}
        feats = {}
        e1, e2, e3 = e.get("EXT_SOURCE_1"), e.get("EXT_SOURCE_2"), e.get("EXT_SOURCE_3")
        # pairwise 積
        if e1 is not None and e2 is not None: feats["EXT_POLY_12_PROD"] = e1 * e2
        if e1 is not None and e3 is not None: feats["EXT_POLY_13_PROD"] = e1 * e3
        if e2 is not None and e3 is not None: feats["EXT_POLY_23_PROD"] = e2 * e3
        # pairwise 比・差
        if e1 is not None and e2 is not None:
            feats["EXT_POLY_12_RATIO"] = e1 / (e2 + EPS)
            feats["EXT_POLY_12_DIFF"] = e1 - e2
        if e2 is not None and e3 is not None:
            feats["EXT_POLY_23_RATIO"] = e2 / (e3 + EPS)
            feats["EXT_POLY_23_DIFF"] = e2 - e3
        # 二乗
        for k, v in e.items():
            feats[f"EXT_POLY_{k[-1]}_SQ"] = v * v
        # 加重（EXT_SOURCE_2が最も予測力が高いとされ重み2）
        if e1 is not None and e2 is not None and e3 is not None:
            feats["EXT_POLY_WEIGHTED"] = e1 * 1.0 + e2 * 2.0 + e3 * 1.0
        # ×金額（EXT_SOURCE_MEANとの交互作用。既存と重複しないもの）
        emean = df.get("EXT_SOURCE_MEAN")
        if emean is not None:
            emean = pd.to_numeric(emean, errors="coerce")
            if "AMT_CREDIT" in df.columns:
                feats["EXT_POLY_MEAN_x_CREDIT"] = emean * pd.to_numeric(df["AMT_CREDIT"], errors="coerce")
            if "AMT_ANNUITY" in df.columns:
                feats["EXT_POLY_MEAN_x_ANNUITY"] = emean * pd.to_numeric(df["AMT_ANNUITY"], errors="coerce")
        if feats:
            add = pd.DataFrame(feats, index=df.index).replace([np.inf, -np.inf], np.nan)
            for c in add.columns:
                df[c] = add[c].astype(np.float32)
            n_made = len(feats)
    print(f"  [ext_poly] EXT高次/相互作用 {n_made}個を付与")
    return app_train, app_test


# =====================================================================
# 2. グループ相対（peer比較・z-score）  ※target非依存
# =====================================================================
GROUP_KEYS = ["ORGANIZATION_TYPE", "OCCUPATION_TYPE", "NAME_INCOME_TYPE"]
GROUP_TARGETS = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
                 "EXT_SOURCE_MEAN", "DAYS_BIRTH", "CREDIT_ANNUITY_RATIO"]


def add_group_relative_features(app_train: pd.DataFrame, app_test: pd.DataFrame) -> tuple:
    keys = [k for k in GROUP_KEYS if k in app_train.columns and k in app_test.columns]
    tgts = [t for t in GROUP_TARGETS if t in app_train.columns and t in app_test.columns]
    if not keys or not tgts:
        print("  [group] 対象キー/数値が無いためスキップ")
        return app_train, app_test

    n_tr = len(app_train)
    full = pd.concat([app_train[keys + tgts], app_test[keys + tgts]], axis=0, ignore_index=True)
    for t in tgts:
        full[t] = pd.to_numeric(full[t], errors="coerce")

    new = {}
    for k in keys:
        g = full.groupby(k, observed=True)[tgts]
        gmean = g.transform("mean")
        gstd = g.transform("std")
        for t in tgts:
            new[f"GRP_{k}_{t}_DEV"] = (full[t] - gmean[t]).values            # グループ平均からの乖離
            new[f"GRP_{k}_{t}_Z"] = ((full[t] - gmean[t]) / (gstd[t] + EPS)).values  # z-score
    add = pd.DataFrame(new).replace([np.inf, -np.inf], np.nan)
    add_tr = add.iloc[:n_tr].reset_index(drop=True).astype(np.float32)
    add_te = add.iloc[n_tr:].reset_index(drop=True).astype(np.float32)
    for c in add.columns:
        app_train[c] = add_tr[c].values
        app_test[c] = add_te[c].values
    print(f"  [group] グループ相対特徴 {add.shape[1]}個を付与 (keys={keys})")
    return app_train, app_test


# =====================================================================
# 3. k-means クラスタ距離 + クラスタID  ※教師なし＝リーク無し
# =====================================================================
KMEANS_COLS = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "EXT_SOURCE_MEAN",
               "CREDIT_ANNUITY_RATIO", "DAYS_BIRTH", "AMT_CREDIT", "AMT_ANNUITY",
               "AMT_INCOME_TOTAL"]
KMEANS_K = 8


def add_kmeans_features(app_train: pd.DataFrame, app_test: pd.DataFrame, k: int = KMEANS_K) -> tuple:
    from sklearn.preprocessing import StandardScaler
    try:
        from sklearn.cluster import MiniBatchKMeans
        KM = lambda: MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3, batch_size=4096)
    except Exception:
        from sklearn.cluster import KMeans
        KM = lambda: KMeans(n_clusters=k, random_state=42, n_init=3)

    cols = [c for c in KMEANS_COLS if c in app_train.columns and c in app_test.columns]
    if len(cols) < 3:
        print("  [kmeans] 対象列が不足のためスキップ")
        return app_train, app_test

    n_tr = len(app_train)
    full = pd.concat([app_train[cols], app_test[cols]], axis=0, ignore_index=True).astype(np.float32)
    full = full.replace([np.inf, -np.inf], np.nan)
    full = full.fillna(full.median())
    X = StandardScaler().fit_transform(full.values)

    km = KM()
    cluster_id = km.fit_predict(X)
    dist = km.transform(X)   # (n, k) 各セントロイドへの距離

    cols_dist = [f"KMEANS_DIST_{i}" for i in range(dist.shape[1])]
    out = pd.DataFrame(dist, columns=cols_dist).astype(np.float32)
    out["KMEANS_CLUSTER"] = cluster_id.astype(np.int16)
    out["KMEANS_MIN_DIST"] = dist.min(axis=1).astype(np.float32)

    for c in out.columns:
        app_train[c] = out[c].values[:n_tr]
        app_test[c] = out[c].values[n_tr:]
    print(f"  [kmeans] クラスタ距離+ID {out.shape[1]}個を付与 (k={k}, cols={len(cols)})")
    return app_train, app_test


def add_all(app_train: pd.DataFrame, app_test: pd.DataFrame) -> tuple:
    app_train, app_test = add_ext_poly_features(app_train, app_test)
    app_train, app_test = add_group_relative_features(app_train, app_test)
    app_train, app_test = add_kmeans_features(app_train, app_test)
    return app_train, app_test
