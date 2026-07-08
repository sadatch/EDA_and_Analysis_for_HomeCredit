"""
day4 / day5 / day6 の提出ファイル(いずれも高品質だが微妙に違う組み合わせの
stackアンサンブル)をrank平均でさらにブレンドする。

狙い: 3つはそれぞれ違うモデル組み合わせ・パラメータでできているため、
誤差の相関が低ければ単純平均でPrivateのブレを削れる可能性がある。
再学習不要でCSVを混ぜるだけなので数秒で終わる。

使い方:
  python3 blend_final.py
"""
import pandas as pd

files = [
    "submissions/submission_day4_stack_cv0.80203.csv",
    "submissions/submission_day5_diversity_stack_cv0.80217.csv",
    "submissions/submission_day6_tuned_stack_cv0.80211.csv",
]

dfs = [pd.read_csv(f).set_index("SK_ID_CURR")["TARGET"] for f in files]

# rank平均(0-1正規化)にしてから単純平均。スケールの違いを吸収するため。
ranked = [d.rank(pct=True) for d in dfs]
blend = sum(ranked) / len(ranked)

out = blend.reset_index()
out.columns = ["SK_ID_CURR", "TARGET"]
out_path = "submissions/submission_final_blend3.csv"
out.to_csv(out_path, index=False)
print(f"saved: {out_path}")
print(f"blended {len(files)} files: {files}")
