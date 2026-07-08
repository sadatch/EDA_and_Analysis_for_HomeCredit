import pandas as pd

d4 = pd.read_csv("submissions/submission_day4_stack_cv0.80203.csv").set_index("SK_ID_CURR")["TARGET"]
d6 = pd.read_csv("submissions/submission_day6_tuned_stack_cv0.80211.csv").set_index("SK_ID_CURR")["TARGET"]

blend = ((d4.rank(pct=True) + d6.rank(pct=True)) / 2).reset_index()
blend.columns = ["SK_ID_CURR", "TARGET"]
blend.to_csv("submissions/submission_final_blend_d4d6.csv", index=False)
print("saved: submissions/submission_final_blend_d4d6.csv")
