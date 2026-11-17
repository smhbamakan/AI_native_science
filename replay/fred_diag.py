"""Exploratory diagnostics added after the confirmatory run of fred_replay.py (not pre-registered)."""
import json, numpy as np, pandas as pd
from scipy.stats import norm
h1 = pd.read_csv("fred_HP1_rows.csv")
def auc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, float); ok = np.isfinite(s) & np.isfinite(y); s, y = s[ok], y[ok]
    P, N = (y == 1).sum(), (y == 0).sum()
    if P == 0 or N == 0: return np.nan
    r = pd.Series(s).rank().values; return (r[y == 1].sum() - P * (P + 1) / 2) / (P * N)
D = {}
cnt = h1.groupby("paper").size().sort_values(ascending=False)
D["rows_per_paper_top5"] = cnt.head(5).tolist(); D["share_top1"] = float(cnt.iloc[0] / len(h1)); D["share_top3"] = float(cnt.head(3).sum() / len(h1))
# paper-level analysis: one point per paper (mean outcome, mean S_L), Spearman
pl = h1.groupby("paper").agg(y=("y", "mean"), S_L=("S_L", "mean"), S_S=("S_S", "mean"), n=("y", "size"))
from scipy.stats import spearmanr
D["paper_level_spearman_S_L"] = float(spearmanr(pl.S_L, pl.y).statistic); D["paper_level_spearman_S_S"] = float(spearmanr(pl.S_S, pl.y, nan_policy="omit").statistic)
# weighted AUC: each paper total weight 1
w = 1 / h1.groupby("paper")["y"].transform("size")
def wauc(s, y, w):
    s, y, w = map(lambda a: np.asarray(a, float), (s, y, w)); ok = np.isfinite(s); s, y, w = s[ok], y[ok], w[ok]
    pos, neg = y == 1, y == 0; num = 0.0
    for si, wi in zip(s[pos], w[pos]): num += wi * (w[neg] * ((si > s[neg]) + 0.5 * (si == s[neg]))).sum()
    return num / (w[pos].sum() * w[neg].sum())
D["paper_weighted_auc"] = dict(S_L=wauc(h1.S_L, h1.y, w), S_S=wauc(h1.S_S, h1.y, w), z=wauc(h1.z, h1.y, w), n_o=wauc(h1.n_o, h1.y, w))
# leave-one-paper-out for the largest paper
big = cnt.index[0]; x = h1[h1.paper != big]
D["without_largest_paper"] = dict(n=len(x), papers=int(x.paper.nunique()), S_L=auc(x.S_L, x.y), S_S=auc(x.S_S, x.y), z=auc(x.z, x.y), n_o=auc(x.n_o, x.y))
# Brier on complete rows (corrects NaN in the secondary analysis)
c = h1.dropna(subset=["S_L", "S_S", "S_F"])
D["brier_HP1_complete"] = dict(n=len(c), S_L=float(((c.S_L - c.y) ** 2).mean()), S_S=float(((c.S_S - c.y) ** 2).mean()), S_F=float(((c.S_F - c.y) ** 2).mean()), base=float(((c.y.mean() - c.y) ** 2).mean()))
D["S_L_by_prev_reports"] = h1.groupby("n_prev_reports").agg(n=("y", "size"), success=("y", "mean"), S_L=("S_L", "mean")).reset_index().round(3).to_dict("records")
json.dump(D, open("fred_diagnostics.json", "w"), indent=1, default=float); print(json.dumps(D, indent=1, default=float))
