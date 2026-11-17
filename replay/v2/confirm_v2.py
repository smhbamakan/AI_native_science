"""Confirmatory v2 analysis (PREREGISTRATION_FReD_v2.md). Usage: python3 -I confirm_v2.py [development|holdout]"""
import sys, json; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from common import *
from sklearn.linear_model import LogisticRegression

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "development"
TAU = 0.264
d = load(SPLIT)
d["z"] = d.z_es.fillna(d.z_p)
k = TAU ** 2 * (d.n_o - 3) / (TAU ** 2 * (d.n_o - 3) + 1)
d["A"] = -np.sqrt(k) * d.z
d["log_ratio"] = np.log(d.n_r / d.n_o)
d["ov"] = (d.overlap > 0).astype(float).where(d.overlap.notna())
R = {"split": SPLIT}
std = lambda s: (s - s.mean()) / s.std()


def coef(x, ycol, cols):
    X = np.c_[[std(x[c]) if c != "ov" else x[c] for c in cols]].T
    if len(np.unique(x[ycol])) < 2: return np.nan
    return LogisticRegression(C=1e6, max_iter=1000).fit(X, x[ycol]).coef_[0][0]


def summ(est, bs, q=(2.5, 97.5)):
    bs = bs[np.isfinite(bs)]
    return dict(est=float(est), lo=float(np.percentile(bs, q[0])), hi=float(np.percentile(bs, q[1])), p5=float(np.percentile(bs, 5)),
                p95=float(np.percentile(bs, 95)), p_one=float((bs <= 0).mean()))


# T1
t1 = d[d.y.notna() & d.A.notna() & d.log_ratio.notna()].reset_index(drop=True); t1["fail"] = 1 - t1.y
bs = cluster_boot(t1, "paper", lambda x: coef(x, "fail", ["A", "log_ratio"]), seed=11)
R["T1"] = dict(n=len(t1), papers=int(t1.paper.nunique()), fail_rate=float(t1.fail.mean()), coef=summ(coef(t1, "fail", ["A", "log_ratio"]), bs))
bsa = cluster_boot(t1, "paper", lambda x: auc(x.A, x.fail), seed=12)
R["T1"]["auc_A"] = summ(auc(t1.A, t1.fail), bsa)
R["T1"]["supported"] = bool(R["T1"]["coef"]["p5"] > 0)
# T2
t2 = d[d.y.notna() & d.A.notna()].reset_index(drop=True); t2["fail"] = 1 - t2.y
st2 = lambda x: auc(x.A, x.fail) - auc(-x.z, x.fail)
bs = cluster_boot(t2, "paper", st2, seed=13)
R["T2"] = dict(n=len(t2), papers=int(t2.paper.nunique()), auc_A=auc(t2.A, t2.fail), auc_negz=auc(-t2.z, t2.fail), diff=summ(st2(t2), bs, (5, 95)))
R["T2"]["equivalent_within_0.03"] = bool(R["T2"]["diff"]["lo"] > -0.03 and R["T2"]["diff"]["hi"] < 0.03)
# T3
t3 = d[d.y.notna() & d.ov.notna() & d.log_ratio.notna()].reset_index(drop=True)
bs = cluster_boot(t3, "report", lambda x: coef(x, "y", ["ov", "log_ratio"]), seed=14)
R["T3"] = dict(n=len(t3), reports=int(t3.report.nunique()), n_overlap=int(t3.ov.sum()), rate_ov=float(t3[t3.ov == 1].y.mean()),
               rate_no=float(t3[t3.ov == 0].y.mean()), coef=summ(coef(t3, "y", ["ov", "log_ratio"]), bs))
R["T3"]["odds_ratio"] = float(np.exp(R["T3"]["coef"]["est"]))
R["T3"]["supported"] = bool(R["T3"]["coef"]["p5"] > 0)
ps = {"T1": R["T1"]["coef"]["p_one"], "T3": R["T3"]["coef"]["p_one"]}
o = sorted(ps, key=ps.get); R["holm"] = {o[0]: min(1, 2 * ps[o[0]])}; R["holm"][o[1]] = max(R["holm"][o[0]], ps[o[1]])
# sensitivity: mixed as failure
dm = d.copy(); dm["y"] = dm["y_mixed0"]
s1 = dm[dm.y.notna() & dm.A.notna() & dm.log_ratio.notna()].assign(fail=lambda x: 1 - x.y)
s3 = dm[dm.y.notna() & dm.ov.notna() & dm.log_ratio.notna()]
R["sens_mixed_as_failure"] = dict(T1_coef=coef(s1, "fail", ["A", "log_ratio"]), T1_auc=auc(s1.A, s1.fail), T3_coef=coef(s3, "y", ["ov", "log_ratio"]))
# descriptive: attestation-log lens (v1 H-P1 definition) on this split
dd = d[d.year_r.notna()].copy(); dd["vote"] = dd.reported_success.map({"successful": 1.0, "failed": 0.0, "mixed": 0.5, "statistically successful but flawed": 0.5})
rep = dd[dd.vote.notna()].groupby(["paper", "report"]).agg(year=("year_r", "min"), v=("vote", "mean")).reset_index()
by = {k_: g for k_, g in rep.groupby("paper")}
SL = []
for k_, yr, r in zip(d.paper, d.year_r, d.report):
    g = by.get(k_)
    if g is None or pd.isna(yr): SL.append(np.nan); continue
    g = g[(g.year < yr) & (g.report != r)]
    SL.append((2 * 0.5 + 0.5 + g.v.sum()) / (2.5 + len(g)) if len(g) else np.nan)
d["S_L"] = SL
h = d[d.y.notna() & d.S_L.notna()].reset_index(drop=True)
if len(h) > 10 and h.y.nunique() == 2:
    bsh = cluster_boot(h, "paper", lambda x: auc(x.S_L, x.y), B=2000, seed=15)
    R["descriptive_log_lens"] = dict(n=len(h), papers=int(h.paper.nunique()), auc=summ(auc(h.S_L, h.y), bsh))
else:
    R["descriptive_log_lens"] = dict(n=len(h), papers=int(h.paper.nunique()) if len(h) else 0, auc=None)
json.dump(R, open(f"results_v2_{SPLIT}.json", "w"), indent=1, default=float)
print(json.dumps(R, indent=1, default=lambda v: round(float(v), 4)))
