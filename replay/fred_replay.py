"""FReD pilot: replaying replication records through KDP lenses. Implements PREREGISTRATION_FReD_pilot.md v1.0.
Run: python3 -I fred_replay.py  (needs pyreadr, numpy, pandas, scipy)."""
import re, json, sys
import numpy as np, pandas as pd, pyreadr
from scipy.stats import norm

rng = np.random.default_rng(20261116)
B = 10000
df = list(pyreadr.read_r("fred_snapshot_data.RDS").values())[0].copy()
nz = lambda s: re.sub(r"\W+", " ", str(s).lower()).strip()

# ---------------- coding (section 3)
OUT = {"successful": 1.0, "failed": 0.0}
VOTE = {"successful": 1.0, "failed": 0.0, "mixed": 0.5, "statistically successful but flawed": 0.5}
df["y"] = df["reported_success"].map(OUT)
df["y_mixed0"] = df["reported_success"].map({**OUT, "mixed": 0.0})
df["vote"] = df["reported_success"].map(VOTE)
df["paper"] = df["doi_o"].str.lower().str.strip()
df["claim_desc"] = df["paper"] + "|" + df["description"].map(nz)
df["report"] = df["ref_r"].map(lambda s: nz(s)[:60])
df["year_r"] = pd.to_numeric(df["year_r"], errors="coerce")
df["n_o"] = pd.to_numeric(df["n_o"], errors="coerce"); df.loc[df.n_o <= 3, "n_o"] = np.nan
p = pd.to_numeric(df["pval_value_o"], errors="coerce").clip(1e-12, 0.999)
df["z"] = norm.ppf(1 - p / 2)
df["overlap"] = pd.to_numeric(df["author_overlap"], errors="coerce")

logistic = lambda x: 1 / (1 + np.exp(-x))


def lenses(d, tau=0.2):
    d = d.copy()
    d["S_F"] = logistic(d.z / 1.96)
    d["z_S"] = d.z * tau ** 2 * d.n_o / (tau ** 2 * d.n_o + 1)
    d["S_S"] = logistic(d.z_S / 1.96)
    d["D"] = d.S_F - d.S_S
    return d


def log_lens(d, key="paper", mu0=0.5, n0=2.0):
    """S_L for each row from earlier replication reports of the same claim (strictly earlier year, other report)."""
    S = np.full(len(d), np.nan); nprev = np.zeros(len(d))
    dd = d[d.year_r.notna() & d.vote.notna()]
    rep = dd.groupby([key, "report"]).agg(year=("year_r", "min"), v=("vote", "mean")).reset_index()
    by = {k: g for k, g in rep.groupby(key)}
    for i, (k, yr, r) in enumerate(zip(d[key], d.year_r, d.report)):
        if pd.isna(yr) or k not in by: continue
        g = by[k]; g = g[(g.year < yr) & (g.report != r)]
        if len(g) == 0: continue
        S[i] = (n0 * mu0 + 0.5 * 1.0 + g.v.sum()) / (n0 + 0.5 + len(g)); nprev[i] = len(g)
    d = d.copy(); d["S_L"] = S; d["n_prev_reports"] = nprev
    return d


def auc(score, y):
    s = np.asarray(score, float); y = np.asarray(y, float)
    ok = np.isfinite(s) & np.isfinite(y); s, y = s[ok], y[ok]
    P, N = (y == 1).sum(), (y == 0).sum()
    if P == 0 or N == 0: return np.nan
    r = pd.Series(s).rank().values
    return (r[y == 1].sum() - P * (P + 1) / 2) / (P * N)


def cluster_boot(d, clusters, stat):
    groups = [g.index.values for _, g in d.groupby(clusters)]
    vals = []
    for _ in range(B):
        idx = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        vals.append(stat(d.loc[idx]))
    return np.array(vals)


def summary(est, bs):
    bs = bs[np.isfinite(bs)]
    return dict(est=float(est), lo95=float(np.percentile(bs, 2.5)), hi95=float(np.percentile(bs, 97.5)),
                p5=float(np.percentile(bs, 5)), p_one=float((bs <= 0).mean()), n_boot=int(len(bs)))


R = {}
d0 = lenses(df)
d0 = log_lens(d0)

# ---------------- H-P1
h1 = d0[d0.y.notna() & d0.S_L.notna()].reset_index(drop=True)
def stat_h1(x):
    base = max(auc(x.S_S, x.y), auc(x.z, x.y), auc(x.n_o, x.y))
    return auc(x.S_L, x.y) - base
bs = cluster_boot(h1, "paper", stat_h1)
R["H-P1"] = dict(n_rows=len(h1), n_papers=int(h1.paper.nunique()), success_rate=float(h1.y.mean()),
                 auc_S_L=auc(h1.S_L, h1.y), auc_S_S=auc(h1.S_S, h1.y), auc_z=auc(h1.z, h1.y), auc_n=auc(h1.n_o, h1.y),
                 diff=summary(stat_h1(h1), bs))

# ---------------- H-P2
h2 = d0[d0.y.notna() & d0.overlap.notna()].reset_index(drop=True)
stat_h2 = lambda x: x[x.overlap > 0].y.mean() - x[x.overlap == 0].y.mean()
bs = cluster_boot(h2, "report", stat_h2)
R["H-P2"] = dict(n_rows=len(h2), n_reports=int(h2.report.nunique()), n_overlap=int((h2.overlap > 0).sum()),
                 rate_overlap=float(h2[h2.overlap > 0].y.mean()), rate_none=float(h2[h2.overlap == 0].y.mean()),
                 diff=summary(stat_h2(h2), bs))

# ---------------- H-P3
h3 = d0[d0.y.notna() & d0.D.notna()].reset_index(drop=True)
h3["fail"] = 1 - h3.y
def stat_h3(x):
    return auc(x.D, x.fail) - max(auc(-x.z, x.fail), auc(-x.n_o, x.fail))
bs = cluster_boot(h3, "paper", stat_h3)
bsA = cluster_boot(h3, "paper", lambda x: auc(x.D, x.fail) - 0.5)
R["H-P3"] = dict(n_rows=len(h3), n_papers=int(h3.paper.nunique()), fail_rate=float(h3.fail.mean()),
                 auc_D=auc(h3.D, h3.fail), auc_negz=auc(-h3.z, h3.fail), auc_negn=auc(-h3.n_o, h3.fail),
                 diff=summary(stat_h3(h3), bs), auc_vs_chance=summary(auc(h3.D, h3.fail) - 0.5, bsA))

# Holm over the three one-sided bootstrap P values
ps = {"H-P1": R["H-P1"]["diff"]["p_one"], "H-P2": R["H-P2"]["diff"]["p_one"], "H-P3": max(R["H-P3"]["diff"]["p_one"], R["H-P3"]["auc_vs_chance"]["p_one"])}
order = sorted(ps, key=ps.get); run = 0; holm = {}
for i, k in enumerate(order):
    run = max(run, min(1, (len(ps) - i) * ps[k])); holm[k] = run
R["holm"] = holm
for k in ("H-P1", "H-P2"):
    R[k]["supported"] = bool(R[k]["diff"]["p5"] > 0)
R["H-P3"]["supported"] = bool(R["H-P3"]["diff"]["p5"] > 0 and R["H-P3"]["auc_vs_chance"]["p5"] > 0)

# ---------------- secondary / exploratory
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
X = np.c_[np.log(h3.z.clip(lower=1e-3)), np.log(h3.n_o)]
pred = np.zeros(len(h3))
for tr, te in GroupKFold(10).split(X, h3.fail, h3.paper):
    m = LogisticRegression().fit(X[tr], h3.fail.values[tr]); pred[te] = m.predict_proba(X[te])[:, 1]
h3["cv"] = pred
bs = cluster_boot(h3, "paper", lambda x: auc(x.D, x.fail) - auc(x.cv, x.fail))
R["sec_H-P3_vs_cv_logistic"] = dict(auc_cv=auc(h3.cv, h3.fail), diff=summary(auc(h3.D, h3.fail) - auc(h3.cv, h3.fail), bs))

brier = lambda s, y: float(np.mean((np.asarray(s) - np.asarray(y)) ** 2))
R["sec_brier_HP1_sample"] = {k: brier(h1[k], h1.y) for k in ("S_L", "S_S", "S_F")}
full = d0[d0.y.notna() & d0.S_S.notna()]
R["sec_brier_full"] = {k: brier(full[k], full.y) for k in ("S_S", "S_F")} | {"base_rate": float(full.y.mean()), "brier_base_rate": brier(np.full(len(full), full.y.mean()), full.y)}
cal = full.assign(bin=pd.qcut(full.S_S, 5, duplicates="drop")).groupby("bin", observed=True).agg(mean_S_S=("S_S", "mean"), obs=("y", "mean"), n=("y", "size"))
R["sec_calibration_S_S"] = cal.reset_index(drop=True).round(3).to_dict("records")
calL = h1.assign(bin=pd.qcut(h1.S_L, 4, duplicates="drop")).groupby("bin", observed=True).agg(mean_S_L=("S_L", "mean"), obs=("y", "mean"), n=("y", "size"))
R["sec_calibration_S_L"] = calL.reset_index(drop=True).round(3).to_dict("records")

# sensitivity
sens = {}
dm = d0.copy(); dm["y"] = dm["y_mixed0"]
x1 = dm[dm.y.notna() & dm.S_L.notna()]; sens["HP1_mixed_as_failure"] = dict(n=len(x1), diff=stat_h1(x1), auc_S_L=auc(x1.S_L, x1.y))
x2 = dm[dm.y.notna() & dm.overlap.notna()]; sens["HP2_mixed_as_failure"] = dict(n=len(x2), diff=stat_h2(x2))
x3 = dm[dm.y.notna() & dm.D.notna()].assign(fail=lambda t: 1 - t.y); sens["HP3_mixed_as_failure"] = dict(n=len(x3), diff=stat_h3(x3), auc_D=auc(x3.D, x3.fail))
for tau in (0.1, 0.3):
    dt = lenses(df, tau); dt = log_lens(dt)
    a = dt[dt.y.notna() & dt.D.notna()].assign(fail=lambda t: 1 - t.y)
    sens[f"HP3_tau_{tau}"] = dict(auc_D=auc(a.D, a.fail), diff=stat_h3(a))
dk = log_lens(lenses(df), key="claim_desc"); x = dk[dk.y.notna() & dk.S_L.notna()]
sens["HP1_claim_by_description"] = dict(n=len(x), papers=int(x.paper.nunique()), auc_S_L=auc(x.S_L, x.y), diff=stat_h1(x) if len(x) > 10 else None)
R["sensitivity"] = sens
allr = d0[d0.y.notna()]
R["context_full_sample"] = dict(n=len(allr), success_rate=float(allr.y.mean()), auc_z=auc(allr.z, allr.y), auc_n=auc(allr.n_o, allr.y), auc_S_S=auc(allr.S_S, allr.y))

json.dump(R, open("fred_results.json", "w"), indent=1, default=float)
h1.to_csv("fred_HP1_rows.csv", index=False); h3[["paper", "report", "y", "z", "n_o", "S_F", "S_S", "D", "cv"]].to_csv("fred_HP3_rows.csv", index=False)
print(json.dumps(R, indent=1, default=lambda v: round(float(v), 4)))

