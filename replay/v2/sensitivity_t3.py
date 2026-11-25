"""Post hoc sensitivity analyses for test T3 (author overlap and replication success), added in revision.

These analyses were NOT pre-registered. They check whether the T3 association depends on sparse-data bias,
on how effects are clustered within replication reports and original papers, or on the coding of mixed
outcomes. Usage: python3 -I sensitivity_t3.py [development|holdout]
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load
import numpy as np
from scipy.stats import norm

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "development"


def wilson(k, n, z=1.959964):
    p = k / n; den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return float(c - h), float(c + h)


def design(d):
    lr = d.log_ratio.values; lr = (lr - lr.mean()) / lr.std()
    return np.c_[np.ones(len(d)), d.ov.values, lr]


def logit_fit(X, y, firth=False, w=None, it=200):
    """Newton-Raphson logistic regression; firth=True adds Jeffreys-prior (Firth) bias reduction."""
    w = np.ones(len(y)) if w is None else w
    b = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-X @ b)); W = w * p * (1 - p)
        I = X.T @ (W[:, None] * X)
        U = X.T @ (w * (y - p))
        if firth:
            Ii = np.linalg.inv(I)
            h = np.einsum("ij,jk,ik->i", X * np.sqrt(W)[:, None], Ii, X * np.sqrt(W)[:, None])
            U = U + X.T @ (w * h * (0.5 - p))
        step = np.linalg.solve(I, U); b = b + step
        if np.max(np.abs(step)) < 1e-10: break
    p = 1 / (1 + np.exp(-X @ b)); W = w * p * (1 - p)
    return b, np.linalg.inv(X.T @ (W[:, None] * X)), p


def cluster_se(X, y, b, bread, groups):
    p = 1 / (1 + np.exp(-X @ b)); sc = X * (y - p)[:, None]
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in np.unique(groups):
        s = sc[groups == g].sum(0); meat += np.outer(s, s)
    G = len(np.unique(groups)); adj = G / (G - 1)
    return np.sqrt(np.diag(adj * bread @ meat @ bread))


def summarise(b, se, j=1):
    est = float(b[j]); lo, hi = est - 1.959964 * se[j], est + 1.959964 * se[j]
    return dict(log_odds=est, ci=(float(lo), float(hi)), odds_ratio=float(np.exp(est)), p_two_sided=float(2 * norm.sf(abs(est / se[j]))))


d = load(SPLIT)
d["log_ratio"] = np.log(d.n_r / d.n_o)
d["ov"] = (d.overlap > 0).astype(float).where(d.overlap.notna())
t = d[d.y.notna() & d.ov.notna() & d.log_ratio.notna()].reset_index(drop=True)
y = t.y.values; X = design(t)
R = {"split": SPLIT, "n_effects": len(t), "n_reports": int(t.report.nunique()), "n_papers": int(t.paper.nunique())}

for lab, m in (("overlap", t.ov == 1), ("independent", t.ov == 0)):
    k, n = int(t.y[m].sum()), int(m.sum())
    R[lab] = dict(successes=k, effects=n, rate=k / n, wilson95=wilson(k, n),
                  reports=int(t.report[m].nunique()), papers=int(t.paper[m].nunique()))

b, V, _ = logit_fit(X, y)
R["ml"] = dict(model_se=summarise(b, np.sqrt(np.diag(V))),
               cluster_report=summarise(b, cluster_se(X, y, b, V, t.report.values)),
               cluster_paper=summarise(b, cluster_se(X, y, b, V, t.paper.values)))
bf, Vf, _ = logit_fit(X, y, firth=True)
R["firth"] = dict(model_se=summarise(bf, np.sqrt(np.diag(Vf))),
                  cluster_report=summarise(bf, cluster_se(X, y, bf, Vf, t.report.values)))
# each replication report counts once: effects weighted by 1 / (effects in their report)
wt = 1.0 / t.groupby("report").report.transform("size").values
bw, Vw, _ = logit_fit(X, y, w=wt * len(t) / wt.sum())
R["report_weighted"] = dict(cluster_report=summarise(bw, cluster_se(X, y, bw, Vw, t.report.values)),
                            rate_overlap=float(np.average(y[t.ov == 1], weights=wt[t.ov == 1])),
                            rate_independent=float(np.average(y[t.ov == 0], weights=wt[t.ov == 0])))
# mixed outcomes coded as failures
dm = d.copy(); dm["y"] = dm["y_mixed0"]
tm = dm[dm.y.notna() & dm.ov.notna() & dm.log_ratio.notna()].reset_index(drop=True)
Xm = design(tm); bm, Vm, _ = logit_fit(Xm, tm.y.values, firth=True)
R["mixed_as_failure_firth"] = dict(n_effects=len(tm), **summarise(bm, cluster_se(Xm, tm.y.values, bm, Vm, tm.report.values)),
                                   rate_overlap=float(tm.y[tm.ov == 1].mean()), rate_independent=float(tm.y[tm.ov == 0].mean()))
# covariate balance: replication design by overlap
R["balance_log_ratio"] = dict(overlap_mean=float(t.log_ratio[t.ov == 1].mean()), independent_mean=float(t.log_ratio[t.ov == 0].mean()))

json.dump(R, open(f"sensitivity_t3_{SPLIT}.json", "w"), indent=1)
print(json.dumps(R, indent=1))
