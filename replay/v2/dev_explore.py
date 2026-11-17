"""Development-half exploration (FReD v2). Everything here is exploratory and uses only the development half."""
import sys, json; sys.path.insert(0, ".")
from common import *
d = load("development")
out = {}
# empirical prior SD of true effects (Fisher z) from replication effects
rr = d.dropna(subset=["r_r", "n_r"]); fz = np.arctanh(rr.r_r.clip(upper=0.999))
tau2 = max(fz.var() - (1 / (rr.n_r - 3)).mean(), 1e-4); out["tau_fisher_z_dev"] = float(np.sqrt(tau2)); out["n_for_tau"] = len(rr)
d["z"] = d.z_es.fillna(d.z_p); d["z_src_es"] = d.z_es.notna()
e = d[d.y.notna() & d.z.notna() & d.n_o.notna()].copy(); e["fail"] = 1 - e.y
out["n_rows"] = len(e); out["n_papers"] = int(e.paper.nunique()); out["fail_rate"] = float(e.fail.mean())
for tau in (0.1, 0.15, out["tau_fisher_z_dev"], 0.3, 0.5):
    k = tau ** 2 * (e.n_o - 3) / (tau ** 2 * (e.n_o - 3) + 1)
    e[f"post_z_{tau:.3f}"] = np.sqrt(k) * e.z
    e[f"shr_z_{tau:.3f}"] = k * e.z
res = {c: auc(-e[c], e.fail) for c in e.columns if c.startswith(("post_z", "shr_z"))}
res["-z"] = auc(-e.z, e.fail); res["-z_p"] = auc(-e.z_p, e.fail); res["-n_o"] = auc(-e.n_o, e.fail)
es = e[e.z_src_es]; res["es_subset_n"] = len(es); res["es_subset_-z_es"] = auc(-es.z_es, es.fail); res["es_subset_-r_o"] = auc(-es.r_o, es.fail)
res["es_subset_-post"] = auc(-es[f"post_z_{out['tau_fisher_z_dev']:.3f}"], es.fail)
out["auc_failure"] = res
# replication design: ratio of sample sizes and absolute replication size
e["log_ratio"] = np.log(e.n_r / e.n_o); e["log_nr"] = np.log(e.n_r)
out["auc_fail_log_ratio"] = auc(e.log_ratio, e.fail); out["auc_fail_log_nr"] = auc(e.log_nr, e.fail)
# alarm within tertiles of replication/original size ratio
tc = f"post_z_{out['tau_fisher_z_dev']:.3f}"
e["rbin"] = pd.qcut(e.log_ratio.rank(method="first"), 3, labels=False)
out["auc_alarm_within_ratio_tertiles"] = [round(auc(-g[tc], g.fail), 3) for _, g in e.dropna(subset=["log_ratio"]).groupby("rbin")]
out["auc_-z_within_ratio_tertiles"] = [round(auc(-g.z, g.fail), 3) for _, g in e.dropna(subset=["log_ratio"]).groupby("rbin")]
# logistic: failure ~ alarm + log_ratio (standardized), in-sample coefficients
from sklearn.linear_model import LogisticRegression
f = e.dropna(subset=["log_ratio"]); X = np.c_[(-f[tc] - (-f[tc]).mean()) / f[tc].std(), (f.log_ratio - f.log_ratio.mean()) / f.log_ratio.std()]
m = LogisticRegression().fit(X, f.fail); out["logit_fail_std_coefs_alarm_logratio"] = m.coef_.round(3).tolist()
# H-P2 premise with controls (development)
g = d[d.y.notna() & d.overlap.notna()].copy(); g["ov"] = (g.overlap > 0).astype(float)
out["HP2_dev_raw"] = dict(n=len(g), rate_ov=float(g[g.ov == 1].y.mean()), rate_no=float(g[g.ov == 0].y.mean()))
h = g.dropna(subset=["n_r", "n_o"]).copy(); h["log_ratio"] = np.log(h.n_r / h.n_o); h["log_nr"] = np.log(h.n_r)
Xh = np.c_[h.ov, (h.log_ratio - h.log_ratio.mean()) / h.log_ratio.std()]
m2 = LogisticRegression(C=1e6).fit(Xh, h.y); out["HP2_dev_logit_overlap_logratio"] = m2.coef_.round(3).tolist()
out["HP2_dev_overlap_by_ratio_tertile"] = [dict(n=len(t), ov=float(t[t.ov == 1].y.mean()) if (t.ov == 1).any() else None, no=float(t[t.ov == 0].y.mean())) for _, t in h.assign(b=pd.qcut(h.log_ratio.rank(method="first"), 3, labels=False)).groupby("b")]
json.dump(out, open("dev_results.json", "w"), indent=1, default=float); print(json.dumps(out, indent=1, default=lambda v: round(float(v), 3)))
