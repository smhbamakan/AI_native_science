"""Exploratory design checks after the confirmatory pilot (not pre-registered)."""
import json, numpy as np, pandas as pd, pyreadr
from scipy.stats import spearmanr, norm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
h3 = pd.read_csv("fred_HP3_rows.csv")
df = list(pyreadr.read_r("fred_snapshot_data.RDS").values())[0]
def auc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, float); ok = np.isfinite(s) & np.isfinite(y); s, y = s[ok], y[ok]
    P, N = (y == 1).sum(), (y == 0).sum(); r = pd.Series(s).rank().values
    return (r[y == 1].sum() - P * (P + 1) / 2) / (P * N)
C = {}
h3["fail"] = 1 - h3.y
C["spearman_D_z"] = spearmanr(h3.D, h3.z).statistic; C["spearman_D_n"] = spearmanr(h3.D, h3.n_o).statistic
C["auc_fail_negS_S"] = auc(-h3.S_S, h3.fail); C["auc_fail_negS_F"] = auc(-h3.S_F, h3.fail)
# D within z strata: does divergence help once z is fixed?
h3["zbin"] = pd.qcut(h3.z.rank(method="first"), 4, labels=False)
C["auc_D_within_z_quartiles"] = [round(auc(g.D, g.fail), 3) for _, g in h3.groupby("zbin")]
C["auc_negn_within_z_quartiles"] = [round(auc(-g.n_o, g.fail), 3) for _, g in h3.groupby("zbin")]
# ties from p-value bounds
p = pd.to_numeric(df.pval_value_o, errors="coerce")
C["share_p_reported_as_bound"] = float(df.pval_type_o.isin(["l (<)", "le (<=)"]).mean()); C["share_p_equal_0.001"] = float((p == 0.001).mean())
C["distinct_z_values_in_HP3"] = int(h3.z.nunique()); C["rows_HP3"] = len(h3)
# outcome depends on replication power: replication sample size
d = df.copy(); d["y"] = d.reported_success.map({"successful": 1, "failed": 0}); d["n_r"] = pd.to_numeric(d.n_r, errors="coerce"); d["n_o"] = pd.to_numeric(d.n_o, errors="coerce")
d = d[d.y.notna()]
C["auc_success_n_r"] = auc(d.n_r, d.y); C["auc_success_ratio_nr_no"] = auc(d.n_r / d.n_o, d.y)
# CV logistic anomaly: in-sample vs grouped CV, raw vs log features
X = np.c_[np.log(h3.z.clip(lower=1e-3)), np.log(h3.n_o)]; y = h3.fail.values
m = LogisticRegression().fit(X, y); C["logit_in_sample_auc"] = auc(m.predict_proba(X)[:, 1], y); C["logit_coefs_logz_logn"] = m.coef_.round(3).tolist()
pred = np.zeros(len(y))
for tr, te in GroupKFold(10).split(X, y, h3.paper):
    pred[te] = LogisticRegression().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
C["logit_groupcv_auc_rerun"] = auc(pred, y)
# H-P1 power: approximate SE of AUC difference from the bootstrap interval width
R = json.load(open("fred_results.json")); w = R["H-P1"]["diff"]["hi95"] - R["H-P1"]["diff"]["lo95"]
C["HP1_CI_width"] = w; C["HP1_min_detectable_diff_approx"] = round(2.5 * w / 3.92, 3)
# H-P1 max-of-three baseline selection: expected optimism of the max under the null
C["note"] = "max over three correlated baselines inside each resample biases the baseline upward (conservative test)"
json.dump(C, open("fred_design_check.json", "w"), indent=1, default=float); print(json.dumps(C, indent=1, default=float))
