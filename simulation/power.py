"""Sensitivity (minimum detectable effect) analysis for n = 30 seeds (revision)."""
import numpy as np, pickle
from scipy.stats import fisher_exact, wilcoxon
rng = np.random.default_rng(3)
M = pickle.load(open("main_results.pkl", "rb"))
# Fisher one-sided, baseline 2/30
for p1 in (0.20, 0.25, 0.30, 0.35, 0.40):
    hits = 0
    for _ in range(2000):
        a = rng.binomial(30, 0.067); b = rng.binomial(30, p1)
        hits += fisher_exact([[a, 30 - a], [b, 30 - b]], alternative="less").pvalue < 0.05
    print(f"Fisher power, 0.067 vs {p1}: {hits/2000:.2f}")
# paired signed-rank: dz for 80% power (normal differences)
for dz in (0.3, 0.4, 0.5, 0.55, 0.6):
    hits = sum(wilcoxon(rng.normal(dz, 1, 30)).pvalue < 0.05 for _ in range(2000)); print(f"signed-rank power two-sided dz={dz}: {hits/2000:.2f}")
d = np.array([r["final_regret"] for r in M[("stationary", "NA2")]]) - np.array([r["final_regret"] for r in M[("stationary", "NA0")]])
e = np.array([r["early_regret"] for r in M[("stationary", "ADA")]]) - np.array([r["early_regret"] for r in M[("stationary", "NA2")]])
print("SD paired diff H1 final regret", d.std(ddof=1).round(3), "H2a early", e.std(ddof=1).round(3))
