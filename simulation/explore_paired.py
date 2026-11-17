"""Paired re-analysis of the main exploratory contrasts (revision)."""
import pickle, numpy as np, json
from scipy.stats import wilcoxon
M = pickle.load(open("main_results.pkl", "rb")); E = pickle.load(open("explore_results.pkl", "rb")); E6 = pickle.load(open("e6_results.pkl", "rb"))
rng = np.random.default_rng(11)
def g(D, sc, k, key="final_regret"): return np.array([r[key] for r in D[(sc, k)]], float)
def ci(d):
    bs = [rng.choice(d, len(d)).mean() for _ in range(10000)]; return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
def p2(a, b): return float(wilcoxon(a, b).pvalue) if np.any(a != b) else 1.0
S, H = "stationary", "shift"
rows = {}
def add(name, a, b):
    d = a - b; rows[name] = dict(a=float(a.mean()), b=float(b.mean()), diff=ci(d), p=p2(a, b))
add("NA2 vs DOC, stationary regret", g(M, S, "NA2"), g(M, S, "DOC"))
add("NA2 vs DOC, stationary error", np.array([r["error"][-1] for r in M[(S, "NA2")]]), np.array([r["error"][-1] for r in M[(S, "DOC")]]))
add("NA2 legacy start vs NA2", g(E, S, "E2:NA2-legacy-init-only"), g(M, S, "NA2"))
add("ADA vs ADA no attractor, stationary", g(M, S, "ADA"), g(E, S, "E2:ADA-no-attractor(phi3=0)"))
add("ADA vs ADA no attractor, shift", g(M, H, "ADA"), g(E, H, "E2:ADA-no-attractor(phi3=0)"))
add("ring vs global (NA2 balanced)", g(E, S, "E5:NA2 ring balanced"), g(M, S, "NA2"))
add("NA2 vs DOC, shift", g(M, H, "NA2"), g(M, H, "DOC"))
add("trigger+decay vs NA2, shift", g(E6, H, "E6:trigger+tau_mem25 NA2"), g(M, H, "NA2"))
add("trigger+decay vs NA2, stationary", g(E6, S, "E6:trigger+tau_mem25 NA2"), g(M, S, "NA2"))
div = lambda D, sc, k: np.array([r["diversity"][29] for r in D[(sc, k)]])
add("diversity at t=30, legacy start vs random", div(E, S, "E2:NA2-legacy-init-only"), div(M, S, "NA2"))
fc = lambda D, sc, k: int(sum(r["false_canon"] for r in D[(sc, k)]))
for name, x in rows.items():
    print(f"{name:42s} {x['a']:.3f} vs {x['b']:.3f}  diff {x['diff'][0]:.3f} [{x['diff'][1]:.3f}, {x['diff'][2]:.3f}]  p2={x['p']:.2g}")
print("false canon shift NA2", fc(M, H, "NA2"), "trigger+decay", fc(E6, H, "E6:trigger+tau_mem25 NA2"), "DOC stat", sum(r["canon_art"][-1] for r in M[(S,"DOC")]))
print("ADA no-attractor means", g(E,S,"E2:ADA-no-attractor(phi3=0)").mean(), g(E,H,"E2:ADA-no-attractor(phi3=0)").mean())
print("E6 keys", sorted({k[1] for k in E6})); 
for k in sorted({k[1] for k in E6}): print(k, g(E6,S,k).mean().round(3), g(E6,H,k).mean().round(3), g(E6,H,k,"post_shift_regret").mean().round(3))
for k in ["E4:NA2 tau_mem=25.0","E4:NA2 tau_mem=10.0","E4:NA2 tau_mem=50.0"]: print(k, g(E,S,k).mean().round(3), g(E,H,k).mean().round(3))
json.dump(rows, open("explore_paired.json", "w"), indent=1)
