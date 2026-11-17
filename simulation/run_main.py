import numpy as np, pickle, time
from aepso import run, lens_divergence_auc, credit_validity
CONDS = ["DOC", "NA0", "NA2", "EAG", "FIX", "ADA"]
SEEDS = range(1000, 1030)
out = {}
t0 = time.time()
for sc in ("stationary", "shift"):
    for c in CONDS:
        rows = []
        for s in SEEDS:
            r = run(c, sc, s)
            row = {k: r[k] for k in ("final_regret", "false_canon", "early_regret", "post_shift_regret")}
            row.update(regret=r["regret"], error=r["error"], diversity=r["diversity"], a0=r["a0"], canon_art=r["canon_art"])
            if c == "ADA" and sc == "stationary":
                row["auc"] = lens_divergence_auc(r)
                cv = credit_validity(r, n_perm=100, seed=s)
                row.update(rho_kdp=cv["rho_kdp"], rho_att=cv["rho_att"])
            rows.append(row)
        out[(sc, c)] = rows
        print(sc, c, "done", round(time.time() - t0), "s", flush=True)
pickle.dump(out, open("main_results.pkl", "wb"))
