import numpy as np, pickle, itertools
from multiprocessing import Pool
from aepso import run
SEEDS = list(range(1000, 1030))
jobs = []
def add(tag, cond, **kw):
    for sc in ("stationary", "shift"):
        for s in SEEDS:
            jobs.append((tag, cond, sc, s, kw))
# E2 decomposition of absorption effects
add("E2:ADA-attractor-only(random init)", "ADA", init="random")
add("E2:ADA-no-attractor(phi3=0)", "ADA", phi3=0.0)
add("E2:NA2-legacy-init-only", "NA2", init="legacy")
# E3 alpha sensitivity (one at a time around ADA defaults)
for k in (0.1, 0.5): add(f"E3:k={k}", "ADA", k_radius=k)
for t in (10.0, 25.0, 100.0, float("inf")): add(f"E3:tau={t}", "ADA", tau=t)
add("E3:layer=L1", "ADA", layer="L1")
for th in (1, 4): add(f"E3:theta={th}", "ADA", theta=th)
for kp in (1.0, 10.0): add(f"E3:kappa={kp}", "ADA", kappa=kp)
for a in (0.1, 0.25, 0.75, 1.0): add(f"E3:FIX a0={a}", "FIX", a0_fixed=a)
# E4 integrated memory decay (tau applied to all evidence)
for tm in (50.0, 25.0, 10.0):
    add(f"E4:NA2 tau_mem={tm}", "NA2", tau_mem=tm)
    add(f"E4:ADA tau_mem={tm}", "ADA", tau_mem=tm)
# E5 social structure
for topo in ("global", "ring"):
    for mix in ("balanced", "maverick", "follower"):
        for c in ("NA2", "ADA"):
            add(f"E5:{c} {topo} {mix}", c, topology=topo, mix=mix)
def work(j):
    tag, c, sc, s, kw = j
    r = run(c, sc, s, **kw)
    return tag, sc, dict(final_regret=r["final_regret"], false_canon=r["false_canon"], early_regret=r["early_regret"],
                         post_shift_regret=r["post_shift_regret"], diversity=r["diversity"], regret=r["regret"],
                         err_final=r["error"][-1])
if __name__ == "__main__":
    with Pool() as pool:
        res = pool.map(work, jobs, chunksize=8)
    out = {}
    for tag, sc, d in res:
        out.setdefault((sc, tag), []).append(d)
    pickle.dump(out, open("explore_results.pkl", "wb"))
    print(len(jobs), "runs done")
