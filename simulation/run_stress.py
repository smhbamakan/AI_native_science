"""Stress tests added in revision (exploratory, not pre-registered).

Two failure modes that the confirmatory testbed assumed away:
  * concealed dependence: with probability p_alias an observation is filed under an unlinked alias identity of its
    agent, so dependent evidence is counted as independent;
  * claim-identity error: with probability p_misid an observation is filed under a neighbouring claim.
Outcomes are compared with the error-free run of the same seed (paired). Writes stress_results.json.
"""
import json
import numpy as np
from multiprocessing import Pool
from scipy.stats import wilcoxon
from aepso import run, lens_divergence_auc

SEEDS = list(range(1000, 1030))
LEVELS = (0.1, 0.25, 0.5)
SETTINGS = [("none", {})] + [(f"alias={a}", dict(p_alias=a)) for a in LEVELS] + [(f"misid={m}", dict(p_misid=m)) for m in LEVELS]
CONDS = ("NA2", "ADA")


def work(job):
    cond, tag, kw, seed = job
    r = run(cond, "stationary", seed, **kw)
    return cond, tag, seed, dict(final_regret=float(r["final_regret"]), false_canon=float(r["false_canon"]),
                                 error=float(r["error"][-1]), auc=float(lens_divergence_auc(r)))


def boot_ci(x, B=10000, seed=1):
    rng = np.random.default_rng(seed); x = np.asarray(x)
    m = rng.choice(x, size=(B, len(x))).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


if __name__ == "__main__":
    jobs = [(c, t, kw, s) for c in CONDS for t, kw in SETTINGS for s in SEEDS]
    with Pool() as pool:
        res = pool.map(work, jobs, chunksize=6)
    R = {}
    for c, t, s, d in res:
        R.setdefault(c, {}).setdefault(t, {})[s] = d
    out = {}
    for c in CONDS:
        base = R[c]["none"]
        for t, _ in SETTINGS:
            cur = R[c][t]; row = {}
            for k in ("final_regret", "error", "auc"):
                a = np.array([cur[s][k] for s in SEEDS]); b = np.array([base[s][k] for s in SEEDS])
                ok = np.isfinite(a) & np.isfinite(b); diff = a[ok] - b[ok]
                row[k] = dict(mean=float(np.nanmean(a)), median=float(np.nanmedian(a)))
                if t != "none":
                    p = float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
                    row[k].update(paired_diff=float(diff.mean()), ci=boot_ci(diff), p_paired=p)
            row["false_canon"] = int(sum(cur[s]["false_canon"] for s in SEEDS))
            out[f"{c} {t}"] = row
    json.dump(out, open("stress_results.json", "w"), indent=1)
    for k, v in out.items():
        print(f"{k:16s} regret {v['final_regret']['mean']:.3f} | error {v['error']['mean']:.3f} | AUC {v['auc']['median']:.2f} | false {v['false_canon']}/30",
              "" if "paired_diff" not in v["error"] else f"| d_err {v['error']['paired_diff']:+.3f} p={v['error']['p_paired']:.3g} | d_reg {v['final_regret']['paired_diff']:+.3f} p={v['final_regret']['p_paired']:.3g}")
