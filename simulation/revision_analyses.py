"""Additional exploratory analyses: matched document baselines, legacy-bias sensitivity, extended alarm and credit diagnostics."""
import numpy as np, pickle, time
from multiprocessing import Pool
from scipy.stats import spearmanr
from aepso import run, pooled_lens, replication_lens, MU0, NC

SEEDS = list(range(1000, 1030))


def auc(score, lab):
    lab = np.asarray(lab, bool); score = np.asarray(score, float)
    P, N = lab.sum(), (~lab).sum()
    if P == 0 or N == 0: return np.nan
    order = np.argsort(score, kind="mergesort"); ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    # average ranks for ties
    for v in np.unique(score):
        idx = score == v
        if idx.sum() > 1: ranks[idx] = ranks[idx].mean()
    return (ranks[lab].sum() - P * (P + 1) / 2) / (P * N)


def avg_precision(score, lab):
    lab = np.asarray(lab, bool)
    if lab.sum() == 0: return np.nan
    o = np.argsort(-np.asarray(score, float), kind="mergesort"); l = lab[o]
    tp = np.cumsum(l); prec = tp / np.arange(1, len(l) + 1)
    return float((prec * l).sum() / l.sum())


def h3_ext(res):
    st = res["_state"]; kS, kC, leg, p, rec = st["kS"], st["kC"], st["leg"], st["p"], st["rec"]
    T = st["world"].T_cell
    Sa, ca = pooled_lens(kS, kC, leg, include_legacy=True)
    Sr, adm, n_eff, _, _ = replication_lens(kS, kC, st["lw"], leg, p["n0"], st["m"])
    ss = st["kSS"].copy(); mx = st["kMax"].copy()
    for c, y, t, a in rec:
        ss[int(c)] += y * y; mx[int(c)] = max(mx[int(c)], y)
    tot = kS.sum(1) + leg["S"].sum(1)
    var = np.where(ca > 1, ss / np.maximum(ca, 1) - (tot / np.maximum(ca, 1)) ** 2, 0.0)
    nind = (kC > 0).sum(1) + leg["nag"]
    ev = ca > 0
    scores = {"lens divergence": np.abs(Sa - Sr), "all-evidence estimate": Sa, "maximum report": mx,
              "few independent agents": -nind.astype(float), "observation variance": var}
    out = {}
    lab1 = (Sa - T > 0.2)
    sub2 = ev & (Sa >= 0.4); lab2 = T < 0.3
    for name, sc in scores.items():
        out[("L1", name)] = (auc(sc[ev], lab1[ev]), avg_precision(sc[ev], lab1[ev]))
        out[("L2", name)] = (auc(sc[sub2], lab2[sub2]), avg_precision(sc[sub2], lab2[sub2]))
    out["prev"] = dict(L1=(int(lab1[ev].sum()), int(ev.sum())), L2=(int(lab2[sub2].sum()), int(sub2.sum())))
    return out


def h4_ext(res, seed, n_perm=100):
    st = res["_state"]; kS, kC, p = st["kS"], st["kC"], st["p"]
    T = st["world"].T_cell; n = kS.shape[1]; n0 = p["n0"]
    has = kC > 0; am = np.where(has, kS / np.maximum(kC, 1), 0)
    wT = T / T.sum()

    def run_perm(rng):
        acc = {k: np.zeros(n) for k in ("U_err", "U_disc", "U_werr", "K_l1", "K_l2")}
        for _ in range(n_perm):
            order = rng.permutation(n); s = np.zeros(NC); c = np.zeros(NC)
            S = np.full(NC, MU0)
            prev = dict(U_err=-np.abs(S - T).mean(), U_disc=0.1, U_werr=-(wT * np.abs(S - T)).sum(), K_l1=0.0, K_l2=0.0)
            for j in order:
                s += am[:, j]; c += has[:, j]
                S = MU0 + (s - c * MU0) / (c + n0)
                adm = c >= 2
                disc = T[int(np.argmax(np.where(adm, S, -np.inf)))] if adm.any() else 0.1
                cur = dict(U_err=-np.abs(S - T).mean(), U_disc=disc, U_werr=-(wT * np.abs(S - T)).sum(),
                           K_l1=np.abs(S - MU0).mean(), K_l2=((S - MU0) ** 2).mean())
                for k in acc: acc[k][j] += cur[k] - prev[k]
                prev = cur
        return {k: v / n_perm for k, v in acc.items()}

    A = run_perm(np.random.default_rng(seed)); B = run_perm(np.random.default_rng(seed + 10_000))
    first = st["first"]
    cites = np.zeros(n)
    for cc in range(NC):
        j = first[cc]
        if j >= 0: cites[j] += has[cc].sum() - 1
    base = {"attention": st["attention"], "cumulative uses": cites, "claims explored": (kC > 0).sum(0), "replications": st["nrep"]}
    rho = lambda a, b: spearmanr(a, b).statistic
    out = {}
    for U in ("U_err", "U_disc", "U_werr"):
        out[(U, "KDP credit (L1)")] = rho(A["K_l1"], A[U]); out[(U, "KDP credit (L2)")] = rho(A["K_l2"], A[U])
        for bname, b in base.items(): out[(U, bname)] = rho(b, A[U])
    out["mc_stability_K"] = rho(A["K_l1"], B["K_l1"]); out["mc_stability_U"] = rho(A["U_err"], B["U_err"])
    out["mc_rho_diff"] = abs(rho(A["K_l1"], A["U_err"]) - rho(B["K_l1"], B["U_err"]))
    out["neg_true_share"] = float((A["U_err"] < 0).mean())
    return out


def work(job):
    kind, cond, sc, s, kw = job
    t0 = time.time(); r = run(cond, sc, s, **kw); t_run = time.time() - t0
    d = dict(final_regret=r["final_regret"], false_canon=r["false_canon"], early_regret=r["early_regret"],
             post_shift_regret=r["post_shift_regret"], regret_auc=float(np.mean(r["regret"])), t_run=t_run,
             err_final=r["error"][-1], canon_art=bool(r["canon_art"][-1]))
    if kind == "credit":
        d["h3"] = h3_ext(r); t1 = time.time(); d["h4"] = h4_ext(r, s); d["t_shapley"] = time.time() - t1
    return kind, cond, sc, tuple(sorted(kw.items())), d


if __name__ == "__main__":
    jobs = []
    for sc in ("stationary", "shift"):
        for c in ("DOCM", "DOCR"):
            jobs += [("baseline", c, sc, s, {}) for s in SEEDS]
    for kw in (dict(pub_threshold=0.3), dict(pub_threshold=0.7), dict(p_flaw=0.1), dict(p_flaw=0.5)):
        for c in ("DOC", "DOCR", "NA2", "EAG", "ADA"):
            jobs += [("bias", c, "stationary", s, kw) for s in SEEDS]
        jobs += [("bias", "NA2", "stationary", s, dict(kw, init="legacy")) for s in SEEDS]
    for c, sc in (("ADA", "stationary"), ("NA2", "stationary"), ("EAG", "stationary"), ("ADA", "shift"), ("NA2", "shift")):
        jobs += [("credit", c, sc, s, {}) for s in SEEDS]
    t0 = time.time()
    with Pool() as pool:
        res = pool.map(work, jobs, chunksize=4)
    out = {}
    for kind, c, sc, kw, d in res:
        out.setdefault((kind, c, sc, kw), []).append(d)
    pickle.dump(out, open("revision_results.pkl", "wb"))
    print(len(jobs), "runs in", round(time.time() - t0), "s")
