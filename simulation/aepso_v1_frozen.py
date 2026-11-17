"""
Absorptive Epistemic Particle Swarm (AE-PSO) testbed for the Knowledge Delta Protocol.

A model of knowledge production in a simulated science world with a hidden ground truth.
Implements the pre-registered design in PREREGISTRATION.md.
"""
import numpy as np

G = 20                      # cells per dimension -> 400 addressable claims
NC = G * G
CENT = np.array([[(i + 0.5) / G, (j + 0.5) / G] for i in range(G) for j in range(G)])
MU0 = 0.2                   # prior expectation of a claim's value
PHI_TOT = 4.1               # total acceleration budget (Clerc-Kennedy)
CHI = 0.7298                # constriction factor for PHI_TOT = 4.1
VMAX = 0.15

DEFAULT = dict(
    n_legacy=40, t_legacy=100, n_agents=40, t_kdp=150, t_shift=75,
    p_flaw=0.3, artifact_bias=0.9, sigma=0.15, sigma_hi=0.40, pub_threshold=0.5,
    rho=0.2, m=2, n0=1.0,
    # absorption profile alpha = (k, tau, g, theta, a0)
    k_radius=0.25, tau=50.0, layer="L2", theta=2, a0_fixed=0.5, a0_prior=0.5, kappa=4.0,
    phi3=0.5, topology="global", mix="balanced",
    init="auto", tau_mem=float("inf"),
    trigger=False, trig_frac=0.3, trig_thr=0.2, trig_cool=15,
)


# ---------------------------------------------------------------- world
class World:
    def __init__(self, rng):
        self.rng = rng
        pts = self._spread_points(6, 0.22)
        self.g_peak = pts[0]
        self.locals = [(pts[1], 0.70, 0.07), (pts[2], 0.62, 0.07), (pts[3], 0.55, 0.06)]
        self.noise_center = pts[1]                    # high-noise region around a local peak
        self.art_center = pts[4]                      # artifact region (low truth)
        self.shift_target = pts[5]
        self.g_height, self.g_width = 1.0, 0.07
        self.old_peak = None
        self.T_cell = self._cell_truth()

    def _spread_points(self, n, dmin):
        while True:
            P = self.rng.uniform(0.1, 0.9, size=(n, 2))
            D = np.linalg.norm(P[:, None] - P[None], axis=-1) + np.eye(n)
            if D.min() > dmin:
                return P

    def truth(self, X):
        X = np.atleast_2d(X)
        vals = [np.full(len(X), 0.1)]
        vals.append(self.g_height * np.exp(-np.sum((X - self.g_peak) ** 2, 1) / (2 * self.g_width ** 2)))
        for c, h, w in self.locals:
            vals.append(h * np.exp(-np.sum((X - c) ** 2, 1) / (2 * w ** 2)))
        if self.old_peak is not None:
            vals.append(0.4 * np.exp(-np.sum((X - self.old_peak) ** 2, 1) / (2 * self.g_width ** 2)))
        return np.max(np.vstack(vals), 0)

    def _cell_truth(self):
        sub = np.array([(a, b) for a in (-1 / 3, 0, 1 / 3) for b in (-1 / 3, 0, 1 / 3)]) / G
        return np.mean([self.truth(CENT + s) for s in sub], 0)

    def observe(self, X, flaw, p):
        X = np.atleast_2d(X)
        art = p["artifact_bias"] * np.exp(-np.sum((X - self.art_center) ** 2, 1) / (2 * 0.05 ** 2))
        sig = np.where(np.linalg.norm(X - self.noise_center, axis=1) < 0.12, p["sigma_hi"], p["sigma"])
        return self.truth(X) + flaw * art + self.rng.normal(0, 1, len(X)) * sig

    def shift(self):
        self.old_peak = self.g_peak.copy()
        self.g_peak = self.shift_target.copy()
        self.T_cell = self._cell_truth()

    def is_artifact_cell(self, c):
        return np.linalg.norm(CENT[c] - self.art_center) < 0.1


def cell_of(X):
    X = np.atleast_2d(X)
    ij = np.minimum((X * G).astype(int), G - 1)
    return ij[:, 0] * G + ij[:, 1]


def pso_step(rng, x, v, p, g, h, a0h, mixc):
    """Constricted velocity update with a fixed acceleration budget shared by
    self (c1), community (c2) and absorbed history (a0 * c3)."""
    n = len(x)
    phi3 = np.where(np.isnan(h[:, 0]), 0.0, a0h)
    rest = PHI_TOT - phi3
    phi1, phi2 = rest * mixc, rest * (1 - mixc)
    hh = np.where(np.isnan(h), x, h)
    r1, r2, r3 = (rng.uniform(size=(n, 2)) for _ in range(3))
    v = CHI * (v + phi1[:, None] * r1 * (p - x) + phi2[:, None] * r2 * (g - x) + phi3[:, None] * r3 * (hh - x))
    v = np.clip(v, -VMAX, VMAX)
    x = x + v
    over = (x < 0) | (x > 1)                   # reflecting boundaries
    x = np.where(x < 0, -x, x)
    x = np.where(x > 1, 2 - x, x)
    v = np.where(over, -v, v)
    return x, v


def mix_vector(rng, n, mix):
    if mix == "balanced":
        return np.full(n, 0.5)
    hi = 0.75 if mix == "maverick" else 0.25
    lo = 1 - hi
    k = int(round(0.7 * n))
    out = np.array([hi] * k + [lo] * (n - k))
    rng.shuffle(out)
    return out


# ---------------------------------------------------------------- legacy (document-centric) phase
def legacy_phase(world, rng, p):
    n, T = p["n_legacy"], p["t_legacy"]
    flaw = (rng.uniform(size=n) < p["p_flaw"]).astype(float)
    x = rng.uniform(size=(n, 2)); v = rng.uniform(-0.05, 0.05, size=(n, 2))
    pb_x, pb_y = x.copy(), np.full(n, -np.inf)
    rec = []                                   # published records: (cell, y, t, agent)
    best_pub_x, best_pub_y = None, -np.inf
    mixc = np.full(n, 0.5)
    for t in range(T):
        y = world.observe(x, flaw, p)
        imp = y > pb_y
        pb_x[imp], pb_y[imp] = x[imp], y[imp]
        pub = y > p["pub_threshold"]
        cells = cell_of(x)
        for a in np.where(pub)[0]:
            rec.append((cells[a], y[a], t - T, a))
            if y[a] > best_pub_y:
                best_pub_y, best_pub_x = y[a], x[a].copy()
        g = np.tile(best_pub_x if best_pub_x is not None else pb_x[np.argmax(pb_y)], (n, 1))
        x, v = pso_step(rng, x, v, pb_x, g, np.full((n, 2), np.nan), np.zeros(n), mixc)
    return np.array(rec, dtype=float).reshape(-1, 4)


def legacy_aggregate(rec, n_leg):
    """Layer L2: per-claim aggregation of the legacy literature (one vote per legacy agent)."""
    nrec = np.zeros(NC); nag = np.zeros(NC); mean = np.zeros(NC); last = np.full(NC, -1e9); mx = np.full(NC, -np.inf)
    S = np.zeros((NC, n_leg)); C = np.zeros((NC, n_leg))
    for c, y, t, a in rec:
        c, a = int(c), int(a)
        S[c, a] += y; C[c, a] += 1; nrec[c] += 1; last[c] = max(last[c], t); mx[c] = max(mx[c], y)
    has = C > 0
    nag = has.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        am = np.where(has, S / np.maximum(C, 1), 0)
        mean = np.where(nag > 0, am.sum(1) / np.maximum(nag, 1), 0)
    return dict(nrec=nrec, nag=nag, mean=mean, last=last, max=mx, S=S, C=C)


# ---------------------------------------------------------------- lenses
def replication_lens(kS, kC, leg_w, leg, n0, m):
    """Replication-weighted lens: one vote per independent agent, absorbed legacy votes
    weighted by leg_w (a0 * decay, or 1 for eager import, 0 if not absorbed)."""
    has = kC > 1e-9
    am = np.where(has, kS / np.maximum(kC, 1e-9), 0)
    wv = np.minimum(1.0, kC)
    n_k = wv.sum(1)
    sum_k = (wv * am).sum(1)
    n_l = leg_w * leg["nag"]
    sum_l = leg_w * leg["nag"] * leg["mean"]
    n_eff = n_k + n_l
    S = MU0 + (sum_k + sum_l - n_eff * MU0) / (n_eff + n0)
    admitted = n_eff >= max(m, 1e-9) if m > 0 else n_eff > 0
    return S, admitted, n_eff, n_k, np.where(n_k > 0, sum_k / np.maximum(n_k, 1), np.nan)


def pooled_lens(kS, kC, leg, include_legacy=True, n0=0.5):
    """All-evidence lens: every observation at face value."""
    s = kS.sum(1) + (leg["S"].sum(1) if include_legacy else 0)
    c = kC.sum(1) + (leg["C"].sum(1) if include_legacy else 0)
    return MU0 + (s - c * MU0) / (c + n0), c


# ---------------------------------------------------------------- KDP phase
def run(cond, scenario, seed, **over):
    p = dict(DEFAULT); p.update(over)
    rng = np.random.default_rng(seed)
    world = World(rng)
    rec = legacy_phase(world, rng, p)
    leg = legacy_aggregate(rec, p["n_legacy"])
    n, T = p["n_agents"], p["t_kdp"]
    flaw = (rng.uniform(size=n) < p["p_flaw"]).astype(float)
    mixc = mix_vector(rng, n, p["mix"])
    absorb = cond in ("FIX", "ADA")
    m = 0 if cond == "NA0" else p["m"]
    rho = 0.0 if cond == "DOC" else p["rho"]

    # initial positions: absorption conditions and literature readers start from the legacy record
    use_leg_init = (cond in ("FIX", "ADA", "EAG", "DOC")) if p["init"] == "auto" else (p["init"] == "legacy")
    if use_leg_init and len(rec):
        w = np.maximum(rec[:, 1], 0.01)
        idx = rng.choice(len(rec), size=n, p=w / w.sum())
        x = np.clip(CENT[rec[idx, 0].astype(int)] + rng.normal(0, 0.05, (n, 2)), 0, 1)
    else:
        x = rng.uniform(size=(n, 2))
    v = rng.uniform(-0.05, 0.05, size=(n, 2))

    kS = np.zeros((NC, n)); kC = np.zeros((NC, n))
    first = np.full(NC, -1)
    attention = np.zeros(n)
    pub_best_y, pub_best_c = (leg["max"].max(), int(np.argmax(leg["max"]))) if len(rec) else (-np.inf, -1)
    pub_S, pub_C = leg["S"].sum(1).copy(), leg["C"].sum(1).copy()
    theta_ok = leg["nrec"] >= p["theta"]
    hist = dict(regret=[], error=[], diversity=[], a0=[], canon_art=[], triggers=[])
    anom, last_trig = 0.0, -10**9

    def legacy_weights(t, kmean, nk):
        decay = 2.0 ** (-(t - leg["last"]) / p["tau"]) if np.isfinite(p["tau"]) else np.ones(NC)
        if cond == "EAG":
            return np.ones(NC), np.ones(NC)
        if not absorb:
            return np.zeros(NC), np.zeros(NC)
        if cond == "FIX":
            a0c = np.full(NC, p["a0_fixed"])
        else:  # ADA: conflict-adaptive trust, per claim
            conf = np.where(nk > 0, np.abs(leg["mean"] - np.nan_to_num(kmean)) * nk / (nk + 1), np.nan)
            a0c = np.where(np.isnan(conf), p["a0_prior"], np.exp(-p["kappa"] * np.nan_to_num(conf)))
        a0c = np.where(theta_ok, a0c, 0.0)
        return a0c * decay, a0c

    for t in range(T):
        if scenario == "shift" and t == p["t_shift"]:
            world.shift()
        # ---- current knowledge state
        nk_tmp = (kC > 0).sum(1)
        km_tmp = np.where(nk_tmp > 0, (np.where(kC > 0, kS / np.maximum(kC, 1), 0)).sum(1) / np.maximum(nk_tmp, 1), np.nan)
        lw, a0c = legacy_weights(t, km_tmp, nk_tmp)
        if cond == "DOC":
            S = MU0 + (pub_S - pub_C * MU0) / (pub_C + 0.5)
            admitted = pub_C > 0
            canon = pub_best_c
        else:
            S, admitted, n_eff, n_k, kmean = replication_lens(kS, kC, lw, leg, p["n0"], m)
            canon = int(np.argmax(np.where(admitted, S, -np.inf))) if admitted.any() else -1
        Tmax = world.T_cell.max()
        hist["regret"].append(Tmax - (world.T_cell[canon] if canon >= 0 else 0.1))
        hist["canon_art"].append(bool(canon >= 0 and world.is_artifact_cell(canon)))
        known = admitted if cond != "DOC" else pub_C > 0
        hist["error"].append(np.abs(S - world.T_cell)[known].mean() if known.any() else np.nan)
        hist["diversity"].append(np.mean(np.linalg.norm(x[:, None] - x[None], axis=-1)))
        hist["a0"].append(np.nanmean(np.where(theta_ok, a0c, np.nan)) if absorb else np.nan)

        # ---- attractors
        visited = kC > 0
        Sv = np.where(visited, S[:, None], -np.inf)
        pc = np.argmax(Sv, 0)
        pbx = np.where((visited.any(0))[:, None], CENT[pc], x)
        if cond == "DOC":
            gx = np.tile(CENT[pub_best_c], (n, 1)) if pub_best_c >= 0 else pbx.copy()
            gsrc = np.full(n, pub_best_c)
        elif p["topology"] == "global":
            gx = np.tile(CENT[canon], (n, 1)) if canon >= 0 else pbx.copy()
            gsrc = np.full(n, canon)
        else:  # ring of radius 1 over agents' personal bests
            pS = np.where(visited.any(0) & admitted[pc], S[pc], -np.inf)
            nb = np.stack([np.roll(np.arange(n), 1), np.arange(n), np.roll(np.arange(n), -1)], 1)
            best = nb[np.arange(n), np.argmax(pS[nb], 1)]
            ok = np.isfinite(pS[best])
            gx = np.where(ok[:, None], pbx[best], pbx)
            gsrc = np.where(ok, pc[best], -1)
        for i in range(n):                          # attention: social-attractor use, priority attribution
            j = first[gsrc[i]] if gsrc[i] >= 0 else -1
            if j >= 0 and j != i:
                attention[j] += 1

        hx = np.full((n, 2), np.nan); a0h = np.zeros(n)
        if absorb:
            base = leg["mean"] if p["layer"] == "L2" else np.where(np.isfinite(leg["max"]), leg["max"], 0)
            score = np.where(theta_ok, lw * base, -np.inf)
            d = np.linalg.norm(x[:, None] - CENT[None], axis=-1)
            sc = np.where(d <= p["k_radius"], score[None], -np.inf)
            hc = np.argmax(sc, 1)
            okh = np.isfinite(sc[np.arange(n), hc]) & (sc[np.arange(n), hc] > 0)
            hx[okh] = CENT[hc[okh]]
            a0h[okh] = p["phi3"] * a0c[hc[okh]]

        # ---- act: explore (PSO move) or replicate (attestation)
        rep = rng.uniform(size=n) < rho
        x_new, v_new = pso_step(rng, x, v, pbx, gx, hx, a0h, mixc)
        x = np.where(rep[:, None], x, x_new); v = np.where(rep[:, None], v, v_new)
        obs_x = x.copy()
        if rep.any():
            nkk = (kC > 0).sum(1)
            cand = np.where((nkk < 3) & (S > MU0), S, -np.inf)
            top = np.argsort(-cand)[:5]
            top = top[np.isfinite(cand[top])]
            for i in np.where(rep)[0]:
                pool = [c for c in top if kC[c, i] == 0]
                c = rng.choice(pool) if pool else (canon if canon >= 0 else cell_of(x[i])[0])
                obs_x[i] = CENT[c] + rng.uniform(-0.5, 0.5, 2) / G
        if np.isfinite(p["tau_mem"]) and cond != "DOC":
            dmem = 2.0 ** (-1.0 / p["tau_mem"])
            kS *= dmem; kC *= dmem
        y = world.observe(obs_x, flaw, p)
        cells = cell_of(obs_x)
        if p["trigger"] and cond != "DOC" and canon >= 0:
            hit = cells == canon
            if hit.any():                       # attestation residuals on the canonised claim
                anom = 0.7 * anom + 0.3 * float(np.mean(y[hit] - S[canon]))
            if anom < -p["trig_thr"] and t - last_trig > p["trig_cool"]:
                k = rng.choice(n, size=int(p["trig_frac"] * n), replace=False)
                x[k] = rng.uniform(size=(len(k), 2)); v[k] = rng.uniform(-0.05, 0.05, size=(len(k), 2))
                last_trig, anom = t, 0.0
                hist["triggers"].append(t)
        for i in range(n):
            c = cells[i]
            kS[c, i] += y[i]; kC[c, i] += 1
            if first[c] < 0:
                first[c] = i
            if cond == "DOC" and y[i] > p["pub_threshold"]:
                pub_S[c] += y[i]; pub_C[c] += 1
                if y[i] > pub_best_y:
                    pub_best_y, pub_best_c = y[i], c

    out = {k: np.array(vv, dtype=float) for k, vv in hist.items()}
    out.update(final_regret=out["regret"][-1], false_canon=float(out["regret"][-1] > 0.3),
               early_regret=np.nanmean(out["regret"][:30]),
               post_shift_regret=np.nanmean(out["regret"][p["t_shift"]:]))
    out["_state"] = dict(kS=kS, kC=kC, leg=leg, lw=lw, world=world, attention=attention, p=p, m=m)
    return out


# ---------------------------------------------------------------- H3: lens divergence alarm
def lens_divergence_auc(res):
    st = res["_state"]; kS, kC, leg, p = st["kS"], st["kC"], st["leg"], st["p"]
    T = st["world"].T_cell
    Sa, ca = pooled_lens(kS, kC, leg, include_legacy=True)
    Sr, adm, n_eff, _, _ = replication_lens(kS, kC, st["lw"], leg, p["n0"], st["m"])
    ev = ca > 0
    lab = (Sa - T > 0.2)[ev]
    score = np.abs(Sa - Sr)[ev]
    if lab.sum() == 0 or lab.sum() == len(lab):
        return np.nan
    from scipy.stats import mannwhitneyu
    u = mannwhitneyu(score[lab], score[~lab], alternative="two-sided").statistic
    return u / (lab.sum() * (~lab).sum())


# ---------------------------------------------------------------- H4: credit validity
def credit_validity(res, n_perm=100, seed=0):
    from scipy.stats import spearmanr
    st = res["_state"]; kS, kC, p = st["kS"], st["kC"], st["p"]
    T = st["world"].T_cell
    n = kS.shape[1]
    has = kC > 0
    am = np.where(has, kS / np.maximum(kC, 1), 0)
    n0 = p["n0"]
    base_err = np.abs(MU0 - T).mean()
    rng = np.random.default_rng(seed)
    phi_true = np.zeros(n); phi_kdp = np.zeros(n)
    for _ in range(n_perm):
        order = rng.permutation(n)
        s = np.zeros(NC); c = np.zeros(NC)
        prev_err, prev_chg = base_err, 0.0
        for j in order:
            s += am[:, j]; c += has[:, j]
            S = MU0 + (s - c * MU0) / (c + n0)
            err = np.abs(S - T).mean(); chg = np.abs(S - MU0).mean()
            phi_true[j] += prev_err - err; phi_kdp[j] += chg - prev_chg
            prev_err, prev_chg = err, chg
    phi_true /= n_perm; phi_kdp /= n_perm
    att = st["attention"]
    return dict(rho_kdp=spearmanr(phi_kdp, phi_true).statistic,
                rho_att=spearmanr(att, phi_true).statistic,
                phi_true=phi_true, phi_kdp=phi_kdp, attention=att)
