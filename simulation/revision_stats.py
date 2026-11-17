import pickle, numpy as np, json
from scipy.stats import wilcoxon, mannwhitneyu, beta
M = pickle.load(open("main_results.pkl", "rb")); E = pickle.load(open("explore_results.pkl", "rb"))
R = pickle.load(open("revision_results.pkl", "rb"))
rng = np.random.default_rng(7)
OUT = {}


def boot_ci(x, f=np.mean, B=10000):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    bs = [f(rng.choice(x, len(x))) for _ in range(B)]
    return float(f(x)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def cp(k, n):
    lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return k, n, float(lo), float(hi)


v = lambda D, k, key="final_regret": np.array([r[key] for r in D[k]], float)
S, H = "stationary", "shift"

# ---- A. confirmatory tests re-analysed with seed pairing, effect sizes, Holm
tests = {
    "H1": (v(M, (S, "NA2")), v(M, (S, "NA0"))),
    "H2a": (v(M, (S, "ADA"), "early_regret"), v(M, (S, "NA2"), "early_regret")),
    "H2c_FIX": (v(M, (H, "ADA"), "post_shift_regret"), v(M, (H, "FIX"), "post_shift_regret")),
    "H2c_EAG": (v(M, (H, "ADA"), "post_shift_regret"), v(M, (H, "EAG"), "post_shift_regret")),
}
conf = {}
for k, (a, b) in tests.items():
    d = a - b
    conf[k] = dict(mean_diff=boot_ci(d), p_rank_one=mannwhitneyu(a, b, alternative="less").pvalue,
                   p_signed_two=wilcoxon(a, b).pvalue if np.any(d != 0) else 1.0,
                   W=float(wilcoxon(a, b).statistic) if np.any(d != 0) else np.nan)
fc = lambda D, k: int(v(D, k, "false_canon").sum())
conf["H1_fc"] = dict(NA2=cp(fc(M, (S, "NA2")), 30), NA0=cp(fc(M, (S, "NA0")), 30))
conf["H2b_fc"] = dict(ADA=cp(fc(M, (S, "ADA")), 30), EAG=cp(fc(M, (S, "EAG")), 30))
auc = v(M, (S, "ADA"), "auc"); conf["H3"] = dict(median=boot_ci(auc, np.median), share_above_070=float((auc > 0.7).mean()))
rk, ra = v(M, (S, "ADA"), "rho_kdp"), v(M, (S, "ADA"), "rho_att")
conf["H4"] = dict(kdp=boot_ci(rk, np.median), att=boot_ci(ra, np.median), diff=boot_ci(rk - ra, np.median),
                  p_one=wilcoxon(rk, ra, alternative="greater").pvalue, p_two=wilcoxon(rk, ra).pvalue, W=float(wilcoxon(rk, ra).statistic))
pre = {"H1": 0.1706, "H1_fc": 0.6940, "H2a": 0.0925, "H2b": 0.7881, "H2c_FIX": 0.5529, "H2c_EAG": 0.4676, "H4": conf["H4"]["p_one"]}
order = sorted(pre, key=pre.get); m = len(pre); holm = {}; run_max = 0
for i, k in enumerate(order):
    run_max = max(run_max, min(1, (m - i) * pre[k])); holm[k] = run_max
conf["holm"] = holm
OUT["confirmatory"] = conf

# ---- B. initialization x absorption factorial
fact = {}
for sc in (S, H):
    cells = {("random", "none"): v(M, (sc, "NA2")), ("legacy", "none"): v(E, (sc, "E2:NA2-legacy-init-only")),
             ("random", "adaptive"): v(E, (sc, "E2:ADA-attractor-only(random init)")), ("legacy", "adaptive"): v(M, (sc, "ADA"))}
    fact[sc] = {f"{a}|{b}": boot_ci(x) for (a, b), x in cells.items()}
    inter = (cells[("legacy", "adaptive")] - cells[("random", "adaptive")]) - (cells[("legacy", "none")] - cells[("random", "none")])
    fact[sc]["interaction"] = boot_ci(inter)
    fact[sc]["init_effect_none"] = boot_ci(cells[("legacy", "none")] - cells[("random", "none")])
    fact[sc]["abs_effect_random"] = boot_ci(cells[("random", "adaptive")] - cells[("random", "none")])
    fact[sc]["abs_effect_legacy"] = boot_ci(cells[("legacy", "adaptive")] - cells[("legacy", "none")])
    fact[sc]["p_init_none"] = wilcoxon(cells[("legacy", "none")], cells[("random", "none")]).pvalue
    fact[sc]["p_abs_random"] = wilcoxon(cells[("random", "adaptive")], cells[("random", "none")]).pvalue
    fact[sc]["p_abs_legacy"] = wilcoxon(cells[("legacy", "adaptive")], cells[("legacy", "none")]).pvalue
OUT["factorial"] = fact

# ---- C. matched document baselines
base = {}
for sc in (S, H):
    rows = {"DOC": M[(sc, "DOC")], "DOCM": R[("baseline", "DOCM", sc, ())], "DOCR": R[("baseline", "DOCR", sc, ())],
            "NA2 legacy start": E[(sc, "E2:NA2-legacy-init-only")], "NA2": M[(sc, "NA2")], "ADA": M[(sc, "ADA")]}
    base[sc] = {k: dict(regret=boot_ci([r["final_regret"] for r in x]), false=cp(int(sum(r["false_canon"] for r in x)), 30),
                        err=boot_ci([r["err_final"] if "err_final" in r else r["error"][-1] for r in x])) for k, x in rows.items()}
    a = v(R, ("baseline", "DOCR", sc, ())); b = v(E, (sc, "E2:NA2-legacy-init-only"))
    base[sc]["DOCR_vs_NA2legacy"] = dict(diff=boot_ci(b - a), p=wilcoxon(b, a).pvalue)
OUT["baselines"] = base

# ---- D. legacy-bias sensitivity
bias = {}
for kw in ((("pub_threshold", 0.3),), (("pub_threshold", 0.7),), (("p_flaw", 0.1),), (("p_flaw", 0.5),)):
    row = {}
    for c in ("DOC", "DOCR", "NA2", "EAG", "ADA"):
        x = R[("bias", c, S, kw)]; row[c] = dict(regret=boot_ci([r["final_regret"] for r in x]), false=int(sum(r["false_canon"] for r in x)))
    x = R[("bias", "NA2", S, tuple(sorted(dict(kw, init="legacy").items())))]
    row["NA2 legacy start"] = dict(regret=boot_ci([r["final_regret"] for r in x]), false=int(sum(r["false_canon"] for r in x)))
    bias[str(dict(kw))] = row
OUT["bias"] = bias

# ---- E/F. H3 and H4 extended
h3, h4 = {}, {}
for (kind, c, sc, kw), rows in R.items():
    if kind != "credit": continue
    tag = f"{c} {sc}"
    names = [k for k in rows[0]["h3"] if k != "prev"]
    h3[tag] = {f"{lab}|{nm}": dict(auc=boot_ci([r["h3"][(lab, nm)][0] for r in rows], np.median),
                                   ap=boot_ci([r["h3"][(lab, nm)][1] for r in rows], np.median)) for lab, nm in names}
    h3[tag]["prevalence"] = {L: [float(np.median([r["h3"]["prev"][L][0] for r in rows])), float(np.median([r["h3"]["prev"][L][1] for r in rows])),
                                 int(sum(r["h3"]["prev"][L][0] == 0 for r in rows))] for L in ("L1", "L2")}
    keys = [k for k in rows[0]["h4"] if isinstance(k, tuple)]
    h4[tag] = {f"{U}|{nm}": boot_ci([r["h4"][(U, nm)] for r in rows], np.nanmedian) for U, nm in keys}
    h4[tag]["mc_stability_K"] = float(np.median([r["h4"]["mc_stability_K"] for r in rows]))
    h4[tag]["mc_stability_U"] = float(np.median([r["h4"]["mc_stability_U"] for r in rows]))
    h4[tag]["mc_rho_diff_median"] = float(np.median([r["h4"]["mc_rho_diff"] for r in rows]))
    h4[tag]["t_run"] = float(np.mean([r["t_run"] for r in rows])); h4[tag]["t_shapley"] = float(np.mean([r["t_shapley"] for r in rows]))
OUT["h3"] = h3; OUT["h4"] = h4
OUT["cost"] = dict(mean_run_s=float(np.mean([r["t_run"] for x in R.values() for r in x])))
json.dump(OUT, open("revision_stats.json", "w"), indent=1, default=float)


def f3(t): return f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"
print("== Confirmatory, paired re-analysis")
for k in ("H1", "H2a", "H2c_FIX", "H2c_EAG"):
    c = conf[k]; print(k, "mean diff", f3(c["mean_diff"]), "| prereg one-sided rank-sum p=%.3f | paired two-sided signed-rank p=%.3f W=%.0f" % (c["p_rank_one"], c["p_signed_two"], c["W"]))
print("H1 false canon", conf["H1_fc"]); print("H2b false canon", conf["H2b_fc"])
print("H3 median AUC", f3(conf["H3"]["median"]), "share>0.7", conf["H3"]["share_above_070"])
print("H4 kdp", f3(conf["H4"]["kdp"]), "att", f3(conf["H4"]["att"]), "diff", f3(conf["H4"]["diff"]), "p1=%.2e p2=%.2e W=%.0f" % (conf["H4"]["p_one"], conf["H4"]["p_two"], conf["H4"]["W"]))
print("Holm-adjusted:", {k: round(x, 4) for k, x in holm.items()})
print("\n== Factorial (final regret)")
for sc in (S, H):
    print(sc, {k: (f3(x) if isinstance(x, tuple) else round(x, 4)) for k, x in fact[sc].items()})
print("\n== Matched baselines")
for sc in (S, H):
    for k, x in base[sc].items():
        if k == "DOCR_vs_NA2legacy": print(sc, k, f3(x["diff"]), "p=%.4f" % x["p"]); continue
        print(sc, f"{k:18s}", "regret", f3(x["regret"]), "| false", x["false"][:2], "| err", f3(x["err"]))
print("\n== Legacy bias")
for kw, row in bias.items():
    print(kw, {c: (round(x["regret"][0], 3), x["false"]) for c, x in row.items()})
print("\n== H3 extended (median AUC / median AP)")
for tag, d in h3.items():
    print(tag, "prev", d["prevalence"])
    for k, x in d.items():
        if k != "prevalence": print("   ", f"{k:34s}", "AUC", f3(x["auc"]), "AP", f3(x["ap"]))
print("\n== H4 extended (median Spearman with true utility)")
for tag, d in h4.items():
    print(tag, "MC stability K=%.3f U=%.3f |rho diff| median=%.3f  t_run=%.2fs t_shapley=%.2fs" % (d["mc_stability_K"], d["mc_stability_U"], d["mc_rho_diff_median"], d["t_run"], d["t_shapley"]))
    for k, x in d.items():
        if isinstance(x, tuple): print("   ", f"{k:34s}", f3(x))
print("cost", OUT["cost"])
