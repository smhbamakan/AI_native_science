"""Figures from the stored simulation results (run the simulation scripts first)."""
import pickle, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdbd6"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
BASE = "#8a8984"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.8, "figure.facecolor": "white",
                     "axes.titlesize": 9.5, "xtick.labelsize": 8, "ytick.labelsize": 8})
M = pickle.load(open("main_results.pkl", "rb")); E = pickle.load(open("explore_results.pkl", "rb"))
E6 = pickle.load(open("e6_results.pkl", "rb")); R = pickle.load(open("revision_results.pkl", "rb"))
S, H = "stationary", "shift"
rng = np.random.default_rng(0)


def clean(ax, title, grid="y"):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis=grid, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontweight="bold", color=INK)


def band(ax, rows, color, label, ls="-", dy=0.0):
    Y = np.array([r["regret"] for r in rows]); t = np.arange(Y.shape[1]) + 1
    m = Y.mean(0); se = Y.std(0, ddof=1) / np.sqrt(len(Y))
    ax.fill_between(t, m - 1.96 * se, m + 1.96 * se, color=color, alpha=0.15, lw=0)
    ax.plot(t, m, color=color, lw=1.8, ls=ls)
    ax.text(t[-1] + 3, m[-1] + dy, label, color=INK, va="center", fontsize=8)


def boot_med(x, B=4000):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    bs = np.median(rng.choice(x, (B, len(x))), 1); return np.median(x), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def mci(x):
    x = np.asarray(x, float); return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x))


fr = lambda rows: np.array([r["final_regret"] for r in rows], float)

# ================================================================ Figure 2 (confirmatory)
fig, axs = plt.subplots(2, 2, figsize=(9.2, 6.6), gridspec_kw={"hspace": 0.62, "wspace": 0.95})
a, b, c, d = axs.ravel()
band(a, M[(S, "DOC")], BASE, "Document-centric", "--")
band(a, M[(S, "EAG")], ORANGE, "KDP, eager import", dy=-0.014)
band(a, M[(S, "ADA")], AQUA, "KDP, adaptive\nabsorption", dy=0.028)
band(a, M[(S, "NA2")], BLUE, "KDP, no absorption", dy=-0.012)
a.set_xlim(0, 150); a.set_ylim(0, 0.45); a.set_xlabel("Iteration of KDP phase"); a.set_ylabel("Regret of canonized claim")
clean(a, "A  Stationary world (n = 30 per condition)")

band(b, M[(H, "DOC")], BASE, "Document-centric", "--")
band(b, M[(H, "NA2")], BLUE, "KDP, no absorption", dy=0.05)
band(b, M[(H, "ADA")], AQUA, "KDP, adaptive\nabsorption", dy=-0.06)
band(b, E6[(H, "E6:trigger+tau_mem25 NA2")], YELLOW, "No absorption, memory\ndecay and anomaly trigger", dy=-0.09)
b.axvline(75, color=INK2, lw=0.8, ls=":"); b.text(76.5, 0.03, "peak moves", color=INK2, fontsize=8)
b.set_xlim(0, 150); b.set_ylim(0, 0.75); b.set_xlabel("Iteration of KDP phase"); b.set_ylabel("Regret of canonized claim")
clean(b, "B  Paradigm-shift world (n = 30 per condition)")

auc = np.array([r["auc"] for r in M[(S, "ADA")]], float); auc = auc[np.isfinite(auc)]
c.scatter(rng.uniform(-0.12, 0.12, len(auc)), auc, s=16, color=BLUE, edgecolor="white", lw=0.6, zorder=3)
c.hlines(np.median(auc), -0.25, 0.25, color=INK, lw=1.5)
c.axhline(0.70, color=INK2, lw=0.8, ls="--"); c.text(0.32, 0.70, "pre-registered\nthreshold", color=INK2, fontsize=8, va="center")
c.axhline(0.5, color=GRID, lw=0.9); c.text(0.32, 0.5, "chance", color=INK2, fontsize=8, va="center")
c.text(0.27, np.median(auc) + 0.005, f"median {np.median(auc):.2f}", fontsize=8, color=INK, va="bottom")
c.set_xlim(-0.4, 1.0); c.set_xticks([]); c.set_ylim(0.4, 1.02); c.set_ylabel("AUC for over-stated claims")
c.set_xlabel(f"one point per run (n = {len(auc)})")
clean(c, "C  Lens divergence as an alarm (H3)")

rk = np.array([r["rho_kdp"] for r in M[(S, "ADA")]], float); ra = np.array([r["rho_att"] for r in M[(S, "ADA")]], float)
ok = np.isfinite(rk) & np.isfinite(ra)
for x1, x2 in zip(ra[ok], rk[ok]): d.plot([0, 1], [x1, x2], color=GRID, lw=0.8, zorder=1)
d.scatter(np.zeros(ok.sum()), ra[ok], s=16, color=BASE, edgecolor="white", lw=0.6, zorder=3)
d.scatter(np.ones(ok.sum()), rk[ok], s=16, color=BLUE, edgecolor="white", lw=0.6, zorder=3)
for xx, vv in ((0, ra[ok]), (1, rk[ok])):
    d.hlines(np.median(vv), xx - 0.18, xx + 0.18, color=INK, lw=1.5)
    d.text(xx + 0.22, np.median(vv), f"{np.median(vv):.2f}", fontsize=8, va="center")
d.axhline(0, color=INK2, lw=0.6)
d.set_xticks([0, 1]); d.set_xticklabels(["Attention\n(citation analogue)", "KDP credit\n(Shapley)"])
d.set_xlim(-0.4, 1.5); d.set_ylabel("Spearman ρ with true contribution")
clean(d, f"D  Credit validity (H4), n = {ok.sum()} paired runs")
fig.savefig("fig2_results.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# ================================================================ Figure 3 (mechanisms; exploratory)
fig, axs = plt.subplots(1, 3, figsize=(10.2, 3.6), gridspec_kw={"wspace": 0.75, "width_ratios": [1.15, 1.2, 1.25]})
a, b, c = axs
# a: 2x2 factorial, interaction plot
for sc, col, mk, lab in ((S, BLUE, "o", "Stationary"), (H, ORANGE, "D", "Paradigm shift")):
    cells = {("random", "none"): M[(sc, "NA2")], ("legacy", "none"): E[(sc, "E2:NA2-legacy-init-only")],
             ("random", "adaptive"): E[(sc, "E2:ADA-attractor-only(random init)")], ("legacy", "adaptive"): M[(sc, "ADA")]}
    for k, (ab, ls) in enumerate((("none", "--"), ("adaptive", "-"))):
        ms, cis = zip(*[mci(fr(cells[(ini, ab)])) for ini in ("random", "legacy")])
        xs = np.array([0, 1]) + (k - 0.5) * 0.08
        a.errorbar(xs, ms, yerr=cis, color=col, ls=ls, marker=mk, ms=5, mfc=col if ab == "adaptive" else "white",
                   mec=col, lw=1.3, capsize=2)
a.set_xticks([0, 1]); a.set_xticklabels(["Random\nstart", "Legacy\nstart"]); a.set_xlim(-0.35, 1.35)
a.set_ylabel("Final regret (mean, 95% CI)"); a.set_ylim(0, 0.62)
h = [Line2D([], [], color=BLUE, marker="o", lw=0, label="Stationary"), Line2D([], [], color=ORANGE, marker="D", lw=0, label="Paradigm shift"),
     Line2D([], [], color=INK2, ls="--", marker="o", mfc="white", label="No absorption"), Line2D([], [], color=INK2, ls="-", marker="o", label="Adaptive absorption")]
a.legend(handles=h, frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, handletextpad=0.3, columnspacing=0.8)
clean(a, "A  Start × absorption")

base_s = fr(M[(S, "ADA")]).mean(); base_h = fr(M[(H, "ADA")]).mean()
sens = [("k = 0.1 (0.25)", "E3:k=0.1"), ("k = 0.5 (0.25)", "E3:k=0.5"), ("τ = 10 (50)", "E3:tau=10.0"), ("τ = ∞ (50)", "E3:tau=inf"),
        ("γ = L1 (L2)", "E3:layer=L1"), ("θ = 1 (2)", "E3:theta=1"), ("θ = 4 (2)", "E3:theta=4"), ("η = 1 (4)", "E3:kappa=1.0"), ("η = 10 (4)", "E3:kappa=10.0")]
for i, (lab, tag) in enumerate(sens):
    ds = fr(E[(S, tag)]).mean() - base_s; dh = fr(E[(H, tag)]).mean() - base_h
    b.plot([ds, dh], [i, i], color=GRID, lw=1, zorder=1)
    b.scatter(ds, i, s=26, color=BLUE, edgecolor="white", lw=0.6, zorder=3)
    b.scatter(dh, i, s=26, color=ORANGE, marker="D", edgecolor="white", lw=0.6, zorder=3)
b.axvline(0, color=INK2, lw=0.8)
b.set_yticks(range(len(sens))); b.set_yticklabels([s[0] for s in sens]); b.invert_yaxis()
b.set_xlabel("Change in final regret vs default α")
b.legend(handles=h[:2], frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, handletextpad=0.2)
clean(b, "B  Sensitivity to α", grid="x")

mixes = ["balanced", "maverick", "follower"]
for j, (topo, col, mk) in enumerate((("global", BLUE, "o"), ("ring", ORANGE, "D"))):
    for i, mx in enumerate(mixes):
        y = fr(M[(S, "NA2")] if (topo == "global" and mx == "balanced") else E[(S, f"E5:NA2 {topo} {mx}")])
        xx = i + (j - 0.5) * 0.32
        c.scatter(xx + rng.uniform(-0.06, 0.06, len(y)), y, s=7, color=col, alpha=0.35, lw=0, zorder=2)
        m, ci = mci(y)
        c.errorbar(xx, m, yerr=ci, fmt=mk, color=col, ms=5.5, mec="white", mew=0.6, lw=1.1, capsize=2, zorder=4,
                   label=("Global network" if topo == "global" else "Ring network") if i == 0 else None)
c.set_xticks(range(3)); c.set_xticklabels(["Balanced", "Maverick-\nheavy", "Follower-\nheavy"])
c.set_ylabel("Final regret (stationary)"); c.set_xlim(-0.5, 2.5)
c.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, handletextpad=0.2)
clean(c, "C  Social structure (KDP, no absorption)")
fig.savefig("fig3_mechanisms.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# ================================================================ Figure 4 (robustness added in revision)
fig, axs = plt.subplots(1, 3, figsize=(10.4, 3.7), gridspec_kw={"wspace": 0.85, "width_ratios": [1.25, 1.1, 1.1]})
a, b, c = axs
rows = [("Document-centric", M[(S, "DOC")], BASE), ("Doc + meta-analysis", R[("baseline", "DOCM", S, ())], BASE),
        ("Doc + replication", R[("baseline", "DOCR", S, ())], BASE), ("KDP, legacy start", E[(S, "E2:NA2-legacy-init-only")], BLUE),
        ("KDP, adaptive absorption", M[(S, "ADA")], BLUE), ("KDP, random start", M[(S, "NA2")], BLUE)]
for i, (lab, rr, col) in enumerate(rows):
    y = fr(rr); m, ci = mci(y)
    err = np.array([x["err_final"] if "err_final" in x else x["error"][-1] for x in rr], float); me, cie = mci(err)
    a.errorbar(m, i - 0.13, xerr=ci, fmt="o", color=col, ms=5, mec="white", mew=0.6, lw=1.1, capsize=2)
    a.errorbar(me, i + 0.13, xerr=cie, fmt="s", color=col, mfc="white", ms=4.5, lw=1.1, capsize=2)
a.set_yticks(range(len(rows))); a.set_yticklabels([r[0] for r in rows]); a.invert_yaxis()
a.set_xlabel("Mean (95% CI), stationary world"); a.set_xlim(0, 0.34)
a.legend(handles=[Line2D([], [], color=INK2, marker="o", lw=0, label="Regret of canon"),
                  Line2D([], [], color=INK2, marker="s", mfc="white", lw=0, label="Knowledge-state error")],
         frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, handletextpad=0.2)
clean(a, "A  Matched baselines", grid="x")

scores = ["lens divergence", "all-evidence estimate", "maximum report", "few independent agents", "observation variance"]
short = ["Lens divergence", "All-evidence est.", "Maximum report", "Few independent\nagents", "Obs. variance"]
for sc, col, mk, off in ((S, BLUE, "o", -0.14), (H, ORANGE, "D", 0.14)):
    rr = R[("credit", "ADA", sc, ())]
    for i, nm in enumerate(scores):
        m, lo, hi = boot_med([r["h3"][("L2", nm)][0] for r in rr])
        b.errorbar(m, i + off, xerr=[[m - lo], [hi - m]], fmt=mk, color=col, ms=5, mec="white", mew=0.6, lw=1.1, capsize=2)
b.axvline(0.5, color=INK2, lw=0.8, ls=":")
b.set_yticks(range(len(scores))); b.set_yticklabels(short); b.invert_yaxis(); b.set_xlim(0, 1)
b.set_xlabel("Median AUC, decoupled label (L2)")
b.legend(handles=h[:2], frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, handletextpad=0.2)
clean(b, "B  Alarm vs simple scores", grid="x")

meas = ["KDP credit (L1)", "claims explored", "replications", "cumulative uses", "attention"]
mshort = ["KDP credit", "Claims explored", "Replications", "Cumulative uses", "Attention"]
for sc, col, mk, off in ((S, BLUE, "o", -0.14), (H, ORANGE, "D", 0.14)):
    rr = R[("credit", "ADA", sc, ())]
    for i, nm in enumerate(meas):
        m, lo, hi = boot_med([r["h4"][("U_err", nm)] for r in rr])
        c.errorbar(m, i + off, xerr=[[m - lo], [hi - m]], fmt=mk, color=col, ms=5, mec="white", mew=0.6, lw=1.1, capsize=2)
c.axvline(0, color=INK2, lw=0.8, ls=":")
c.set_yticks(range(len(meas))); c.set_yticklabels(mshort); c.invert_yaxis(); c.set_xlim(-0.75, 0.6)
c.set_xlabel("Median Spearman ρ with true contribution")
c.legend(handles=h[:2], frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, handletextpad=0.2)
clean(c, "C  Credit vs simple measures", grid="x")
fig.savefig("fig4_robustness.png", dpi=300, bbox_inches="tight"); plt.close(fig)
print("ok")
