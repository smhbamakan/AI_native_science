import numpy as np, pickle
from scipy.stats import mannwhitneyu, fisher_exact, wilcoxon
R = pickle.load(open("main_results.pkl", "rb"))
def col(sc, c, k): return np.array([r[k] for r in R[(sc, c)]], dtype=float)
print("condition summary (mean ± sd; false canonisation count /30)")
for sc in ("stationary", "shift"):
    for c in ["DOC","NA0","NA2","EAG","FIX","ADA"]:
        fr=col(sc,c,"final_regret"); fc=col(sc,c,"false_canon"); er=col(sc,c,"early_regret"); ps=col(sc,c,"post_shift_regret")
        err=np.array([np.nanmean(r["error"][-1:]) for r in R[(sc,c)]])
        art=np.array([r["canon_art"][-1] for r in R[(sc,c)]])
        print(f"{sc:10s} {c}: final {fr.mean():.3f}±{fr.std():.3f} | false {int(fc.sum()):2d} | early {er.mean():.3f} | post-shift {ps.mean():.3f} | err {np.nanmean(err):.3f} | artifact canon {int(art.sum())}")
def mw(a,b,alt="less"): return mannwhitneyu(a,b,alternative=alt).pvalue
def fe(a,b): return fisher_exact([[a.sum(),len(a)-a.sum()],[b.sum(),len(b)-b.sum()]],alternative="less").pvalue
S="stationary"
print("\nH1 NA2 vs NA0 (stationary): regret p=%.4f, false-canon Fisher p=%.4f" % (mw(col(S,"NA2","final_regret"),col(S,"NA0","final_regret")), fe(col(S,"NA2","false_canon"),col(S,"NA0","false_canon"))))
print("H1 NA2 vs NA0 (shift):      regret p=%.4f, false-canon Fisher p=%.4f" % (mw(col("shift","NA2","final_regret"),col("shift","NA0","final_regret")), fe(col("shift","NA2","false_canon"),col("shift","NA0","false_canon"))))
print("H2a ADA vs NA2 early regret (stationary) p=%.4f" % mw(col(S,"ADA","early_regret"),col(S,"NA2","early_regret")))
print("H2b ADA vs EAG false canon (stationary) Fisher p=%.4f ; regret p=%.4f" % (fe(col(S,"ADA","false_canon"),col(S,"EAG","false_canon")), mw(col(S,"ADA","final_regret"),col(S,"EAG","final_regret"))))
print("H2c shift post-shift regret ADA vs FIX p=%.4f ; ADA vs EAG p=%.4f" % (mw(col("shift","ADA","post_shift_regret"),col("shift","FIX","post_shift_regret")), mw(col("shift","ADA","post_shift_regret"),col("shift","EAG","post_shift_regret"))))
auc=col(S,"ADA","auc"); print("H3 AUC median %.3f IQR [%.3f, %.3f] n=%d" % (np.nanmedian(auc), np.nanpercentile(auc,25), np.nanpercentile(auc,75), np.isfinite(auc).sum()))
rk=col(S,"ADA","rho_kdp"); ra=col(S,"ADA","rho_att"); ok=np.isfinite(rk)&np.isfinite(ra)
print("H4 Spearman with true contribution: KDP credit median %.3f, attention median %.3f; Wilcoxon p=%.2e" % (np.median(rk[ok]), np.median(ra[ok]), wilcoxon(rk[ok],ra[ok],alternative="greater").pvalue))
