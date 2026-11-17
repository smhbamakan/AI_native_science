"""Export per-seed results from the pickles to CSV (one row per setting x world x seed)."""
import pickle, csv, numpy as np
M = pickle.load(open("main_results.pkl", "rb")); E = pickle.load(open("explore_results.pkl", "rb"))
E6 = pickle.load(open("e6_results.pkl", "rb")); R = pickle.load(open("revision_results.pkl", "rb"))
with open("results_per_seed.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["source", "setting", "world", "overrides", "seed", "final_regret", "false_canon", "early_regret", "post_shift_regret", "final_error", "auc_H3", "rho_kdp_H4", "rho_att_H4"])
    for src, D in (("confirmatory", M), ("exploratory", E), ("exploratory_E6", E6)):
        for (sc, c), rows in D.items():
            for k, r in enumerate(rows):
                err = r["error"][-1] if "error" in r else ""
                w.writerow([src, c, sc, "", 1000 + k, r["final_regret"], r["false_canon"], r["early_regret"], r["post_shift_regret"], err, r.get("auc", ""), r.get("rho_kdp", ""), r.get("rho_att", "")])
    for (kind, c, sc, kw), rows in R.items():
        for k, r in enumerate(rows):
            w.writerow(["revision_" + kind, c, sc, ";".join(f"{a}={b}" for a, b in kw), 1000 + k, r["final_regret"], r["false_canon"], r["early_regret"], r["post_shift_regret"], r["err_final"], "", "", ""])
with open("results_H3_H4_revision.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["condition", "world", "seed", "measure", "label_or_utility", "score_name", "value", "pr_auc"])
    for (kind, c, sc, kw), rows in R.items():
        if kind != "credit": continue
        for k, r in enumerate(rows):
            for key, val in r["h3"].items():
                if key == "prev": continue
                w.writerow([c, sc, 1000 + k, "H3_AUC", key[0], key[1], val[0], val[1]])
            for key, val in r["h4"].items():
                if isinstance(key, tuple): w.writerow([c, sc, 1000 + k, "H4_spearman", key[0], key[1], val, ""])
print("csv ok")
