"""Regression test: default settings reproduce stored confirmatory results exactly (first three seeds of each condition)."""
import pickle, numpy as np
from aepso import run
M = pickle.load(open("main_results.pkl", "rb"))
for sc in ("stationary", "shift"):
    for c in ("DOC", "NA0", "NA2", "EAG", "FIX", "ADA"):
        for k, s in enumerate(range(1000, 1003)):
            r = run(c, sc, s)
            assert np.allclose(r["regret"], M[(sc, c)][k]["regret"]), (sc, c, s)
print("regression ok")
