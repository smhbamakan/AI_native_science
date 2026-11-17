# AI_native_science

Code, pre-registrations and results for the manuscript **"From documents to epistemic transactions: a computable layer for AI-native science"**, which proposes the Knowledge Delta Protocol (KDP): contributions are signed transactions on scoped, addressable claims; public, versioned lenses compute states of knowledge; related work, epistemic impact and credit are computed rather than written; and legacy literature enters through a declared absorption profile α = (k, τ, γ, θ, a₀).

The repository contains three independent parts.

| Folder | What it is |
|---|---|
| `prototype/` | Minimal reference implementation of KDP: transaction schema, the worked example log (Table S4), Bayesian and replication-weighted lenses, retraction and qualification semantics, epistemic impact and exact Shapley credit |
| `simulation/` | AE-PSO testbed (absorptive epistemic particle swarm) with a hidden ground truth: pre-registered confirmatory runs, exploratory and revision analyses, statistics, figures and unit tests |
| `replay/` | Two-stage pre-registered replay of 2,164 original–replication pairs from the FORRT Replication Database (FReD) |
| `docs/` | Reference audit for the manuscript |

## Requirements

Python 3.11+ with the packages in `requirements.txt`. The replay additionally needs `pandas`, `pyreadr` and `scikit-learn`:

```
pip install -r requirements.txt pandas pyreadr scikit-learn
```

All runs are deterministic given their seeds.

## Quick checks

```
cd simulation && python3 test_equations.py && python3 test_regression.py
cd ../prototype && python3 kdp_prototype.py
```

`test_equations.py` links the manuscript's equations to the code, including the worked example of the shrinkage lens (S = 0.52). In the Patterns version of the manuscript, the AE-PSO attractor and velocity/budget equations are (9)–(10), the observation model is (11), the shrinkage lens is (12) and adaptive trust is (13); test names follow the earlier numbering (`test_eq9`/`10`/`11` = AE-PSO, `test_eq12` = observation, `test_eq13` = shrinkage, `test_eq14` = trust).

## Simulation (`simulation/`)

* `PREREGISTRATION.md`: pre-registered plan with amendment log (SHA-256 `d0f48e654daa5d1dcfe82e60098a4cba473e342188f5e8252de4095658be07c3`). The plan was kept locally before the confirmatory runs; the hash fixes its content from release onward and does not by itself prove when it was written.
* `aepso.py`: world, legacy phase, lenses, absorption and swarm. `aepso_v1_frozen.py` is the version used for the confirmatory runs, before revision-only options were added; defaults reproduce the confirmatory results exactly (`test_regression.py`).
* Runs: `run_main.py` (360 confirmatory runs), `run_explore.py` and `run_e6.py` (exploratory), `revision_analyses.py` (990 runs added during peer review). Stored outputs: `main_results.pkl`, `explore_results.pkl`, `e6_results.pkl`, `revision_results.pkl`, with per-seed CSV exports from `export_csv.py`.
* Statistics: `analyze.py`, `revision_stats.py`, `explore_paired.py`, `power.py` (outputs `RESULTS_*.txt`, `*.json`).
* Figures: `make_figures_patterns.py` (Figures 2–4 of the Patterns manuscript), `make_figures_v4.py` (earlier version).
* `configs_index.json`: identifiers for every row of the full settings table (Table S5).

Full reproduction (a few minutes on a multicore machine):

```
cd simulation
python3 run_main.py && python3 run_explore.py && python3 run_e6.py && python3 revision_analyses.py
python3 analyze.py && python3 revision_stats.py && python3 explore_paired.py && python3 power.py
python3 make_figures_patterns.py && python3 export_csv.py
```

All analyses after the confirmatory runs are exploratory. Known deviation: epistemic error is computed over admitted claims, not over all claims with evidence as written in the plan; it is not used in any confirmatory test.

## Replay of real replication records (`replay/`)

Data: the snapshot shipped with the FReD R package (https://github.com/forrtproject/FReD, commit `052d314465b9502dcb8a7940388eab1270a6cd0c`, 12 November 2026), included as `fred_snapshot_data.RDS` under the FReD MIT licence (`FReD_LICENSE.md`). Please also cite FReD: Hartmann, H., Röseler, L., Wallrich, L., et al. FReD: FORRT Replication Database, version 2.4.1. OSF, https://doi.org/10.17605/OSF.IO/9R62X.

**Stage 1 (pilot on all data).** `PREREGISTRATION_FReD_pilot.md` and the data file were hashed before analysis (`HASHES.txt`). `fred_replay.py` runs the three pre-registered tests; `fred_diag.py` and `fred_design_check.py` are diagnostics added afterwards (exploratory).

```
cd replay && python3 -I fred_replay.py && python3 -I fred_diag.py && python3 -I fred_design_check.py
```

**Stage 2 (refinement and held-out confirmation, `replay/v2/`).** Original papers were split at random (`split.csv`, hash in `SPLIT_HASH.txt`). `common.py` refuses to load the held-out half unless `PREREG_V2_HASH.txt` exists; that file records the hashes of the stage-2 plan, code and split made before the single held-out run. `dev_explore.py` uses the development half only; `confirm_v2.py` implements the pre-registered tests.

```
cd replay/v2 && python3 -I confirm_v2.py holdout
```

| Stage | Test | Result |
|---|---|---|
| 1 | Author-overlap replications succeed more often | Supported (0.86 vs 0.50) |
| 1 | Lens divergence on single studies flags failures | Not supported (score reduced to sample size) |
| 1 | Attestation log predicts later replications | Not supported (no usable power) |
| 2 (held-out) | Sceptical lens predicts failure beyond replication design | Supported, weak (AUC 0.57) |
| 2 (held-out) | Sceptical lens equivalent to raw original test statistic | Equivalent (ΔAUC 0.008) |
| 2 (held-out) | Author overlap, adjusted for replication design | Supported (0.94 vs 0.46) |

## Prototype (`prototype/`)

`kdp_prototype.py` prints the state trajectories of the worked example under two lenses and writes `kdp_example_log.json`. Signatures are placeholders; nothing in the prototype is cryptographically secure.

## Citation

See `CITATION.cff`. A licence for this repository has not been chosen yet.
