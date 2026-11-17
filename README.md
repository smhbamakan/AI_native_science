# AI_native_science

Code accompanying a manuscript currently under peer review. It is provided so that reviewers and editors can run and inspect the analyses. Descriptions of the study, its pre-registration documents and its results are not included here and will be added after publication.

## Contents

| Folder | Contents |
|---|---|
| `prototype/` | Minimal reference implementation of the Knowledge Delta Protocol (transaction schema, two lenses, retraction and qualification semantics, impact and exact Shapley credit) with an example log |
| `simulation/` | Agent-based simulation (absorptive particle swarm on a synthetic epistemic landscape), run scripts, statistics, figure script and tests |
| `replay/` | Analysis of replication records from the FORRT Replication Database (FReD), stage 1 and stage 2 (`v2/`) |

## Requirements

Python 3.11 or later.

```
pip install -r requirements.txt pandas pyreadr scikit-learn
```

All runs are deterministic given their seeds (1000–1029).

## Running

Prototype:

```
cd prototype && python3 kdp_prototype.py
```

Simulation (outputs are written to `simulation/`):

```
cd simulation
python3 test_equations.py
python3 run_main.py && python3 test_regression.py
python3 run_explore.py && python3 run_e6.py && python3 revision_analyses.py
python3 analyze.py && python3 revision_stats.py && python3 explore_paired.py && python3 power.py
python3 make_figures.py && python3 export_csv.py
```

`aepso.py` is the simulation; `aepso_v1_frozen.py` is the version used for the confirmatory runs, before exploratory options were added. `test_regression.py` checks that the current code reproduces the confirmatory runs.

Replication-record analysis:

```
cd replay && python3 -I fred_replay.py && python3 -I fred_diag.py && python3 -I fred_design_check.py
cd v2 && python3 -I dev_explore.py && python3 -I confirm_v2.py holdout
```

`replay/v2/common.py` refuses to load the held-out half unless `PREREG_V2_HASH.txt` is present; that file and `replay/HASHES.txt` record SHA-256 hashes of the analysis plans, code, data split and data snapshot made before the corresponding analyses. The plans themselves are available to the journal on request.

## Data

`replay/fred_snapshot_data.RDS` is the data snapshot shipped with the FReD R package (https://github.com/forrtproject/FReD, commit `052d314465b9502dcb8a7940388eab1270a6cd0c`), redistributed under its MIT licence (`replay/FReD_LICENSE.md`). Please cite FReD: Hartmann, H., Röseler, L., Wallrich, L., et al. FReD: FORRT Replication Database, version 2.4.1. OSF, https://doi.org/10.17605/OSF.IO/9R62X.

## Licence

A licence for this code has not yet been chosen.
