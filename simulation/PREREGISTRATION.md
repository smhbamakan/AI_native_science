# Pre-registration: absorptive swarm testbed for the Knowledge Delta Protocol

Registered before any simulation run. Date: 2026-11-16. Any deviation from this plan
will be reported as such in the manuscript.

## 1. Purpose

To test, in a simulated science world with a hidden ground truth, whether the core
mechanisms of the Knowledge Delta Protocol (KDP) — replication-gated admission,
lens-relative knowledge states, tunable absorption of legacy literature and
counterfactual (Shapley) credit — behave as the conceptual framework claims.
The testbed is an Absorptive Epistemic Particle Swarm (AE-PSO). It is a model of
knowledge production, not a proposed optimisation algorithm.

## 2. Simulated world

* Hypothesis space X = [0,1]^2, discretised into 20 x 20 = 400 addressable claims (cells).
* Ground truth T(x): one global peak (height 1.0), three local peaks (0.55-0.70), baseline 0.1.
* Artifact region: a low-truth region (T ≈ 0.15) in which agents using a flawed method
  (probability p_flaw = 0.3 per agent, fixed for life) observe an upward bias of +0.9
  (analogue of data leakage or a shared methodological artifact).
* Heteroscedastic noise: sigma = 0.15 everywhere, 0.40 in a high-noise region around one
  local peak (source of winner's-curse false positives).
* Shift scenario: halfway through the KDP phase the global peak moves to a new location and
  the old peak decays to 0.4 (analogue of a paradigm shift that makes legacy knowledge partly
  obsolete).

## 3. Phases

1. Legacy phase (100 iterations, 40 agents): document-centric dynamics. Single evaluations,
   publication bias (only observations with y > 0.5 are published), community attractor =
   best single published report. Published records form the legacy literature H.
2. KDP phase (150 iterations, 40 new agents): one of the conditions below.

Equal experimental budget in every condition: one observation per agent per iteration.
In KDP conditions a fraction rho = 0.2 of observations are replications.

## 4. Conditions

| Code | Name | Legacy use | Admission | Lens for attractors |
|---|---|---|---|---|
| DOC | Document-centric | read at face value as reports | none | best single report |
| NA0 | KDP, no absorption, m = 0 | ignored | none | replication lens |
| NA2 | KDP, no absorption, m = 2 | ignored | >= 2 independent agents | replication lens |
| EAG | KDP, eager import | all legacy records imported into the log as transactions | m = 2 | replication lens |
| FIX | KDP, fixed absorption | absorbed attractor, a0 = 0.5 fixed | m = 2 | replication lens |
| ADA | KDP, adaptive absorption | absorbed attractor, a0 adapted to conflict | m = 2 | replication lens |

Default absorption profile alpha = (k: radius 0.25, tau: 50 iterations, g: layer L2
(cell-aggregated legacy evidence), theta: >= 2 legacy records, a0 as per condition).

Scenarios: stationary and shift. Seeds: 30 per condition x scenario.

## 5. Hypotheses and tests

* **H1 (replication gating).** NA2 has a lower false-canonisation rate and lower final regret
  than NA0. Test: one-sided Wilcoxon rank-sum over seeds (alpha = 0.05), plus Fisher exact test
  on false-canonisation counts.
* **H2 (tunable absorption, origin problem).**
  (a) ADA reaches lower regret than NA2 early in the KDP phase (mean regret over iterations 1-30);
  (b) ADA has a lower false-canonisation rate than EAG at the end of the KDP phase;
  (c) in the shift scenario ADA has lower mean regret after the shift than FIX and EAG.
  Tests: one-sided Wilcoxon rank-sum; Fisher exact for (b).
* **H3 (lens divergence as an alarm).** The absolute divergence between an all-evidence lens
  (legacy + new evidence at face value) and the replication lens identifies claims whose
  all-evidence estimate overstates the truth by more than 0.2 with AUC > 0.70 (median over seeds,
  ADA condition, stationary).
* **H4 (credit validity).** Across agents within a run, KDP credit (Shapley value of each agent's
  transactions with respect to the change they induce in the replication-lens state) correlates
  more strongly with true epistemic contribution (Shapley value with respect to the reduction of
  error against ground truth) than attention does (number of times an agent's first-asserted claims
  served as another agent's social attractor; a citation analogue with priority attribution).
  Test: paired one-sided Wilcoxon signed-rank on per-run Spearman correlations (ADA, stationary).

## 6. Primary metrics

* Regret = max T - T(c*), where c* is the canonised claim (highest admitted state value).
* False canonisation = regret > 0.3 at the end of the KDP phase.
* Epistemic error = mean |S(c) - T(c)| over claims with at least one observation.
* Swarm diversity = mean pairwise distance between agents.

## 7. Exploratory (not confirmatory) analyses

* One-at-a-time sensitivity of alpha components (k, tau, g, theta, a0) in ADA and FIX.
* Social structure: topology (global vs ring) x researcher mix (balanced, maverick-heavy,
  follower-heavy) in ADA.

## Amendment log (added after registration; nothing above was changed)

* 2026-11-16: Before the confirmatory runs, one single-seed smoke test checked that the code
  executed, and world geometry was checked against legacy-phase-only criteria (legacy literature
  locates the true peak in ~73% of worlds, contains the artifact in 40%, best single report
  misleading in 40%). No parameter was changed as a result.
* Confirmatory results are stored unchanged in RESULTS_confirmatory.txt.
* All analyses run after the confirmatory results are labelled exploratory.
