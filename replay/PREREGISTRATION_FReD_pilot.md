# Pre-registration: replaying replication records through KDP lenses (FReD pilot)

Version 1.0, written 16 November 2026, before any analysis that relates predictors to replication outcomes.

## 1. What has been seen before writing this plan
Data: FORRT Replication Database (FReD), R-package snapshot `inst/extdata/snapshot/data.RDS` from https://github.com/forrtproject/FReD (last commit 12 November 2026; commit 052d314). Before writing, the analyst inspected only descriptive structure: column names; 2,164 rows (original effect x replication effect pairs); 786 distinct original DOIs; 1,116 replication reports (`fred_id`); marginal counts of `reported_success` (985 successful, 787 failed, 359 mixed, 18 flawed, 10 descriptive only, 5 uninformative); marginal counts of `author_overlap`; availability of sample sizes, p values and years; and the number of replications that have an earlier replication of the same original paper by another report (194 rows, 48 papers). No association between any predictor and any outcome was computed or viewed.

## 2. Purpose
The simulation in the manuscript tested KDP mechanisms against a known ground truth. This pilot asks whether three of those mechanisms are informative on real replication records. It is a pilot: the data come mainly from psychology and neighbouring social sciences, claim identity is taken from FReD's coding, and conclusions apply to this dataset only.

## 3. Units, outcome and coding
* Unit of prediction: one row (an original effect paired with one replication effect).
* Outcome: replication success, 1 = `successful`, 0 = `failed`. Rows coded `mixed`, `statistically successful but flawed`, `descriptive only` or `uninformative` are excluded from the outcome set (primary analysis). Sensitivity: `mixed` coded 0.
* Claim (for the log in H-P1): the original paper (`doi_o`, lower-cased). Replication report: `fred_id` with any suffix after the first underscore-delimited number kept as is; reports sharing the same reference text (`ref_r`, first 60 normalized characters) are one report.
* Original evidence strength: two-sided p value `pval_value_o` converted to a z score, z = Φ⁻¹(1 − p/2), with p clipped to [1e-12, 0.999]. Bounds ("<", "<=") are used at the bound; ">" and ">=" values are used as given. Non-numeric p values are missing. Original sample size: `n_o` (numeric, > 3).

## 4. Lenses (fixed before analysis)
* **Face-value lens** (document-centric reading): S_F = logistic(z / 1.96).
* **Sceptical shrinkage lens** (a single-study analogue of the replication-weighted lens): the observed standardized effect r = z/√n is shrunk towards 0 under a normal prior with standard deviation τ = 0.2 on the correlation scale, giving z_S = z · τ²n / (τ²n + 1) and S_S = logistic(z_S / 1.96).
* **Replication-weighted log lens** (equation (13) analogue, prior μ0 = 0.5, n0 = 2): each earlier transaction on the claim is a vote. The original publication is one vote of value 1 and weight 0.5 (a non-replicating assertion). Each earlier replication report is one vote with weight 1 whose value is the mean of its effects' outcomes (successful 1, failed 0, mixed or flawed 0.5; descriptive-only and uninformative effects ignored). S_L = (n0 μ0 + Σ w v) / (n0 + Σ w).
* "Earlier" means a replication year (`year_r`) strictly smaller than the year of the target row. Rows without `year_r` cannot be targets in H-P1 and cannot contribute votes.

## 5. Hypotheses and tests (confirmatory)
**H-P1 (attestation log improves the knowledge state).** Among target rows whose original paper has at least one earlier replication by another report, the replication-weighted log lens S_L ranks successful replications above failed ones better than the best original-evidence score. Original-evidence scores: S_S, z (equivalently S_F), and n_o. Test: AUC(S_L) − max AUC(original-evidence scores), with a 95% cluster bootstrap interval (10,000 resamples of original papers); supported if the one-sided lower 95% bound (5th percentile) is above 0. The "max" is taken within each bootstrap resample.

**H-P2 (dependence premise).** Replications that share at least one author with the original (`author_overlap` > 0) report success more often than replications without overlap. Test: difference in success proportions, one-sided cluster bootstrap over replication reports (10,000 resamples); supported if the 5th percentile is above 0.

**H-P3 (lens divergence as an alarm on single studies).** Divergence D = S_F − S_S (non-negative by construction) ranks failed replications above successful ones (AUC for failure > 0.5), and does so better than each single original-evidence baseline taken alone (z, n_o). Test: AUC(D, failure) − max(AUC(−z, failure), AUC(−n_o, failure)), cluster bootstrap over original papers; supported if the 5th percentile of the difference is above 0 and the 5th percentile of AUC(D) is above 0.5.

Multiplicity: Holm adjustment over the three bootstrap one-sided P values (P = share of resamples ≤ 0). All three are reported whatever the result.

## 6. Secondary and exploratory analyses (labelled as such)
* H-P3 against a cross-validated logistic model using both log z and log n_o (10-fold, folds by original paper), the fairest baseline that uses the same information as D.
* Brier score and calibration of S_L, S_S and S_F as probabilities of success (H-P1 sample and full sample).
* Sensitivity: mixed coded as failure; τ = 0.1 and 0.3; H-P1 with claims keyed by original effect description instead of paper.
* Overall success rate and AUCs of z and n_o on the full sample (context).

## 7. Exclusions and stopping
No outliers removed. Rows with a missing input are excluded only from analyses needing that input. Analysis is run once on the snapshot above; any deviation will be reported as such.

## 8. What would count against KDP
If H-P1 fails, the replay gives no evidence that recording attestations improves prediction beyond the original document. If H-P3 fails against single baselines or the cross-validated model, lens divergence on single studies adds nothing beyond its inputs. If H-P2 fails, the dependence rule's empirical premise is not supported in these data.
