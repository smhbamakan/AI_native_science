# Pre-registration v2: refined replay tests on the FReD holdout half

Version 2.0, written 16 November 2026 after development on one half of the data and before any access to the holdout half.

## 1. History
Pilot v1 (PREREGISTRATION_FReD_pilot.md, SHA-256 7b11f62f…9251d) was run on all of FReD. One of three tests was supported (author overlap). Post hoc checks found two design problems: (i) the H-P3 divergence score was almost a function of original sample size alone (Spearman −0.95), so the test compared a noisy version of n with n; (ii) H-P1 had no usable power (bootstrap interval width 0.66 in AUC). The analyst also saw, on the full data, that the ratio of replication to original sample size relates to the outcome. Original papers were then split at random (seed 20261117) into a development and a holdout half (split.csv, SHA-256 335a9098…0477). All refinement used only the development half. The holdout half is locked in code until this document is hashed.

## 2. What development showed (development half only)
900 effects from 337 papers with outcomes and inputs. A sceptical single-study lens (posterior z under a normal prior on the Fisher-z scale with τ estimated from replication effects, τ = 0.264) ranked failures with AUC 0.558, against 0.561 for the original z alone and 0.506 for original n. The replication/original sample-size ratio predicted failure (AUC 0.579); the alarm kept its association after adjustment for it (standardized logistic coefficient 0.31). Author overlap remained associated with success after adjustment for the size ratio (log odds 1.14), within every tertile of the ratio.

## 3. Frozen definitions (code: common.py, confirm_v2.py)
* Outcome: successful = 1, failed = 0; other classes excluded. Sensitivity: mixed = 0.
* Original evidence z: Fisher-z test statistic atanh(|r_o|)·√(n_o − 3) when the original effect size converts to r (FReD conversion rules ported in common.py; chi-squared not converted); otherwise the two-sided original P value converted to z.
* Sceptical lens (KDP shrinkage lens for a single study): k = τ²(n_o − 3)/(τ²(n_o − 3) + 1) with τ = 0.264 fixed from development; posterior z_post = √k · z. Alarm A = −z_post.
* Replication design: log(n_r / n_o).
* Author overlap: `author_overlap` > 0.

## 4. Confirmatory tests on the holdout half
**T1 (lens alarm carries information beyond replication design).** In a logistic regression of failure on standardized A and standardized log(n_r/n_o), the coefficient of A is positive. Test: one-sided, 10,000 cluster-bootstrap resamples over original papers; supported if the 5th percentile is above 0. Also reported: AUC of A for failure with 95% interval.

**T2 (no advantage of the sceptical lens over the raw original z; equivalence).** The difference AUC(A) − AUC(−z) lies within ±0.03. Test: 90% cluster-bootstrap interval (over original papers) entirely inside [−0.03, 0.03]. This tests our development finding that shrinking single-study evidence adds no ranking information beyond z; if it holds, single-study lens plurality cannot serve as an alarm in such data.

**T3 (dependence premise, adjusted).** In a logistic regression of success on author overlap (0/1) and standardized log(n_r/n_o), the overlap coefficient is positive. Test: one-sided, 10,000 cluster-bootstrap resamples over replication reports; supported if the 5th percentile is above 0. Also reported: raw success proportions.

Multiplicity: Holm across T1 and T3 (one-sided bootstrap P = share of resamples ≤ 0); T2 is an equivalence test reported separately.

## 5. Descriptive (no test)
The attestation-log lens of pilot v1 (H-P1) on the holdout half: AUC with 95% interval and sample size, reported as descriptive because development and v1 showed it cannot reach usable power in FReD.

## 6. Run once
confirm_v2.py is run once on the holdout half after this file is hashed. Results are reported whatever they are; any deviation is reported as such.
