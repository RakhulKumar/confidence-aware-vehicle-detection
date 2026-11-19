# Theoretical foundations

This is the reference for every formula in `src/confdet/`. Each section names the function that implements it. If a formula changes in code, change it here in the same commit.

Notation: an image has annotated vehicles (ground truth) `g_j` and the detector's raw candidates `(b_k, c_k, s_k)` (box, class, confidence), collected down to a near-zero confidence floor. `tau` is the standard operating threshold (0.25).

---

## 1. Occlusion severity `o_j` (annotation side)

For an annotated box `b_j`, with the other annotated boxes of the same image:

```
o_j = area( b_j ∩ ⋃ { b_i : b_i nearer than b_j } ) / area(b_j)        in [0, 1]
```

"Nearer" has no depth data behind it and is a proxy: with `bottom_y` (default), `b_i` is nearer when its bottom edge is lower in the frame, which is the geometry of an oblique fixed camera; with `area`, when it is larger. Ties are "not nearer".

`o_j` is an **analysis covariate**. It exists only where annotations exist, so no decision rule may use it.

Known weaknesses, to state in a paper: box overlap over-estimates pixel occlusion; an unlabelled occluder is invisible to it; the depth proxy can be wrong for elevated or tall vehicles; and heavily hidden vehicles are less likely to have been annotated at all. The first real run shows the size of this: at annotated severity 0.8-1.0 YOLO26 still found 69% of vehicles, while with 80% of the box actually covered (controlled probe) it found 28%.

Code: `data/occlusion.py::occlusion_severity_array` (reference version: `compute_occlusion_severity`).

## 2. Occlusion context `ô_k` (prediction side, deployable)

Let `D = { k : s_k ≥ tau_ctx }` be the detections the detector is already confident about (`tau_ctx = posthoc.context_conf`). For any candidate `k`:

```
dup_k = max over d in D, d ≠ k of  IoU(b_k, b_d)
ô_k   = area( b_k ∩ ⋃ { b_d : d in D, d ≠ k, b_d nearer than b_k, IoU(b_k, b_d) < 0.5 } ) / area(b_k)
```

`ô_k` answers "how much of this candidate sits behind a vehicle the detector is sure about". The IoU exclusion keeps a second box on the *same* vehicle from being read as an occluded neighbour; that case is described by `dup_k`.

Both use detector outputs only, so they exist at test time. Every decision rule in this project conditions on `ô`, never on `o`. How well `ô` tracks `o` is itself reported (RQ1, `spearman_occlusion_vs_deployable_context`).

Code: `data/occlusion.py::occlusion_context`.

## 3. Labelling, object scores, and why nothing is ever re-matched

Per image, candidates are processed in descending confidence. Each takes the still-unmatched annotated vehicle of its class with the highest IoU, if that IoU ≥ 0.5 (COCO protocol). A matched candidate is a true positive, an unmatched one a false positive.

Each annotated vehicle `j` therefore has at most one candidate. Its **object score** is

```
S_j = confidence of its matched candidate,      S_j = −∞ if nothing was proposed ("unproposed").
```

Because matching runs in confidence order, the matching of the subset `{s_k ≥ t}` is exactly the prefix of the full matching. Hence, for any threshold `t`:

```
vehicle j is found at t        ⇔  S_j ≥ t
candidate k is a false alarm   ⇔  k is a false positive and s_k ≥ t
```

Every operating point, policy and metric is computed from these fixed labels (`tests/test_evaluation.py` checks the prefix property against an independent matcher).

**Matching modes.** What counts as "found" is a choice, made explicit:

| mode | a vehicle is found when | use |
|---|---|---|
| `class_agnostic` | any candidate box matches it | primary. "Silently missed" means no output at all |
| `coarse` | a candidate of the same *merged* class matches it (classes mapped by name into `evaluation.common_classes`) | when coarse type matters |
| `native` | a candidate of the same fine-grained class matches it | the confusion analysis (RQ5); comparison with standard mAP |

In the first real run 47% of native-mode misses were a detected vehicle with the wrong fine class. Those are naming errors, not silent failures, and they diluted every occlusion result. Each mode in `evaluation.matching_modes` is analysed into its own folder; the first is primary. On a dataset with a different taxonomy `native` falls back to `coarse`: class indices of two taxonomies are never compared. A class with no counterpart becomes an ignore region.

**Operating threshold.** `tau` is per detector: with `standard_operating_threshold: max_f1` it is the F1-optimal threshold on the calib role (searched on a 0.01 grid), stored in `calibration.json` and used by every later stage. One shared number means different operating regimes for different architectures (at 0.25: 3.4 false alarms per image for YOLO, 9.0 for RT-DETR).

Code: `evaluation/labeling.py::build_tables`, `pipeline.py::tables_from_records`, `pipeline.py::max_f1_threshold`.

## 4. Miss-aware conformal thresholds

The system keeps every candidate whose score is at least `t`. With `n` calibration vehicles and `k = ⌊α(n+1)⌋`:

```
t̂ = k-th smallest of { S_1, …, S_n }          (unproposed vehicles count as −∞)
```

If calibration and test vehicles are exchangeable, a new vehicle is found with probability at least `1 − α`:

```
P( S_test ≥ t̂ ) ≥ 1 − α.
```

*Why:* `S_test < t̂` requires `S_test` to rank among the `k` smallest of the `n+1` values, which has probability at most `k/(n+1) ≤ α`.

The target can be unreachable, and the code reports it instead of returning a threshold that looks fine:

- `k = 0`: too few calibration vehicles for this `α`.
- the `k`-th smallest value is `−∞`: more than a fraction `α` of vehicles have no candidate at all. No threshold recovers a vehicle that was never proposed. The smallest certifiable miss rate is `(n_unproposed + 1)/(n + 1)`. This number is a result in its own right: it is the floor on silent failures that no post-hoc method can go below.

**Occlusion-conditional (Mondrian) version.** A single threshold controls the miss rate on average only. Vehicles are stratified by occlusion context `ô` (edges `posthoc.context_edges`) and one threshold is fitted per stratum, giving `P(S ≥ t̂_g | stratum g) ≥ 1 − α` in each. A vehicle's stratum is its matched candidate's `ô`; an unproposed vehicle is assigned by evaluating `ô` at its annotated box (bookkeeping only; it is a miss under any threshold), with the same exclusion of confident boxes that sit on that very vehicle (IoU ≥ 0.5). A stratum with fewer than `min_group_size` calibration vehicles is merged with its neighbour before fitting; merging depends on counts, not scores, so validity is kept.

**Rank transform.** Within stratum `g` with calibration scores `S^(g)`,

```
r_k = ( #{ S^(g) ≤ s_k } + 1 ) / ( n_g + 1 )
```

and candidate `k` is kept by the level-`α` rule exactly when `r_k > α`. Sweeping one threshold over `r` traces the whole family of conditional rules, which makes `r` a policy score comparable to confidence at any cost (RQ2).

**Deployed output, three tiers.** `accepted`: `s_k ≥ tau` (the detector's normal output, never altered). `flagged`: below `tau` but kept by the conformal rule. `rejected`: the rest. If the detector's recall at `tau` already exceeds `1 − α`, the rule's threshold is stricter than `tau` and it flags nothing; the report says so.

**Assumptions to state.** (i) Exchangeability of calibration and test vehicles: guaranteed by the random, group-aware split within one dataset, not across datasets (RQ4 measures the break). (ii) Vehicles in one image are dependent; the guarantee is exact for exchangeable vehicles and is verified empirically with image-level bootstrap intervals.

Code: `uq/conformal.py`.

## 5. Occlusion-aware candidate score

```
P(true positive | k) = σ( β0 + β1 z + β2 ô + β3 dup + β4 z·ô + β5 z·dup ),      z = logit(s_k)
```

fitted by logistic regression on calibration candidates (features standardised, small ridge).

- **Nested baseline.** With `β2..β5 = 0` the score is monotone in confidence, i.e. the baseline ranking. The baseline is a special case, so the fitted score can rank worse only by estimation noise.
- **Monotone recalibration cannot help.** Temperature, Platt and isotonic scaling leave the ranking unchanged and so trace the same recall-versus-false-alarm curve as raw confidence. Only side information can move the curve. (`features="confidence"` fits Platt scaling so this can be checked.)

The score and the conformal thresholds applied to it are fitted on **disjoint halves** of the calib images.

Whether `ô` carries information for a given detector is empirical. RQ2 measures it either way.

Code: `uq/calibrator.py`, `pipeline.py::fit_posthoc`.

## 6. Calibration metrics

**D-ECE** (Küppers et al., 2020). Emitted detections (`s_k ≥ tau`) are binned by confidence:

```
D-ECE = Σ_m (|B_m| / N_det) · | precision(B_m) − mean confidence(B_m) |
```

A vehicle the detector says nothing about never enters the sum.

**OCE** (Park, Sobolewski & Azizan, arXiv:2412.01782). Mean over annotated vehicles of a Brier score between the vehicle's label and the class probabilities of the kept detection(s) overlapping it; a vehicle with no overlapping kept detection scores 1. It sees misses; a false alarm on empty road overlaps no vehicle and is not penalised. Implemented with one class and one score per box (what Ultralytics exposes): a detection's probability vector is its confidence on its predicted class and 0 elsewhere. This is an approximation of the published metric and is labelled as such.

**MA-ECE** (this project). Every emitted detection is the claim "a vehicle is here" with stated probability = confidence. Every missed vehicle is the same claim with stated probability 0 and outcome 1. Pooling both:

```
MA-ECE = ( N_det · D-ECE + N_miss ) / ( N_det + N_miss ) = (1 − φ) · D-ECE + φ,
φ = N_miss / (N_det + N_miss)                                  ("miss mass")
gap = MA-ECE − D-ECE = φ · (1 − D-ECE) ≥ 0
```

`φ` is a sum over missed vehicles, so occlusion bucket `b` accounts for exactly `N_miss,b / N_miss` of the gap. Compared with the bucket's share of vehicles, this states how over-represented occluded vehicles are in what D-ECE cannot see.

MA-ECE is deliberately simple: it is D-ECE plus an honest accounting term, on the same scale, penalising both miscalibrated detections (through D-ECE) and misses (through `φ`). The miss-aware Brier score is its proper-scoring counterpart.

**What MA-ECE cannot do.** Like any calibration metric it does not penalise a *calibrated* low-confidence box: a box at confidence 0.05 that is right 5% of the time costs nothing. On real detectors it therefore keeps improving as the threshold is lowered (it preferred 0.05 for all three in the first run, at up to 77 false alarms per image). It corrects D-ECE's blindness to misses; it is not a criterion for choosing an operating point and must not be presented as one.

**What is and is not new.** Object-level, miss-penalising calibration already exists (OCE), as does the observation that prediction-level ECE rewards discarding predictions (Kuzucu et al., 2024; Park et al.). What this project adds is the exact occlusion decomposition of the blind spot and its measurement in dense fixed-camera traffic. Claim that, not the idea of a miss-aware metric.

Code: `calibration/detection_ece.py`.

## 7. The research questions

### 7.1 RQ1: does uncertainty track occlusion?

Per vehicle: `u_j = 1 − S_j` (1 if unproposed), `missed_j = [S_j < tau]`. Reported: `n` per occlusion bucket first; miss rate and mean uncertainty per bucket; Spearman `ρ(o, u)`; the **partial** `ρ` given log box size and class; the logistic model `missed ~ o + log size + class` as an odds ratio per +0.1 severity.

The partial correlation and the regression exist because of a confound: under the bottom-edge proxy, occluded vehicles are on average farther and smaller, and small objects get lower confidence regardless of occlusion.

For a causal statement there is a controlled probe (`scripts/09`): the same vehicles are shown at severity 0…0.9 by pasting a real vehicle crop over them, holding everything else fixed.

Code: `evaluation/rq1_correlation.py`, `data/synthetic_occlusion.py`.

### 7.2 RQ2: recovery at matched cost

Below `tau` lies a pool of weak candidates. A policy is a score on them. For a budget of `b` extra false alarms per image, the recovery of a policy is the share of the standard threshold's misses it brings back, over all misses and over occluded misses (`o ≥ 0.2`). Policies: confidence (lower the threshold everywhere), Mondrian rank (section 4), occlusion-aware score (section 5). Differences from the baseline carry paired bootstrap intervals.

The **ceiling** is the share of misses that have any candidate; the rest are unrecoverable by any post-hoc rule.

Conformal validity is reported separately: recall of each rule alone (what is guaranteed) and of the deployed system (accepted + flagged), overall, per context stratum, and per annotated occlusion bucket.

Code: `evaluation/rq2_recovery.py`.

### 7.3 RQ3: what prediction-only calibration misses

At `tau`: D-ECE, MA-ECE, gap, `φ`, miss-aware Brier, OCE. Across a threshold sweep: the threshold each metric would choose (D-ECE is free to prefer a threshold at which most vehicles are dropped). The exact decomposition of the gap over occlusion buckets. Across detectors (`scripts/08`): ranking under each metric and the paired-bootstrap probability that D-ECE and MA-ECE disagree on order, computed twice: with each detector at its own operating threshold, and with all detectors moved to the same number of false alarms per image. Only a ranking difference that survives the second comparison is a difference in reliability; one that appears only in the first is a difference in operating point.

Code: `evaluation/rq3_calibration_gap.py`.

### 7.4 RQ4: transfer without retraining

On an independent dataset: detection quality first (gate: a candidate for at least half the vehicles), then conformal recall in three settings (if the target dataset's annotations are incomplete, precision, false alarms, D-ECE and MA-ECE on it are not measurements; only recall-side quantities are): standard threshold; **frozen** (thresholds fitted on the development dataset; the guarantee does not carry over a distribution shift, and this measures by how much); **recalibrated** (thresholds re-fitted on the transfer dataset's calib role, detector untouched). Reported in the transfer counterpart of each matching mode (`class_agnostic`, or `coarse` for the class-aware modes).

Code: `evaluation/rq4_cross_dataset.py`, `scripts/07`.

### 7.5 RQ5: occlusion misses versus unfamiliarity misses (secondary)

Each vehicle missed at `tau` gets one cause from the frozen detector's raw output: `confused` (a confident box of another class, belonging to no other vehicle, sits on it), `suppressed` (its own candidate scored below `tau`), `merged` (only a neighbour's box covers it), `unproposed` (nothing overlaps it), `weak_other`. The last three before `weak_other` are visibility failures; `confused` is a familiarity failure.

Two logistic models over all test vehicles, `P(visibility miss)` and `P(confusion miss)`, each on occlusion, class rarity (`−log` training share) and size. The causes are separated if visibility misses follow occlusion and not rarity, and confusion misses follow rarity and not occlusion.

This replaces the plan's evidential aleatoric/epistemic split with a post-hoc analysis that needs no retraining. It is a proxy for that split, not the same thing, and should be described as such.

Code: `evaluation/rq5_miss_causes.py`.

## 8. Statistics

The sampling unit is the **image**: vehicles in one frame share lighting, congestion and camera, so their outcomes are correlated. Every interval is a percentile interval from an image-level bootstrap (`evaluation.bootstrap_replicates` resamples). Comparisons between policies or detectors reuse the same resamples, so they are paired.

Gates, applied before anything is interpreted:

- a split with fewer than `min_occluded_objects` vehicles at `o ≥ 0.2` is flagged as unable to support occlusion-stratified statements;
- a bucket with fewer than `min_bucket_n` vehicles is marked unreliable;
- a dataset on which the detector proposes a candidate for fewer than half the vehicles is flagged as unusable.

Code: `evaluation/stats.py`, `evaluation/detection_metrics.py`.

## 9. References

Check each entry against the source before citing.

- F. Küppers, J. Kronenberger, A. Shantia, A. Haselhoff. Multivariate confidence calibration for object detection. CVPR Workshops, 2020. (D-ECE)
- Y.-J. Park, C. Sobolewski, N. Azizan. Quantifying the reliability of predictions in detection transformers: object-level calibration and image-level uncertainty. arXiv:2412.01782. (OCE)
- S. Kuzucu, K. Oksuz, J. Sadeghi, P. K. Dokania. On calibration of object detectors: pitfalls, evaluation and baselines. ECCV, 2024.
- K. Oksuz, T. Joy, P. K. Dokania. Towards building self-aware object detectors via reliable uncertainty quantification and calibration. CVPR, 2023. (LaECE)
- V. Vovk, A. Gammerman, G. Shafer. Algorithmic Learning in a Random World. Springer, 2005. (conformal and Mondrian conformal prediction)
- A. N. Angelopoulos, S. Bates. Conformal prediction: a gentle introduction. Foundations and Trends in Machine Learning, 2023.
- A. N. Angelopoulos, S. Bates, A. Fisch, L. Lei, T. Schuster. Conformal risk control. ICLR, 2024. (the natural image-level extension of section 4)
- BMD-45: Bengaluru Mobility Dataset. arXiv:2604.24419; dataset at huggingface.co/datasets/iisc-aim/BMD-45.

Prior work on conformal prediction for object detection exists (recall- and box-level guarantees via conformal risk control and related methods). Survey it before claiming novelty for section 4; the specific element here is conditioning on a deployable occlusion context and auditing the result by annotated occlusion in dense fixed-camera traffic.
