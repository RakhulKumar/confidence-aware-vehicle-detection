# Audit of the first full run, and what replaced each part

Every finding below comes from the files of the first run (`outputs/results/*.json`, `outputs/run_all.log`, the training artefacts) or from the code that produced them. "Was" describes version 0.1; "Now" describes this version.

## 1. The primary dataset does not contain the phenomenon under study

**Evidence.** The validation mosaics written by training show product and web photographs plus frames of one roadside video. Counts: about 1.3 vehicles per image; 16% of test images with no vehicle; 278 of 286 test vehicles with occlusion severity below 0.2, and buckets with 0, 1 and 1 vehicles. Many visible vehicles are unlabelled (only the "subject" of each photo is boxed), so an occlusion measure built from labels cannot see their occluders, and correct detections of them count as false alarms.

**Consequence.** RQ1, RQ3-by-occlusion and RQ2-on-occluded-misses are unanswerable on this dataset. Removing Roboflow's augmentation does not change the images, so the new export has the same limit.

**Now.** The roles are config-driven. The recommended config trains, calibrates and tests on BMD-45 (dense Bengaluru CCTV; about 27,000 of its 106,000 val vehicles have severity ≥ 0.2) and uses AU-INV-PALS as the transfer target. `scripts/03` prints the occlusion histogram per role and a pass/fail power gate before any GPU time is spent. The original arrangement is kept as `configs/experiment_pals_dev.yaml`.

## 2. Leakage

**Evidence.** (a) `scripts/05` calibrated on the detector's own training images. (b) File names such as `video_illumination_correction_mp4-147_jpg.rf.…` show frames of one video on both sides of Roboflow's random split. (c) The first export contained roughly three augmented copies of each source image, also split at random.

**Consequence.** The reported 0.93 mAP50 and the 89.8% "coverage" are optimistic and cannot be trusted.

**Now.** Four roles (fit / val / calib / test). Whole leakage groups are assigned to a role: all copies of one source image, runs of consecutive video frames, and a camera id where one is available. `scripts/02` fails if any group appears in two roles. The candidate score and the conformal thresholds are fitted on disjoint halves of calib.

## 3. The cross-dataset result (RQ4) compared unrelated class indices

**Evidence.** The detector had 7 classes (index 0 = AutoRikshaw), BMD-45 has 13 (index 0 = Hatchback), and matching required equal indices. `rq4_transfer_bmd45.json` reports mean "uncertainty" of 0.996 to 0.9997 per bucket on BMD-45: more than 99% of its vehicles had no matching candidate.

**Consequence.** The "clean monotonic trend in the opposite direction" was an artefact of items 3 and 4 together, not a finding.

**Now.** Classes are mapped by name into `evaluation.common_classes`; unmapped classes become ignore regions; a class-agnostic evaluation is reported alongside. A gate flags any dataset on which fewer than half the vehicles get a candidate. The class maps in the config are a judgement call (for example Maxi Cab ↔ Tempo-traveller/Van); review them.

## 4. The "object-level ECE" was not a calibration metric

**Evidence (code).** A vehicle's outcome was "did its best raw candidate survive the threshold", which is `1[confidence ≥ 0.25]`, a function of the confidence itself. A vehicle with no candidate scored (confidence 0, outcome 0), i.e. perfectly calibrated. Per-bucket numbers pooled *every* detection of an image into each bucket and matched pooled lists across images with colliding ids, which made D-ECE in sparse buckets large and the "gap" negative.

**Also.** The design was attributed to the OCE paper (arXiv:2412.01782), which defines something different: a Brier score per annotated object, 1.0 when nothing overlaps it. A miss-penalising object-level metric therefore already exists.

**Now.** D-ECE, OCE (top-1 approximation, labelled as such) and MA-ECE = (1 − φ)·D-ECE + φ with an exact per-bucket decomposition of the gap. The plan's contribution 2 should be reframed: the new part is the occlusion decomposition and its measurement in this setting, not the idea of a miss-aware metric.

## 5. The conformal layer guaranteed nothing about misses and decided nothing

**Evidence (code).** Scores were `1 − confidence` of vehicles the detector had already found, from a made-up two-slot "softmax"; missed vehicles were excluded "by design"; `avg_set_size` was 0.92. The fitted quantiles were used only to pick which bucket to "relax". Buckets were defined by ground-truth occlusion, which a deployed system does not have.

**Now.** Thresholds are fitted on object scores that include never-proposed vehicles, give `P(found) ≥ 1 − α`, and report when a target cannot be certified. Conditioning uses occlusion context computed from the detector's own output. The output is three-tiered (accepted / flagged / rejected).

## 6. RQ1 and RQ2 design

**Was.** RQ1: a correlation on a covariate with almost no variance, no control for object size. RQ2: recoveries gated by the ground-truth occlusion of the object being recovered, false alarms counted everywhere, no baseline at equal cost.

**Now.** RQ1: bucket counts first, image-bootstrap intervals, partial correlation and logistic model controlling size and class, plus a within-vehicle controlled probe. RQ2: policies compared at matched extra false alarms per image, with paired intervals and the recoverable ceiling.

## 7. Training was about ten times slower than necessary

**Evidence.** `run_all.log`: `GPU_mem 20.3G–20.6G` on a 16,376 MiB card, `6.6 s/it`, 21.6 h for 50 epochs. The batch (16 at 960 px) did not fit and was paged into shared system memory. Also: `optimizer=auto` silently ignored the configured `lr0`; a second training run would have written to `yolo26-2` while later stages kept reading `yolo26`.

**Now.** Batch 8 (4 for RT-DETR); training stops with an explanation if reserved memory exceeds the card; the optimizer is passed explicitly where `lr0` matters; the best checkpoint is always copied to one canonical path.

## 8. Things that were fine and are kept

The occlusion-severity definition and its tests; the Roboflow layout handling; per-dataset (never guessed) polygon handling, now a config field; the BMD-45 downloader (with two small fixes); the practice of testing every statistic against constructed data.

## 9. Found by the first real run (fixed in v0.3; details in `docs/FIRST_RUN_REVIEW.md`)

- **Native class-aware matching as the only lens.** 47% of "misses" were misnamed, detected vehicles. Matching mode is now explicit; class-agnostic is primary.
- **Stratum bookkeeping bug.** An unproposed vehicle was assigned to the heavily-occluded context stratum whenever a confident box sat on that same vehicle. Fixed with the same-vehicle exclusion used for candidates.
- **One confidence threshold for all architectures.** Now per detector (F1-optimal on calib), with a matched-cost comparison across detectors.
- **Windows paths of 260+ characters read as missing files.** Fixed during the run (`long_path`).

## 10. Not done, stated plainly

- **Evidential head.** Still a template, moved to `scripts/experimental/`, never run. RQ5 is answered post hoc instead.
- **The v0.3 changes have not been run on real predictions.** v0.2 ran end to end on real data (6-7 October 2026). The v0.3 additions (matching modes, per-detector threshold, matched-cost ranking) are verified on the simulated detector and the stand-in only; they need no GPU to run on the cached predictions.
- **One seed per detector.**
- **No promise about results.** Whether occlusion context improves on plain confidence for a real detector is an empirical question. The design reports the answer either way.
