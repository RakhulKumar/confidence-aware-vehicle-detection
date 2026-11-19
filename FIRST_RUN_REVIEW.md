# Review of the first real run (6-7 October 2026) and what changed in v0.3

Source: the `handoff_first_real_run` bundle (results JSON for three detectors on BMD-45, the PALS-dev run, and the console logs). Numbers below are from those files. "Run 1" analysed everything with native class-aware matching at a fixed threshold of 0.25.

## What ran

- BMD-45 val split, re-split 6,088 / 1,026 / 1,539 / 1,541 images (fit / val / calib / test). Test: 16,093 vehicles, 4,072 with occlusion severity ≥ 0.2.
- YOLO26m, YOLOv8m, RT-DETR-L; all stopped early (best epochs 49, 49, 38); about 3 hours each for the YOLO models, 7-11 GB of GPU memory, no overflow.
- One seed per detector.

## What holds

**1. Misses rise with occlusion, and the effect survives controls (RQ1).** Miss rate from the least to the most occluded bucket: YOLO26 14.2% → 30.9%, YOLOv8 14.9% → 33.9%, RT-DETR 7.2% → 16.6%. Odds of a miss per +0.1 severity, controlling size and class: 1.135 [1.115, 1.161], 1.134 [1.109, 1.163], 1.112 [1.082, 1.146]. The controlled probe (same 127 vehicles, pasted occluder) takes YOLO26's miss rate from 0.00 to 0.78.

**2. The conformal thresholds deliver what they promise, where they can promise it.** YOLO26 global rule, target → delivered: 0.90 → 0.895 [0.889, 0.901], 0.85 → 0.849, 0.80 → 0.797, 0.70 → 0.705. RT-DETR: 0.95 → 0.947, 0.90 → 0.900, 0.85 → 0.857, 0.80 → 0.806. Occlusion-conditional groups that were certifiable delivered 0.894-0.909 against 0.90.

**3. A global rule meets the average and under-serves occluded vehicles.** At a 0.90 target the most occluded bucket got 0.853 (YOLO26), 0.847 (YOLOv8), 0.801 (RT-DETR, rule alone).

**4. Prediction-only calibration is blind to misses (RQ3, first half).** YOLO26: D-ECE 0.047 against MA-ECE 0.162. Left to itself D-ECE would choose a threshold of 0.95, at which 96% of vehicles are missed.

**5. Two kinds of miss, cleanly different (RQ5).** For the YOLO models 47-48% of native-mode misses are a detected vehicle given the wrong fine class; these follow class rarity (odds ×3.0-3.1 per SD) and not occlusion. Suppressed and unproposed misses follow occlusion (odds ×1.18-1.19 per +0.1) and not rarity.

**6. The deployable occlusion context tracks annotated occlusion** (Spearman 0.82-0.87), so conditioning on it is conditioning on something real.

**7. The original arrangement cannot work.** Trained on PALS, the detector proposes a candidate for 14.6% of BMD-45 vehicles. On a leak-free PALS split its AP50 is 0.772 (version 0.1 reported 0.929 with leakage and augmented copies).

## What does not hold

**A. The occlusion-aware score does not beat plain confidence (RQ2).** Difference in mean recovery, all misses: −0.0005 [−0.0019, 0.0007] (YOLO26), −0.0009 [−0.0019, −0.0000] (YOLOv8), +0.0036 [−0.0036, 0.0101] (RT-DETR). The fitted context coefficients are ≈ 0. The conditional rank policy moves recovery toward occluded misses (+4 to +5 points) and pays for it elsewhere (−0.7 to −2.0 points overall): a redistribution, not a gain.

**B. The ranking "flip" is mostly an operating-point effect (RQ3, second half).** At 0.25, RT-DETR emitted 9.0 false alarms per image against 3.4 for the YOLO models. From the threshold sweeps, at matched cost (≈3.4 false alarms per image: RT-DETR at 0.50) RT-DETR is worse than YOLO26 on D-ECE (0.100 vs 0.047), MA-ECE (0.199 vs 0.162) and miss-aware Brier (0.255 vs 0.247), and better only on OCE (0.236 vs 0.261). MA-ECE never disagreed with D-ECE on order.

**C. Occlusion is a minority of the story under native matching.** Vehicles with severity ≥ 0.2 are 25% of objects and 33% of misses; two thirds of the D-ECE blind spot sits on unoccluded vehicles, and half of that is class confusion.

**D. Box-overlap severity understates real occlusion effects.** At "severity 0.8-1.0" YOLO26 still finds 69% of vehicles; with 80% of the box actually covered in the probe it finds 28%. Overlapping boxes often hide little, and heavily hidden vehicles are less likely to have been annotated at all. The probe overstates in the other direction (pasted edges, wheels covered). The truth lies between; say so.

**E. PALS precision-side numbers are not measurements.** The BMD-trained detectors emit 4.3-8.4 "false alarms" per PALS image against 1.3 annotated vehicles (precision 0.10-0.20): they are finding the unlabelled vehicles. Only recall and conformal coverage are meaningful on PALS.

## Three problems in the analysis itself, fixed in v0.3

**1. Native class-aware matching was the wrong primary lens.** The research question is about vehicles a system says *nothing* about. Under native matching a Sedan boxed as a Hatchback is a "miss". Class-agnostic recall is 0.916 at 0.25 and 0.981 at the confidence floor (YOLO26), against 0.841 and 0.906 natively. Consequences in run 1: 47% of "misses" were misnamings; the recoverable ceiling looked like 41% (class-agnostically it is about 77% from these figures); a 0.90 target sat at the edge of feasibility and cost 10-28 flags per image.
*v0.3:* three matching modes (`class_agnostic`, `coarse`, `native`), each analysed into its own folder; `class_agnostic` is primary. Native stays, because the confusion finding lives there. This is a change of primary analysis made after seeing data; report both and give the reason above.

**2. A bug of mine in stratum bookkeeping.** An unproposed vehicle is assigned an occlusion-context stratum from its annotated box. That code did not exclude a confident detection sitting on the *same* vehicle, so a misnamed vehicle whose box edge was a pixel higher than its detection was filed as heavily occluded. That is why the high-context stratum showed 21.9% unproposed (5% elsewhere) and "could not be certified". Annotated-occlusion buckets, unaffected by this, show a flat 9-12%.
*v0.3:* the same-vehicle exclusion is applied (`occlusion_context_at_boxes`), with a test.

**3. One threshold for three architectures.** 0.25 is a different operating regime for RT-DETR.
*v0.3:* `standard_operating_threshold: max_f1` picks each detector's F1-optimal threshold on calib; the detector comparison is reported both at own thresholds and at matched false alarms per image, and states whether a flip survives.

Also merged from the run: the Windows long-path fix (`long_path`), `processed_dir` outside OneDrive.

## One discrepancy to resolve

Ultralytics reported mAP50 0.754 / 0.750 / 0.759 on the val role; this project's class-aware AP50 on test is 0.668 / 0.661 / 0.728. The gap is large for the YOLO models and concentrated in confusable classes (Hatchback 0.70 vs 0.60, Sedan 0.76 vs 0.60), small for RT-DETR. Likely cause, not verified: Ultralytics' validator keeps several class hypotheses per box, while `predict()` keeps the best class only, so a vehicle whose second-best class is correct has no correctly-classed candidate here. `scripts/check_ultralytics_val.py` settles it in a minute. Class-agnostic numbers are unaffected either way.

## A limit of MA-ECE the run exposed

MA-ECE and miss-aware Brier both prefer the lowest threshold tried (0.05), even for RT-DETR at 77 false alarms per image. That is coherent for a calibration metric (a box at confidence 0.05 that is right 5% of the time is calibrated) but it means MA-ECE cannot choose an operating point. Claim only what the data shows: D-ECE's verdicts change once misses are counted. Do not present MA-ECE as a selection criterion.

## Where this leaves the paper's claims

| Planned claim | Status after run 1 |
|---|---|
| Uncertainty tracks occlusion (RQ1) | Supported, modest in observational data, strong in the probe |
| Occlusion-conditioned rule recovers misses better than a plain threshold (RQ2) | Not supported as a gain in efficiency. Supported as: guaranteed recall, and equal treatment of occluded vehicles, at a stated cost |
| Prediction-only metrics understate unreliability (RQ3a) | Supported |
| Miss-aware metric changes detector ranking (RQ3b) | Not supported so far; re-check at matched cost in v0.3 |
| Calibration transfers (RQ4) | Frozen thresholds do not carry over; re-fitting on a small labelled set restores coverage. Target dataset too sparse and too incompletely labelled for more |
| Occlusion vs unfamiliarity separable (RQ5) | Supported for the YOLO models |

Every row needs the class-agnostic re-analysis and more than one seed before it goes in a paper.
