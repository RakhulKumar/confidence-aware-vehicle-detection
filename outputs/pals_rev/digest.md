# Results digest: experiment `pals_rev`

Intervals are 95% image-level bootstrap intervals. Read the gates first.

## Matching: class_agnostic  (primary)

### Gates

- yolo26_rev / detector_usable_on_development: pass
- yolo26_rev / enough_occlusion: FAIL -- only 50 objects with occlusion >= 0.2 (need >= 200): occlusion-stratified results on this split are NOT interpretable
- yolo26_rev / detector_usable_on_bmd45 / class_agnostic: FAIL -- the detector proposes a candidate for only 37.7% of the vehicles (need >= 50%); check the class mapping and whether the detector transfers to this dataset at all before reading anything else
- rtdetr_rev / detector_usable_on_development: pass
- rtdetr_rev / enough_occlusion: FAIL -- only 50 objects with occlusion >= 0.2 (need >= 200): occlusion-stratified results on this split are NOT interpretable
- rtdetr_rev / detector_usable_on_bmd45 / class_agnostic: pass

### Detection at each detector's operating threshold

| detector | threshold | AP50 | recall | precision | false alarms/image | recall at floor |
|---|---|---|---|---|---|---|
| yolo26_rev | 0.29 | 0.690 | 0.703 | 0.625 | 0.55 | 0.959 |
| rtdetr_rev | 0.63 | 0.661 | 0.716 | 0.637 | 0.53 | 0.965 |

### RQ1: miss rate by annotated occlusion severity

| detector | 0.0-0.2 (n=1248) | 0.2-0.4 (n=36) | 0.4-0.6 (n=8) | 0.6-0.8 (n=2) | 0.8-1.0 (n=4) | odds of a miss per +0.1 (size, class controlled) | partial Spearman | context vs annotated occlusion |
|---|---|---|---|---|---|---|---|---|
| yolo26_rev | 0.285 | 0.583 | 0.500* | 1.000* | 0.750* | 1.385 [1.197, 1.700] | 0.156 [0.098, 0.211] | 0.58 |
| rtdetr_rev | 0.276 | 0.500 | 0.125* | 1.000* | 0.750* | 1.225 [1.054, 1.436] | 0.079 [0.010, 0.144] | 0.62 |

`*` marks a bucket with fewer objects than `min_bucket_n`; do not read it.

#### RQ1 by object size (tertiles of box area relative to the image)

| detector | tertile edges (relative area) | small | medium | large |
|---|---|---|---|---|
| yolo26_rev | 1.60e-02 / 5.15e-02 | 0.400 [0.353, 0.447] (n=433) | 0.236 [0.192, 0.284] (n=432) | 0.256 [0.212, 0.300] (n=433) |
| rtdetr_rev | 1.60e-02 / 5.15e-02 | 0.409 [0.360, 0.458] (n=433) | 0.236 [0.196, 0.278] (n=432) | 0.206 [0.160, 0.250] (n=433) |

Miss rate by occlusion bucket within each size tertile (n in brackets; `*` = fewer than `min_bucket_n`):

| detector | size | 0.0-0.2 | 0.2-0.4 | 0.4-0.6 | 0.6-0.8 | 0.8-1.0 |
|---|---|---|---|---|---|---|
| yolo26_rev | small | 0.393 (417) | 0.500 (12)* | 0.500 (2)* | n/a (0)* | 1.000 (2)* |
| yolo26_rev | medium | 0.220 (410) | 0.533 (15)* | 0.500 (6)* | n/a (0)* | 1.000 (1)* |
| yolo26_rev | large | 0.242 (421) | 0.778 (9)* | n/a (0)* | 1.000 (2)* | 0.000 (1)* |
| rtdetr_rev | small | 0.408 (417) | 0.417 (12)* | 0.000 (2)* | n/a (0)* | 1.000 (2)* |
| rtdetr_rev | medium | 0.229 (410) | 0.400 (15)* | 0.167 (6)* | n/a (0)* | 1.000 (1)* |
| rtdetr_rev | large | 0.190 (421) | 0.778 (9)* | n/a (0)* | 1.000 (2)* | 0.000 (1)* |

### Conformal layer

| detector | rule | target | recall of rule alone | recall of system | system recall, most occluded bucket | flags/image | certifiable |
|---|---|---|---|---|---|---|---|
| yolo26_rev | standard_threshold | n/a | 0.703 [0.674, 0.730] | 0.703 | 0.250 | n/a |  |
| yolo26_rev | global_confidence | 0.90 | 0.959 [0.946, 0.971] | 0.959 | 0.750 | 12.44 | NO |
| yolo26_rev | mondrian_confidence | 0.90 | 0.959 [0.946, 0.971] | 0.959 | 0.750 | 12.44 | NO |
| yolo26_rev | global, duplicate- and context-aware score | 0.90 | 0.959 [0.946, 0.971] | 0.959 | 0.750 | 12.44 | NO |
| rtdetr_rev | standard_threshold | n/a | 0.716 [0.692, 0.741] | 0.716 | 0.250 | n/a |  |
| rtdetr_rev | global_confidence | 0.90 | 0.854 [0.835, 0.875] | 0.854 | 0.750 | 4.02 | yes |
| rtdetr_rev | mondrian_confidence | 0.90 | 0.854 [0.835, 0.875] | 0.854 | 0.750 | 4.02 | yes |
| rtdetr_rev | global, duplicate- and context-aware score | 0.90 | 0.865 [0.848, 0.886] | 0.865 | 0.250 | 2.49 | yes |

### RQ2: recovery of missed vehicles at matched extra false alarms

| detector | policy | misses | recoverable ceiling | mean recovery | difference from baseline, all misses | difference from baseline, occluded misses |
|---|---|---|---|---|---|---|
| yolo26_rev | confidence | 386 | 0.863 | 0.303 [0.262, 0.338] | baseline | baseline |
| yolo26_rev | mondrian_rank | 386 | 0.863 | 0.283 [0.244, 0.317] | -0.0196 [-0.0249, -0.0159] | -0.0025 [-0.0077, 0.0000] |
| yolo26_rev | duplicate- and context-aware score | 386 | 0.863 | 0.365 [0.323, 0.407] | 0.0625 [0.0463, 0.0796] | -0.0275 [-0.1009, 0.0131] |
| rtdetr_rev | confidence | 368 | 0.875 | 0.280 [0.235, 0.325] | baseline | baseline |
| rtdetr_rev | mondrian_rank | 368 | 0.875 | 0.224 [0.185, 0.265] | -0.0558 [-0.0677, -0.0433] | -0.0625 [-0.1202, -0.0167] |
| rtdetr_rev | duplicate- and context-aware score | 368 | 0.875 | 0.338 [0.296, 0.379] | 0.0583 [0.0365, 0.0873] | -0.0542 [-0.1536, 0.0254] |

### RQ3: calibration metrics at the operating threshold

| detector | D-ECE | MA-ECE | miss mass | OCE | threshold D-ECE alone would choose | occluded vehicles: share of objects / share of D-ECE blind spot |
|---|---|---|---|---|---|---|
| yolo26_rev | 0.033 [0.023, 0.058] | 0.235 [0.216, 0.266] | 0.209 | 0.399 | 0.05 (recall 0.85) | 0.04 / 0.08 |
| rtdetr_rev | 0.192 [0.170, 0.214] | 0.355 [0.331, 0.377] | 0.201 | 0.302 | 0.9 (recall 0.26) | 0.04 / 0.07 |

### RQ5: why vehicles are missed (native classes, whatever the mode)

| detector | misses | confused | suppressed | unproposed | merged | visibility misses: odds per +0.1 occlusion | confusion misses: odds per +1 SD rarity | confusion misses: odds per +0.1 occlusion |
|---|---|---|---|---|---|---|---|---|
| yolo26_rev | 484 | 0.20 | 0.60 | 0.11 | 0.00 | 1.333 [1.173, 1.546] | 1.007 [0.799, 1.277] | 0.866 [0.523, 1.043] |
| rtdetr_rev | 436 | 0.16 | 0.69 | 0.11 | 0.00 | 1.225 [1.062, 1.430] | 0.858 [0.706, 1.088] | 1.151 [0.957, 1.363] |

### RQ4: transfer with the detector frozen (recall-side quantities only)

Each coverage is followed by (flags per image; certifiable). A recalibrated coverage that is not certifiable is the best the rule could do, not a guarantee.

| detector | dataset | matching | recall, development -> transfer | rule | target | coverage on development | coverage on transfer, frozen | coverage on transfer, recalibrated |
|---|---|---|---|---|---|---|---|---|
| yolo26_rev | bmd45 | class_agnostic | 0.703 -> 0.090 | global_confidence | 0.90 | 0.959 (12.44; NO) | 0.377 (29.90; NO) | 0.377 (29.90; NO) |
| yolo26_rev | bmd45 | class_agnostic | 0.703 -> 0.090 | mondrian_confidence | 0.90 | 0.959 (12.44; NO) | 0.377 (29.90; NO) | 0.377 (29.90; NO) |
| yolo26_rev | bmd45 | class_agnostic | 0.703 -> 0.090 | global_occlusion_aware | 0.90 | 0.959 (12.44; NO) | 0.377 (29.90; NO) | 0.377 (29.90; NO) |
| rtdetr_rev | bmd45 | class_agnostic | 0.716 -> 0.179 | global_confidence | 0.90 | 0.854 (4.02; yes) | 0.328 (7.12; yes) | 0.632 (297.26; NO) |
| rtdetr_rev | bmd45 | class_agnostic | 0.716 -> 0.179 | mondrian_confidence | 0.90 | 0.854 (4.02; yes) | 0.328 (7.12; yes) | 0.632 (297.26; NO) |
| rtdetr_rev | bmd45 | class_agnostic | 0.716 -> 0.179 | global_occlusion_aware | 0.90 | 0.865 (2.49; yes) | 0.341 (7.71; yes) | 0.632 (297.26; NO) |

### Detector ranking (best first)

| comparison | metric | ranking |
|---|---|---|
| each at its own threshold | dece | yolo26_rev < rtdetr_rev |
| each at its own threshold | ma_ece | yolo26_rev < rtdetr_rev |
| each at its own threshold | ma_brier | yolo26_rev < rtdetr_rev |
| each at its own threshold | oce_max | rtdetr_rev < yolo26_rev |
| at matched false alarms per image | dece | yolo26_rev < rtdetr_rev |
| at matched false alarms per image | ma_ece | yolo26_rev < rtdetr_rev |
| at matched false alarms per image | ma_brier | yolo26_rev < rtdetr_rev |
| at matched false alarms per image | oce_max | rtdetr_rev < yolo26_rev |

Point-estimate orders above can differ by noise. Below, a metric *separates* a pair only if its 95% paired-difference interval excludes zero; a *flip* needs both D-ECE and the other metric to separate the pair, in opposite directions.

**each at its own threshold**

| pair (difference = first minus second) | D-ECE | MA-ECE | MA-Brier | OCE | flips vs D-ECE |
|---|---|---|---|---|---|
| rtdetr_rev vs yolo26_rev | yolo26_rev better [0.1185, 0.1769] | yolo26_rev better [0.0808, 0.1437] | yolo26_rev better [0.0104, 0.0505] | rtdetr_rev better [-0.1207, -0.0761] | oce_max |


**at matched false alarms per image** (budget 0.54 false alarms/image)

| pair (difference = first minus second) | D-ECE | MA-ECE | MA-Brier | OCE | flips vs D-ECE |
|---|---|---|---|---|---|
| rtdetr_rev vs yolo26_rev | yolo26_rev better [0.1171, 0.1751] | yolo26_rev better [0.0684, 0.1310] | yolo26_rev better [0.0003, 0.0418] | rtdetr_rev better [-0.1312, -0.0860] | oce_max |


Ranking flip relative to D-ECE (both intervals exclude zero, opposite signs) survives matched cost: **True**

#### MA-Brier decomposition

Each term is its contribution to MA-Brier (share of MA-Brier in brackets); the three sum to MA-Brier. OCE averages over vehicles only, so it has no false-alarm term.

| detector | where | threshold | false alarms/image | MA-Brier | false-alarm term | matched-detection term | miss term | OCE |
|---|---|---|---|---|---|---|---|---|
| yolo26_rev | own threshold | 0.29 | 0.55 | 0.3644 | 0.0874 (0.24) | 0.0679 (0.19) | 0.2091 (0.57) | 0.3986 |
| rtdetr_rev | own threshold | 0.63 | 0.53 | 0.3964 | 0.1809 (0.46) | 0.0140 (0.04) | 0.2014 (0.51) | 0.3018 |
| yolo26_rev | matched cost | 0.29 | 0.54 | 0.3675 | 0.0874 (0.24) | 0.0666 (0.18) | 0.2135 (0.58) | 0.4009 |
| rtdetr_rev | matched cost | 0.62 | 0.54 | 0.3902 | 0.1819 (0.47) | 0.0150 (0.04) | 0.1934 (0.50) | 0.2933 |

## Matching: native

### Gates

- yolo26_rev / detector_usable_on_development: pass
- yolo26_rev / enough_occlusion: FAIL -- only 50 objects with occlusion >= 0.2 (need >= 200): occlusion-stratified results on this split are NOT interpretable
- yolo26_rev / detector_usable_on_bmd45 / coarse: FAIL -- the detector proposes a candidate for only 29.3% of the vehicles (need >= 50%); check the class mapping and whether the detector transfers to this dataset at all before reading anything else
- rtdetr_rev / detector_usable_on_development: pass
- rtdetr_rev / enough_occlusion: FAIL -- only 50 objects with occlusion >= 0.2 (need >= 200): occlusion-stratified results on this split are NOT interpretable
- rtdetr_rev / detector_usable_on_bmd45 / coarse: pass

### Detection at each detector's operating threshold

| detector | threshold | AP50 | recall | precision | false alarms/image | recall at floor |
|---|---|---|---|---|---|---|
| yolo26_rev | 0.32 | 0.588 | 0.610 | 0.573 | 0.59 | 0.891 |
| rtdetr_rev | 0.74 | 0.621 | 0.597 | 0.663 | 0.39 | 0.946 |

### RQ1: miss rate by annotated occlusion severity

| detector | 0.0-0.2 (n=1248) | 0.2-0.4 (n=36) | 0.4-0.6 (n=8) | 0.6-0.8 (n=2) | 0.8-1.0 (n=4) | odds of a miss per +0.1 (size, class controlled) | partial Spearman | context vs annotated occlusion |
|---|---|---|---|---|---|---|---|---|
| yolo26_rev | 0.378 | 0.667 | 0.625* | 1.000* | 0.750* | 1.354 [1.155, 1.681] | 0.141 [0.082, 0.194] | 0.58 |
| rtdetr_rev | 0.392 | 0.722 | 0.375* | 1.000* | 0.750* | 1.294 [1.105, 1.558] | 0.095 [0.033, 0.155] | 0.62 |

`*` marks a bucket with fewer objects than `min_bucket_n`; do not read it.

#### RQ1 by object size (tertiles of box area relative to the image)

| detector | tertile edges (relative area) | small | medium | large |
|---|---|---|---|---|
| yolo26_rev | 1.60e-02 / 5.15e-02 | 0.469 [0.418, 0.517] (n=433) | 0.301 [0.249, 0.348] (n=432) | 0.400 [0.354, 0.445] (n=433) |
| rtdetr_rev | 1.60e-02 / 5.15e-02 | 0.533 [0.484, 0.586] (n=433) | 0.373 [0.323, 0.419] (n=432) | 0.303 [0.259, 0.348] (n=433) |

Miss rate by occlusion bucket within each size tertile (n in brackets; `*` = fewer than `min_bucket_n`):

| detector | size | 0.0-0.2 | 0.2-0.4 | 0.4-0.6 | 0.6-0.8 | 0.8-1.0 |
|---|---|---|---|---|---|---|
| yolo26_rev | small | 0.463 (417) | 0.583 (12)* | 0.500 (2)* | n/a (0)* | 1.000 (2)* |
| yolo26_rev | medium | 0.283 (410) | 0.600 (15)* | 0.667 (6)* | n/a (0)* | 1.000 (1)* |
| yolo26_rev | large | 0.387 (421) | 0.889 (9)* | n/a (0)* | 1.000 (2)* | 0.000 (1)* |
| rtdetr_rev | small | 0.528 (417) | 0.667 (12)* | 0.500 (2)* | n/a (0)* | 1.000 (2)* |
| rtdetr_rev | medium | 0.361 (410) | 0.667 (15)* | 0.333 (6)* | n/a (0)* | 1.000 (1)* |
| rtdetr_rev | large | 0.287 (421) | 0.889 (9)* | n/a (0)* | 1.000 (2)* | 0.000 (1)* |

### Conformal layer

| detector | rule | target | recall of rule alone | recall of system | system recall, most occluded bucket | flags/image | certifiable |
|---|---|---|---|---|---|---|---|
| yolo26_rev | standard_threshold | n/a | 0.610 [0.579, 0.640] | 0.610 | 0.250 | n/a |  |
| yolo26_rev | global_confidence | 0.90 | 0.891 [0.870, 0.910] | 0.891 | 0.750 | 12.51 | NO |
| yolo26_rev | mondrian_confidence | 0.90 | 0.891 [0.870, 0.910] | 0.891 | 0.750 | 12.51 | NO |
| yolo26_rev | global, duplicate- and context-aware score | 0.90 | 0.891 [0.870, 0.910] | 0.891 | 0.750 | 12.51 | NO |
| rtdetr_rev | standard_threshold | n/a | 0.597 [0.571, 0.626] | 0.597 | 0.250 | n/a |  |
| rtdetr_rev | global_confidence | 0.90 | 0.845 [0.827, 0.866] | 0.845 | 0.750 | 8.98 | yes |
| rtdetr_rev | mondrian_confidence | 0.90 | 0.845 [0.827, 0.866] | 0.845 | 0.750 | 8.98 | yes |
| rtdetr_rev | global, duplicate- and context-aware score | 0.90 | 0.853 [0.834, 0.873] | 0.853 | 0.750 | 12.00 | yes |

### RQ2: recovery of missed vehicles at matched extra false alarms

| detector | policy | misses | recoverable ceiling | mean recovery | difference from baseline, all misses | difference from baseline, occluded misses |
|---|---|---|---|---|---|---|
| yolo26_rev | confidence | 506 | 0.719 | 0.256 [0.221, 0.291] | baseline | baseline |
| yolo26_rev | mondrian_rank | 506 | 0.719 | 0.232 [0.200, 0.264] | -0.0242 [-0.0301, -0.0179] | -0.0059 [-0.0099, -0.0009] |
| yolo26_rev | duplicate- and context-aware score | 506 | 0.719 | 0.263 [0.228, 0.295] | 0.0067 [-0.0050, 0.0168] | -0.0301 [-0.0864, 0.0036] |
| rtdetr_rev | confidence | 523 | 0.866 | 0.290 [0.250, 0.329] | baseline | baseline |
| rtdetr_rev | mondrian_rank | 523 | 0.866 | 0.261 [0.222, 0.296] | -0.0296 [-0.0385, -0.0238] | -0.0390 [-0.0681, -0.0095] |
| rtdetr_rev | duplicate- and context-aware score | 523 | 0.866 | 0.306 [0.271, 0.341] | 0.0159 [0.0011, 0.0332] | -0.0301 [-0.0782, 0.0162] |

### RQ3: calibration metrics at the operating threshold

| detector | D-ECE | MA-ECE | miss mass | OCE | threshold D-ECE alone would choose | occluded vehicles: share of objects / share of D-ECE blind spot |
|---|---|---|---|---|---|---|
| yolo26_rev | 0.063 [0.041, 0.091] | 0.314 [0.290, 0.346] | 0.268 | 0.518 | 0.9 (recall 0.07) | 0.04 / 0.07 |
| rtdetr_rev | 0.202 [0.174, 0.226] | 0.449 [0.421, 0.472] | 0.309 | 0.437 | 0.9 (recall 0.26) | 0.04 / 0.07 |

### RQ5: why vehicles are missed (native classes, whatever the mode)

| detector | misses | confused | suppressed | unproposed | merged | visibility misses: odds per +0.1 occlusion | confusion misses: odds per +1 SD rarity | confusion misses: odds per +0.1 occlusion |
|---|---|---|---|---|---|---|---|---|
| yolo26_rev | 506 | 0.18 | 0.63 | 0.10 | 0.00 | 1.350 [1.179, 1.603] | 0.939 [0.759, 1.178] | 0.876 [0.517, 1.052] |
| rtdetr_rev | 523 | 0.08 | 0.78 | 0.09 | 0.00 | 1.260 [1.094, 1.495] | 0.832 [0.709, 1.016] | 1.014 [0.544, 1.209] |

### RQ4: transfer with the detector frozen (recall-side quantities only)

Each coverage is followed by (flags per image; certifiable). A recalibrated coverage that is not certifiable is the best the rule could do, not a guarantee.

| detector | dataset | matching | recall, development -> transfer | rule | target | coverage on development | coverage on transfer, frozen | coverage on transfer, recalibrated |
|---|---|---|---|---|---|---|---|---|
| yolo26_rev | bmd45 | coarse | 0.610 -> 0.069 | global_confidence | 0.90 | 0.891 (12.51; NO) | 0.293 (29.99; NO) | 0.293 (29.99; NO) |
| yolo26_rev | bmd45 | coarse | 0.610 -> 0.069 | mondrian_confidence | 0.90 | 0.891 (12.51; NO) | 0.293 (29.99; NO) | 0.293 (29.99; NO) |
| yolo26_rev | bmd45 | coarse | 0.610 -> 0.069 | global_occlusion_aware | 0.90 | 0.891 (12.51; NO) | 0.293 (29.99; NO) | 0.293 (29.99; NO) |
| rtdetr_rev | bmd45 | coarse | 0.597 -> 0.096 | global_confidence | 0.90 | 0.845 (8.98; yes) | 0.289 (12.94; yes) | 0.566 (297.90; NO) |
| rtdetr_rev | bmd45 | coarse | 0.597 -> 0.096 | mondrian_confidence | 0.90 | 0.845 (8.98; yes) | 0.289 (12.94; yes) | 0.566 (297.90; NO) |
| rtdetr_rev | bmd45 | coarse | 0.597 -> 0.096 | global_occlusion_aware | 0.90 | 0.853 (12.00; yes) | 0.324 (26.73; yes) | 0.566 (297.90; NO) |

### Detector ranking (best first)

| comparison | metric | ranking |
|---|---|---|
| each at its own threshold | dece | yolo26_rev < rtdetr_rev |
| each at its own threshold | ma_ece | yolo26_rev < rtdetr_rev |
| each at its own threshold | ma_brier | yolo26_rev < rtdetr_rev |
| each at its own threshold | oce_max | rtdetr_rev < yolo26_rev |
| at matched false alarms per image | dece | yolo26_rev < rtdetr_rev |
| at matched false alarms per image | ma_ece | yolo26_rev < rtdetr_rev |
| at matched false alarms per image | ma_brier | yolo26_rev < rtdetr_rev |
| at matched false alarms per image | oce_max | rtdetr_rev < yolo26_rev |

Point-estimate orders above can differ by noise. Below, a metric *separates* a pair only if its 95% paired-difference interval excludes zero; a *flip* needs both D-ECE and the other metric to separate the pair, in opposite directions.

**each at its own threshold**

| pair (difference = first minus second) | D-ECE | MA-ECE | MA-Brier | OCE | flips vs D-ECE |
|---|---|---|---|---|---|
| rtdetr_rev vs yolo26_rev | yolo26_rev better [0.1095, 0.1535] | yolo26_rev better [0.1009, 0.1564] | yolo26_rev better [0.0366, 0.0778] | rtdetr_rev better [-0.1112, -0.0546] | oce_max |


**at matched false alarms per image** (budget 0.49 false alarms/image)

| pair (difference = first minus second) | D-ECE | MA-ECE | MA-Brier | OCE | flips vs D-ECE |
|---|---|---|---|---|---|
| rtdetr_rev vs yolo26_rev | yolo26_rev better [0.1287, 0.1753] | yolo26_rev better [0.0470, 0.1081] | cannot separate [-0.0141, 0.0288] | rtdetr_rev better [-0.1482, -0.0907] | oce_max |


Ranking flip relative to D-ECE (both intervals exclude zero, opposite signs) survives matched cost: **True**

#### MA-Brier decomposition

Each term is its contribution to MA-Brier (share of MA-Brier in brackets); the three sum to MA-Brier. OCE averages over vehicles only, so it has no false-alarm term.

| detector | where | threshold | false alarms/image | MA-Brier | false-alarm term | matched-detection term | miss term | OCE |
|---|---|---|---|---|---|---|---|---|
| yolo26_rev | own threshold | 0.32 | 0.59 | 0.4228 | 0.1036 (0.25) | 0.0511 (0.12) | 0.2680 (0.63) | 0.5176 |
| rtdetr_rev | own threshold | 0.74 | 0.39 | 0.4805 | 0.1634 (0.34) | 0.0080 (0.02) | 0.3091 (0.64) | 0.4366 |
| yolo26_rev | matched cost | 0.38 | 0.49 | 0.4518 | 0.1027 (0.23) | 0.0447 (0.10) | 0.3045 (0.67) | 0.5294 |
| rtdetr_rev | matched cost | 0.69 | 0.49 | 0.4597 | 0.1826 (0.40) | 0.0095 (0.02) | 0.2676 (0.58) | 0.4121 |
