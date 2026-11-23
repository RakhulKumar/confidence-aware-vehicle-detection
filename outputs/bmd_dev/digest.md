# Results digest: experiment `bmd_dev`

Intervals are 95% image-level bootstrap intervals. Read the gates first.

## Matching: class_agnostic  (primary)

### Gates

- yolo26 / detector_usable_on_development: pass
- yolo26 / enough_occlusion: pass
- yolo26 / detector_usable_on_pals / class_agnostic: pass
- yolov8 / detector_usable_on_development: pass
- yolov8 / enough_occlusion: pass
- yolov8 / detector_usable_on_pals / class_agnostic: pass
- rtdetr / detector_usable_on_development: pass
- rtdetr / enough_occlusion: pass
- rtdetr / detector_usable_on_pals / class_agnostic: pass
- yolo26_ens / detector_usable_on_development: pass
- yolo26_ens / enough_occlusion: pass
- yolo26_ens / detector_usable_on_pals / class_agnostic: pass
- rtdetr_ens / detector_usable_on_development: pass
- rtdetr_ens / enough_occlusion: pass
- rtdetr_ens / detector_usable_on_pals / class_agnostic: pass

### Detection at each detector's operating threshold

| detector | threshold | AP50 | recall | precision | false alarms/image | recall at floor |
|---|---|---|---|---|---|---|
| yolo26 | 0.46 | 0.915 | 0.869 | 0.862 | 1.45 | 0.981 |
| yolov8 | 0.47 | 0.913 | 0.863 | 0.870 | 1.35 | 0.981 |
| rtdetr | 0.65 | 0.909 | 0.859 | 0.865 | 1.40 | 0.986 |
| yolo26_ens | 0.42 | 0.927 | 0.883 | 0.882 | 1.23 | 0.989 |
| rtdetr_ens | 0.63 | 0.921 | 0.880 | 0.879 | 1.27 | 0.990 |

### RQ1: miss rate by annotated occlusion severity

| detector | 0.0-0.2 (n=12021) | 0.2-0.4 (n=2152) | 0.4-0.6 (n=1169) | 0.6-0.8 (n=444) | 0.8-1.0 (n=307) | odds of a miss per +0.1 (size, class controlled) | partial Spearman | context vs annotated occlusion |
|---|---|---|---|---|---|---|---|---|
| yolo26 | 0.108 | 0.152 | 0.202 | 0.329 | 0.332 | 1.193 [1.170, 1.220] | 0.207 [0.189, 0.225] | 0.88 |
| yolov8 | 0.113 | 0.161 | 0.221 | 0.304 | 0.362 | 1.198 [1.178, 1.226] | 0.195 [0.174, 0.211] | 0.88 |
| rtdetr | 0.115 | 0.168 | 0.228 | 0.340 | 0.378 | 1.219 [1.195, 1.247] | 0.265 [0.245, 0.281] | 0.89 |
| yolo26_ens | 0.094 | 0.136 | 0.188 | 0.320 | 0.306 | 1.206 [1.182, 1.236] | 0.214 [0.195, 0.232] | 0.90 |
| rtdetr_ens | 0.096 | 0.137 | 0.192 | 0.329 | 0.345 | 1.229 [1.205, 1.261] | 0.260 [0.242, 0.275] | 0.90 |

`*` marks a bucket with fewer objects than `min_bucket_n`; do not read it.

#### RQ1 by object size (tertiles of box area relative to the image)

| detector | tertile edges (relative area) | small | medium | large |
|---|---|---|---|---|
| yolo26 | 4.14e-03 / 1.41e-02 | 0.198 [0.185, 0.210] (n=5362) | 0.119 [0.109, 0.130] (n=5366) | 0.076 [0.068, 0.084] (n=5365) |
| yolov8 | 4.14e-03 / 1.41e-02 | 0.203 [0.190, 0.215] (n=5362) | 0.127 [0.116, 0.138] (n=5366) | 0.082 [0.074, 0.090] (n=5365) |
| rtdetr | 4.14e-03 / 1.41e-02 | 0.228 [0.215, 0.241] (n=5362) | 0.129 [0.119, 0.139] (n=5366) | 0.067 [0.060, 0.075] (n=5365) |
| yolo26_ens | 4.14e-03 / 1.41e-02 | 0.183 [0.171, 0.195] (n=5362) | 0.105 [0.096, 0.116] (n=5366) | 0.062 [0.055, 0.070] (n=5365) |
| rtdetr_ens | 4.14e-03 / 1.41e-02 | 0.197 [0.184, 0.210] (n=5362) | 0.108 [0.098, 0.118] (n=5366) | 0.054 [0.047, 0.061] (n=5365) |

Miss rate by occlusion bucket within each size tertile (n in brackets; `*` = fewer than `min_bucket_n`):

| detector | size | 0.0-0.2 | 0.2-0.4 | 0.4-0.6 | 0.6-0.8 | 0.8-1.0 |
|---|---|---|---|---|---|---|
| yolo26 | small | 0.169 (4021) | 0.244 (733) | 0.269 (353) | 0.449 (138) | 0.393 (117) |
| yolo26 | medium | 0.090 (3730) | 0.129 (785) | 0.192 (520) | 0.313 (214) | 0.274 (117) |
| yolo26 | large | 0.065 (4270) | 0.074 (634) | 0.139 (296) | 0.185 (92) | 0.329 (73) |
| yolov8 | small | 0.170 (4021) | 0.267 (733) | 0.309 (353) | 0.406 (138) | 0.368 (117) |
| yolov8 | medium | 0.100 (3730) | 0.131 (785) | 0.206 (520) | 0.280 (214) | 0.333 (117) |
| yolov8 | large | 0.071 (4270) | 0.074 (634) | 0.142 (296) | 0.207 (92) | 0.397 (73) |
| rtdetr | small | 0.193 (4021) | 0.286 (733) | 0.320 (353) | 0.478 (138) | 0.504 (117) |
| rtdetr | medium | 0.099 (3730) | 0.135 (785) | 0.219 (520) | 0.304 (214) | 0.308 (117) |
| rtdetr | large | 0.055 (4270) | 0.071 (634) | 0.135 (296) | 0.217 (92) | 0.288 (73) |
| yolo26_ens | small | 0.153 (4021) | 0.232 (733) | 0.266 (353) | 0.413 (138) | 0.359 (117) |
| yolo26_ens | medium | 0.080 (3730) | 0.103 (785) | 0.171 (520) | 0.313 (214) | 0.265 (117) |
| yolo26_ens | large | 0.051 (4270) | 0.065 (634) | 0.125 (296) | 0.196 (92) | 0.288 (73) |
| rtdetr_ens | small | 0.165 (4021) | 0.244 (733) | 0.275 (353) | 0.464 (138) | 0.444 (117) |
| rtdetr_ens | medium | 0.084 (3730) | 0.101 (785) | 0.177 (520) | 0.304 (214) | 0.282 (117) |
| rtdetr_ens | large | 0.042 (4270) | 0.058 (634) | 0.118 (296) | 0.185 (92) | 0.288 (73) |

### Conformal layer

| detector | rule | target | recall of rule alone | recall of system | system recall, most occluded bucket | flags/image | certifiable |
|---|---|---|---|---|---|---|---|
| yolo26 | standard_threshold | n/a | 0.869 [0.862, 0.876] | 0.869 | 0.668 | n/a |  |
| yolo26 | global_confidence | 0.95 | 0.951 [0.945, 0.956] | 0.951 | 0.821 | 4.56 | yes |
| yolo26 | mondrian_confidence | 0.95 | 0.951 [0.946, 0.956] | 0.951 | 0.902 | 6.18 | yes |
| yolo26 | global, duplicate- and context-aware score | 0.95 | 0.951 [0.945, 0.956] | 0.951 | 0.827 | 4.32 | yes |
| yolov8 | standard_threshold | n/a | 0.863 [0.856, 0.870] | 0.863 | 0.638 | n/a |  |
| yolov8 | global_confidence | 0.95 | 0.948 [0.943, 0.952] | 0.948 | 0.834 | 4.50 | yes |
| yolov8 | mondrian_confidence | 0.95 | 0.951 [0.945, 0.955] | 0.951 | 0.886 | 6.71 | yes |
| yolov8 | global, duplicate- and context-aware score | 0.95 | 0.948 [0.943, 0.953] | 0.948 | 0.834 | 4.41 | yes |
| rtdetr | standard_threshold | n/a | 0.859 [0.852, 0.865] | 0.859 | 0.622 | n/a |  |
| rtdetr | global_confidence | 0.95 | 0.947 [0.943, 0.952] | 0.947 | 0.860 | 5.64 | yes |
| rtdetr | mondrian_confidence | 0.95 | 0.950 [0.945, 0.955] | 0.950 | 0.919 | 8.80 | yes |
| rtdetr | global, duplicate- and context-aware score | 0.95 | 0.948 [0.944, 0.953] | 0.948 | 0.857 | 4.63 | yes |
| yolo26_ens | standard_threshold | n/a | 0.883 [0.876, 0.889] | 0.883 | 0.694 | n/a |  |
| yolo26_ens | global_confidence | 0.95 | 0.948 [0.943, 0.953] | 0.948 | 0.834 | 3.31 | yes |
| yolo26_ens | mondrian_confidence | 0.95 | 0.949 [0.944, 0.954] | 0.949 | 0.902 | 3.92 | yes |
| yolo26_ens | global, duplicate- and context-aware score | 0.95 | 0.948 [0.942, 0.952] | 0.948 | 0.827 | 2.55 | yes |
| rtdetr_ens | standard_threshold | n/a | 0.880 [0.874, 0.887] | 0.880 | 0.655 | n/a |  |
| rtdetr_ens | global_confidence | 0.95 | 0.948 [0.943, 0.953] | 0.948 | 0.831 | 4.10 | yes |
| rtdetr_ens | mondrian_confidence | 0.95 | 0.949 [0.944, 0.954] | 0.949 | 0.906 | 5.36 | yes |
| rtdetr_ens | global, duplicate- and context-aware score | 0.95 | 0.948 [0.943, 0.952] | 0.948 | 0.827 | 2.74 | yes |

### RQ2: recovery of missed vehicles at matched extra false alarms

| detector | policy | misses | recoverable ceiling | mean recovery | difference from baseline, all misses | difference from baseline, occluded misses |
|---|---|---|---|---|---|---|
| yolo26 | confidence | 2107 | 0.855 | 0.191 [0.173, 0.210] | baseline | baseline |
| yolo26 | mondrian_rank | 2107 | 0.855 | 0.144 [0.131, 0.162] | -0.0462 [-0.0609, -0.0323] | 0.1151 [0.0963, 0.1384] |
| yolo26 | duplicate- and context-aware score | 2107 | 0.855 | 0.235 [0.217, 0.255] | 0.0446 [0.0385, 0.0511] | 0.0188 [0.0118, 0.0270] |
| yolov8 | confidence | 2210 | 0.861 | 0.194 [0.176, 0.211] | baseline | baseline |
| yolov8 | mondrian_rank | 2210 | 0.861 | 0.156 [0.139, 0.172] | -0.0377 [-0.0516, -0.0215] | 0.1146 [0.0901, 0.1371] |
| yolov8 | duplicate- and context-aware score | 2210 | 0.861 | 0.226 [0.206, 0.245] | 0.0324 [0.0253, 0.0395] | 0.0022 [-0.0054, 0.0107] |
| rtdetr | confidence | 2272 | 0.903 | 0.195 [0.177, 0.212] | baseline | baseline |
| rtdetr | mondrian_rank | 2272 | 0.903 | 0.150 [0.135, 0.164] | -0.0447 [-0.0591, -0.0325] | 0.0982 [0.0795, 0.1187] |
| rtdetr | duplicate- and context-aware score | 2272 | 0.903 | 0.255 [0.237, 0.274] | 0.0604 [0.0526, 0.0672] | 0.0615 [0.0507, 0.0718] |
| yolo26_ens | confidence | 1880 | 0.907 | 0.215 [0.195, 0.235] | baseline | baseline |
| yolo26_ens | mondrian_rank | 1880 | 0.907 | 0.154 [0.138, 0.172] | -0.0617 [-0.0763, -0.0447] | 0.0943 [0.0701, 0.1215] |
| yolo26_ens | duplicate- and context-aware score | 1880 | 0.907 | 0.244 [0.223, 0.266] | 0.0293 [0.0235, 0.0359] | 0.0089 [0.0003, 0.0176] |
| rtdetr_ens | confidence | 1926 | 0.920 | 0.195 [0.175, 0.215] | baseline | baseline |
| rtdetr_ens | mondrian_rank | 1926 | 0.920 | 0.158 [0.142, 0.175] | -0.0364 [-0.0511, -0.0209] | 0.1113 [0.0909, 0.1319] |
| rtdetr_ens | duplicate- and context-aware score | 1926 | 0.920 | 0.249 [0.226, 0.272] | 0.0543 [0.0458, 0.0627] | 0.0475 [0.0370, 0.0581] |

### RQ2 ablation: which features of the occlusion-aware score carry the gain

Chosen by 5-fold cross-validation over calib images (criterion fixed in advance: mean recovery of all misses over the budget range); every set then evaluated once on test against plain confidence. The configured default calibrator is unchanged.

| detector | feature set | calib CV: mean recovery | calib CV: occluded | test: mean recovery | test: difference from confidence, all misses | test: difference, occluded misses |
|---|---|---|---|---|---|---|
| yolo26 | z_only | 0.1745 | 0.1547 | 0.191 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| yolo26 | z_dup | 0.2168 | 0.1668 | 0.229 | 0.0386 [0.0319, 0.0456] | 0.0053 [-0.0026, 0.0142] |
| yolo26 | z_ctx | 0.1733 | 0.1441 | 0.188 | -0.0025 [-0.0046, -0.0002] | -0.0247 [-0.0286, -0.0207] |
| yolo26 | full | 0.2168 | 0.1771 | 0.229 | 0.0378 [0.0307, 0.0447] | 0.0041 [-0.0045, 0.0136] |
| yolo26 | full_strict **(chosen on calib)** | 0.2209 | 0.1839 | 0.235 | 0.0446 [0.0385, 0.0511] | 0.0188 [0.0118, 0.0270] |
| yolov8 | z_only | 0.1945 | 0.1868 | 0.194 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| yolov8 | z_dup | 0.2324 | 0.1876 | 0.224 | 0.0306 [0.0234, 0.0374] | 0.0044 [-0.0037, 0.0126] |
| yolov8 | z_ctx | 0.1942 | 0.1662 | 0.192 | -0.0020 [-0.0047, 0.0010] | -0.0293 [-0.0345, -0.0243] |
| yolov8 | full | 0.2317 | 0.1879 | 0.223 | 0.0295 [0.0226, 0.0369] | -0.0035 [-0.0119, 0.0060] |
| yolov8 | full_strict **(chosen on calib)** | 0.2377 | 0.2017 | 0.226 | 0.0324 [0.0253, 0.0395] | 0.0022 [-0.0054, 0.0107] |
| rtdetr | z_only | 0.1788 | 0.1773 | 0.195 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| rtdetr | z_dup | 0.2260 | 0.1950 | 0.248 | 0.0529 [0.0438, 0.0603] | 0.0277 [0.0165, 0.0376] |
| rtdetr | z_ctx | 0.1794 | 0.1989 | 0.196 | 0.0012 [-0.0001, 0.0025] | 0.0128 [0.0104, 0.0152] |
| rtdetr | full | 0.2258 | 0.2264 | 0.248 | 0.0531 [0.0452, 0.0608] | 0.0502 [0.0391, 0.0614] |
| rtdetr | full_strict **(chosen on calib)** | 0.2305 | 0.2401 | 0.255 | 0.0604 [0.0526, 0.0672] | 0.0615 [0.0507, 0.0718] |

### Baselines from the literature, next to our rules (v0.6 Phase 1)

Every rule is fitted on the same calib images (the threshold half used by our conformal rules) and evaluated once on test. Coverage = object recall of the rule alone; flags = candidates the rule keeps below the operating threshold, per image; o_j = annotated occlusion, o_hat = occlusion context.

**Target 1-a = 0.90**  (primary, D6)

| detector | rule | guarantee | coverage | per-image FNR | flags/image | coverage, most occluded o_j bucket | lowest coverage over o_hat strata |
|---|---|---|---|---|---|---|---|
| yolo26 | mondrian_confidence | recall >= 1-a in every occlusion-context stratum (ours, headline) | 0.901 [0.894, 0.907] | 0.103 [0.095, 0.111] | 1.30 | 0.792 (n=307) | 0.895 (0.30-1.00) |
| yolo26 | global_confidence | recall >= 1-a, marginal (ours, reference) | 0.901 [0.895, 0.907] | 0.099 [0.092, 0.107] | 1.00 | 0.720 (n=307) | 0.830 (0.30-1.00) |
| yolo26 | global_occlusion_aware | recall >= 1-a, marginal; duplicate- and context-aware score (ours) | 0.902 [0.896, 0.908] | 0.098 [0.091, 0.106] | 0.97 | 0.710 (n=307) | 0.826 (0.30-1.00) |
| yolo26 | global_gilg_iou_aware | recall >= 1-a, marginal; Gilg IoU-aware score | 0.900 [0.894, 0.907] | 0.100 [0.092, 0.107] | 0.77 | 0.713 (n=307) | 0.823 (0.30-1.00) |
| yolo26 | crc_fnr | E[per-image FNR] <= a (CRC; Hulsman 2024) | 0.899 [0.893, 0.905] | 0.101 [0.094, 0.109] | 0.92 | 0.717 (n=307) | 0.828 (0.30-1.00) |
| yolo26 | seqcrc_box_count_recall | box-count recall loss <= a (SeqCRC; no spatial matching) | 0.838 [0.831, 0.846] | 0.158 [0.150, 0.167] | 0.00 | 0.632 (n=307) | 0.738 (0.30-1.00) |
| yolo26 | seqcrc_box_count_threshold | P(fewer boxes than objects) <= a (SeqCRC) | 0.919 [0.913, 0.925] | 0.081 [0.075, 0.088] | 1.81 | 0.752 (n=307) | 0.858 (0.30-1.00) |
| yolo26 | pac | P(object miss rate > a) <= delta (PAC; Li 2022) | 0.907 [0.901, 0.913] | 0.093 [0.086, 0.101] | 1.25 | 0.730 (n=307) | 0.838 (0.30-1.00) |
| yolov8 | mondrian_confidence | recall >= 1-a in every occlusion-context stratum (ours, headline) | 0.905 [0.899, 0.911] | 0.099 [0.092, 0.106] | 1.55 | 0.782 (n=307) | 0.891 (0.30-1.00) |
| yolov8 | global_confidence | recall >= 1-a, marginal (ours, reference) | 0.903 [0.897, 0.909] | 0.098 [0.091, 0.105] | 1.32 | 0.697 (n=307) | 0.825 (0.30-1.00) |
| yolov8 | global_occlusion_aware | recall >= 1-a, marginal; duplicate- and context-aware score (ours) | 0.903 [0.896, 0.909] | 0.099 [0.092, 0.106] | 1.23 | 0.674 (n=307) | 0.816 (0.30-1.00) |
| yolov8 | global_gilg_iou_aware | recall >= 1-a, marginal; Gilg IoU-aware score | 0.905 [0.898, 0.911] | 0.096 [0.090, 0.104] | 1.10 | 0.700 (n=307) | 0.826 (0.30-1.00) |
| yolov8 | crc_fnr | E[per-image FNR] <= a (CRC; Hulsman 2024) | 0.900 [0.893, 0.905] | 0.102 [0.095, 0.109] | 1.14 | 0.697 (n=307) | 0.819 (0.30-1.00) |
| yolov8 | seqcrc_box_count_recall | box-count recall loss <= a (SeqCRC; no spatial matching) | 0.838 [0.831, 0.846] | 0.158 [0.150, 0.167] | 0.00 | 0.596 (n=307) | 0.733 (0.30-1.00) |
| yolov8 | seqcrc_box_count_threshold | P(fewer boxes than objects) <= a (SeqCRC) | 0.920 [0.914, 0.925] | 0.081 [0.075, 0.088] | 2.07 | 0.733 (n=307) | 0.852 (0.30-1.00) |
| yolov8 | pac | P(object miss rate > a) <= delta (PAC; Li 2022) | 0.908 [0.902, 0.914] | 0.093 [0.086, 0.101] | 1.51 | 0.713 (n=307) | 0.835 (0.30-1.00) |
| rtdetr | mondrian_confidence | recall >= 1-a in every occlusion-context stratum (ours, headline) | 0.901 [0.896, 0.907] | 0.099 [0.093, 0.106] | 1.65 | 0.795 (n=307) | 0.892 (0.30-1.00) |
| rtdetr | global_confidence | recall >= 1-a, marginal (ours, reference) | 0.902 [0.895, 0.908] | 0.095 [0.088, 0.101] | 1.31 | 0.723 (n=307) | 0.838 (0.30-1.00) |
| rtdetr | global_occlusion_aware | recall >= 1-a, marginal; duplicate- and context-aware score (ours) | 0.902 [0.896, 0.908] | 0.095 [0.088, 0.101] | 1.12 | 0.720 (n=307) | 0.838 (0.30-1.00) |
| rtdetr | global_gilg_iou_aware | recall >= 1-a, marginal; Gilg IoU-aware score | 0.901 [0.894, 0.907] | 0.096 [0.089, 0.103] | 0.95 | 0.713 (n=307) | 0.835 (0.30-1.00) |
| rtdetr | crc_fnr | E[per-image FNR] <= a (CRC; Hulsman 2024) | 0.894 [0.888, 0.900] | 0.102 [0.095, 0.108] | 1.01 | 0.697 (n=307) | 0.822 (0.30-1.00) |
| rtdetr | seqcrc_box_count_recall | box-count recall loss <= a (SeqCRC; no spatial matching) | 0.836 [0.828, 0.843] | 0.156 [0.147, 0.164] | 0.00 | 0.599 (n=307) | 0.728 (0.30-1.00) |
| rtdetr | seqcrc_box_count_threshold | P(fewer boxes than objects) <= a (SeqCRC) | 0.920 [0.914, 0.925] | 0.078 [0.072, 0.084] | 2.29 | 0.756 (n=307) | 0.865 (0.30-1.00) |
| rtdetr | pac | P(object miss rate > a) <= delta (PAC; Li 2022) | 0.906 [0.900, 0.912] | 0.091 [0.084, 0.097] | 1.54 | 0.726 (n=307) | 0.845 (0.30-1.00) |
| yolo26_ens | mondrian_confidence | recall >= 1-a in every occlusion-context stratum (ours, headline) | 0.901 [0.894, 0.908] | 0.100 [0.092, 0.107] | 0.75 | 0.765 (n=307) | 0.889 (0.30-1.00) |
| yolo26_ens | global_confidence | recall >= 1-a, marginal (ours, reference) | 0.900 [0.894, 0.907] | 0.097 [0.090, 0.105] | 0.48 | 0.726 (n=307) | 0.822 (0.30-1.00) |
| yolo26_ens | global_occlusion_aware | recall >= 1-a, marginal; duplicate- and context-aware score (ours) | 0.900 [0.893, 0.906] | 0.097 [0.090, 0.105] | 0.47 | 0.726 (n=307) | 0.817 (0.30-1.00) |
| yolo26_ens | global_gilg_iou_aware | recall >= 1-a, marginal; Gilg IoU-aware score | 0.900 [0.893, 0.906] | 0.099 [0.091, 0.106] | 0.45 | 0.736 (n=307) | 0.821 (0.30-1.00) |
| yolo26_ens | crc_fnr | E[per-image FNR] <= a (CRC; Hulsman 2024) | 0.898 [0.891, 0.904] | 0.100 [0.093, 0.108] | 0.40 | 0.723 (n=307) | 0.819 (0.30-1.00) |
| yolo26_ens | seqcrc_box_count_recall | box-count recall loss <= a (SeqCRC; no spatial matching) | 0.850 [0.843, 0.857] | 0.146 [0.138, 0.154] | 0.00 | 0.645 (n=307) | 0.745 (0.30-1.00) |
| yolo26_ens | seqcrc_box_count_threshold | P(fewer boxes than objects) <= a (SeqCRC) | 0.930 [0.925, 0.936] | 0.068 [0.062, 0.074] | 1.74 | 0.759 (n=307) | 0.873 (0.30-1.00) |
| yolo26_ens | pac | P(object miss rate > a) <= delta (PAC; Li 2022) | 0.905 [0.898, 0.911] | 0.093 [0.086, 0.100] | 0.61 | 0.739 (n=307) | 0.829 (0.30-1.00) |
| rtdetr_ens | mondrian_confidence | recall >= 1-a in every occlusion-context stratum (ours, headline) | 0.899 [0.893, 0.905] | 0.101 [0.094, 0.108] | 0.94 | 0.782 (n=307) | 0.895 (0.30-1.00) |
| rtdetr_ens | global_confidence | recall >= 1-a, marginal (ours, reference) | 0.901 [0.895, 0.907] | 0.095 [0.089, 0.102] | 0.60 | 0.717 (n=307) | 0.832 (0.30-1.00) |
| rtdetr_ens | global_occlusion_aware | recall >= 1-a, marginal; duplicate- and context-aware score (ours) | 0.900 [0.894, 0.906] | 0.097 [0.091, 0.104] | 0.55 | 0.710 (n=307) | 0.829 (0.30-1.00) |
| rtdetr_ens | global_gilg_iou_aware | recall >= 1-a, marginal; Gilg IoU-aware score | 0.900 [0.893, 0.906] | 0.097 [0.091, 0.104] | 0.49 | 0.717 (n=307) | 0.830 (0.30-1.00) |
| rtdetr_ens | crc_fnr | E[per-image FNR] <= a (CRC; Hulsman 2024) | 0.895 [0.888, 0.901] | 0.101 [0.095, 0.108] | 0.40 | 0.697 (n=307) | 0.818 (0.30-1.00) |
| rtdetr_ens | seqcrc_box_count_recall | box-count recall loss <= a (SeqCRC; no spatial matching) | 0.844 [0.837, 0.851] | 0.149 [0.141, 0.157] | 0.00 | 0.603 (n=307) | 0.726 (0.30-1.00) |
| rtdetr_ens | seqcrc_box_count_threshold | P(fewer boxes than objects) <= a (SeqCRC) | 0.922 [0.916, 0.928] | 0.076 [0.069, 0.082] | 1.61 | 0.769 (n=307) | 0.868 (0.30-1.00) |
| rtdetr_ens | pac | P(object miss rate > a) <= delta (PAC; Li 2022) | 0.906 [0.899, 0.912] | 0.092 [0.085, 0.099] | 0.77 | 0.726 (n=307) | 0.839 (0.30-1.00) |

**Target 1-a = 0.95**

| detector | rule | guarantee | coverage | per-image FNR | flags/image | coverage, most occluded o_j bucket | lowest coverage over o_hat strata |
|---|---|---|---|---|---|---|---|
| yolo26 | mondrian_confidence | recall >= 1-a in every occlusion-context stratum (ours, headline) | 0.951 [0.946, 0.956] | 0.052 [0.047, 0.058] | 6.18 | 0.902 (n=307) | 0.950 (0.00-0.05) |
| yolo26 | global_confidence | recall >= 1-a, marginal (ours, reference) | 0.951 [0.945, 0.956] | 0.050 [0.045, 0.056] | 4.56 | 0.821 (n=307) | 0.906 (0.30-1.00) |
| yolo26 | global_occlusion_aware | recall >= 1-a, marginal; duplicate- and context-aware score (ours) | 0.951 [0.945, 0.956] | 0.050 [0.045, 0.056] | 4.32 | 0.827 (n=307) | 0.906 (0.30-1.00) |
| yolo26 | global_gilg_iou_aware | recall >= 1-a, marginal; Gilg IoU-aware score | 0.950 [0.944, 0.954] | 0.050 [0.045, 0.056] | 3.43 | 0.824 (n=307) | 0.899 (0.30-1.00) |
| yolo26 | crc_fnr | E[per-image FNR] <= a (CRC; Hulsman 2024) | 0.951 [0.946, 0.956] | 0.050 [0.044, 0.055] | 4.64 | 0.824 (n=307) | 0.906 (0.30-1.00) |
| yolo26 | seqcrc_box_count_recall | box-count recall loss <= a (SeqCRC; no spatial matching) | 0.887 [0.880, 0.893] | 0.111 [0.104, 0.119] | 0.47 | 0.687 (n=307) | 0.805 (0.30-1.00) |
| yolo26 | seqcrc_box_count_threshold | P(fewer boxes than objects) <= a (SeqCRC) | 0.937 [0.932, 0.943] | 0.063 [0.058, 0.069] | 2.99 | 0.785 (n=307) | 0.884 (0.30-1.00) |
| yolo26 | pac | P(object miss rate > a) <= delta (PAC; Li 2022) | 0.953 [0.947, 0.958] | 0.048 [0.043, 0.053] | 4.95 | 0.834 (n=307) | 0.910 (0.30-1.00) |
| yolov8 | mondrian_confidence | recall >= 1-a in every occlusion-context stratum (ours, headline) | 0.951 [0.945, 0.955] | 0.052 [0.047, 0.058] | 6.71 | 0.886 (n=307) | 0.947 (0.30-1.00) |
| yolov8 | global_confidence | recall >= 1-a, marginal (ours, reference) | 0.948 [0.943, 0.952] | 0.054 [0.048, 0.060] | 4.50 | 0.834 (n=307) | 0.904 (0.30-1.00) |
| yolov8 | global_occlusion_aware | recall >= 1-a, marginal; duplicate- and context-aware score (ours) | 0.948 [0.943, 0.953] | 0.054 [0.048, 0.060] | 4.41 | 0.834 (n=307) | 0.905 (0.30-1.00) |
| yolov8 | global_gilg_iou_aware | recall >= 1-a, marginal; Gilg IoU-aware score | 0.949 [0.944, 0.953] | 0.053 [0.048, 0.058] | 3.69 | 0.811 (n=307) | 0.904 (0.30-1.00) |
| yolov8 | crc_fnr | E[per-image FNR] <= a (CRC; Hulsman 2024) | 0.950 [0.945, 0.955] | 0.052 [0.047, 0.057] | 4.86 | 0.837 (n=307) | 0.908 (0.30-1.00) |
| yolov8 | seqcrc_box_count_recall | box-count recall loss <= a (SeqCRC; no spatial matching) | 0.885 [0.878, 0.891] | 0.115 [0.107, 0.123] | 0.61 | 0.678 (n=307) | 0.797 (0.30-1.00) |
| yolov8 | seqcrc_box_count_threshold | P(fewer boxes than objects) <= a (SeqCRC) | 0.937 [0.931, 0.942] | 0.064 [0.059, 0.071] | 3.16 | 0.779 (n=307) | 0.882 (0.30-1.00) |
| yolov8 | pac | P(object miss rate > a) <= delta (PAC; Li 2022) | 0.951 [0.946, 0.956] | 0.051 [0.046, 0.056] | 5.04 | 0.837 (n=307) | 0.908 (0.30-1.00) |
| rtdetr | mondrian_confidence | recall >= 1-a in every occlusion-context stratum (ours, headline) | 0.950 [0.945, 0.955] | 0.051 [0.046, 0.057] | 8.80 | 0.919 (n=307) | 0.949 (0.00-0.05) |
| rtdetr | global_confidence | recall >= 1-a, marginal (ours, reference) | 0.947 [0.943, 0.952] | 0.053 [0.048, 0.058] | 5.64 | 0.860 (n=307) | 0.910 (0.30-1.00) |
| rtdetr | global_occlusion_aware | recall >= 1-a, marginal; duplicate- and context-aware score (ours) | 0.948 [0.944, 0.953] | 0.052 [0.047, 0.057] | 4.63 | 0.857 (n=307) | 0.910 (0.30-1.00) |
| rtdetr | global_gilg_iou_aware | recall >= 1-a, marginal; Gilg IoU-aware score | 0.948 [0.943, 0.952] | 0.053 [0.048, 0.058] | 3.38 | 0.853 (n=307) | 0.909 (0.30-1.00) |
| rtdetr | crc_fnr | E[per-image FNR] <= a (CRC; Hulsman 2024) | 0.949 [0.944, 0.954] | 0.051 [0.046, 0.056] | 5.87 | 0.860 (n=307) | 0.911 (0.30-1.00) |
| rtdetr | seqcrc_box_count_recall | box-count recall loss <= a (SeqCRC; no spatial matching) | 0.880 [0.873, 0.886] | 0.115 [0.108, 0.122] | 0.57 | 0.671 (n=307) | 0.799 (0.30-1.00) |
| rtdetr | seqcrc_box_count_threshold | P(fewer boxes than objects) <= a (SeqCRC) | 0.936 [0.930, 0.941] | 0.063 [0.057, 0.068] | 3.70 | 0.801 (n=307) | 0.887 (0.30-1.00) |
| rtdetr | pac | P(object miss rate > a) <= delta (PAC; Li 2022) | 0.952 [0.947, 0.956] | 0.049 [0.044, 0.054] | 6.61 | 0.866 (n=307) | 0.917 (0.30-1.00) |
| yolo26_ens | mondrian_confidence | recall >= 1-a in every occlusion-context stratum (ours, headline) | 0.949 [0.944, 0.954] | 0.053 [0.048, 0.059] | 3.92 | 0.902 (n=307) | 0.948 (0.00-0.05) |
| yolo26_ens | global_confidence | recall >= 1-a, marginal (ours, reference) | 0.948 [0.943, 0.953] | 0.052 [0.047, 0.058] | 3.31 | 0.834 (n=307) | 0.905 (0.30-1.00) |
| yolo26_ens | global_occlusion_aware | recall >= 1-a, marginal; duplicate- and context-aware score (ours) | 0.948 [0.942, 0.952] | 0.053 [0.048, 0.058] | 2.55 | 0.827 (n=307) | 0.903 (0.30-1.00) |
| yolo26_ens | global_gilg_iou_aware | recall >= 1-a, marginal; Gilg IoU-aware score | 0.946 [0.941, 0.951] | 0.053 [0.048, 0.059] | 2.42 | 0.805 (n=307) | 0.900 (0.30-1.00) |
| yolo26_ens | crc_fnr | E[per-image FNR] <= a (CRC; Hulsman 2024) | 0.949 [0.944, 0.954] | 0.051 [0.046, 0.057] | 3.43 | 0.834 (n=307) | 0.907 (0.30-1.00) |
| yolo26_ens | seqcrc_box_count_recall | box-count recall loss <= a (SeqCRC; no spatial matching) | 0.896 [0.889, 0.902] | 0.102 [0.095, 0.110] | 0.33 | 0.723 (n=307) | 0.816 (0.30-1.00) |
| yolo26_ens | seqcrc_box_count_threshold | P(fewer boxes than objects) <= a (SeqCRC) | 0.941 [0.936, 0.946] | 0.059 [0.053, 0.065] | 2.54 | 0.785 (n=307) | 0.892 (0.30-1.00) |
| yolo26_ens | pac | P(object miss rate > a) <= delta (PAC; Li 2022) | 0.952 [0.947, 0.957] | 0.049 [0.044, 0.054] | 3.86 | 0.837 (n=307) | 0.912 (0.30-1.00) |
| rtdetr_ens | mondrian_confidence | recall >= 1-a in every occlusion-context stratum (ours, headline) | 0.949 [0.944, 0.954] | 0.052 [0.047, 0.058] | 5.36 | 0.906 (n=307) | 0.947 (0.05-0.30) |
| rtdetr_ens | global_confidence | recall >= 1-a, marginal (ours, reference) | 0.948 [0.943, 0.953] | 0.051 [0.046, 0.056] | 4.10 | 0.831 (n=307) | 0.907 (0.30-1.00) |
| rtdetr_ens | global_occlusion_aware | recall >= 1-a, marginal; duplicate- and context-aware score (ours) | 0.948 [0.943, 0.952] | 0.051 [0.046, 0.056] | 2.74 | 0.827 (n=307) | 0.906 (0.30-1.00) |
| rtdetr_ens | global_gilg_iou_aware | recall >= 1-a, marginal; Gilg IoU-aware score | 0.947 [0.942, 0.952] | 0.052 [0.046, 0.057] | 2.40 | 0.827 (n=307) | 0.906 (0.30-1.00) |
| rtdetr_ens | crc_fnr | E[per-image FNR] <= a (CRC; Hulsman 2024) | 0.949 [0.944, 0.954] | 0.050 [0.044, 0.055] | 4.30 | 0.840 (n=307) | 0.910 (0.30-1.00) |
| rtdetr_ens | seqcrc_box_count_recall | box-count recall loss <= a (SeqCRC; no spatial matching) | 0.891 [0.884, 0.897] | 0.105 [0.098, 0.112] | 0.27 | 0.691 (n=307) | 0.809 (0.30-1.00) |
| rtdetr_ens | seqcrc_box_count_threshold | P(fewer boxes than objects) <= a (SeqCRC) | 0.941 [0.936, 0.946] | 0.058 [0.052, 0.063] | 3.26 | 0.808 (n=307) | 0.896 (0.30-1.00) |
| rtdetr_ens | pac | P(object miss rate > a) <= delta (PAC; Li 2022) | 0.952 [0.947, 0.956] | 0.047 [0.042, 0.053] | 4.67 | 0.847 (n=307) | 0.914 (0.30-1.00) |

#### IoU-aware calibration (Gilg et al.) vs our duplicate-aware scores

Recovery of missed vehicles vs plain confidence at matched extra false alarms (RQ2 protocol), all misses / occluded misses. Gilg's score is fitted on the calib score half (no fit-role predictions exist).

| detector | Spearman: Gilg vs z_dup (weak candidates) | Spearman: Gilg's u vs our dup feature | Gilg IoU-aware | our z_dup | our default (duplicate- and context-aware) |
|---|---|---|---|---|---|
| yolo26 | 0.621 | 0.294 | 0.0597 [0.0511, 0.0693] / occl. 0.0541 | 0.0386 [0.0319, 0.0456] / occl. 0.0053 | 0.0446 [0.0385, 0.0511] / occl. 0.0188 |
| yolov8 | 0.675 | 0.251 | 0.0515 [0.0427, 0.0608] / occl. 0.0549 | 0.0306 [0.0234, 0.0374] / occl. 0.0044 | 0.0324 [0.0253, 0.0395] / occl. 0.0022 |
| rtdetr | 0.472 | 0.202 | 0.0783 [0.0692, 0.0862] / occl. 0.0853 | 0.0529 [0.0438, 0.0603] / occl. 0.0277 | 0.0604 [0.0526, 0.0672] / occl. 0.0615 |
| yolo26_ens | 0.741 | 0.244 | 0.0298 [0.0223, 0.0372] / occl. 0.0273 | 0.0200 [0.0123, 0.0292] / occl. -0.0272 | 0.0293 [0.0235, 0.0359] / occl. 0.0089 |
| rtdetr_ens | 0.707 | 0.121 | 0.0607 [0.0512, 0.0703] / occl. 0.0638 | 0.0427 [0.0327, 0.0534] / occl. 0.0006 | 0.0543 [0.0458, 0.0627] / occl. 0.0475 |

#### Localisation-aware calibration and LRP (Kuzucu et al. 2024)

LaECE_0 / LaACE_0: 25 bins, target = IoU with the matched vehicle (any IoU > 0), at the operating threshold. LRP at tau = 0.5; the LRP-optimal threshold is chosen on calib.

| detector | LaECE_0 | LaACE_0 | D-ECE | LRP at F1-optimal threshold | LRP at LRP-optimal threshold | recall / false alarms per image at LRP-optimal |
|---|---|---|---|---|---|---|
| yolo26 | 0.0418 [0.0367, 0.0477] | 0.0439 [0.0385, 0.0496] | 0.0497 | 0.46: 0.3709 [0.3628, 0.3806] | 0.54: 0.3706 [0.3617, 0.3799] | 0.843 / 1.13 |
| yolov8 | 0.0357 [0.0307, 0.0437] | 0.0369 [0.0324, 0.0443] | 0.0490 | 0.47: 0.3733 [0.3652, 0.3829] | 0.47: 0.3733 [0.3652, 0.3829] | 0.863 / 1.35 |
| rtdetr | 0.0878 [0.0807, 0.0948] | 0.0878 [0.0807, 0.0948] | 0.0489 | 0.65: 0.3766 [0.3675, 0.3852] | 0.67: 0.3753 [0.3663, 0.3845] | 0.848 / 1.25 |
| yolo26_ens | 0.0211 [0.0172, 0.0270] | 0.0216 [0.0183, 0.0265] | 0.0928 | 0.42: 0.3439 [0.3353, 0.3533] | 0.46: 0.3440 [0.3350, 0.3534] | 0.869 / 1.07 |
| rtdetr_ens | 0.0662 [0.0591, 0.0731] | 0.0662 [0.0594, 0.0731] | 0.0474 | 0.63: 0.3512 [0.3423, 0.3600] | 0.65: 0.3503 [0.3419, 0.3596] | 0.871 / 1.15 |

### RQ3: calibration metrics at the operating threshold

| detector | D-ECE | MA-ECE | miss mass | OCE | threshold D-ECE alone would choose | occluded vehicles: share of objects / share of D-ECE blind spot |
|---|---|---|---|---|---|---|
| yolo26 | 0.050 [0.044, 0.056] | 0.159 [0.152, 0.167] | 0.115 | 0.161 | 0.95 (recall 0.04) | 0.25 / 0.38 |
| yolov8 | 0.049 [0.043, 0.056] | 0.165 [0.157, 0.173] | 0.122 | 0.167 | 0.95 (recall 0.07) | 0.25 / 0.38 |
| rtdetr | 0.049 [0.044, 0.055] | 0.167 [0.160, 0.174] | 0.124 | 0.155 | 0.95 (recall 0.18) | 0.25 / 0.39 |
| yolo26_ens | 0.093 [0.085, 0.101] | 0.188 [0.180, 0.196] | 0.105 | 0.161 | 0.95 (recall 0.01) | 0.25 / 0.40 |
| rtdetr_ens | 0.047 [0.041, 0.053] | 0.149 [0.142, 0.156] | 0.107 | 0.137 | 0.95 (recall 0.16) | 0.25 / 0.40 |

### RQ5: why vehicles are missed (native classes, whatever the mode)

| detector | misses | confused | suppressed | unproposed | merged | visibility misses: odds per +0.1 occlusion | confusion misses: odds per +1 SD rarity | confusion misses: odds per +0.1 occlusion |
|---|---|---|---|---|---|---|---|---|
| yolo26 | 3271 | 0.36 | 0.47 | 0.09 | 0.00 | 1.188 [1.163, 1.218] | 2.997 [2.857, 3.153] | 1.015 [0.978, 1.047] |
| yolov8 | 3369 | 0.34 | 0.47 | 0.09 | 0.00 | 1.190 [1.166, 1.220] | 2.914 [2.770, 3.045] | 0.996 [0.962, 1.027] |
| rtdetr | 3275 | 0.31 | 0.62 | 0.07 | 0.00 | 1.207 [1.186, 1.233] | 2.890 [2.719, 3.054] | 1.027 [0.986, 1.059] |
| yolo26_ens | 3080 | 0.39 | 0.49 | 0.06 | 0.00 | 1.198 [1.174, 1.229] | 2.967 [2.829, 3.128] | 1.032 [1.003, 1.063] |
| rtdetr_ens | 3022 | 0.36 | 0.56 | 0.05 | 0.00 | 1.217 [1.193, 1.247] | 2.889 [2.739, 3.043] | 1.021 [0.987, 1.052] |

### RQ4: transfer with the detector frozen (recall-side quantities only)

Each coverage is followed by (flags per image; certifiable). A recalibrated coverage that is not certifiable is the best the rule could do, not a guarantee.

| detector | dataset | matching | recall, development -> transfer | rule | target | coverage on development | coverage on transfer, frozen | coverage on transfer, recalibrated |
|---|---|---|---|---|---|---|---|---|
| yolo26 | pals | class_agnostic | 0.869 -> 0.718 | global_confidence | 0.95 | 0.950 (4.47; yes) | 0.889 (2.76; yes) | 0.949 (38.62; NO) |
| yolo26 | pals | class_agnostic | 0.869 -> 0.718 | mondrian_confidence | 0.95 | 0.951 (6.11; yes) | 0.865 (3.43; yes) | 0.934 (9.80; NO) |
| yolo26 | pals | class_agnostic | 0.869 -> 0.718 | global_occlusion_aware | 0.95 | 0.950 (4.24; yes) | 0.889 (2.58; yes) | 0.949 (38.62; NO) |
| yolo26 | pals | class_agnostic | 0.869 -> 0.718 | global_confidence | 0.90 | 0.901 (0.98; yes) | 0.782 (0.58; yes) | 0.902 (3.59; yes) |
| yolo26 | pals | class_agnostic | 0.869 -> 0.718 | mondrian_confidence | 0.90 | 0.901 (1.29; yes) | 0.747 (0.70; yes) | 0.898 (7.42; NO) |
| yolo26 | pals | class_agnostic | 0.869 -> 0.718 | global_occlusion_aware | 0.90 | 0.902 (0.96; yes) | 0.793 (0.55; yes) | 0.896 (2.86; yes) |
| yolov8 | pals | class_agnostic | 0.863 -> 0.711 | global_confidence | 0.95 | 0.947 (4.43; yes) | 0.858 (2.69; yes) | 0.931 (26.04; NO) |
| yolov8 | pals | class_agnostic | 0.863 -> 0.711 | mondrian_confidence | 0.95 | 0.950 (6.63; yes) | 0.851 (3.66; yes) | 0.931 (26.04; NO) |
| yolov8 | pals | class_agnostic | 0.863 -> 0.711 | global_occlusion_aware | 0.95 | 0.948 (4.34; yes) | 0.860 (2.63; yes) | 0.931 (26.04; NO) |
| yolov8 | pals | class_agnostic | 0.863 -> 0.711 | global_confidence | 0.90 | 0.903 (1.30; yes) | 0.775 (0.78; yes) | 0.914 (8.53; yes) |
| yolov8 | pals | class_agnostic | 0.863 -> 0.711 | mondrian_confidence | 0.90 | 0.905 (1.54; yes) | 0.753 (0.86; yes) | 0.906 (10.08; NO) |
| yolov8 | pals | class_agnostic | 0.863 -> 0.711 | global_occlusion_aware | 0.90 | 0.902 (1.21; yes) | 0.777 (0.75; yes) | 0.911 (7.43; yes) |
| rtdetr | pals | class_agnostic | 0.859 -> 0.630 | global_confidence | 0.95 | 0.948 (5.55; yes) | 0.825 (4.17; yes) | 0.959 (246.62; yes) |
| rtdetr | pals | class_agnostic | 0.859 -> 0.630 | mondrian_confidence | 0.95 | 0.951 (8.71; yes) | 0.814 (5.08; yes) | 0.959 (246.62; yes) |
| rtdetr | pals | class_agnostic | 0.859 -> 0.630 | global_occlusion_aware | 0.95 | 0.949 (4.56; yes) | 0.826 (3.47; yes) | 0.961 (285.24; yes) |
| rtdetr | pals | class_agnostic | 0.859 -> 0.630 | global_confidence | 0.90 | 0.902 (1.29; yes) | 0.700 (0.83; yes) | 0.916 (25.16; yes) |
| rtdetr | pals | class_agnostic | 0.859 -> 0.630 | mondrian_confidence | 0.90 | 0.902 (1.63; yes) | 0.683 (0.77; yes) | 0.916 (25.16; yes) |
| rtdetr | pals | class_agnostic | 0.859 -> 0.630 | global_occlusion_aware | 0.90 | 0.902 (1.10; yes) | 0.698 (0.75; yes) | 0.911 (19.76; yes) |
| yolo26_ens | pals | class_agnostic | 0.883 -> 0.764 | global_confidence | 0.95 | 0.948 (3.26; yes) | 0.882 (2.12; yes) | 0.958 (13.74; yes) |
| yolo26_ens | pals | class_agnostic | 0.883 -> 0.764 | mondrian_confidence | 0.95 | 0.949 (3.87; yes) | 0.872 (2.35; yes) | 0.955 (17.58; NO) |
| yolo26_ens | pals | class_agnostic | 0.883 -> 0.764 | global_occlusion_aware | 0.95 | 0.947 (2.51; yes) | 0.882 (1.48; yes) | 0.964 (49.76; yes) |
| yolo26_ens | pals | class_agnostic | 0.883 -> 0.764 | global_confidence | 0.90 | 0.900 (0.48; yes) | 0.808 (0.27; yes) | 0.898 (2.99; yes) |
| yolo26_ens | pals | class_agnostic | 0.883 -> 0.764 | mondrian_confidence | 0.90 | 0.901 (0.75; yes) | 0.770 (0.39; yes) | 0.909 (15.46; NO) |
| yolo26_ens | pals | class_agnostic | 0.883 -> 0.764 | global_occlusion_aware | 0.90 | 0.900 (0.46; yes) | 0.804 (0.26; yes) | 0.891 (1.51; yes) |
| rtdetr_ens | pals | class_agnostic | 0.880 -> 0.693 | global_confidence | 0.95 | 0.948 (4.05; yes) | 0.858 (2.92; yes) | 0.968 (132.34; yes) |
| rtdetr_ens | pals | class_agnostic | 0.880 -> 0.693 | mondrian_confidence | 0.95 | 0.948 (5.32; yes) | 0.846 (3.24; yes) | 0.966 (164.99; NO) |
| rtdetr_ens | pals | class_agnostic | 0.880 -> 0.693 | global_occlusion_aware | 0.95 | 0.947 (2.70; yes) | 0.859 (1.92; yes) | 0.963 (179.93; yes) |
| rtdetr_ens | pals | class_agnostic | 0.880 -> 0.693 | global_confidence | 0.90 | 0.901 (0.59; yes) | 0.754 (0.41; yes) | 0.914 (10.51; yes) |
| rtdetr_ens | pals | class_agnostic | 0.880 -> 0.693 | mondrian_confidence | 0.90 | 0.899 (0.93; yes) | 0.699 (0.39; yes) | 0.913 (23.53; yes) |
| rtdetr_ens | pals | class_agnostic | 0.880 -> 0.693 | global_occlusion_aware | 0.90 | 0.900 (0.54; yes) | 0.754 (0.40; yes) | 0.911 (7.33; yes) |

### Detector ranking (best first)

| comparison | metric | ranking |
|---|---|---|
| each at its own threshold | dece | rtdetr_ens < rtdetr < yolov8 < yolo26 < yolo26_ens |
| each at its own threshold | ma_ece | rtdetr_ens < yolo26 < yolov8 < rtdetr < yolo26_ens |
| each at its own threshold | ma_brier | rtdetr_ens < yolo26_ens < yolo26 < yolov8 < rtdetr |
| each at its own threshold | oce_max | rtdetr_ens < rtdetr < yolo26_ens < yolo26 < yolov8 |
| at matched false alarms per image | dece | rtdetr < yolov8 < rtdetr_ens < yolo26 < yolo26_ens |
| at matched false alarms per image | ma_ece | rtdetr_ens < yolov8 < yolo26 < rtdetr < yolo26_ens |
| at matched false alarms per image | ma_brier | yolo26_ens < rtdetr_ens < yolo26 < yolov8 < rtdetr |
| at matched false alarms per image | oce_max | rtdetr_ens < yolo26_ens < rtdetr < yolo26 < yolov8 |

Point-estimate orders above can differ by noise. Below, a metric *separates* a pair only if its 95% paired-difference interval excludes zero; a *flip* needs both D-ECE and the other metric to separate the pair, in opposite directions.

**each at its own threshold**

| pair (difference = first minus second) | D-ECE | MA-ECE | MA-Brier | OCE | flips vs D-ECE |
|---|---|---|---|---|---|
| rtdetr vs rtdetr_ens | cannot separate [-0.0028, 0.0061] | rtdetr_ens better [0.0133, 0.0235] | rtdetr_ens better [0.0206, 0.0258] | rtdetr_ens better [0.0157, 0.0215] | none |
| rtdetr vs yolo26 | cannot separate [-0.0074, 0.0045] | yolo26 better [0.0010, 0.0147] | yolo26 better [0.0054, 0.0130] | rtdetr better [-0.0105, -0.0023] | none |
| rtdetr vs yolo26_ens | rtdetr better [-0.0517, -0.0372] | rtdetr better [-0.0269, -0.0144] | yolo26_ens better [0.0180, 0.0248] | rtdetr better [-0.0092, -0.0022] | ma_brier |
| rtdetr vs yolov8 | cannot separate [-0.0075, 0.0053] | cannot separate [-0.0046, 0.0088] | yolov8 better [0.0008, 0.0090] | rtdetr better [-0.0166, -0.0080] | none |
| rtdetr_ens vs yolo26 | cannot separate [-0.0092, 0.0028] | rtdetr_ens better [-0.0180, -0.0040] | rtdetr_ens better [-0.0176, -0.0102] | rtdetr_ens better [-0.0293, -0.0207] | none |
| rtdetr_ens vs yolo26_ens | rtdetr_ens better [-0.0529, -0.0384] | rtdetr_ens better [-0.0464, -0.0318] | cannot separate [-0.0049, 0.0014] | rtdetr_ens better [-0.0272, -0.0208] | none |
| rtdetr_ens vs yolov8 | cannot separate [-0.0092, 0.0039] | rtdetr_ens better [-0.0236, -0.0094] | rtdetr_ens better [-0.0220, -0.0141] | rtdetr_ens better [-0.0350, -0.0265] | none |
| yolo26 vs yolo26_ens | yolo26 better [-0.0470, -0.0376] | yolo26 better [-0.0329, -0.0230] | yolo26_ens better [0.0097, 0.0147] | cannot separate [-0.0018, 0.0033] | ma_brier |
| yolo26 vs yolov8 | cannot separate [-0.0039, 0.0053] | yolo26 better [-0.0117, -0.0001] | yolo26 better [-0.0086, -0.0006] | yolo26 better [-0.0105, -0.0018] | none |
| yolo26_ens vs yolov8 | yolov8 better [0.0392, 0.0471] | yolov8 better [0.0182, 0.0267] | yolo26_ens better [-0.0198, -0.0131] | yolo26_ens better [-0.0104, -0.0030] | ma_brier, oce_max |

- D-ECE cannot separate rtdetr and rtdetr_ens (each at its own threshold).
- D-ECE cannot separate rtdetr and yolo26 (each at its own threshold).
- D-ECE cannot separate rtdetr and yolov8 (each at its own threshold).
- D-ECE cannot separate rtdetr_ens and yolo26 (each at its own threshold).
- D-ECE cannot separate rtdetr_ens and yolov8 (each at its own threshold).
- D-ECE cannot separate yolo26 and yolov8 (each at its own threshold).

**at matched false alarms per image** (budget 1.35 false alarms/image)

| pair (difference = first minus second) | D-ECE | MA-ECE | MA-Brier | OCE | flips vs D-ECE |
|---|---|---|---|---|---|
| rtdetr vs rtdetr_ens | cannot separate [-0.0064, 0.0026] | rtdetr_ens better [0.0194, 0.0301] | rtdetr_ens better [0.0252, 0.0312] | rtdetr_ens better [0.0241, 0.0303] | none |
| rtdetr vs yolo26 | cannot separate [-0.0087, 0.0036] | cannot separate [-0.0026, 0.0113] | yolo26 better [0.0036, 0.0115] | rtdetr better [-0.0116, -0.0026] | none |
| rtdetr vs yolo26_ens | rtdetr better [-0.0511, -0.0365] | rtdetr better [-0.0150, -0.0015] | yolo26_ens better [0.0254, 0.0321] | yolo26_ens better [0.0001, 0.0071] | ma_brier, oce_max |
| rtdetr vs yolov8 | cannot separate [-0.0085, 0.0040] | cannot separate [-0.0009, 0.0115] | yolov8 better [0.0032, 0.0117] | rtdetr better [-0.0125, -0.0036] | none |
| rtdetr_ens vs yolo26 | cannot separate [-0.0070, 0.0051] | rtdetr_ens better [-0.0274, -0.0133] | rtdetr_ens better [-0.0246, -0.0164] | rtdetr_ens better [-0.0386, -0.0296] | none |
| rtdetr_ens vs yolo26_ens | rtdetr_ens better [-0.0493, -0.0340] | rtdetr_ens better [-0.0407, -0.0258] | cannot separate [-0.0026, 0.0037] | rtdetr_ens better [-0.0266, -0.0202] | none |
| rtdetr_ens vs yolov8 | cannot separate [-0.0071, 0.0066] | rtdetr_ens better [-0.0266, -0.0120] | rtdetr_ens better [-0.0247, -0.0166] | rtdetr_ens better [-0.0393, -0.0309] | none |
| yolo26 vs yolo26_ens | yolo26 better [-0.0453, -0.0361] | yolo26 better [-0.0170, -0.0082] | yolo26_ens better [0.0186, 0.0238] | yolo26_ens better [0.0078, 0.0134] | ma_brier, oce_max |
| yolo26 vs yolov8 | cannot separate [-0.0036, 0.0055] | cannot separate [-0.0042, 0.0067] | cannot separate [-0.0044, 0.0039] | cannot separate [-0.0057, 0.0032] | none |
| yolo26_ens vs yolov8 | yolov8 better [0.0374, 0.0454] | yolov8 better [0.0093, 0.0178] | yolo26_ens better [-0.0245, -0.0178] | yolo26_ens better [-0.0152, -0.0078] | ma_brier, oce_max |

- D-ECE cannot separate rtdetr and rtdetr_ens (at matched false alarms per image).
- D-ECE cannot separate rtdetr and yolo26 (at matched false alarms per image).
- D-ECE cannot separate rtdetr and yolov8 (at matched false alarms per image).
- D-ECE cannot separate rtdetr_ens and yolo26 (at matched false alarms per image).
- D-ECE cannot separate rtdetr_ens and yolov8 (at matched false alarms per image).
- D-ECE cannot separate yolo26 and yolov8 (at matched false alarms per image).

Ranking flip relative to D-ECE (both intervals exclude zero, opposite signs) survives matched cost: **True**

#### MA-Brier decomposition

Each term is its contribution to MA-Brier (share of MA-Brier in brackets); the three sum to MA-Brier. OCE averages over vehicles only, so it has no false-alarm term.

| detector | where | threshold | false alarms/image | MA-Brier | false-alarm term | matched-detection term | miss term | OCE |
|---|---|---|---|---|---|---|---|---|
| yolo26 | own threshold | 0.46 | 1.45 | 0.2010 | 0.0575 (0.29) | 0.0285 (0.14) | 0.1150 (0.57) | 0.1613 |
| yolov8 | own threshold | 0.47 | 1.35 | 0.2053 | 0.0550 (0.27) | 0.0287 (0.14) | 0.1216 (0.59) | 0.1673 |
| rtdetr | own threshold | 0.65 | 1.40 | 0.2103 | 0.0716 (0.34) | 0.0143 (0.07) | 0.1245 (0.59) | 0.1551 |
| yolo26_ens | own threshold | 0.42 | 1.23 | 0.1889 | 0.0434 (0.23) | 0.0409 (0.22) | 0.1045 (0.55) | 0.1606 |
| rtdetr_ens | own threshold | 0.63 | 1.27 | 0.1872 | 0.0637 (0.34) | 0.0168 (0.09) | 0.1067 (0.57) | 0.1366 |
| yolo26 | matched cost | 0.48 | 1.35 | 0.2052 | 0.0561 (0.27) | 0.0271 (0.13) | 0.1220 (0.59) | 0.1662 |
| yolov8 | matched cost | 0.47 | 1.35 | 0.2053 | 0.0550 (0.27) | 0.0287 (0.14) | 0.1216 (0.59) | 0.1673 |
| rtdetr | matched cost | 0.66 | 1.35 | 0.2129 | 0.0700 (0.33) | 0.0139 (0.07) | 0.1291 (0.61) | 0.1593 |
| yolo26_ens | matched cost | 0.39 | 1.35 | 0.1841 | 0.0447 (0.24) | 0.0430 (0.23) | 0.0964 (0.52) | 0.1556 |
| rtdetr_ens | matched cost | 0.62 | 1.35 | 0.1847 | 0.0659 (0.36) | 0.0173 (0.09) | 0.1015 (0.55) | 0.1321 |

## Matching: native

### Gates

- yolo26 / detector_usable_on_development: pass
- yolo26 / enough_occlusion: pass
- yolo26 / detector_usable_on_pals / coarse: pass
- yolov8 / detector_usable_on_development: pass
- yolov8 / enough_occlusion: pass
- yolov8 / detector_usable_on_pals / coarse: pass
- rtdetr / detector_usable_on_development: pass
- rtdetr / enough_occlusion: pass
- rtdetr / detector_usable_on_pals / coarse: pass
- yolo26_ens / detector_usable_on_development: pass
- yolo26_ens / enough_occlusion: pass
- yolo26_ens / detector_usable_on_pals / coarse: pass
- rtdetr_ens / detector_usable_on_development: pass
- rtdetr_ens / enough_occlusion: pass
- rtdetr_ens / detector_usable_on_pals / coarse: pass

### Detection at each detector's operating threshold

| detector | threshold | AP50 | recall | precision | false alarms/image | recall at floor |
|---|---|---|---|---|---|---|
| yolo26 | 0.49 | 0.668 | 0.789 | 0.800 | 2.06 | 0.906 |
| yolov8 | 0.47 | 0.661 | 0.791 | 0.797 | 2.10 | 0.898 |
| rtdetr | 0.65 | 0.728 | 0.796 | 0.802 | 2.05 | 0.984 |
| yolo26_ens | 0.47 | 0.696 | 0.794 | 0.824 | 1.77 | 0.926 |
| rtdetr_ens | 0.66 | 0.727 | 0.799 | 0.824 | 1.78 | 0.973 |

### RQ1: miss rate by annotated occlusion severity

| detector | 0.0-0.2 (n=12021) | 0.2-0.4 (n=2152) | 0.4-0.6 (n=1169) | 0.6-0.8 (n=444) | 0.8-1.0 (n=307) | odds of a miss per +0.1 (size, class controlled) | partial Spearman | context vs annotated occlusion |
|---|---|---|---|---|---|---|---|---|
| yolo26 | 0.186 | 0.248 | 0.281 | 0.401 | 0.391 | 1.155 [1.136, 1.180] | 0.184 [0.166, 0.199] | 0.88 |
| yolov8 | 0.186 | 0.243 | 0.287 | 0.349 | 0.404 | 1.153 [1.134, 1.177] | 0.172 [0.154, 0.188] | 0.88 |
| rtdetr | 0.176 | 0.241 | 0.285 | 0.394 | 0.433 | 1.180 [1.160, 1.203] | 0.238 [0.220, 0.253] | 0.89 |
| yolo26_ens | 0.180 | 0.242 | 0.284 | 0.401 | 0.384 | 1.161 [1.140, 1.185] | 0.196 [0.178, 0.212] | 0.90 |
| rtdetr_ens | 0.174 | 0.239 | 0.281 | 0.408 | 0.414 | 1.181 [1.158, 1.206] | 0.233 [0.215, 0.248] | 0.90 |

`*` marks a bucket with fewer objects than `min_bucket_n`; do not read it.

#### RQ1 by object size (tertiles of box area relative to the image)

| detector | tertile edges (relative area) | small | medium | large |
|---|---|---|---|---|
| yolo26 | 4.14e-03 / 1.41e-02 | 0.258 [0.243, 0.271] (n=5362) | 0.198 [0.187, 0.212] (n=5366) | 0.177 [0.166, 0.189] (n=5365) |
| yolov8 | 4.14e-03 / 1.41e-02 | 0.251 [0.238, 0.264] (n=5362) | 0.199 [0.187, 0.211] (n=5366) | 0.178 [0.168, 0.189] (n=5365) |
| rtdetr | 4.14e-03 / 1.41e-02 | 0.270 [0.256, 0.284] (n=5362) | 0.193 [0.182, 0.206] (n=5366) | 0.148 [0.138, 0.157] (n=5365) |
| yolo26_ens | 4.14e-03 / 1.41e-02 | 0.255 [0.242, 0.269] (n=5362) | 0.194 [0.182, 0.206] (n=5366) | 0.169 [0.158, 0.181] (n=5365) |
| rtdetr_ens | 4.14e-03 / 1.41e-02 | 0.265 [0.251, 0.278] (n=5362) | 0.192 [0.180, 0.204] (n=5366) | 0.147 [0.137, 0.157] (n=5365) |

Miss rate by occlusion bucket within each size tertile (n in brackets; `*` = fewer than `min_bucket_n`):

| detector | size | 0.0-0.2 | 0.2-0.4 | 0.4-0.6 | 0.6-0.8 | 0.8-1.0 |
|---|---|---|---|---|---|---|
| yolo26 | small | 0.226 (4021) | 0.326 (733) | 0.326 (353) | 0.486 (138) | 0.436 (117) |
| yolo26 | medium | 0.165 (3730) | 0.234 (785) | 0.263 (520) | 0.407 (214) | 0.350 (117) |
| yolo26 | large | 0.167 (4270) | 0.174 (634) | 0.257 (296) | 0.261 (92) | 0.384 (73) |
| yolov8 | small | 0.220 (4021) | 0.325 (733) | 0.331 (353) | 0.435 (138) | 0.410 (117) |
| yolov8 | medium | 0.170 (3730) | 0.217 (785) | 0.279 (520) | 0.336 (214) | 0.385 (117) |
| yolov8 | large | 0.167 (4270) | 0.180 (634) | 0.247 (296) | 0.250 (92) | 0.425 (73) |
| rtdetr | small | 0.234 (4021) | 0.344 (733) | 0.337 (353) | 0.514 (138) | 0.547 (117) |
| rtdetr | medium | 0.161 (3730) | 0.210 (785) | 0.285 (520) | 0.364 (214) | 0.393 (117) |
| rtdetr | large | 0.135 (4270) | 0.161 (634) | 0.223 (296) | 0.283 (92) | 0.315 (73) |
| yolo26_ens | small | 0.222 (4021) | 0.337 (733) | 0.323 (353) | 0.464 (138) | 0.419 (117) |
| yolo26_ens | medium | 0.163 (3730) | 0.205 (785) | 0.269 (520) | 0.416 (214) | 0.368 (117) |
| yolo26_ens | large | 0.156 (4270) | 0.178 (634) | 0.264 (296) | 0.272 (92) | 0.356 (73) |
| rtdetr_ens | small | 0.230 (4021) | 0.340 (733) | 0.331 (353) | 0.507 (138) | 0.521 (117) |
| rtdetr_ens | medium | 0.161 (3730) | 0.199 (785) | 0.277 (520) | 0.407 (214) | 0.359 (117) |
| rtdetr_ens | large | 0.132 (4270) | 0.174 (634) | 0.226 (296) | 0.261 (92) | 0.329 (73) |

### Conformal layer

| detector | rule | target | recall of rule alone | recall of system | system recall, most occluded bucket | flags/image | certifiable |
|---|---|---|---|---|---|---|---|
| yolo26 | standard_threshold | n/a | 0.789 [0.781, 0.796] | 0.789 | 0.609 | n/a |  |
| yolo26 | global_confidence | 0.95 | 0.906 [0.900, 0.911] | 0.906 | 0.886 | 33.71 | NO |
| yolo26 | mondrian_confidence | 0.95 | 0.906 [0.900, 0.911] | 0.906 | 0.886 | 33.71 | NO |
| yolo26 | global, duplicate- and context-aware score | 0.95 | 0.906 [0.900, 0.911] | 0.906 | 0.886 | 33.71 | NO |
| yolov8 | standard_threshold | n/a | 0.791 [0.783, 0.797] | 0.791 | 0.596 | n/a |  |
| yolov8 | global_confidence | 0.95 | 0.898 [0.892, 0.903] | 0.898 | 0.853 | 37.79 | NO |
| yolov8 | mondrian_confidence | 0.95 | 0.898 [0.892, 0.903] | 0.898 | 0.853 | 37.79 | NO |
| yolov8 | global, duplicate- and context-aware score | 0.95 | 0.898 [0.892, 0.903] | 0.898 | 0.853 | 37.79 | NO |
| rtdetr | standard_threshold | n/a | 0.796 [0.789, 0.804] | 0.796 | 0.567 | n/a |  |
| rtdetr | global_confidence | 0.95 | 0.947 [0.941, 0.951] | 0.947 | 0.896 | 16.42 | yes |
| rtdetr | mondrian_confidence | 0.95 | 0.947 [0.942, 0.952] | 0.947 | 0.919 | 20.97 | yes |
| rtdetr | global, duplicate- and context-aware score | 0.95 | 0.946 [0.940, 0.950] | 0.946 | 0.873 | 14.47 | yes |
| yolo26_ens | standard_threshold | n/a | 0.794 [0.786, 0.802] | 0.794 | 0.616 | n/a |  |
| yolo26_ens | global_confidence | 0.95 | 0.926 [0.920, 0.931] | 0.926 | 0.912 | 80.98 | NO |
| yolo26_ens | mondrian_confidence | 0.95 | 0.926 [0.920, 0.931] | 0.926 | 0.912 | 80.98 | NO |
| yolo26_ens | global, duplicate- and context-aware score | 0.95 | 0.926 [0.920, 0.931] | 0.926 | 0.912 | 80.98 | NO |
| rtdetr_ens | standard_threshold | n/a | 0.799 [0.791, 0.806] | 0.799 | 0.586 | n/a |  |
| rtdetr_ens | global_confidence | 0.95 | 0.947 [0.941, 0.952] | 0.947 | 0.919 | 23.99 | yes |
| rtdetr_ens | mondrian_confidence | 0.95 | 0.945 [0.939, 0.950] | 0.945 | 0.925 | 29.29 | yes |
| rtdetr_ens | global, duplicate- and context-aware score | 0.95 | 0.946 [0.941, 0.951] | 0.946 | 0.919 | 22.94 | yes |

### RQ2: recovery of missed vehicles at matched extra false alarms

| detector | policy | misses | recoverable ceiling | mean recovery | difference from baseline, all misses | difference from baseline, occluded misses |
|---|---|---|---|---|---|---|
| yolo26 | confidence | 3395 | 0.553 | 0.117 [0.106, 0.128] | baseline | baseline |
| yolo26 | mondrian_rank | 3395 | 0.553 | 0.092 [0.083, 0.101] | -0.0257 [-0.0358, -0.0164] | 0.0847 [0.0700, 0.0997] |
| yolo26 | duplicate- and context-aware score | 3395 | 0.553 | 0.113 [0.102, 0.124] | -0.0045 [-0.0081, -0.0009] | -0.0045 [-0.0096, 0.0006] |
| yolov8 | confidence | 3369 | 0.514 | 0.100 [0.089, 0.109] | baseline | baseline |
| yolov8 | mondrian_rank | 3369 | 0.514 | 0.080 [0.071, 0.088] | -0.0198 [-0.0290, -0.0100] | 0.0816 [0.0666, 0.0976] |
| yolov8 | duplicate- and context-aware score | 3369 | 0.514 | 0.098 [0.088, 0.108] | -0.0013 [-0.0036, 0.0012] | -0.0098 [-0.0132, -0.0061] |
| rtdetr | confidence | 3275 | 0.921 | 0.130 [0.117, 0.142] | baseline | baseline |
| rtdetr | mondrian_rank | 3275 | 0.921 | 0.102 [0.092, 0.112] | -0.0280 [-0.0373, -0.0183] | 0.0856 [0.0724, 0.1020] |
| rtdetr | duplicate- and context-aware score | 3275 | 0.921 | 0.125 [0.112, 0.137] | -0.0046 [-0.0080, -0.0010] | 0.0017 [-0.0039, 0.0068] |
| yolo26_ens | confidence | 3314 | 0.641 | 0.121 [0.110, 0.133] | baseline | baseline |
| yolo26_ens | mondrian_rank | 3314 | 0.641 | 0.091 [0.081, 0.101] | -0.0302 [-0.0402, -0.0205] | 0.0856 [0.0695, 0.1010] |
| yolo26_ens | duplicate- and context-aware score | 3314 | 0.641 | 0.122 [0.110, 0.135] | 0.0010 [-0.0023, 0.0042] | 0.0037 [-0.0003, 0.0072] |
| rtdetr_ens | confidence | 3240 | 0.866 | 0.131 [0.121, 0.143] | baseline | baseline |
| rtdetr_ens | mondrian_rank | 3240 | 0.866 | 0.105 [0.093, 0.116] | -0.0263 [-0.0358, -0.0161] | 0.0835 [0.0704, 0.0970] |
| rtdetr_ens | duplicate- and context-aware score | 3240 | 0.866 | 0.133 [0.123, 0.145] | 0.0024 [0.0011, 0.0034] | 0.0137 [0.0117, 0.0161] |

### RQ3: calibration metrics at the operating threshold

| detector | D-ECE | MA-ECE | miss mass | OCE | threshold D-ECE alone would choose | occluded vehicles: share of objects / share of D-ECE blind spot |
|---|---|---|---|---|---|---|
| yolo26 | 0.042 [0.037, 0.049] | 0.211 [0.205, 0.219] | 0.176 | 0.282 | 0.95 (recall 0.04) | 0.25 / 0.34 |
| yolov8 | 0.037 [0.031, 0.043] | 0.205 [0.198, 0.212] | 0.174 | 0.281 | 0.95 (recall 0.07) | 0.25 / 0.34 |
| rtdetr | 0.069 [0.061, 0.077] | 0.227 [0.219, 0.236] | 0.170 | 0.268 | 0.95 (recall 0.18) | 0.25 / 0.35 |
| yolo26_ens | 0.047 [0.041, 0.053] | 0.215 [0.208, 0.222] | 0.176 | 0.273 | 0.95 (recall 0.00) | 0.25 / 0.35 |
| rtdetr_ens | 0.059 [0.053, 0.065] | 0.221 [0.213, 0.228] | 0.172 | 0.260 | 0.9 (recall 0.44) | 0.25 / 0.36 |

### RQ5: why vehicles are missed (native classes, whatever the mode)

| detector | misses | confused | suppressed | unproposed | merged | visibility misses: odds per +0.1 occlusion | confusion misses: odds per +1 SD rarity | confusion misses: odds per +0.1 occlusion |
|---|---|---|---|---|---|---|---|---|
| yolo26 | 3395 | 0.34 | 0.49 | 0.09 | 0.00 | 1.190 [1.165, 1.220] | 2.936 [2.795, 3.081] | 1.007 [0.970, 1.035] |
| yolov8 | 3369 | 0.34 | 0.47 | 0.09 | 0.00 | 1.190 [1.166, 1.220] | 2.914 [2.770, 3.045] | 0.996 [0.962, 1.027] |
| rtdetr | 3275 | 0.31 | 0.62 | 0.07 | 0.00 | 1.207 [1.186, 1.233] | 2.890 [2.719, 3.054] | 1.027 [0.986, 1.059] |
| yolo26_ens | 3314 | 0.34 | 0.53 | 0.05 | 0.00 | 1.191 [1.168, 1.222] | 2.944 [2.813, 3.083] | 1.030 [0.999, 1.058] |
| rtdetr_ens | 3240 | 0.33 | 0.60 | 0.05 | 0.00 | 1.220 [1.196, 1.247] | 2.846 [2.704, 2.998] | 1.012 [0.976, 1.046] |

### RQ4: transfer with the detector frozen (recall-side quantities only)

Each coverage is followed by (flags per image; certifiable). A recalibrated coverage that is not certifiable is the best the rule could do, not a guarantee.

| detector | dataset | matching | recall, development -> transfer | rule | target | coverage on development | coverage on transfer, frozen | coverage on transfer, recalibrated |
|---|---|---|---|---|---|---|---|---|
| yolo26 | pals | coarse | 0.840 -> 0.592 | global_confidence | 0.95 | 0.957 (33.26; NO) | 0.802 (38.75; NO) | 0.802 (38.75; NO) |
| yolo26 | pals | coarse | 0.840 -> 0.592 | mondrian_confidence | 0.95 | 0.957 (33.26; NO) | 0.802 (38.75; NO) | 0.802 (38.75; NO) |
| yolo26 | pals | coarse | 0.840 -> 0.592 | global_occlusion_aware | 0.95 | 0.957 (33.26; NO) | 0.802 (38.75; NO) | 0.802 (38.75; NO) |
| yolo26 | pals | coarse | 0.840 -> 0.592 | global_confidence | 0.90 | 0.946 (12.22; yes) | 0.773 (6.88; yes) | 0.802 (38.75; NO) |
| yolo26 | pals | coarse | 0.840 -> 0.592 | mondrian_confidence | 0.90 | 0.945 (17.06; NO) | 0.756 (7.94; NO) | 0.802 (38.75; NO) |
| yolo26 | pals | coarse | 0.840 -> 0.592 | global_occlusion_aware | 0.90 | 0.947 (12.27; yes) | 0.773 (6.85; yes) | 0.802 (38.75; NO) |
| yolov8 | pals | coarse | 0.843 -> 0.611 | global_confidence | 0.95 | 0.954 (37.23; NO) | 0.793 (26.04; NO) | 0.793 (26.04; NO) |
| yolov8 | pals | coarse | 0.843 -> 0.611 | mondrian_confidence | 0.95 | 0.954 (37.23; NO) | 0.793 (26.04; NO) | 0.793 (26.04; NO) |
| yolov8 | pals | coarse | 0.843 -> 0.611 | global_occlusion_aware | 0.95 | 0.954 (37.23; NO) | 0.793 (26.04; NO) | 0.793 (26.04; NO) |
| yolov8 | pals | coarse | 0.843 -> 0.611 | global_confidence | 0.90 | 0.952 (29.57; yes) | 0.789 (18.84; yes) | 0.793 (26.04; NO) |
| yolov8 | pals | coarse | 0.843 -> 0.611 | mondrian_confidence | 0.90 | 0.946 (19.88; NO) | 0.762 (9.99; NO) | 0.793 (26.04; NO) |
| yolov8 | pals | coarse | 0.843 -> 0.611 | global_occlusion_aware | 0.90 | 0.951 (25.65; yes) | 0.784 (14.75; yes) | 0.793 (26.04; NO) |
| rtdetr | pals | coarse | 0.842 -> 0.497 | global_confidence | 0.95 | 0.959 (16.07; yes) | 0.773 (12.88; yes) | 0.958 (294.59; NO) |
| rtdetr | pals | coarse | 0.842 -> 0.497 | mondrian_confidence | 0.95 | 0.960 (20.61; yes) | 0.770 (14.20; yes) | 0.958 (294.59; NO) |
| rtdetr | pals | coarse | 0.842 -> 0.497 | global_occlusion_aware | 0.95 | 0.956 (14.17; yes) | 0.761 (11.33; yes) | 0.958 (294.59; NO) |
| rtdetr | pals | coarse | 0.842 -> 0.497 | global_confidence | 0.90 | 0.929 (5.22; yes) | 0.665 (3.92; yes) | 0.924 (98.78; yes) |
| rtdetr | pals | coarse | 0.842 -> 0.497 | mondrian_confidence | 0.90 | 0.931 (5.94; yes) | 0.655 (3.82; yes) | 0.924 (98.78; yes) |
| rtdetr | pals | coarse | 0.842 -> 0.497 | global_occlusion_aware | 0.90 | 0.927 (5.11; yes) | 0.666 (3.81; yes) | 0.922 (90.00; yes) |
| yolo26_ens | pals | coarse | 0.845 -> 0.651 | global_confidence | 0.95 | 0.969 (79.58; NO) | 0.876 (58.83; NO) | 0.876 (58.83; NO) |
| yolo26_ens | pals | coarse | 0.845 -> 0.651 | mondrian_confidence | 0.95 | 0.969 (79.58; NO) | 0.876 (58.83; NO) | 0.876 (58.83; NO) |
| yolo26_ens | pals | coarse | 0.845 -> 0.651 | global_occlusion_aware | 0.95 | 0.969 (79.58; NO) | 0.876 (58.83; NO) | 0.876 (58.83; NO) |
| yolo26_ens | pals | coarse | 0.845 -> 0.651 | global_confidence | 0.90 | 0.941 (6.60; yes) | 0.794 (4.18; yes) | 0.876 (58.83; NO) |
| yolo26_ens | pals | coarse | 0.845 -> 0.651 | mondrian_confidence | 0.90 | 0.940 (8.73; yes) | 0.771 (5.08; yes) | 0.876 (58.83; NO) |
| yolo26_ens | pals | coarse | 0.845 -> 0.651 | global_occlusion_aware | 0.90 | 0.941 (6.67; yes) | 0.800 (4.28; yes) | 0.876 (58.83; NO) |
| rtdetr_ens | pals | coarse | 0.847 -> 0.542 | global_confidence | 0.95 | 0.966 (23.47; yes) | 0.865 (19.16; yes) | 0.971 (503.89; NO) |
| rtdetr_ens | pals | coarse | 0.847 -> 0.542 | mondrian_confidence | 0.95 | 0.965 (28.79; yes) | 0.861 (19.71; yes) | 0.971 (503.89; NO) |
| rtdetr_ens | pals | coarse | 0.847 -> 0.542 | global_occlusion_aware | 0.95 | 0.965 (22.43; yes) | 0.865 (18.25; yes) | 0.971 (503.89; NO) |
| rtdetr_ens | pals | coarse | 0.847 -> 0.542 | global_confidence | 0.90 | 0.935 (5.61; yes) | 0.738 (4.12; yes) | 0.949 (135.69; yes) |
| rtdetr_ens | pals | coarse | 0.847 -> 0.542 | mondrian_confidence | 0.90 | 0.936 (6.27; yes) | 0.725 (4.07; yes) | 0.949 (135.69; yes) |
| rtdetr_ens | pals | coarse | 0.847 -> 0.542 | global_occlusion_aware | 0.90 | 0.933 (5.37; yes) | 0.734 (3.93; yes) | 0.948 (128.85; yes) |

### Detector ranking (best first)

| comparison | metric | ranking |
|---|---|---|
| each at its own threshold | dece | yolov8 < yolo26 < yolo26_ens < rtdetr_ens < rtdetr |
| each at its own threshold | ma_ece | yolov8 < yolo26 < yolo26_ens < rtdetr_ens < rtdetr |
| each at its own threshold | ma_brier | yolo26_ens < rtdetr_ens < yolov8 < yolo26 < rtdetr |
| each at its own threshold | oce_max | rtdetr_ens < rtdetr < yolo26_ens < yolov8 < yolo26 |
| at matched false alarms per image | dece | yolov8 < yolo26 < yolo26_ens < rtdetr_ens < rtdetr |
| at matched false alarms per image | ma_ece | yolo26_ens < yolov8 < rtdetr_ens < yolo26 < rtdetr |
| at matched false alarms per image | ma_brier | yolo26_ens < rtdetr_ens < yolov8 < yolo26 < rtdetr |
| at matched false alarms per image | oce_max | rtdetr_ens < yolo26_ens < rtdetr < yolov8 < yolo26 |

Point-estimate orders above can differ by noise. Below, a metric *separates* a pair only if its 95% paired-difference interval excludes zero; a *flip* needs both D-ECE and the other metric to separate the pair, in opposite directions.

**each at its own threshold**

| pair (difference = first minus second) | D-ECE | MA-ECE | MA-Brier | OCE | flips vs D-ECE |
|---|---|---|---|---|---|
| rtdetr vs rtdetr_ens | rtdetr_ens better [0.0032, 0.0173] | rtdetr_ens better [0.0004, 0.0134] | rtdetr_ens better [0.0061, 0.0128] | rtdetr_ens better [0.0036, 0.0131] | none |
| rtdetr vs yolo26 | yolo26 better [0.0157, 0.0366] | yolo26 better [0.0060, 0.0244] | cannot separate [-0.0032, 0.0060] | rtdetr better [-0.0214, -0.0085] | oce_max |
| rtdetr vs yolo26_ens | yolo26_ens better [0.0109, 0.0329] | yolo26_ens better [0.0024, 0.0229] | yolo26_ens better [0.0094, 0.0181] | cannot separate [-0.0117, 0.0002] | none |
| rtdetr vs yolov8 | yolov8 better [0.0216, 0.0422] | yolov8 better [0.0131, 0.0322] | cannot separate [-0.0019, 0.0072] | rtdetr better [-0.0199, -0.0074] | oce_max |
| rtdetr_ens vs yolo26 | yolo26 better [0.0075, 0.0224] | yolo26 better [0.0016, 0.0165] | rtdetr_ens better [-0.0127, -0.0036] | rtdetr_ens better [-0.0291, -0.0170] | ma_brier, oce_max |
| rtdetr_ens vs yolo26_ens | yolo26_ens better [0.0027, 0.0189] | cannot separate [-0.0030, 0.0133] | yolo26_ens better [0.0002, 0.0084] | rtdetr_ens better [-0.0192, -0.0085] | oce_max |
| rtdetr_ens vs yolov8 | yolov8 better [0.0143, 0.0284] | yolov8 better [0.0083, 0.0230] | rtdetr_ens better [-0.0113, -0.0014] | rtdetr_ens better [-0.0277, -0.0154] | ma_brier, oce_max |
| yolo26 vs yolo26_ens | cannot separate [-0.0096, 0.0011] | cannot separate [-0.0095, 0.0022] | yolo26_ens better [0.0091, 0.0151] | yolo26_ens better [0.0049, 0.0126] | none |
| yolo26 vs yolov8 | yolov8 better [0.0000, 0.0116] | cannot separate [-0.0005, 0.0137] | cannot separate [-0.0032, 0.0058] | cannot separate [-0.0049, 0.0075] | none |
| yolo26_ens vs yolov8 | yolov8 better [0.0044, 0.0174] | yolov8 better [0.0042, 0.0172] | yolo26_ens better [-0.0148, -0.0066] | yolo26_ens better [-0.0130, -0.0019] | ma_brier, oce_max |

- D-ECE cannot separate yolo26 and yolo26_ens (each at its own threshold).

**at matched false alarms per image** (budget 2.05 false alarms/image)

| pair (difference = first minus second) | D-ECE | MA-ECE | MA-Brier | OCE | flips vs D-ECE |
|---|---|---|---|---|---|
| rtdetr vs rtdetr_ens | cannot separate [-0.0047, 0.0094] | rtdetr_ens better [0.0094, 0.0226] | rtdetr_ens better [0.0152, 0.0219] | rtdetr_ens better [0.0152, 0.0248] | none |
| rtdetr vs yolo26 | yolo26 better [0.0159, 0.0367] | yolo26 better [0.0059, 0.0242] | cannot separate [-0.0034, 0.0058] | rtdetr better [-0.0216, -0.0087] | oce_max |
| rtdetr vs yolo26_ens | yolo26_ens better [0.0111, 0.0320] | yolo26_ens better [0.0192, 0.0383] | yolo26_ens better [0.0206, 0.0289] | cannot separate [-0.0019, 0.0097] | none |
| rtdetr vs yolov8 | yolov8 better [0.0224, 0.0427] | yolov8 better [0.0112, 0.0301] | cannot separate [-0.0036, 0.0055] | rtdetr better [-0.0211, -0.0085] | oce_max |
| rtdetr_ens vs yolo26 | yolo26 better [0.0160, 0.0311] | cannot separate [-0.0080, 0.0062] | rtdetr_ens better [-0.0220, -0.0127] | rtdetr_ens better [-0.0414, -0.0290] | ma_brier, oce_max |
| rtdetr_ens vs yolo26_ens | yolo26_ens better [0.0105, 0.0269] | yolo26_ens better [0.0041, 0.0199] | yolo26_ens better [0.0021, 0.0104] | rtdetr_ens better [-0.0210, -0.0107] | oce_max |
| rtdetr_ens vs yolov8 | yolov8 better [0.0230, 0.0374] | cannot separate [-0.0033, 0.0117] | rtdetr_ens better [-0.0224, -0.0123] | rtdetr_ens better [-0.0404, -0.0282] | ma_brier, oce_max |
| yolo26 vs yolo26_ens | cannot separate [-0.0103, 0.0003] | yolo26_ens better [0.0073, 0.0191] | yolo26_ens better [0.0208, 0.0263] | yolo26_ens better [0.0150, 0.0227] | none |
| yolo26 vs yolov8 | yolov8 better [0.0008, 0.0125] | cannot separate [-0.0022, 0.0116] | cannot separate [-0.0046, 0.0047] | cannot separate [-0.0055, 0.0068] | none |
| yolo26_ens vs yolov8 | yolov8 better [0.0055, 0.0181] | yolo26_ens better [-0.0145, -0.0016] | yolo26_ens better [-0.0278, -0.0196] | yolo26_ens better [-0.0236, -0.0129] | ma_ece, ma_brier, oce_max |

- D-ECE cannot separate rtdetr and rtdetr_ens (at matched false alarms per image).
- D-ECE cannot separate yolo26 and yolo26_ens (at matched false alarms per image).

Ranking flip relative to D-ECE (both intervals exclude zero, opposite signs) survives matched cost: **True**

#### MA-Brier decomposition

Each term is its contribution to MA-Brier (share of MA-Brier in brackets); the three sum to MA-Brier. OCE averages over vehicles only, so it has no false-alarm term.

| detector | where | threshold | false alarms/image | MA-Brier | false-alarm term | matched-detection term | miss term | OCE |
|---|---|---|---|---|---|---|---|---|
| yolo26 | own threshold | 0.49 | 2.06 | 0.2854 | 0.0882 (0.31) | 0.0210 (0.07) | 0.1762 (0.62) | 0.2823 |
| yolov8 | own threshold | 0.47 | 2.10 | 0.2840 | 0.0873 (0.31) | 0.0224 (0.08) | 0.1743 (0.61) | 0.2809 |
| rtdetr | own threshold | 0.65 | 2.05 | 0.2870 | 0.1051 (0.37) | 0.0119 (0.04) | 0.1701 (0.59) | 0.2678 |
| yolo26_ens | own threshold | 0.47 | 1.77 | 0.2734 | 0.0693 (0.25) | 0.0281 (0.10) | 0.1761 (0.64) | 0.2734 |
| rtdetr_ens | own threshold | 0.66 | 1.78 | 0.2777 | 0.0934 (0.34) | 0.0123 (0.04) | 0.1720 (0.62) | 0.2596 |
| yolo26 | matched cost | 0.49 | 2.05 | 0.2857 | 0.0881 (0.31) | 0.0209 (0.07) | 0.1766 (0.62) | 0.2825 |
| yolov8 | matched cost | 0.48 | 2.05 | 0.2857 | 0.0867 (0.30) | 0.0219 (0.08) | 0.1770 (0.62) | 0.2820 |
| rtdetr | matched cost | 0.65 | 2.05 | 0.2870 | 0.1051 (0.37) | 0.0119 (0.04) | 0.1701 (0.59) | 0.2678 |
| yolo26_ens | matched cost | 0.41 | 2.05 | 0.2621 | 0.0720 (0.27) | 0.0317 (0.12) | 0.1583 (0.60) | 0.2635 |
| rtdetr_ens | matched cost | 0.62 | 2.05 | 0.2686 | 0.1003 (0.37) | 0.0138 (0.05) | 0.1545 (0.58) | 0.2480 |

## Training-seed variability (matching: class_agnostic)

Mean, SD across seeds (n-1 denominator) and [min, max]. With three seeds the SD is itself rough: read it as the size of run-to-run noise.

| quantity | yolo26 (yolo26, yolo26_s1, yolo26_s2) | rtdetr (rtdetr, rtdetr_s1, rtdetr_s2) |
|---|---|---|
| operating threshold | 0.44 ± 0.02 [0.42, 0.46] (n=3) | 0.67 ± 0.02 [0.65, 0.69] (n=3) |
| AP50 (class-agnostic) | 0.9157 ± 0.0006 [0.9150, 0.9162] (n=3) | 0.9096 ± 0.0018 [0.9082, 0.9116] (n=3) |
| recall at threshold | 0.87 ± 0.00 [0.86, 0.87] (n=3) | 0.85 ± 0.00 [0.85, 0.86] (n=3) |
| false alarms/image at threshold | 1.44 ± 0.09 [1.34, 1.52] (n=3) | 1.31 ± 0.08 [1.24, 1.40] (n=3) |
| RQ1 odds of a miss per +0.1 occlusion | 1.1881 ± 0.0046 [1.1839, 1.1929] (n=3) | 1.2153 ± 0.0066 [1.2078, 1.2192] (n=3) |
| RQ1 partial Spearman | 0.2063 ± 0.0018 [0.2042, 0.2078] (n=3) | 0.2536 ± 0.0103 [0.2442, 0.2647] (n=3) |
| conformal global_confidence, target 0.95: coverage | 0.9488 ± 0.0019 [0.9467, 0.9505] (n=3) | 0.9474 ± 0.0002 [0.9473, 0.9477] (n=3) |
| conformal global_confidence, target 0.95: flags/image | 4.28 ± 0.34 [3.90, 4.56] (n=3) | 5.69 ± 0.52 [5.19, 6.24] (n=3) |
| conformal mondrian_confidence, target 0.95: coverage | 0.9491 ± 0.0021 [0.9468, 0.9508] (n=3) | 0.9488 ± 0.0014 [0.9476, 0.9503] (n=3) |
| conformal mondrian_confidence, target 0.95: flags/image | 5.79 ± 0.76 [4.91, 6.28] (n=3) | 7.70 ± 1.04 [6.74, 8.80] (n=3) |
| conformal global_occlusion_aware, target 0.95: coverage | 0.9486 ± 0.0020 [0.9466, 0.9505] (n=3) | 0.9475 ± 0.0007 [0.9471, 0.9483] (n=3) |
| conformal global_occlusion_aware, target 0.95: flags/image | 4.02 ± 0.38 [3.60, 4.32] (n=3) | 4.60 ± 0.27 [4.32, 4.86] (n=3) |
| conformal global_confidence, target 0.90: coverage | 0.9018 ± 0.0016 [0.9006, 0.9036] (n=3) | 0.9008 ± 0.0006 [0.9003, 0.9015] (n=3) |
| conformal global_confidence, target 0.90: flags/image | 1.07 ± 0.14 [0.99, 1.23] (n=3) | 1.44 ± 0.15 [1.31, 1.61] (n=3) |
| conformal mondrian_confidence, target 0.90: coverage | 0.9020 ± 0.0017 [0.9005, 0.9039] (n=3) | 0.9017 ± 0.0016 [0.9004, 0.9035] (n=3) |
| conformal mondrian_confidence, target 0.90: flags/image | 1.38 ± 0.14 [1.30, 1.54] (n=3) | 1.73 ± 0.10 [1.65, 1.84] (n=3) |
| conformal global_occlusion_aware, target 0.90: coverage | 0.9024 ± 0.0010 [0.9016, 0.9035] (n=3) | 0.9015 ± 0.0005 [0.9010, 0.9019] (n=3) |
| conformal global_occlusion_aware, target 0.90: flags/image | 1.02 ± 0.05 [0.97, 1.06] (n=3) | 1.28 ± 0.14 [1.12, 1.39] (n=3) |
| RQ2 mondrian_rank vs confidence, all misses | -0.0513 ± 0.0094 [-0.0621, -0.0454] (n=3) | -0.0361 ± 0.0082 [-0.0447, -0.0283] (n=3) |
| RQ2 mondrian_rank vs confidence, occluded misses | 0.1106 ± 0.0039 [0.1077, 0.1151] (n=3) | 0.1091 ± 0.0112 [0.0982, 0.1206] (n=3) |
| RQ2 default score vs confidence, all misses | 0.0474 ± 0.0024 [0.0446, 0.0491] (n=3) | 0.0553 ± 0.0072 [0.0470, 0.0604] (n=3) |
| RQ2 default score vs confidence, occluded misses | 0.0213 ± 0.0048 [0.0183, 0.0268] (n=3) | 0.0439 ± 0.0152 [0.0338, 0.0615] (n=3) |
| D-ECE | 0.0662 ± 0.0145 [0.0497, 0.0768] (n=3) | 0.0449 ± 0.0038 [0.0413, 0.0489] (n=3) |
| MA-ECE | 0.1757 ± 0.0155 [0.1589, 0.1894] (n=3) | 0.1686 ± 0.0027 [0.1669, 0.1717] (n=3) |
| MA-Brier | 0.2048 ± 0.0036 [0.2010, 0.2080] (n=3) | 0.2125 ± 0.0029 [0.2103, 0.2158] (n=3) |
| OCE | 0.1716 ± 0.0090 [0.1613, 0.1781] (n=3) | 0.1588 ± 0.0038 [0.1551, 0.1628] (n=3) |
