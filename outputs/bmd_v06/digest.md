# Results digest: experiment `bmd_v06`

Intervals are 95% image-level bootstrap intervals. Read the gates first.

## Matching: class_agnostic  (primary)

### Gates

- yolo26 / detector_usable_on_development: pass
- yolo26 / enough_occlusion: pass
- yolo26 / detector_usable_on_pals / class_agnostic: pass
- rtdetr / detector_usable_on_development: pass
- rtdetr / enough_occlusion: pass
- rtdetr / detector_usable_on_pals / class_agnostic: pass
- yolov8 / detector_usable_on_development: pass
- yolov8 / enough_occlusion: pass
- yolov8 / detector_usable_on_pals / class_agnostic: pass
- yolo26_s1 / detector_usable_on_development: pass
- yolo26_s1 / enough_occlusion: pass
- yolo26_s1 / detector_usable_on_pals / class_agnostic: pass
- yolo26_s2 / detector_usable_on_development: pass
- yolo26_s2 / enough_occlusion: pass
- yolo26_s2 / detector_usable_on_pals / class_agnostic: pass
- rtdetr_s1 / detector_usable_on_development: pass
- rtdetr_s1 / enough_occlusion: pass
- rtdetr_s1 / detector_usable_on_pals / class_agnostic: pass
- rtdetr_s2 / detector_usable_on_development: pass
- rtdetr_s2 / enough_occlusion: pass
- rtdetr_s2 / detector_usable_on_pals / class_agnostic: pass
- yolo26_evid / detector_usable_on_development: pass
- yolo26_evid / enough_occlusion: pass
- yolo26_evid / detector_usable_on_pals / class_agnostic: pass
- yolo26_evid_s1 / detector_usable_on_development: pass
- yolo26_evid_s1 / enough_occlusion: pass
- yolo26_evid_s1 / detector_usable_on_pals / class_agnostic: pass
- yolo26_evid_s2 / detector_usable_on_development: pass
- yolo26_evid_s2 / enough_occlusion: pass
- yolo26_evid_s2 / detector_usable_on_pals / class_agnostic: pass
- yolo26_ens / detector_usable_on_development: pass
- yolo26_ens / enough_occlusion: pass
- yolo26_ens / detector_usable_on_pals / class_agnostic: pass
- rtdetr_ens / detector_usable_on_development: pass
- rtdetr_ens / enough_occlusion: pass
- rtdetr_ens / detector_usable_on_pals / class_agnostic: pass

### Detection at each detector's operating threshold

| detector | threshold | AP50 | recall | precision | false alarms/image | recall at floor |
|---|---|---|---|---|---|---|
| yolo26 | 0.46 | 0.936 | 0.889 | 0.881 | 1.26 | 0.989 |
| rtdetr | 0.69 | 0.924 | 0.880 | 0.872 | 1.35 | 0.990 |
| yolov8 | 0.47 | 0.933 | 0.888 | 0.875 | 1.32 | 0.989 |
| yolo26_s1 | 0.46 | 0.935 | 0.889 | 0.882 | 1.24 | 0.989 |
| yolo26_s2 | 0.46 | 0.936 | 0.885 | 0.888 | 1.16 | 0.989 |
| rtdetr_s1 | 0.70 | 0.924 | 0.878 | 0.878 | 1.27 | 0.989 |
| rtdetr_s2 | 0.70 | 0.925 | 0.876 | 0.880 | 1.24 | 0.990 |
| yolo26_evid | 0.45 | 0.931 | 0.883 | 0.882 | 1.23 | 0.990 |
| yolo26_evid_s1 | 0.43 | 0.933 | 0.883 | 0.886 | 1.19 | 0.990 |
| yolo26_evid_s2 | 0.45 | 0.930 | 0.880 | 0.879 | 1.26 | 0.990 |
| yolo26_ens | 0.44 | 0.941 | 0.896 | 0.886 | 1.21 | 0.991 |
| rtdetr_ens | 0.68 | 0.928 | 0.889 | 0.879 | 1.28 | 0.991 |

### RQ1: miss rate by annotated occlusion severity

| detector | 0.0-0.2 (n=12021) | 0.2-0.4 (n=2152) | 0.4-0.6 (n=1169) | 0.6-0.8 (n=444) | 0.8-1.0 (n=307) | odds of a miss per +0.1 (size, class controlled) | partial Spearman | context vs annotated occlusion |
|---|---|---|---|---|---|---|---|---|
| yolo26 | 0.092 | 0.126 | 0.172 | 0.268 | 0.277 | 1.186 [1.161, 1.213] | 0.238 [0.218, 0.254] | 0.90 |
| rtdetr | 0.098 | 0.145 | 0.185 | 0.288 | 0.303 | 1.198 [1.174, 1.230] | 0.251 [0.230, 0.266] | 0.90 |
| yolov8 | 0.091 | 0.134 | 0.178 | 0.279 | 0.261 | 1.190 [1.166, 1.216] | 0.228 [0.208, 0.245] | 0.90 |
| yolo26_s1 | 0.092 | 0.126 | 0.179 | 0.273 | 0.287 | 1.196 [1.172, 1.225] | 0.238 [0.218, 0.255] | 0.90 |
| yolo26_s2 | 0.096 | 0.126 | 0.177 | 0.286 | 0.293 | 1.190 [1.166, 1.222] | 0.240 [0.221, 0.257] | 0.90 |
| rtdetr_s1 | 0.100 | 0.140 | 0.206 | 0.293 | 0.313 | 1.208 [1.183, 1.236] | 0.248 [0.229, 0.263] | 0.90 |
| rtdetr_s2 | 0.102 | 0.145 | 0.198 | 0.297 | 0.329 | 1.204 [1.178, 1.229] | 0.249 [0.229, 0.265] | 0.90 |
| yolo26_evid | 0.098 | 0.134 | 0.183 | 0.286 | 0.283 | 1.190 [1.165, 1.219] | 0.232 [0.212, 0.249] | 0.90 |
| yolo26_evid_s1 | 0.098 | 0.136 | 0.175 | 0.284 | 0.277 | 1.183 [1.160, 1.210] | 0.231 [0.211, 0.248] | 0.90 |
| yolo26_evid_s2 | 0.098 | 0.135 | 0.191 | 0.297 | 0.313 | 1.202 [1.177, 1.231] | 0.238 [0.218, 0.255] | 0.90 |
| yolo26_ens | 0.085 | 0.116 | 0.167 | 0.273 | 0.287 | 1.203 [1.174, 1.232] | 0.240 [0.220, 0.257] | 0.90 |
| rtdetr_ens | 0.091 | 0.127 | 0.175 | 0.277 | 0.300 | 1.202 [1.178, 1.235] | 0.251 [0.230, 0.266] | 0.90 |

`*` marks a bucket with fewer objects than `min_bucket_n`; do not read it.

#### RQ1 by object size (tertiles of box area relative to the image)

| detector | tertile edges (relative area) | small | medium | large |
|---|---|---|---|---|
| yolo26 | 4.14e-03 / 1.41e-02 | 0.174 [0.163, 0.186] (n=5362) | 0.102 [0.093, 0.113] (n=5366) | 0.055 [0.048, 0.064] (n=5365) |
| rtdetr | 4.14e-03 / 1.41e-02 | 0.197 [0.184, 0.209] (n=5362) | 0.112 [0.102, 0.123] (n=5366) | 0.052 [0.045, 0.058] (n=5365) |
| yolov8 | 4.14e-03 / 1.41e-02 | 0.178 [0.168, 0.191] (n=5362) | 0.103 [0.093, 0.113] (n=5366) | 0.054 [0.047, 0.062] (n=5365) |
| yolo26_s1 | 4.14e-03 / 1.41e-02 | 0.180 [0.169, 0.192] (n=5362) | 0.102 [0.092, 0.113] (n=5366) | 0.052 [0.045, 0.060] (n=5365) |
| yolo26_s2 | 4.14e-03 / 1.41e-02 | 0.184 [0.172, 0.197] (n=5362) | 0.104 [0.094, 0.115] (n=5366) | 0.057 [0.049, 0.066] (n=5365) |
| rtdetr_s1 | 4.14e-03 / 1.41e-02 | 0.201 [0.189, 0.214] (n=5362) | 0.113 [0.103, 0.124] (n=5366) | 0.053 [0.046, 0.060] (n=5365) |
| rtdetr_s2 | 4.14e-03 / 1.41e-02 | 0.203 [0.191, 0.216] (n=5362) | 0.115 [0.105, 0.127] (n=5366) | 0.055 [0.048, 0.063] (n=5365) |
| yolo26_evid | 4.14e-03 / 1.41e-02 | 0.194 [0.182, 0.206] (n=5362) | 0.106 [0.097, 0.118] (n=5366) | 0.052 [0.045, 0.059] (n=5365) |
| yolo26_evid_s1 | 4.14e-03 / 1.41e-02 | 0.193 [0.181, 0.206] (n=5362) | 0.102 [0.093, 0.113] (n=5366) | 0.055 [0.048, 0.063] (n=5365) |
| yolo26_evid_s2 | 4.14e-03 / 1.41e-02 | 0.192 [0.181, 0.206] (n=5362) | 0.111 [0.100, 0.121] (n=5366) | 0.056 [0.049, 0.063] (n=5365) |
| yolo26_ens | 4.14e-03 / 1.41e-02 | 0.169 [0.157, 0.181] (n=5362) | 0.095 [0.086, 0.105] (n=5366) | 0.049 [0.042, 0.058] (n=5365) |
| rtdetr_ens | 4.14e-03 / 1.41e-02 | 0.187 [0.176, 0.200] (n=5362) | 0.100 [0.091, 0.111] (n=5366) | 0.047 [0.040, 0.054] (n=5365) |

Miss rate by occlusion bucket within each size tertile (n in brackets; `*` = fewer than `min_bucket_n`):

| detector | size | 0.0-0.2 | 0.2-0.4 | 0.4-0.6 | 0.6-0.8 | 0.8-1.0 |
|---|---|---|---|---|---|---|
| yolo26 | small | 0.150 (4021) | 0.216 (733) | 0.244 (353) | 0.355 (138) | 0.316 (117) |
| yolo26 | medium | 0.078 (3730) | 0.104 (785) | 0.171 (520) | 0.271 (214) | 0.248 (117) |
| yolo26 | large | 0.049 (4270) | 0.049 (634) | 0.088 (296) | 0.130 (92) | 0.260 (73) |
| rtdetr | small | 0.169 (4021) | 0.247 (733) | 0.266 (353) | 0.391 (138) | 0.402 (117) |
| rtdetr | medium | 0.087 (3730) | 0.121 (785) | 0.175 (520) | 0.276 (214) | 0.256 (117) |
| rtdetr | large | 0.042 (4270) | 0.055 (634) | 0.105 (296) | 0.163 (92) | 0.219 (73) |
| yolov8 | small | 0.151 (4021) | 0.233 (733) | 0.244 (353) | 0.391 (138) | 0.308 (117) |
| yolov8 | medium | 0.080 (3730) | 0.106 (785) | 0.177 (520) | 0.248 (214) | 0.239 (117) |
| yolov8 | large | 0.045 (4270) | 0.054 (634) | 0.101 (296) | 0.185 (92) | 0.219 (73) |
| yolo26_s1 | small | 0.156 (4021) | 0.218 (733) | 0.244 (353) | 0.377 (138) | 0.350 (117) |
| yolo26_s1 | medium | 0.077 (3730) | 0.102 (785) | 0.179 (520) | 0.266 (214) | 0.256 (117) |
| yolo26_s1 | large | 0.044 (4270) | 0.050 (634) | 0.101 (296) | 0.130 (92) | 0.233 (73) |
| yolo26_s2 | small | 0.161 (4021) | 0.213 (733) | 0.244 (353) | 0.391 (138) | 0.376 (117) |
| yolo26_s2 | medium | 0.081 (3730) | 0.097 (785) | 0.181 (520) | 0.276 (214) | 0.222 (117) |
| yolo26_s2 | large | 0.049 (4270) | 0.062 (634) | 0.091 (296) | 0.152 (92) | 0.274 (73) |
| rtdetr_s1 | small | 0.172 (4021) | 0.246 (733) | 0.292 (353) | 0.399 (138) | 0.427 (117) |
| rtdetr_s1 | medium | 0.087 (3730) | 0.112 (785) | 0.196 (520) | 0.285 (214) | 0.256 (117) |
| rtdetr_s1 | large | 0.043 (4270) | 0.052 (634) | 0.122 (296) | 0.152 (92) | 0.219 (73) |
| rtdetr_s2 | small | 0.174 (4021) | 0.248 (733) | 0.292 (353) | 0.406 (138) | 0.402 (117) |
| rtdetr_s2 | medium | 0.089 (3730) | 0.125 (785) | 0.187 (520) | 0.262 (214) | 0.299 (117) |
| rtdetr_s2 | large | 0.045 (4270) | 0.052 (634) | 0.105 (296) | 0.217 (92) | 0.260 (73) |
| yolo26_evid | small | 0.168 (4021) | 0.247 (733) | 0.252 (353) | 0.406 (138) | 0.333 (117) |
| yolo26_evid | medium | 0.082 (3730) | 0.103 (785) | 0.190 (520) | 0.266 (214) | 0.256 (117) |
| yolo26_evid | large | 0.046 (4270) | 0.043 (634) | 0.088 (296) | 0.152 (92) | 0.247 (73) |
| yolo26_evid_s1 | small | 0.167 (4021) | 0.248 (733) | 0.252 (353) | 0.399 (138) | 0.342 (117) |
| yolo26_evid_s1 | medium | 0.081 (3730) | 0.097 (785) | 0.169 (520) | 0.266 (214) | 0.222 (117) |
| yolo26_evid_s1 | large | 0.047 (4270) | 0.055 (634) | 0.091 (296) | 0.152 (92) | 0.260 (73) |
| yolo26_evid_s2 | small | 0.165 (4021) | 0.241 (733) | 0.258 (353) | 0.399 (138) | 0.385 (117) |
| yolo26_evid_s2 | medium | 0.085 (3730) | 0.106 (785) | 0.194 (520) | 0.280 (214) | 0.265 (117) |
| yolo26_evid_s2 | large | 0.047 (4270) | 0.049 (634) | 0.105 (296) | 0.185 (92) | 0.274 (73) |
| yolo26_ens | small | 0.146 (4021) | 0.201 (733) | 0.235 (353) | 0.355 (138) | 0.359 (117) |
| yolo26_ens | medium | 0.071 (3730) | 0.092 (785) | 0.165 (520) | 0.280 (214) | 0.231 (117) |
| yolo26_ens | large | 0.041 (4270) | 0.049 (634) | 0.088 (296) | 0.130 (92) | 0.260 (73) |
| rtdetr_ens | small | 0.159 (4021) | 0.232 (733) | 0.266 (353) | 0.377 (138) | 0.393 (117) |
| rtdetr_ens | medium | 0.079 (3730) | 0.097 (785) | 0.160 (520) | 0.262 (214) | 0.248 (117) |
| rtdetr_ens | large | 0.038 (4270) | 0.044 (634) | 0.095 (296) | 0.163 (92) | 0.233 (73) |

### Conformal layer

| detector | rule | target | recall of rule alone | recall of system | system recall, most occluded bucket | flags/image | certifiable |
|---|---|---|---|---|---|---|---|
| yolo26 | standard_threshold | n/a | 0.889 [0.882, 0.896] | 0.889 | 0.723 | n/a |  |
| yolo26 | global_confidence | 0.90 | 0.902 [0.895, 0.908] | 0.902 | 0.739 | 0.32 | yes |
| yolo26 | mondrian_confidence | 0.90 | 0.903 [0.896, 0.909] | 0.906 | 0.798 | 0.53 | yes |
| yolo26 | global, duplicate- and context-aware score | 0.90 | 0.901 [0.894, 0.907] | 0.906 | 0.736 | 0.39 | yes |
| rtdetr | standard_threshold | n/a | 0.880 [0.873, 0.886] | 0.880 | 0.697 | n/a |  |
| rtdetr | global_confidence | 0.90 | 0.902 [0.896, 0.908] | 0.902 | 0.736 | 0.63 | yes |
| rtdetr | mondrian_confidence | 0.90 | 0.903 [0.897, 0.909] | 0.903 | 0.808 | 0.82 | yes |
| rtdetr | global, duplicate- and context-aware score | 0.90 | 0.902 [0.895, 0.907] | 0.907 | 0.743 | 0.64 | yes |
| yolov8 | standard_threshold | n/a | 0.888 [0.882, 0.895] | 0.888 | 0.739 | n/a |  |
| yolov8 | global_confidence | 0.90 | 0.900 [0.893, 0.906] | 0.900 | 0.759 | 0.27 | yes |
| yolov8 | mondrian_confidence | 0.90 | 0.901 [0.894, 0.907] | 0.906 | 0.824 | 0.55 | yes |
| yolov8 | global, duplicate- and context-aware score | 0.90 | 0.899 [0.892, 0.905] | 0.904 | 0.749 | 0.35 | yes |
| yolo26_s1 | standard_threshold | n/a | 0.889 [0.881, 0.895] | 0.889 | 0.713 | n/a |  |
| yolo26_s1 | global_confidence | 0.90 | 0.902 [0.895, 0.908] | 0.902 | 0.739 | 0.37 | yes |
| yolo26_s1 | mondrian_confidence | 0.90 | 0.901 [0.894, 0.906] | 0.906 | 0.801 | 0.58 | yes |
| yolo26_s1 | global, duplicate- and context-aware score | 0.90 | 0.901 [0.894, 0.906] | 0.906 | 0.723 | 0.40 | yes |
| yolo26_s2 | standard_threshold | n/a | 0.885 [0.877, 0.892] | 0.885 | 0.707 | n/a |  |
| yolo26_s2 | global_confidence | 0.90 | 0.899 [0.891, 0.906] | 0.899 | 0.723 | 0.37 | yes |
| yolo26_s2 | mondrian_confidence | 0.90 | 0.900 [0.892, 0.906] | 0.901 | 0.795 | 0.51 | yes |
| yolo26_s2 | global, duplicate- and context-aware score | 0.90 | 0.900 [0.893, 0.906] | 0.904 | 0.717 | 0.45 | yes |
| rtdetr_s1 | standard_threshold | n/a | 0.878 [0.872, 0.884] | 0.878 | 0.687 | n/a |  |
| rtdetr_s1 | global_confidence | 0.90 | 0.902 [0.896, 0.907] | 0.902 | 0.726 | 0.73 | yes |
| rtdetr_s1 | mondrian_confidence | 0.90 | 0.903 [0.897, 0.909] | 0.903 | 0.814 | 0.87 | yes |
| rtdetr_s1 | global, duplicate- and context-aware score | 0.90 | 0.901 [0.895, 0.907] | 0.908 | 0.733 | 0.74 | yes |
| rtdetr_s2 | standard_threshold | n/a | 0.876 [0.870, 0.882] | 0.876 | 0.671 | n/a |  |
| rtdetr_s2 | global_confidence | 0.90 | 0.904 [0.898, 0.911] | 0.904 | 0.723 | 0.76 | yes |
| rtdetr_s2 | mondrian_confidence | 0.90 | 0.903 [0.897, 0.909] | 0.903 | 0.798 | 0.83 | yes |
| rtdetr_s2 | global, duplicate- and context-aware score | 0.90 | 0.905 [0.898, 0.911] | 0.909 | 0.733 | 0.75 | yes |
| yolo26_evid | standard_threshold | n/a | 0.883 [0.876, 0.889] | 0.883 | 0.717 | n/a |  |
| yolo26_evid | global_confidence | 0.90 | 0.902 [0.895, 0.908] | 0.902 | 0.736 | 0.48 | yes |
| yolo26_evid | mondrian_confidence | 0.90 | 0.903 [0.896, 0.909] | 0.905 | 0.824 | 0.69 | yes |
| yolo26_evid | global, duplicate- and context-aware score | 0.90 | 0.902 [0.896, 0.908] | 0.908 | 0.723 | 0.59 | yes |
| yolo26_evid_s1 | standard_threshold | n/a | 0.883 [0.876, 0.889] | 0.883 | 0.723 | n/a |  |
| yolo26_evid_s1 | global_confidence | 0.90 | 0.898 [0.890, 0.903] | 0.898 | 0.743 | 0.39 | yes |
| yolo26_evid_s1 | mondrian_confidence | 0.90 | 0.896 [0.889, 0.902] | 0.899 | 0.808 | 0.53 | yes |
| yolo26_evid_s1 | global, duplicate- and context-aware score | 0.90 | 0.897 [0.890, 0.903] | 0.902 | 0.726 | 0.44 | yes |
| yolo26_evid_s2 | standard_threshold | n/a | 0.880 [0.874, 0.887] | 0.880 | 0.687 | n/a |  |
| yolo26_evid_s2 | global_confidence | 0.90 | 0.901 [0.894, 0.907] | 0.901 | 0.730 | 0.50 | yes |
| yolo26_evid_s2 | mondrian_confidence | 0.90 | 0.901 [0.895, 0.908] | 0.902 | 0.827 | 0.69 | yes |
| yolo26_evid_s2 | global, duplicate- and context-aware score | 0.90 | 0.899 [0.893, 0.905] | 0.905 | 0.707 | 0.54 | yes |
| yolo26_ens | standard_threshold | n/a | 0.896 [0.889, 0.902] | 0.896 | 0.713 | n/a |  |
| yolo26_ens | global_confidence | 0.90 | 0.901 [0.894, 0.907] | 0.901 | 0.730 | 0.12 | yes |
| yolo26_ens | mondrian_confidence | 0.90 | 0.900 [0.893, 0.907] | 0.910 | 0.801 | 0.42 | yes |
| yolo26_ens | global, duplicate- and context-aware score | 0.90 | 0.900 [0.893, 0.906] | 0.904 | 0.720 | 0.20 | yes |
| rtdetr_ens | standard_threshold | n/a | 0.889 [0.882, 0.895] | 0.889 | 0.700 | n/a |  |
| rtdetr_ens | global_confidence | 0.90 | 0.904 [0.898, 0.910] | 0.904 | 0.730 | 0.43 | yes |
| rtdetr_ens | mondrian_confidence | 0.90 | 0.902 [0.896, 0.908] | 0.907 | 0.801 | 0.65 | yes |
| rtdetr_ens | global, duplicate- and context-aware score | 0.90 | 0.903 [0.897, 0.909] | 0.908 | 0.736 | 0.47 | yes |

### RQ2: recovery of missed vehicles at matched extra false alarms

| detector | policy | misses | recoverable ceiling | mean recovery | difference from baseline, all misses | difference from baseline, occluded misses |
|---|---|---|---|---|---|---|
| yolo26 | confidence | 1779 | 0.903 | 0.240 [0.221, 0.260] | baseline | baseline |
| yolo26 | mondrian_rank | 1779 | 0.903 | 0.199 [0.179, 0.218] | -0.0414 [-0.0559, -0.0261] | 0.1144 [0.0954, 0.1362] |
| yolo26 | duplicate- and context-aware score | 1779 | 0.903 | 0.270 [0.248, 0.292] | 0.0296 [0.0231, 0.0358] | 0.0026 [-0.0046, 0.0100] |
| rtdetr | confidence | 1932 | 0.915 | 0.213 [0.192, 0.234] | baseline | baseline |
| rtdetr | mondrian_rank | 1932 | 0.915 | 0.176 [0.158, 0.195] | -0.0376 [-0.0517, -0.0230] | 0.1244 [0.1026, 0.1460] |
| rtdetr | duplicate- and context-aware score | 1932 | 0.915 | 0.270 [0.246, 0.292] | 0.0570 [0.0498, 0.0650] | 0.0529 [0.0435, 0.0627] |
| yolov8 | confidence | 1798 | 0.902 | 0.261 [0.239, 0.280] | baseline | baseline |
| yolov8 | mondrian_rank | 1798 | 0.902 | 0.207 [0.187, 0.225] | -0.0534 [-0.0681, -0.0389] | 0.1148 [0.0921, 0.1359] |
| yolov8 | duplicate- and context-aware score | 1798 | 0.902 | 0.283 [0.259, 0.303] | 0.0217 [0.0158, 0.0270] | -0.0057 [-0.0134, 0.0022] |
| yolo26_s1 | confidence | 1791 | 0.903 | 0.234 [0.214, 0.256] | baseline | baseline |
| yolo26_s1 | mondrian_rank | 1791 | 0.903 | 0.194 [0.174, 0.214] | -0.0393 [-0.0533, -0.0269] | 0.1186 [0.0965, 0.1383] |
| yolo26_s1 | duplicate- and context-aware score | 1791 | 0.903 | 0.271 [0.250, 0.294] | 0.0375 [0.0300, 0.0439] | 0.0143 [0.0061, 0.0227] |
| yolo26_s2 | confidence | 1853 | 0.903 | 0.233 [0.212, 0.253] | baseline | baseline |
| yolo26_s2 | mondrian_rank | 1853 | 0.903 | 0.208 [0.188, 0.228] | -0.0252 [-0.0386, -0.0117] | 0.1177 [0.0961, 0.1384] |
| yolo26_s2 | duplicate- and context-aware score | 1853 | 0.903 | 0.265 [0.245, 0.289] | 0.0322 [0.0254, 0.0386] | 0.0063 [-0.0027, 0.0150] |
| rtdetr_s1 | confidence | 1968 | 0.914 | 0.204 [0.185, 0.222] | baseline | baseline |
| rtdetr_s1 | mondrian_rank | 1968 | 0.914 | 0.176 [0.159, 0.193] | -0.0279 [-0.0416, -0.0136] | 0.1360 [0.1134, 0.1572] |
| rtdetr_s1 | duplicate- and context-aware score | 1968 | 0.914 | 0.261 [0.238, 0.282] | 0.0569 [0.0490, 0.0642] | 0.0505 [0.0411, 0.0593] |
| rtdetr_s2 | confidence | 2001 | 0.918 | 0.230 [0.210, 0.248] | baseline | baseline |
| rtdetr_s2 | mondrian_rank | 2001 | 0.918 | 0.197 [0.179, 0.213] | -0.0330 [-0.0461, -0.0206] | 0.1175 [0.0939, 0.1351] |
| rtdetr_s2 | duplicate- and context-aware score | 2001 | 0.918 | 0.287 [0.263, 0.308] | 0.0574 [0.0492, 0.0654] | 0.0542 [0.0438, 0.0644] |
| yolo26_evid | confidence | 1890 | 0.912 | 0.251 [0.232, 0.274] | baseline | baseline |
| yolo26_evid | mondrian_rank | 1890 | 0.912 | 0.204 [0.184, 0.225] | -0.0471 [-0.0608, -0.0333] | 0.1079 [0.0882, 0.1276] |
| yolo26_evid | duplicate- and context-aware score | 1890 | 0.912 | 0.282 [0.261, 0.305] | 0.0305 [0.0250, 0.0375] | 0.0040 [-0.0028, 0.0122] |
| yolo26_evid_s1 | confidence | 1882 | 0.915 | 0.238 [0.215, 0.259] | baseline | baseline |
| yolo26_evid_s1 | mondrian_rank | 1882 | 0.915 | 0.186 [0.166, 0.205] | -0.0517 [-0.0661, -0.0359] | 0.0994 [0.0780, 0.1193] |
| yolo26_evid_s1 | duplicate- and context-aware score | 1882 | 0.915 | 0.270 [0.248, 0.292] | 0.0323 [0.0257, 0.0383] | 0.0093 [0.0023, 0.0158] |
| yolo26_evid_s2 | confidence | 1924 | 0.913 | 0.247 [0.228, 0.270] | baseline | baseline |
| yolo26_evid_s2 | mondrian_rank | 1924 | 0.913 | 0.201 [0.182, 0.220] | -0.0465 [-0.0605, -0.0321] | 0.1160 [0.0935, 0.1376] |
| yolo26_evid_s2 | duplicate- and context-aware score | 1924 | 0.913 | 0.281 [0.259, 0.303] | 0.0335 [0.0259, 0.0408] | -0.0016 [-0.0109, 0.0066] |
| yolo26_ens | confidence | 1679 | 0.911 | 0.254 [0.231, 0.277] | baseline | baseline |
| yolo26_ens | mondrian_rank | 1679 | 0.911 | 0.212 [0.190, 0.234] | -0.0420 [-0.0573, -0.0277] | 0.1045 [0.0859, 0.1240] |
| yolo26_ens | duplicate- and context-aware score | 1679 | 0.911 | 0.272 [0.248, 0.296] | 0.0182 [0.0122, 0.0233] | -0.0074 [-0.0159, 0.0000] |
| rtdetr_ens | confidence | 1792 | 0.917 | 0.210 [0.191, 0.228] | baseline | baseline |
| rtdetr_ens | mondrian_rank | 1792 | 0.917 | 0.172 [0.155, 0.190] | -0.0376 [-0.0523, -0.0230] | 0.1165 [0.0956, 0.1387] |
| rtdetr_ens | duplicate- and context-aware score | 1792 | 0.917 | 0.269 [0.246, 0.291] | 0.0595 [0.0507, 0.0681] | 0.0566 [0.0463, 0.0670] |

### RQ2 ablation: which features of the occlusion-aware score carry the gain

Chosen by 5-fold cross-validation over calib images (criterion fixed in advance: mean recovery of all misses over the budget range); every set then evaluated once on test against plain confidence. The configured default calibrator is unchanged.

| detector | feature set | calib CV: mean recovery | calib CV: occluded | test: mean recovery | test: difference from confidence, all misses | test: difference, occluded misses |
|---|---|---|---|---|---|---|
| yolo26 | z_only | 0.2317 | 0.2299 | 0.240 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| yolo26 | z_dup | 0.2650 | 0.2348 | 0.264 | 0.0236 [0.0172, 0.0303] | -0.0082 [-0.0162, -0.0002] |
| yolo26 | z_ctx | 0.2318 | 0.2128 | 0.239 | -0.0013 [-0.0043, 0.0014] | -0.0264 [-0.0313, -0.0217] |
| yolo26 | full | 0.2648 | 0.2371 | 0.263 | 0.0229 [0.0165, 0.0299] | -0.0116 [-0.0198, -0.0030] |
| yolo26 | full_strict **(chosen on calib)** | 0.2692 | 0.2448 | 0.270 | 0.0296 [0.0231, 0.0358] | 0.0026 [-0.0046, 0.0100] |
| rtdetr | z_only | 0.2042 | 0.1990 | 0.213 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| rtdetr | z_dup | 0.2570 | 0.2161 | 0.259 | 0.0452 [0.0370, 0.0544] | 0.0103 [0.0002, 0.0217] |
| rtdetr | z_ctx | 0.2038 | 0.2166 | 0.214 | 0.0009 [-0.0003, 0.0017] | 0.0089 [0.0068, 0.0103] |
| rtdetr | full | 0.2574 | 0.2456 | 0.261 | 0.0475 [0.0393, 0.0564] | 0.0358 [0.0246, 0.0478] |
| rtdetr | full_strict **(chosen on calib)** | 0.2617 | 0.2578 | 0.270 | 0.0570 [0.0498, 0.0650] | 0.0529 [0.0435, 0.0627] |
| yolov8 | z_only | 0.2486 | 0.2405 | 0.261 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| yolov8 | z_dup | 0.2706 | 0.2414 | 0.277 | 0.0164 [0.0114, 0.0212] | -0.0082 [-0.0136, -0.0016] |
| yolov8 | z_ctx | 0.2488 | 0.2254 | 0.260 | -0.0004 [-0.0025, 0.0015] | -0.0197 [-0.0229, -0.0162] |
| yolov8 | full | 0.2700 | 0.2409 | 0.277 | 0.0158 [0.0107, 0.0205] | -0.0120 [-0.0179, -0.0058] |
| yolov8 | full_strict **(chosen on calib)** | 0.2773 | 0.2499 | 0.283 | 0.0217 [0.0158, 0.0270] | -0.0057 [-0.0134, 0.0022] |
| yolo26_s1 | z_only | 0.2397 | 0.2261 | 0.234 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| yolo26_s1 | z_dup | 0.2719 | 0.2331 | 0.267 | 0.0333 [0.0262, 0.0396] | 0.0063 [-0.0023, 0.0140] |
| yolo26_s1 | z_ctx | 0.2394 | 0.2133 | 0.233 | -0.0005 [-0.0026, 0.0017] | -0.0213 [-0.0247, -0.0173] |
| yolo26_s1 | full | 0.2719 | 0.2396 | 0.266 | 0.0321 [0.0257, 0.0388] | 0.0047 [-0.0032, 0.0133] |
| yolo26_s1 | full_strict **(chosen on calib)** | 0.2749 | 0.2478 | 0.271 | 0.0375 [0.0300, 0.0439] | 0.0143 [0.0061, 0.0227] |
| yolo26_s2 | z_only | 0.2480 | 0.2481 | 0.233 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| yolo26_s2 | z_dup | 0.2828 | 0.2506 | 0.261 | 0.0282 [0.0206, 0.0353] | -0.0008 [-0.0096, 0.0080] |
| yolo26_s2 | z_ctx | 0.2476 | 0.2309 | 0.232 | -0.0017 [-0.0042, 0.0003] | -0.0249 [-0.0286, -0.0206] |
| yolo26_s2 | full | 0.2830 | 0.2568 | 0.261 | 0.0278 [0.0202, 0.0347] | 0.0004 [-0.0086, 0.0092] |
| yolo26_s2 | full_strict **(chosen on calib)** | 0.2868 | 0.2659 | 0.265 | 0.0322 [0.0254, 0.0386] | 0.0063 [-0.0027, 0.0150] |
| rtdetr_s1 | z_only | 0.2201 | 0.2174 | 0.204 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| rtdetr_s1 | z_dup | 0.2736 | 0.2396 | 0.251 | 0.0464 [0.0375, 0.0543] | 0.0123 [0.0013, 0.0216] |
| rtdetr_s1 | z_ctx | 0.2211 | 0.2360 | 0.205 | 0.0010 [0.0001, 0.0018] | 0.0091 [0.0073, 0.0109] |
| rtdetr_s1 | full | 0.2748 | 0.2714 | 0.253 | 0.0489 [0.0407, 0.0568] | 0.0377 [0.0266, 0.0486] |
| rtdetr_s1 | full_strict **(chosen on calib)** | 0.2797 | 0.2828 | 0.261 | 0.0569 [0.0490, 0.0642] | 0.0505 [0.0411, 0.0593] |
| rtdetr_s2 | z_only | 0.2255 | 0.2139 | 0.230 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| rtdetr_s2 | z_dup | 0.2757 | 0.2334 | 0.279 | 0.0497 [0.0404, 0.0589] | 0.0122 [0.0009, 0.0233] |
| rtdetr_s2 | z_ctx | 0.2269 | 0.2359 | 0.231 | 0.0010 [-0.0001, 0.0020] | 0.0108 [0.0087, 0.0124] |
| rtdetr_s2 | full | 0.2792 | 0.2723 | 0.281 | 0.0516 [0.0433, 0.0611] | 0.0408 [0.0293, 0.0526] |
| rtdetr_s2 | full_strict **(chosen on calib)** | 0.2823 | 0.2799 | 0.287 | 0.0574 [0.0492, 0.0654] | 0.0542 [0.0438, 0.0644] |
| yolo26_evid | z_only | 0.2340 | 0.2351 | 0.251 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| yolo26_evid | z_dup | 0.2670 | 0.2413 | 0.278 | 0.0266 [0.0199, 0.0341] | -0.0069 [-0.0162, 0.0026] |
| yolo26_evid | z_ctx | 0.2329 | 0.2211 | 0.250 | -0.0017 [-0.0038, 0.0008] | -0.0245 [-0.0287, -0.0201] |
| yolo26_evid | full | 0.2677 | 0.2495 | 0.278 | 0.0267 [0.0200, 0.0340] | -0.0050 [-0.0140, 0.0049] |
| yolo26_evid | full_strict **(chosen on calib)** | 0.2717 | 0.2563 | 0.282 | 0.0305 [0.0250, 0.0375] | 0.0040 [-0.0028, 0.0122] |
| yolo26_evid_s1 | z_only | 0.2382 | 0.2221 | 0.238 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| yolo26_evid_s1 | z_dup | 0.2652 | 0.2237 | 0.264 | 0.0269 [0.0203, 0.0333] | 0.0018 [-0.0059, 0.0093] |
| yolo26_evid_s1 | z_ctx | 0.2367 | 0.2045 | 0.238 | 0.0000 [-0.0021, 0.0026] | -0.0208 [-0.0248, -0.0169] |
| yolo26_evid_s1 | full | 0.2658 | 0.2287 | 0.264 | 0.0268 [0.0201, 0.0332] | 0.0005 [-0.0073, 0.0087] |
| yolo26_evid_s1 | full_strict **(chosen on calib)** | 0.2700 | 0.2355 | 0.270 | 0.0323 [0.0257, 0.0383] | 0.0093 [0.0023, 0.0158] |
| yolo26_evid_s2 | z_only | 0.2316 | 0.2320 | 0.247 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| yolo26_evid_s2 | z_dup | 0.2655 | 0.2352 | 0.277 | 0.0302 [0.0231, 0.0369] | -0.0008 [-0.0093, 0.0068] |
| yolo26_evid_s2 | z_ctx | 0.2314 | 0.2123 | 0.244 | -0.0031 [-0.0067, -0.0001] | -0.0369 [-0.0437, -0.0317] |
| yolo26_evid_s2 | full | 0.2656 | 0.2389 | 0.276 | 0.0288 [0.0214, 0.0356] | -0.0104 [-0.0194, -0.0018] |
| yolo26_evid_s2 | full_strict **(chosen on calib)** | 0.2696 | 0.2475 | 0.281 | 0.0335 [0.0259, 0.0408] | -0.0016 [-0.0109, 0.0066] |

### Baselines from the literature, next to our rules (v0.6 Phase 1)

Every rule is fitted on the same calib images (the threshold half used by our conformal rules) and evaluated once on test. Coverage = object recall of the rule alone; flags = candidates the rule keeps below the operating threshold, per image; o_j = annotated occlusion, o_hat = occlusion context.

**Target 1-a = 0.90**  (primary, D6)

| detector | rule | guarantee | coverage | per-image FNR | flags/image | coverage, most occluded o_j bucket | lowest coverage over o_hat strata |
|---|---|---|---|---|---|---|---|
| yolo26 | mondrian_confidence | recall >= 1-a in every occlusion-context stratum (ours, headline) | 0.903 [0.896, 0.909] | 0.097 [0.090, 0.105] | 0.53 | 0.798 (n=307) | 0.896 (0.30-1.00) |
| yolo26 | global_confidence | recall >= 1-a, marginal (ours, reference) | 0.902 [0.895, 0.908] | 0.095 [0.088, 0.102] | 0.32 | 0.739 (n=307) | 0.838 (0.30-1.00) |
| yolo26 | global_occlusion_aware | recall >= 1-a, marginal; duplicate- and context-aware score (ours) | 0.901 [0.894, 0.907] | 0.095 [0.088, 0.102] | 0.39 | 0.736 (n=307) | 0.828 (0.30-1.00) |
| yolo26 | global_gilg_iou_aware | recall >= 1-a, marginal; Gilg IoU-aware score | 0.902 [0.895, 0.908] | 0.095 [0.088, 0.102] | 0.29 | 0.739 (n=307) | 0.838 (0.30-1.00) |
| yolo26 | crc_fnr | E[per-image FNR] <= a (CRC; Hulsman 2024) | 0.896 [0.889, 0.903] | 0.100 [0.093, 0.108] | 0.18 | 0.736 (n=307) | 0.831 (0.30-1.00) |
| yolo26 | seqcrc_box_count_recall | box-count recall loss <= a (SeqCRC; no spatial matching) | 0.853 [0.844, 0.860] | 0.142 [0.134, 0.150] | 0.00 | 0.678 (n=307) | 0.772 (0.30-1.00) |
| yolo26 | seqcrc_box_count_threshold | P(fewer boxes than objects) <= a (SeqCRC) | 0.936 [0.930, 0.942] | 0.063 [0.057, 0.070] | 1.61 | 0.818 (n=307) | 0.896 (0.30-1.00) |
| yolo26 | pac | P(object miss rate > a) <= delta (PAC; Li 2022) | 0.906 [0.899, 0.912] | 0.091 [0.084, 0.098] | 0.45 | 0.752 (n=307) | 0.846 (0.30-1.00) |
| rtdetr | mondrian_confidence | recall >= 1-a in every occlusion-context stratum (ours, headline) | 0.903 [0.897, 0.909] | 0.098 [0.092, 0.105] | 0.82 | 0.808 (n=307) | 0.897 (0.30-1.00) |
| rtdetr | global_confidence | recall >= 1-a, marginal (ours, reference) | 0.902 [0.896, 0.908] | 0.095 [0.089, 0.102] | 0.63 | 0.736 (n=307) | 0.832 (0.30-1.00) |
| rtdetr | global_occlusion_aware | recall >= 1-a, marginal; duplicate- and context-aware score (ours) | 0.902 [0.895, 0.907] | 0.095 [0.089, 0.102] | 0.64 | 0.730 (n=307) | 0.834 (0.30-1.00) |
| rtdetr | global_gilg_iou_aware | recall >= 1-a, marginal; Gilg IoU-aware score | 0.902 [0.896, 0.908] | 0.095 [0.089, 0.102] | 0.50 | 0.736 (n=307) | 0.834 (0.30-1.00) |
| rtdetr | crc_fnr | E[per-image FNR] <= a (CRC; Hulsman 2024) | 0.896 [0.889, 0.902] | 0.101 [0.094, 0.107] | 0.44 | 0.723 (n=307) | 0.826 (0.30-1.00) |
| rtdetr | seqcrc_box_count_recall | box-count recall loss <= a (SeqCRC; no spatial matching) | 0.843 [0.836, 0.850] | 0.152 [0.144, 0.160] | 0.00 | 0.606 (n=307) | 0.744 (0.30-1.00) |
| rtdetr | seqcrc_box_count_threshold | P(fewer boxes than objects) <= a (SeqCRC) | 0.930 [0.924, 0.935] | 0.069 [0.063, 0.075] | 1.79 | 0.808 (n=307) | 0.880 (0.30-1.00) |
| rtdetr | pac | P(object miss rate > a) <= delta (PAC; Li 2022) | 0.907 [0.901, 0.913] | 0.090 [0.084, 0.096] | 0.79 | 0.743 (n=307) | 0.842 (0.30-1.00) |
| yolov8 | mondrian_confidence | recall >= 1-a in every occlusion-context stratum (ours, headline) | 0.901 [0.894, 0.907] | 0.100 [0.093, 0.108] | 0.55 | 0.824 (n=307) | 0.898 (0.00-0.05) |
| yolov8 | global_confidence | recall >= 1-a, marginal (ours, reference) | 0.900 [0.893, 0.906] | 0.097 [0.090, 0.104] | 0.27 | 0.759 (n=307) | 0.834 (0.30-1.00) |
| yolov8 | global_occlusion_aware | recall >= 1-a, marginal; duplicate- and context-aware score (ours) | 0.899 [0.892, 0.905] | 0.098 [0.092, 0.106] | 0.35 | 0.749 (n=307) | 0.827 (0.30-1.00) |
| yolov8 | global_gilg_iou_aware | recall >= 1-a, marginal; Gilg IoU-aware score | 0.900 [0.893, 0.905] | 0.098 [0.091, 0.105] | 0.24 | 0.759 (n=307) | 0.831 (0.30-1.00) |
| yolov8 | crc_fnr | E[per-image FNR] <= a (CRC; Hulsman 2024) | 0.895 [0.888, 0.901] | 0.102 [0.095, 0.109] | 0.16 | 0.749 (n=307) | 0.827 (0.30-1.00) |
| yolov8 | seqcrc_box_count_recall | box-count recall loss <= a (SeqCRC; no spatial matching) | 0.839 [0.830, 0.846] | 0.156 [0.148, 0.164] | 0.00 | 0.645 (n=307) | 0.751 (0.30-1.00) |
| yolov8 | seqcrc_box_count_threshold | P(fewer boxes than objects) <= a (SeqCRC) | 0.934 [0.928, 0.939] | 0.064 [0.059, 0.070] | 1.38 | 0.811 (n=307) | 0.883 (0.30-1.00) |
| yolov8 | pac | P(object miss rate > a) <= delta (PAC; Li 2022) | 0.905 [0.898, 0.911] | 0.093 [0.087, 0.100] | 0.41 | 0.772 (n=307) | 0.841 (0.30-1.00) |
| yolo26_ens | mondrian_confidence | recall >= 1-a in every occlusion-context stratum (ours, headline) | 0.900 [0.893, 0.907] | 0.101 [0.093, 0.109] | 0.42 | 0.798 (n=307) | 0.895 (0.30-1.00) |
| yolo26_ens | global_confidence | recall >= 1-a, marginal (ours, reference) | 0.901 [0.894, 0.907] | 0.095 [0.088, 0.103] | 0.12 | 0.730 (n=307) | 0.837 (0.30-1.00) |
| yolo26_ens | global_occlusion_aware | recall >= 1-a, marginal; duplicate- and context-aware score (ours) | 0.900 [0.893, 0.906] | 0.095 [0.089, 0.103] | 0.20 | 0.700 (n=307) | 0.825 (0.30-1.00) |
| yolo26_ens | global_gilg_iou_aware | recall >= 1-a, marginal; Gilg IoU-aware score | 0.901 [0.894, 0.907] | 0.095 [0.088, 0.103] | 0.14 | 0.730 (n=307) | 0.836 (0.30-1.00) |
| yolo26_ens | crc_fnr | E[per-image FNR] <= a (CRC; Hulsman 2024) | 0.894 [0.887, 0.900] | 0.101 [0.094, 0.109] | 0.00 | 0.713 (n=307) | 0.826 (0.30-1.00) |
| yolo26_ens | seqcrc_box_count_recall | box-count recall loss <= a (SeqCRC; no spatial matching) | 0.853 [0.845, 0.860] | 0.142 [0.134, 0.152] | 0.00 | 0.678 (n=307) | 0.776 (0.30-1.00) |
| yolo26_ens | seqcrc_box_count_threshold | P(fewer boxes than objects) <= a (SeqCRC) | 0.936 [0.930, 0.942] | 0.064 [0.058, 0.071] | 1.35 | 0.818 (n=307) | 0.896 (0.30-1.00) |
| yolo26_ens | pac | P(object miss rate > a) <= delta (PAC; Li 2022) | 0.906 [0.899, 0.912] | 0.090 [0.083, 0.097] | 0.24 | 0.739 (n=307) | 0.846 (0.30-1.00) |
| rtdetr_ens | mondrian_confidence | recall >= 1-a in every occlusion-context stratum (ours, headline) | 0.902 [0.896, 0.908] | 0.101 [0.094, 0.108] | 0.65 | 0.801 (n=307) | 0.890 (0.30-1.00) |
| rtdetr_ens | global_confidence | recall >= 1-a, marginal (ours, reference) | 0.904 [0.898, 0.910] | 0.094 [0.087, 0.100] | 0.43 | 0.730 (n=307) | 0.834 (0.30-1.00) |
| rtdetr_ens | global_occlusion_aware | recall >= 1-a, marginal; duplicate- and context-aware score (ours) | 0.903 [0.897, 0.909] | 0.095 [0.089, 0.102] | 0.47 | 0.730 (n=307) | 0.836 (0.30-1.00) |
| rtdetr_ens | global_gilg_iou_aware | recall >= 1-a, marginal; Gilg IoU-aware score | 0.903 [0.896, 0.908] | 0.096 [0.089, 0.102] | 0.33 | 0.726 (n=307) | 0.834 (0.30-1.00) |
| rtdetr_ens | crc_fnr | E[per-image FNR] <= a (CRC; Hulsman 2024) | 0.897 [0.891, 0.903] | 0.101 [0.094, 0.107] | 0.21 | 0.707 (n=307) | 0.821 (0.30-1.00) |
| rtdetr_ens | seqcrc_box_count_recall | box-count recall loss <= a (SeqCRC; no spatial matching) | 0.847 [0.840, 0.854] | 0.149 [0.142, 0.157] | 0.00 | 0.619 (n=307) | 0.740 (0.30-1.00) |
| rtdetr_ens | seqcrc_box_count_threshold | P(fewer boxes than objects) <= a (SeqCRC) | 0.932 [0.926, 0.937] | 0.067 [0.061, 0.072] | 1.67 | 0.801 (n=307) | 0.879 (0.30-1.00) |
| rtdetr_ens | pac | P(object miss rate > a) <= delta (PAC; Li 2022) | 0.909 [0.902, 0.914] | 0.089 [0.083, 0.096] | 0.58 | 0.736 (n=307) | 0.842 (0.30-1.00) |

**Target 1-a = 0.95**

| detector | rule | guarantee | coverage | per-image FNR | flags/image | coverage, most occluded o_j bucket | lowest coverage over o_hat strata |
|---|---|---|---|---|---|---|---|
| yolo26 | mondrian_confidence | recall >= 1-a in every occlusion-context stratum (ours, headline) | 0.949 [0.944, 0.954] | 0.053 [0.047, 0.059] | 2.66 | 0.896 (n=307) | 0.946 (0.30-1.00) |
| yolo26 | global_confidence | recall >= 1-a, marginal (ours, reference) | 0.949 [0.944, 0.954] | 0.052 [0.047, 0.057] | 2.38 | 0.853 (n=307) | 0.916 (0.30-1.00) |
| yolo26 | global_occlusion_aware | recall >= 1-a, marginal; duplicate- and context-aware score (ours) | 0.949 [0.943, 0.954] | 0.052 [0.047, 0.058] | 2.24 | 0.844 (n=307) | 0.914 (0.30-1.00) |
| yolo26 | global_gilg_iou_aware | recall >= 1-a, marginal; Gilg IoU-aware score | 0.949 [0.944, 0.954] | 0.052 [0.047, 0.058] | 2.02 | 0.847 (n=307) | 0.914 (0.30-1.00) |
| yolo26 | crc_fnr | E[per-image FNR] <= a (CRC; Hulsman 2024) | 0.949 [0.944, 0.954] | 0.052 [0.047, 0.057] | 2.40 | 0.853 (n=307) | 0.917 (0.30-1.00) |
| yolo26 | seqcrc_box_count_recall | box-count recall loss <= a (SeqCRC; no spatial matching) | 0.901 [0.894, 0.907] | 0.096 [0.089, 0.103] | 0.29 | 0.739 (n=307) | 0.837 (0.30-1.00) |
| yolo26 | seqcrc_box_count_threshold | P(fewer boxes than objects) <= a (SeqCRC) | 0.952 [0.947, 0.957] | 0.049 [0.044, 0.055] | 2.62 | 0.857 (n=307) | 0.922 (0.30-1.00) |
| yolo26 | pac | P(object miss rate > a) <= delta (PAC; Li 2022) | 0.952 [0.947, 0.957] | 0.049 [0.044, 0.055] | 2.63 | 0.857 (n=307) | 0.922 (0.30-1.00) |
| rtdetr | mondrian_confidence | recall >= 1-a in every occlusion-context stratum (ours, headline) | 0.950 [0.944, 0.955] | 0.051 [0.046, 0.057] | 3.90 | 0.886 (n=307) | 0.944 (0.30-1.00) |
| rtdetr | global_confidence | recall >= 1-a, marginal (ours, reference) | 0.950 [0.945, 0.955] | 0.049 [0.043, 0.054] | 3.49 | 0.866 (n=307) | 0.913 (0.30-1.00) |
| rtdetr | global_occlusion_aware | recall >= 1-a, marginal; duplicate- and context-aware score (ours) | 0.950 [0.944, 0.955] | 0.049 [0.043, 0.054] | 2.78 | 0.866 (n=307) | 0.913 (0.30-1.00) |
| rtdetr | global_gilg_iou_aware | recall >= 1-a, marginal; Gilg IoU-aware score | 0.949 [0.944, 0.954] | 0.050 [0.045, 0.055] | 2.27 | 0.870 (n=307) | 0.914 (0.30-1.00) |
| rtdetr | crc_fnr | E[per-image FNR] <= a (CRC; Hulsman 2024) | 0.949 [0.943, 0.954] | 0.050 [0.045, 0.056] | 3.28 | 0.863 (n=307) | 0.911 (0.30-1.00) |
| rtdetr | seqcrc_box_count_recall | box-count recall loss <= a (SeqCRC; no spatial matching) | 0.891 [0.885, 0.897] | 0.105 [0.099, 0.112] | 0.29 | 0.717 (n=307) | 0.820 (0.30-1.00) |
| rtdetr | seqcrc_box_count_threshold | P(fewer boxes than objects) <= a (SeqCRC) | 0.940 [0.934, 0.945] | 0.058 [0.053, 0.064] | 2.55 | 0.831 (n=307) | 0.897 (0.30-1.00) |
| rtdetr | pac | P(object miss rate > a) <= delta (PAC; Li 2022) | 0.953 [0.947, 0.958] | 0.047 [0.041, 0.052] | 3.81 | 0.870 (n=307) | 0.918 (0.30-1.00) |
| yolov8 | mondrian_confidence | recall >= 1-a in every occlusion-context stratum (ours, headline) | 0.950 [0.944, 0.955] | 0.052 [0.047, 0.057] | 2.74 | 0.915 (n=307) | 0.950 (0.05-0.30) |
| yolov8 | global_confidence | recall >= 1-a, marginal (ours, reference) | 0.948 [0.943, 0.953] | 0.052 [0.047, 0.058] | 2.15 | 0.853 (n=307) | 0.908 (0.30-1.00) |
| yolov8 | global_occlusion_aware | recall >= 1-a, marginal; duplicate- and context-aware score (ours) | 0.948 [0.942, 0.953] | 0.052 [0.047, 0.057] | 2.09 | 0.850 (n=307) | 0.906 (0.30-1.00) |
| yolov8 | global_gilg_iou_aware | recall >= 1-a, marginal; Gilg IoU-aware score | 0.948 [0.943, 0.953] | 0.052 [0.047, 0.057] | 1.92 | 0.850 (n=307) | 0.908 (0.30-1.00) |
| yolov8 | crc_fnr | E[per-image FNR] <= a (CRC; Hulsman 2024) | 0.948 [0.943, 0.953] | 0.052 [0.047, 0.057] | 2.17 | 0.853 (n=307) | 0.909 (0.30-1.00) |
| yolov8 | seqcrc_box_count_recall | box-count recall loss <= a (SeqCRC; no spatial matching) | 0.891 [0.884, 0.897] | 0.106 [0.099, 0.113] | 0.06 | 0.739 (n=307) | 0.821 (0.30-1.00) |
| yolov8 | seqcrc_box_count_threshold | P(fewer boxes than objects) <= a (SeqCRC) | 0.948 [0.943, 0.953] | 0.052 [0.047, 0.057] | 2.19 | 0.853 (n=307) | 0.909 (0.30-1.00) |
| yolov8 | pac | P(object miss rate > a) <= delta (PAC; Li 2022) | 0.953 [0.948, 0.958] | 0.048 [0.043, 0.053] | 2.47 | 0.860 (n=307) | 0.918 (0.30-1.00) |
| yolo26_ens | mondrian_confidence | recall >= 1-a in every occlusion-context stratum (ours, headline) | 0.948 [0.942, 0.953] | 0.055 [0.049, 0.061] | 2.33 | 0.886 (n=307) | 0.944 (0.30-1.00) |
| yolo26_ens | global_confidence | recall >= 1-a, marginal (ours, reference) | 0.948 [0.943, 0.953] | 0.053 [0.048, 0.059] | 2.09 | 0.853 (n=307) | 0.915 (0.30-1.00) |
| yolo26_ens | global_occlusion_aware | recall >= 1-a, marginal; duplicate- and context-aware score (ours) | 0.948 [0.943, 0.953] | 0.053 [0.047, 0.059] | 1.93 | 0.840 (n=307) | 0.913 (0.30-1.00) |
| yolo26_ens | global_gilg_iou_aware | recall >= 1-a, marginal; Gilg IoU-aware score | 0.948 [0.943, 0.953] | 0.053 [0.048, 0.059] | 1.80 | 0.837 (n=307) | 0.914 (0.30-1.00) |
| yolo26_ens | crc_fnr | E[per-image FNR] <= a (CRC; Hulsman 2024) | 0.949 [0.943, 0.954] | 0.052 [0.047, 0.058] | 2.12 | 0.853 (n=307) | 0.917 (0.30-1.00) |
| yolo26_ens | seqcrc_box_count_recall | box-count recall loss <= a (SeqCRC; no spatial matching) | 0.901 [0.894, 0.907] | 0.095 [0.088, 0.103] | 0.13 | 0.730 (n=307) | 0.838 (0.30-1.00) |
| yolo26_ens | seqcrc_box_count_threshold | P(fewer boxes than objects) <= a (SeqCRC) | 0.951 [0.946, 0.956] | 0.050 [0.044, 0.055] | 2.30 | 0.853 (n=307) | 0.920 (0.30-1.00) |
| yolo26_ens | pac | P(object miss rate > a) <= delta (PAC; Li 2022) | 0.952 [0.946, 0.957] | 0.049 [0.044, 0.055] | 2.35 | 0.853 (n=307) | 0.921 (0.30-1.00) |
| rtdetr_ens | mondrian_confidence | recall >= 1-a in every occlusion-context stratum (ours, headline) | 0.952 [0.946, 0.956] | 0.050 [0.044, 0.055] | 3.57 | 0.899 (n=307) | 0.946 (0.30-1.00) |
| rtdetr_ens | global_confidence | recall >= 1-a, marginal (ours, reference) | 0.949 [0.944, 0.954] | 0.050 [0.045, 0.056] | 3.04 | 0.857 (n=307) | 0.907 (0.30-1.00) |
| rtdetr_ens | global_occlusion_aware | recall >= 1-a, marginal; duplicate- and context-aware score (ours) | 0.950 [0.944, 0.954] | 0.050 [0.044, 0.055] | 2.27 | 0.850 (n=307) | 0.907 (0.30-1.00) |
| rtdetr_ens | global_gilg_iou_aware | recall >= 1-a, marginal; Gilg IoU-aware score | 0.949 [0.944, 0.954] | 0.051 [0.045, 0.056] | 2.01 | 0.857 (n=307) | 0.908 (0.30-1.00) |
| rtdetr_ens | crc_fnr | E[per-image FNR] <= a (CRC; Hulsman 2024) | 0.948 [0.943, 0.953] | 0.051 [0.046, 0.056] | 2.96 | 0.850 (n=307) | 0.906 (0.30-1.00) |
| rtdetr_ens | seqcrc_box_count_recall | box-count recall loss <= a (SeqCRC; no spatial matching) | 0.894 [0.888, 0.900] | 0.103 [0.097, 0.110] | 0.13 | 0.704 (n=307) | 0.817 (0.30-1.00) |
| rtdetr_ens | seqcrc_box_count_threshold | P(fewer boxes than objects) <= a (SeqCRC) | 0.945 [0.939, 0.950] | 0.054 [0.049, 0.060] | 2.64 | 0.844 (n=307) | 0.900 (0.30-1.00) |
| rtdetr_ens | pac | P(object miss rate > a) <= delta (PAC; Li 2022) | 0.952 [0.947, 0.956] | 0.048 [0.043, 0.053] | 3.35 | 0.870 (n=307) | 0.912 (0.30-1.00) |

#### IoU-aware calibration (Gilg et al.) vs our duplicate-aware scores

Recovery of missed vehicles vs plain confidence at matched extra false alarms (RQ2 protocol), all misses / occluded misses. Gilg's score is fitted on the calib score half (no fit-role predictions exist).

| detector | Spearman: Gilg vs z_dup (weak candidates) | Spearman: Gilg's u vs our dup feature | Gilg IoU-aware | our z_dup | our default (duplicate- and context-aware) |
|---|---|---|---|---|---|
| yolo26 | 0.793 | 0.258 | 0.0450 [0.0362, 0.0545] / occl. 0.0434 | 0.0236 [0.0172, 0.0303] / occl. -0.0082 | 0.0296 [0.0231, 0.0358] / occl. 0.0026 |
| rtdetr | 0.567 | 0.115 | 0.0790 [0.0697, 0.0888] / occl. 0.0940 | 0.0452 [0.0370, 0.0544] / occl. 0.0103 | 0.0570 [0.0498, 0.0650] / occl. 0.0529 |
| yolov8 | 0.818 | 0.253 | 0.0382 [0.0303, 0.0451] / occl. 0.0344 | 0.0164 [0.0114, 0.0212] / occl. -0.0082 | 0.0217 [0.0158, 0.0270] / occl. -0.0057 |
| yolo26_ens | 0.805 | 0.221 | 0.0212 [0.0147, 0.0281] / occl. 0.0102 | 0.0130 [0.0066, 0.0196] / occl. -0.0259 | 0.0182 [0.0122, 0.0233] / occl. -0.0074 |
| rtdetr_ens | 0.734 | 0.131 | 0.0667 [0.0565, 0.0775] / occl. 0.0792 | 0.0493 [0.0396, 0.0594] / occl. 0.0096 | 0.0595 [0.0507, 0.0681] / occl. 0.0566 |

#### Localisation-aware calibration and LRP (Kuzucu et al. 2024)

LaECE_0 / LaACE_0: 25 bins, target = IoU with the matched vehicle (any IoU > 0), at the operating threshold. LRP at tau = 0.5; the LRP-optimal threshold is chosen on calib.

| detector | LaECE_0 | LaACE_0 | D-ECE | LRP at F1-optimal threshold | LRP at LRP-optimal threshold | recall / false alarms per image at LRP-optimal |
|---|---|---|---|---|---|---|
| yolo26 | 0.0291 [0.0246, 0.0338] | 0.0295 [0.0252, 0.0345] | 0.0666 | 0.46: 0.3331 [0.3251, 0.3415] | 0.46: 0.3331 [0.3251, 0.3415] | 0.889 / 1.26 |
| rtdetr | 0.0908 [0.0836, 0.0976] | 0.0908 [0.0836, 0.0976] | 0.0468 | 0.69: 0.3490 [0.3407, 0.3577] | 0.72: 0.3472 [0.3391, 0.3559] | 0.860 / 1.09 |
| yolov8 | 0.0309 [0.0262, 0.0367] | 0.0307 [0.0264, 0.0363] | 0.0559 | 0.47: 0.3394 [0.3307, 0.3487] | 0.48: 0.3390 [0.3302, 0.3485] | 0.884 / 1.27 |
| yolo26_ens | 0.0228 [0.0190, 0.0279] | 0.0232 [0.0205, 0.0288] | 0.0779 | 0.44: 0.3233 [0.3148, 0.3323] | 0.46: 0.3225 [0.3145, 0.3314] | 0.889 / 1.12 |
| rtdetr_ens | 0.0817 [0.0748, 0.0890] | 0.0817 [0.0748, 0.0890] | 0.0404 | 0.68: 0.3370 [0.3283, 0.3461] | 0.70: 0.3354 [0.3257, 0.3442] | 0.878 / 1.13 |

### RQ3: calibration metrics at the operating threshold

| detector | D-ECE | MA-ECE | miss mass | OCE | threshold D-ECE alone would choose | occluded vehicles: share of objects / share of D-ECE blind spot |
|---|---|---|---|---|---|---|
| yolo26 | 0.067 [0.062, 0.073] | 0.159 [0.152, 0.167] | 0.099 | 0.145 | 0.95 (recall 0.06) | 0.25 / 0.38 |
| rtdetr | 0.047 [0.042, 0.052] | 0.148 [0.142, 0.155] | 0.106 | 0.131 | 0.95 (recall 0.25) | 0.25 / 0.39 |
| yolov8 | 0.056 [0.050, 0.063] | 0.150 [0.143, 0.157] | 0.099 | 0.144 | 0.95 (recall 0.10) | 0.25 / 0.39 |
| yolo26_s1 | 0.072 [0.067, 0.078] | 0.164 [0.158, 0.172] | 0.099 | 0.147 | 0.95 (recall 0.05) | 0.25 / 0.39 |
| yolo26_s2 | 0.076 [0.071, 0.082] | 0.172 [0.165, 0.180] | 0.104 | 0.150 | 0.95 (recall 0.04) | 0.25 / 0.38 |
| rtdetr_s1 | 0.043 [0.038, 0.048] | 0.147 [0.141, 0.154] | 0.109 | 0.133 | 0.95 (recall 0.24) | 0.25 / 0.39 |
| rtdetr_s2 | 0.042 [0.038, 0.048] | 0.149 [0.142, 0.156] | 0.111 | 0.135 | 0.95 (recall 0.25) | 0.25 / 0.39 |
| yolo26_evid | 0.098 [0.092, 0.105] | 0.193 [0.186, 0.200] | 0.105 | 0.163 | 0.95 (recall 0.00) | 0.25 / 0.38 |
| yolo26_evid_s1 | 0.111 [0.105, 0.117] | 0.204 [0.197, 0.212] | 0.105 | 0.166 | 0.95 (recall 0.00) | 0.25 / 0.38 |
| yolo26_evid_s2 | 0.096 [0.089, 0.103] | 0.193 [0.186, 0.200] | 0.107 | 0.165 | 0.95 (recall 0.00) | 0.25 / 0.39 |
| yolo26_ens | 0.078 [0.072, 0.084] | 0.164 [0.157, 0.172] | 0.094 | 0.142 | 0.95 (recall 0.05) | 0.25 / 0.39 |
| rtdetr_ens | 0.040 [0.037, 0.046] | 0.136 [0.131, 0.143] | 0.099 | 0.123 | 0.8 (recall 0.78) | 0.25 / 0.39 |

### RQ5: why vehicles are missed (native classes, whatever the mode)

| detector | misses | confused | suppressed | unproposed | merged | visibility misses: odds per +0.1 occlusion | confusion misses: odds per +1 SD rarity | confusion misses: odds per +0.1 occlusion |
|---|---|---|---|---|---|---|---|---|
| yolo26 | 2594 | 0.31 | 0.53 | 0.06 | 0.00 | 1.182 [1.153, 1.210] | 2.522 [2.380, 2.672] | 1.014 [0.978, 1.052] |
| rtdetr | 2701 | 0.28 | 0.65 | 0.06 | 0.00 | 1.190 [1.166, 1.220] | 2.543 [2.408, 2.686] | 1.021 [0.982, 1.061] |
| yolov8 | 2639 | 0.32 | 0.51 | 0.06 | 0.00 | 1.179 [1.155, 1.208] | 2.542 [2.400, 2.672] | 1.017 [0.980, 1.052] |
| yolo26_s1 | 2636 | 0.32 | 0.53 | 0.06 | 0.00 | 1.193 [1.167, 1.223] | 2.551 [2.406, 2.699] | 1.018 [0.975, 1.056] |
| yolo26_s2 | 2685 | 0.31 | 0.53 | 0.06 | 0.00 | 1.179 [1.150, 1.210] | 2.538 [2.400, 2.689] | 0.997 [0.952, 1.035] |
| rtdetr_s1 | 2784 | 0.29 | 0.64 | 0.06 | 0.00 | 1.200 [1.174, 1.228] | 2.621 [2.476, 2.757] | 1.016 [0.979, 1.058] |
| rtdetr_s2 | 2820 | 0.29 | 0.65 | 0.06 | 0.00 | 1.195 [1.169, 1.221] | 2.564 [2.425, 2.709] | 1.008 [0.970, 1.047] |
| yolo26_evid | 2691 | 0.30 | 0.55 | 0.06 | 0.00 | 1.177 [1.150, 1.205] | 2.517 [2.369, 2.643] | 1.019 [0.979, 1.057] |
| yolo26_evid_s1 | 2712 | 0.31 | 0.55 | 0.06 | 0.00 | 1.174 [1.151, 1.205] | 2.588 [2.442, 2.740] | 1.033 [0.992, 1.069] |
| yolo26_evid_s2 | 2747 | 0.30 | 0.55 | 0.06 | 0.00 | 1.188 [1.164, 1.216] | 2.465 [2.330, 2.604] | 1.011 [0.970, 1.049] |
| yolo26_ens | 2571 | 0.35 | 0.52 | 0.05 | 0.00 | 1.198 [1.170, 1.230] | 2.554 [2.404, 2.702] | 1.015 [0.978, 1.054] |
| rtdetr_ens | 2644 | 0.32 | 0.61 | 0.06 | 0.00 | 1.194 [1.168, 1.225] | 2.617 [2.472, 2.764] | 1.013 [0.972, 1.059] |

### RQ4: transfer with the detector frozen (recall-side quantities only)

Each coverage is followed by (flags per image; certifiable). A recalibrated coverage that is not certifiable is the best the rule could do, not a guarantee.

| detector | dataset | matching | recall, development -> transfer | rule | target | coverage on development | coverage on transfer, frozen | coverage on transfer, recalibrated |
|---|---|---|---|---|---|---|---|---|
| yolo26 | pals | class_agnostic | 0.889 -> 0.838 | global_confidence | 0.90 | 0.902 (0.32; yes) | 0.859 (0.23; yes) | 0.927 (1.73; yes) |
| yolo26 | pals | class_agnostic | 0.889 -> 0.838 | mondrian_confidence | 0.90 | 0.903 (0.53; yes) | 0.841 (0.30; yes) | 0.927 (7.51; NO) |
| yolo26 | pals | class_agnostic | 0.889 -> 0.838 | global_occlusion_aware | 0.90 | 0.901 (0.38; yes) | 0.857 (0.25; yes) | 0.918 (1.59; yes) |
| rtdetr | pals | class_agnostic | 0.880 -> 0.683 | global_confidence | 0.90 | 0.902 (0.63; yes) | 0.740 (0.44; yes) | 0.924 (9.41; yes) |
| rtdetr | pals | class_agnostic | 0.880 -> 0.683 | mondrian_confidence | 0.90 | 0.903 (0.82; yes) | 0.702 (0.30; yes) | 0.924 (9.41; yes) |
| rtdetr | pals | class_agnostic | 0.880 -> 0.683 | global_occlusion_aware | 0.90 | 0.901 (0.64; yes) | 0.742 (0.47; yes) | 0.924 (7.64; yes) |
| yolov8 | pals | class_agnostic | 0.888 -> 0.820 | global_confidence | 0.90 | 0.899 (0.27; yes) | 0.837 (0.19; yes) | 0.924 (1.90; yes) |
| yolov8 | pals | class_agnostic | 0.888 -> 0.820 | mondrian_confidence | 0.90 | 0.900 (0.55; yes) | 0.822 (0.32; yes) | 0.924 (7.34; NO) |
| yolov8 | pals | class_agnostic | 0.888 -> 0.820 | global_occlusion_aware | 0.90 | 0.898 (0.34; yes) | 0.834 (0.24; yes) | 0.923 (1.83; yes) |
| yolo26_s1 | pals | class_agnostic | 0.889 -> 0.820 | global_confidence | 0.90 | 0.902 (0.37; yes) | 0.840 (0.25; yes) | 0.919 (1.72; yes) |
| yolo26_s1 | pals | class_agnostic | 0.889 -> 0.820 | mondrian_confidence | 0.90 | 0.901 (0.58; yes) | 0.820 (0.35; yes) | 0.922 (7.40; NO) |
| yolo26_s1 | pals | class_agnostic | 0.889 -> 0.820 | global_occlusion_aware | 0.90 | 0.901 (0.40; yes) | 0.837 (0.27; yes) | 0.910 (1.35; yes) |
| yolo26_s2 | pals | class_agnostic | 0.885 -> 0.846 | global_confidence | 0.90 | 0.899 (0.37; yes) | 0.864 (0.22; yes) | 0.921 (1.59; yes) |
| yolo26_s2 | pals | class_agnostic | 0.885 -> 0.846 | mondrian_confidence | 0.90 | 0.900 (0.50; yes) | 0.855 (0.29; yes) | 0.931 (6.84; NO) |
| yolo26_s2 | pals | class_agnostic | 0.885 -> 0.846 | global_occlusion_aware | 0.90 | 0.900 (0.45; yes) | 0.861 (0.27; yes) | 0.918 (1.34; yes) |
| rtdetr_s1 | pals | class_agnostic | 0.878 -> 0.637 | global_confidence | 0.90 | 0.902 (0.72; yes) | 0.694 (0.47; yes) | 0.903 (10.72; yes) |
| rtdetr_s1 | pals | class_agnostic | 0.878 -> 0.637 | mondrian_confidence | 0.90 | 0.903 (0.87; yes) | 0.656 (0.35; yes) | 0.903 (10.72; yes) |
| rtdetr_s1 | pals | class_agnostic | 0.878 -> 0.637 | global_occlusion_aware | 0.90 | 0.901 (0.73; yes) | 0.693 (0.50; yes) | 0.907 (8.41; yes) |
| rtdetr_s2 | pals | class_agnostic | 0.875 -> 0.682 | global_confidence | 0.90 | 0.904 (0.75; yes) | 0.749 (0.48; yes) | 0.904 (6.43; yes) |
| rtdetr_s2 | pals | class_agnostic | 0.875 -> 0.682 | mondrian_confidence | 0.90 | 0.902 (0.83; yes) | 0.713 (0.31; yes) | 0.904 (6.43; yes) |
| rtdetr_s2 | pals | class_agnostic | 0.875 -> 0.682 | global_occlusion_aware | 0.90 | 0.904 (0.74; yes) | 0.754 (0.52; yes) | 0.903 (5.43; yes) |
| yolo26_evid | pals | class_agnostic | 0.882 -> 0.799 | global_confidence | 0.90 | 0.902 (0.47; yes) | 0.838 (0.37; yes) | 0.921 (1.73; yes) |
| yolo26_evid | pals | class_agnostic | 0.882 -> 0.799 | mondrian_confidence | 0.90 | 0.902 (0.69; yes) | 0.810 (0.41; yes) | 0.931 (36.99; NO) |
| yolo26_evid | pals | class_agnostic | 0.882 -> 0.799 | global_occlusion_aware | 0.90 | 0.902 (0.58; yes) | 0.843 (0.45; yes) | 0.923 (1.87; yes) |
| yolo26_evid_s1 | pals | class_agnostic | 0.883 -> 0.817 | global_confidence | 0.90 | 0.898 (0.38; yes) | 0.838 (0.26; yes) | 0.904 (1.51; yes) |
| yolo26_evid_s1 | pals | class_agnostic | 0.883 -> 0.817 | mondrian_confidence | 0.90 | 0.896 (0.53; yes) | 0.818 (0.30; yes) | 0.911 (38.23; NO) |
| yolo26_evid_s1 | pals | class_agnostic | 0.883 -> 0.817 | global_occlusion_aware | 0.90 | 0.897 (0.43; yes) | 0.835 (0.31; yes) | 0.904 (1.54; yes) |
| yolo26_evid_s2 | pals | class_agnostic | 0.880 -> 0.807 | global_confidence | 0.90 | 0.900 (0.49; yes) | 0.838 (0.33; yes) | 0.911 (2.03; yes) |
| yolo26_evid_s2 | pals | class_agnostic | 0.880 -> 0.807 | mondrian_confidence | 0.90 | 0.901 (0.69; yes) | 0.820 (0.38; yes) | 0.924 (37.15; NO) |
| yolo26_evid_s2 | pals | class_agnostic | 0.880 -> 0.807 | global_occlusion_aware | 0.90 | 0.899 (0.54; yes) | 0.828 (0.36; yes) | 0.914 (1.85; yes) |
| yolo26_ens | pals | class_agnostic | 0.896 -> 0.856 | global_confidence | 0.90 | 0.901 (0.12; yes) | 0.862 (0.06; yes) | 0.920 (1.17; yes) |
| yolo26_ens | pals | class_agnostic | 0.896 -> 0.856 | mondrian_confidence | 0.90 | 0.900 (0.41; yes) | 0.851 (0.24; yes) | 0.935 (10.11; NO) |
| yolo26_ens | pals | class_agnostic | 0.896 -> 0.856 | global_occlusion_aware | 0.90 | 0.900 (0.20; yes) | 0.860 (0.12; yes) | 0.911 (1.03; yes) |
| rtdetr_ens | pals | class_agnostic | 0.889 -> 0.701 | global_confidence | 0.90 | 0.904 (0.42; yes) | 0.737 (0.29; yes) | 0.907 (6.37; yes) |
| rtdetr_ens | pals | class_agnostic | 0.889 -> 0.701 | mondrian_confidence | 0.90 | 0.902 (0.65; yes) | 0.685 (0.21; yes) | 0.907 (6.37; yes) |
| rtdetr_ens | pals | class_agnostic | 0.889 -> 0.701 | global_occlusion_aware | 0.90 | 0.903 (0.46; yes) | 0.747 (0.34; yes) | 0.903 (4.05; yes) |

### Detector ranking (best first)

| comparison | metric | ranking |
|---|---|---|
| each at its own threshold | dece | rtdetr_ens < rtdetr_s2 < rtdetr_s1 < rtdetr < yolov8 < yolo26 < yolo26_s1 < yolo26_s2 < yolo26_ens < yolo26_evid_s2 < yolo26_evid < yolo26_evid_s1 |
| each at its own threshold | ma_ece | rtdetr_ens < rtdetr_s1 < rtdetr < rtdetr_s2 < yolov8 < yolo26 < yolo26_ens < yolo26_s1 < yolo26_s2 < yolo26_evid_s2 < yolo26_evid < yolo26_evid_s1 |
| each at its own threshold | ma_brier | yolo26_ens < yolo26 < yolo26_s1 < yolo26_s2 < rtdetr_ens < yolov8 < rtdetr < yolo26_evid < rtdetr_s1 < yolo26_evid_s1 < rtdetr_s2 < yolo26_evid_s2 |
| each at its own threshold | oce_max | rtdetr_ens < rtdetr < rtdetr_s1 < rtdetr_s2 < yolo26_ens < yolov8 < yolo26 < yolo26_s1 < yolo26_s2 < yolo26_evid < yolo26_evid_s2 < yolo26_evid_s1 |
| at matched false alarms per image | dece | rtdetr_ens < rtdetr_s1 < rtdetr_s2 < rtdetr < yolov8 < yolo26 < yolo26_s1 < yolo26_s2 < yolo26_ens < yolo26_evid_s2 < yolo26_evid < yolo26_evid_s1 |
| at matched false alarms per image | ma_ece | rtdetr_ens < rtdetr_s2 < rtdetr_s1 < rtdetr < yolov8 < yolo26 < yolo26_ens < yolo26_s1 < yolo26_s2 < yolo26_evid < yolo26_evid_s2 < yolo26_evid_s1 |
| at matched false alarms per image | ma_brier | yolo26_ens < yolo26_s2 < yolo26 < yolo26_s1 < rtdetr_ens < yolov8 < yolo26_evid_s1 < yolo26_evid < rtdetr_s2 < rtdetr_s1 < yolo26_evid_s2 < rtdetr |
| at matched false alarms per image | oce_max | rtdetr_ens < rtdetr_s2 < rtdetr_s1 < rtdetr < yolo26_ens < yolo26 < yolo26_s2 < yolo26_s1 < yolov8 < yolo26_evid < yolo26_evid_s1 < yolo26_evid_s2 |

Point-estimate orders above can differ by noise. Below, a metric *separates* a pair only if its 95% paired-difference interval excludes zero; a *flip* needs both D-ECE and the other metric to separate the pair, in opposite directions.

**each at its own threshold**

| pair (difference = first minus second) | D-ECE | MA-ECE | MA-Brier | OCE | flips vs D-ECE |
|---|---|---|---|---|---|
| rtdetr vs rtdetr_ens | rtdetr_ens better [0.0006, 0.0092] | rtdetr_ens better [0.0067, 0.0160] | rtdetr_ens better [0.0080, 0.0123] | rtdetr_ens better [0.0054, 0.0098] | none |
| rtdetr vs rtdetr_s1 | rtdetr_s1 better [0.0006, 0.0076] | cannot separate [-0.0035, 0.0056] | cannot separate [-0.0034, 0.0029] | cannot separate [-0.0051, 0.0009] | none |
| rtdetr vs rtdetr_s2 | rtdetr_s2 better [0.0009, 0.0080] | cannot separate [-0.0049, 0.0040] | cannot separate [-0.0032, 0.0023] | rtdetr better [-0.0070, -0.0016] | oce_max |
| rtdetr vs yolo26 | rtdetr better [-0.0289, -0.0114] | rtdetr better [-0.0199, -0.0021] | yolo26 better [0.0082, 0.0165] | rtdetr better [-0.0183, -0.0101] | ma_brier |
| rtdetr vs yolo26_ens | rtdetr better [-0.0397, -0.0222] | rtdetr better [-0.0246, -0.0074] | yolo26_ens better [0.0139, 0.0216] | rtdetr better [-0.0149, -0.0072] | ma_brier |
| rtdetr vs yolo26_evid | rtdetr better [-0.0612, -0.0419] | rtdetr better [-0.0536, -0.0360] | cannot separate [-0.0041, 0.0036] | rtdetr better [-0.0356, -0.0285] | none |
| rtdetr vs yolo26_evid_s1 | rtdetr better [-0.0734, -0.0553] | rtdetr better [-0.0645, -0.0468] | cannot separate [-0.0046, 0.0036] | rtdetr better [-0.0389, -0.0312] | none |
| rtdetr vs yolo26_evid_s2 | rtdetr better [-0.0587, -0.0401] | rtdetr better [-0.0532, -0.0358] | cannot separate [-0.0057, 0.0017] | rtdetr better [-0.0373, -0.0299] | none |
| rtdetr vs yolo26_s1 | rtdetr better [-0.0335, -0.0170] | rtdetr better [-0.0248, -0.0074] | yolo26_s1 better [0.0075, 0.0155] | rtdetr better [-0.0202, -0.0124] | ma_brier |
| rtdetr vs yolo26_s2 | rtdetr better [-0.0379, -0.0218] | rtdetr better [-0.0328, -0.0156] | yolo26_s2 better [0.0059, 0.0142] | rtdetr better [-0.0233, -0.0152] | ma_brier |
| rtdetr vs yolov8 | rtdetr better [-0.0179, -0.0004] | cannot separate [-0.0099, 0.0076] | yolov8 better [0.0063, 0.0135] | rtdetr better [-0.0169, -0.0095] | ma_brier |
| rtdetr_ens vs rtdetr_s1 | cannot separate [-0.0054, 0.0041] | rtdetr_ens better [-0.0153, -0.0051] | rtdetr_ens better [-0.0125, -0.0080] | rtdetr_ens better [-0.0120, -0.0073] | none |
| rtdetr_ens vs rtdetr_s2 | cannot separate [-0.0051, 0.0044] | rtdetr_ens better [-0.0170, -0.0069] | rtdetr_ens better [-0.0129, -0.0085] | rtdetr_ens better [-0.0142, -0.0097] | none |
| rtdetr_ens vs yolo26 | rtdetr_ens better [-0.0323, -0.0185] | rtdetr_ens better [-0.0298, -0.0154] | cannot separate [-0.0014, 0.0061] | rtdetr_ens better [-0.0250, -0.0178] | none |
| rtdetr_ens vs yolo26_ens | rtdetr_ens better [-0.0433, -0.0297] | rtdetr_ens better [-0.0349, -0.0210] | yolo26_ens better [0.0039, 0.0112] | rtdetr_ens better [-0.0219, -0.0150] | ma_brier |
| rtdetr_ens vs yolo26_evid | rtdetr_ens better [-0.0642, -0.0487] | rtdetr_ens better [-0.0634, -0.0494] | rtdetr_ens better [-0.0137, -0.0067] | rtdetr_ens better [-0.0428, -0.0364] | none |
| rtdetr_ens vs yolo26_evid_s1 | rtdetr_ens better [-0.0765, -0.0624] | rtdetr_ens better [-0.0745, -0.0602] | rtdetr_ens better [-0.0145, -0.0068] | rtdetr_ens better [-0.0462, -0.0390] | none |
| rtdetr_ens vs yolo26_evid_s2 | rtdetr_ens better [-0.0617, -0.0472] | rtdetr_ens better [-0.0629, -0.0492] | rtdetr_ens better [-0.0154, -0.0086] | rtdetr_ens better [-0.0444, -0.0379] | none |
| rtdetr_ens vs yolo26_s1 | rtdetr_ens better [-0.0371, -0.0239] | rtdetr_ens better [-0.0357, -0.0209] | cannot separate [-0.0027, 0.0053] | rtdetr_ens better [-0.0273, -0.0204] | none |
| rtdetr_ens vs yolo26_s2 | rtdetr_ens better [-0.0418, -0.0288] | rtdetr_ens better [-0.0430, -0.0288] | cannot separate [-0.0039, 0.0038] | rtdetr_ens better [-0.0310, -0.0226] | none |
| rtdetr_ens vs yolov8 | rtdetr_ens better [-0.0216, -0.0074] | rtdetr_ens better [-0.0198, -0.0056] | cannot separate [-0.0037, 0.0033] | rtdetr_ens better [-0.0242, -0.0174] | none |
| rtdetr_s1 vs rtdetr_s2 | cannot separate [-0.0031, 0.0042] | cannot separate [-0.0063, 0.0030] | cannot separate [-0.0032, 0.0028] | cannot separate [-0.0052, 0.0008] | none |
| rtdetr_s1 vs yolo26 | rtdetr_s1 better [-0.0334, -0.0156] | rtdetr_s1 better [-0.0205, -0.0033] | yolo26 better [0.0084, 0.0166] | rtdetr_s1 better [-0.0158, -0.0080] | ma_brier |
| rtdetr_s1 vs yolo26_ens | rtdetr_s1 better [-0.0444, -0.0262] | rtdetr_s1 better [-0.0256, -0.0082] | yolo26_ens better [0.0141, 0.0219] | rtdetr_s1 better [-0.0127, -0.0051] | ma_brier |
| rtdetr_s1 vs yolo26_evid | rtdetr_s1 better [-0.0650, -0.0458] | rtdetr_s1 better [-0.0546, -0.0369] | cannot separate [-0.0037, 0.0039] | rtdetr_s1 better [-0.0335, -0.0265] | none |
| rtdetr_s1 vs yolo26_evid_s1 | rtdetr_s1 better [-0.0779, -0.0589] | rtdetr_s1 better [-0.0657, -0.0478] | cannot separate [-0.0041, 0.0038] | rtdetr_s1 better [-0.0367, -0.0291] | none |
| rtdetr_s1 vs yolo26_evid_s2 | rtdetr_s1 better [-0.0631, -0.0442] | rtdetr_s1 better [-0.0540, -0.0364] | cannot separate [-0.0051, 0.0016] | rtdetr_s1 better [-0.0348, -0.0282] | none |
| rtdetr_s1 vs yolo26_s1 | rtdetr_s1 better [-0.0380, -0.0205] | rtdetr_s1 better [-0.0260, -0.0081] | yolo26_s1 better [0.0078, 0.0156] | rtdetr_s1 better [-0.0183, -0.0102] | ma_brier |
| rtdetr_s1 vs yolo26_s2 | rtdetr_s1 better [-0.0427, -0.0256] | rtdetr_s1 better [-0.0341, -0.0161] | yolo26_s2 better [0.0064, 0.0146] | rtdetr_s1 better [-0.0212, -0.0130] | ma_brier |
| rtdetr_s1 vs yolov8 | rtdetr_s1 better [-0.0226, -0.0038] | cannot separate [-0.0109, 0.0065] | yolov8 better [0.0066, 0.0138] | rtdetr_s1 better [-0.0148, -0.0078] | ma_brier |
| rtdetr_s2 vs yolo26 | rtdetr_s2 better [-0.0332, -0.0160] | rtdetr_s2 better [-0.0185, -0.0019] | yolo26 better [0.0090, 0.0169] | rtdetr_s2 better [-0.0136, -0.0056] | ma_brier |
| rtdetr_s2 vs yolo26_ens | rtdetr_s2 better [-0.0443, -0.0267] | rtdetr_s2 better [-0.0238, -0.0071] | yolo26_ens better [0.0146, 0.0220] | rtdetr_s2 better [-0.0103, -0.0029] | ma_brier |
| rtdetr_s2 vs yolo26_evid | rtdetr_s2 better [-0.0655, -0.0463] | rtdetr_s2 better [-0.0523, -0.0356] | cannot separate [-0.0035, 0.0041] | rtdetr_s2 better [-0.0311, -0.0242] | none |
| rtdetr_s2 vs yolo26_evid_s1 | rtdetr_s2 better [-0.0775, -0.0595] | rtdetr_s2 better [-0.0640, -0.0468] | cannot separate [-0.0038, 0.0040] | rtdetr_s2 better [-0.0347, -0.0273] | none |
| rtdetr_s2 vs yolo26_evid_s2 | rtdetr_s2 better [-0.0632, -0.0446] | rtdetr_s2 better [-0.0525, -0.0354] | cannot separate [-0.0049, 0.0017] | rtdetr_s2 better [-0.0329, -0.0259] | none |
| rtdetr_s2 vs yolo26_s1 | rtdetr_s2 better [-0.0383, -0.0213] | rtdetr_s2 better [-0.0244, -0.0073] | yolo26_s1 better [0.0082, 0.0161] | rtdetr_s2 better [-0.0158, -0.0079] | ma_brier |
| rtdetr_s2 vs yolo26_s2 | rtdetr_s2 better [-0.0427, -0.0261] | rtdetr_s2 better [-0.0323, -0.0150] | yolo26_s2 better [0.0065, 0.0147] | rtdetr_s2 better [-0.0191, -0.0108] | ma_brier |
| rtdetr_s2 vs yolov8 | rtdetr_s2 better [-0.0230, -0.0047] | cannot separate [-0.0094, 0.0077] | yolov8 better [0.0072, 0.0142] | rtdetr_s2 better [-0.0123, -0.0052] | ma_brier |
| yolo26 vs yolo26_ens | yolo26 better [-0.0129, -0.0091] | yolo26 better [-0.0070, -0.0029] | yolo26_ens better [0.0036, 0.0066] | yolo26_ens better [0.0012, 0.0045] | ma_brier, oce_max |
| yolo26 vs yolo26_evid | yolo26 better [-0.0352, -0.0277] | yolo26 better [-0.0385, -0.0294] | yolo26 better [-0.0155, -0.0102] | yolo26 better [-0.0210, -0.0152] | none |
| yolo26 vs yolo26_evid_s1 | yolo26 better [-0.0466, -0.0412] | yolo26 better [-0.0483, -0.0421] | yolo26 better [-0.0152, -0.0109] | yolo26 better [-0.0235, -0.0188] | none |
| yolo26 vs yolo26_evid_s2 | yolo26 better [-0.0329, -0.0252] | yolo26 better [-0.0379, -0.0284] | yolo26 better [-0.0174, -0.0119] | yolo26 better [-0.0229, -0.0165] | none |
| yolo26 vs yolo26_s1 | yolo26 better [-0.0086, -0.0027] | yolo26 better [-0.0090, -0.0025] | cannot separate [-0.0031, 0.0007] | yolo26 better [-0.0045, -0.0003] | none |
| yolo26 vs yolo26_s2 | yolo26 better [-0.0123, -0.0075] | yolo26 better [-0.0162, -0.0104] | yolo26 better [-0.0046, -0.0006] | yolo26 better [-0.0075, -0.0028] | none |
| yolo26 vs yolov8 | yolov8 better [0.0072, 0.0146] | yolov8 better [0.0054, 0.0134] | cannot separate [-0.0053, 0.0001] | cannot separate [-0.0023, 0.0033] | none |
| yolo26_ens vs yolo26_evid | yolo26_ens better [-0.0239, -0.0168] | yolo26_ens better [-0.0329, -0.0245] | yolo26_ens better [-0.0206, -0.0156] | yolo26_ens better [-0.0241, -0.0184] | none |
| yolo26_ens vs yolo26_evid_s1 | yolo26_ens better [-0.0351, -0.0306] | yolo26_ens better [-0.0428, -0.0373] | yolo26_ens better [-0.0202, -0.0163] | yolo26_ens better [-0.0262, -0.0220] | none |
| yolo26_ens vs yolo26_evid_s2 | yolo26_ens better [-0.0216, -0.0141] | yolo26_ens better [-0.0322, -0.0240] | yolo26_ens better [-0.0226, -0.0172] | yolo26_ens better [-0.0258, -0.0200] | none |
| yolo26_ens vs yolo26_s1 | yolo26_s1 better [0.0029, 0.0080] | cannot separate [-0.0033, 0.0021] | yolo26_ens better [-0.0078, -0.0049] | yolo26_ens better [-0.0069, -0.0036] | ma_brier, oce_max |
| yolo26_ens vs yolo26_s2 | cannot separate [-0.0010, 0.0032] | yolo26_ens better [-0.0106, -0.0059] | yolo26_ens better [-0.0092, -0.0063] | yolo26_ens better [-0.0099, -0.0067] | none |
| yolo26_ens vs yolov8 | yolov8 better [0.0182, 0.0253] | yolov8 better [0.0106, 0.0183] | yolo26_ens better [-0.0104, -0.0053] | cannot separate [-0.0050, 0.0002] | ma_brier |
| yolo26_evid vs yolo26_evid_s1 | yolo26_evid better [-0.0158, -0.0092] | yolo26_evid better [-0.0147, -0.0073] | cannot separate [-0.0027, 0.0021] | yolo26_evid better [-0.0055, -0.0001] | none |
| yolo26_evid vs yolo26_evid_s2 | cannot separate [-0.0003, 0.0046] | cannot separate [-0.0021, 0.0033] | cannot separate [-0.0041, 0.0005] | cannot separate [-0.0040, 0.0010] | none |
| yolo26_evid vs yolo26_s1 | yolo26_s1 better [0.0224, 0.0298] | yolo26_s1 better [0.0239, 0.0330] | yolo26_s1 better [0.0092, 0.0141] | yolo26_s1 better [0.0130, 0.0186] | none |
| yolo26_evid vs yolo26_s2 | yolo26_s2 better [0.0176, 0.0253] | yolo26_s2 better [0.0157, 0.0251] | yolo26_s2 better [0.0077, 0.0130] | yolo26_s2 better [0.0101, 0.0159] | none |
| yolo26_evid vs yolov8 | yolov8 better [0.0391, 0.0455] | yolov8 better [0.0398, 0.0470] | yolov8 better [0.0073, 0.0127] | yolov8 better [0.0162, 0.0217] | none |
| yolo26_evid_s1 vs yolo26_evid_s2 | yolo26_evid_s2 better [0.0112, 0.0183] | yolo26_evid_s2 better [0.0080, 0.0156] | cannot separate [-0.0042, 0.0010] | cannot separate [-0.0014, 0.0040] | none |
| yolo26_evid_s1 vs yolo26_s1 | yolo26_s1 better [0.0351, 0.0414] | yolo26_s1 better [0.0360, 0.0428] | yolo26_s1 better [0.0099, 0.0139] | yolo26_s1 better [0.0166, 0.0209] | none |
| yolo26_evid_s1 vs yolo26_s2 | yolo26_s2 better [0.0312, 0.0367] | yolo26_s2 better [0.0284, 0.0352] | yolo26_s2 better [0.0085, 0.0129] | yolo26_s2 better [0.0135, 0.0185] | none |
| yolo26_evid_s1 vs yolov8 | yolov8 better [0.0512, 0.0579] | yolov8 better [0.0505, 0.0578] | yolov8 better [0.0079, 0.0132] | yolov8 better [0.0192, 0.0247] | none |
| yolo26_evid_s2 vs yolo26_s1 | yolo26_s1 better [0.0198, 0.0278] | yolo26_s1 better [0.0232, 0.0324] | yolo26_s1 better [0.0111, 0.0162] | yolo26_s1 better [0.0146, 0.0205] | none |
| yolo26_evid_s2 vs yolo26_s2 | yolo26_s2 better [0.0151, 0.0229] | yolo26_s2 better [0.0155, 0.0247] | yolo26_s2 better [0.0093, 0.0154] | yolo26_s2 better [0.0115, 0.0179] | none |
| yolo26_evid_s2 vs yolov8 | yolov8 better [0.0369, 0.0433] | yolov8 better [0.0387, 0.0464] | yolov8 better [0.0089, 0.0146] | yolov8 better [0.0171, 0.0233] | none |
| yolo26_s1 vs yolo26_s2 | yolo26_s1 better [-0.0077, -0.0018] | yolo26_s1 better [-0.0113, -0.0043] | cannot separate [-0.0034, 0.0006] | yolo26_s1 better [-0.0052, -0.0006] | none |
| yolo26_s1 vs yolov8 | yolov8 better [0.0123, 0.0199] | yolov8 better [0.0104, 0.0190] | cannot separate [-0.0041, 0.0010] | yolov8 better [0.0003, 0.0058] | none |
| yolo26_s2 vs yolov8 | yolov8 better [0.0170, 0.0242] | yolov8 better [0.0181, 0.0272] | cannot separate [-0.0030, 0.0029] | yolov8 better [0.0028, 0.0092] | none |

- D-ECE cannot separate rtdetr_ens and rtdetr_s1 (each at its own threshold).
- D-ECE cannot separate rtdetr_ens and rtdetr_s2 (each at its own threshold).
- D-ECE cannot separate rtdetr_s1 and rtdetr_s2 (each at its own threshold).
- D-ECE cannot separate yolo26_ens and yolo26_s2 (each at its own threshold).
- D-ECE cannot separate yolo26_evid and yolo26_evid_s2 (each at its own threshold).

**at matched false alarms per image** (budget 1.25 false alarms/image)

| pair (difference = first minus second) | D-ECE | MA-ECE | MA-Brier | OCE | flips vs D-ECE |
|---|---|---|---|---|---|
| rtdetr vs rtdetr_ens | cannot separate [-0.0017, 0.0067] | rtdetr_ens better [0.0094, 0.0192] | rtdetr_ens better [0.0104, 0.0148] | rtdetr_ens better [0.0100, 0.0148] | none |
| rtdetr vs rtdetr_s1 | cannot separate [-0.0026, 0.0046] | cannot separate [-0.0018, 0.0081] | cannot separate [-0.0013, 0.0050] | cannot separate [-0.0007, 0.0053] | none |
| rtdetr vs rtdetr_s2 | cannot separate [-0.0028, 0.0037] | cannot separate [-0.0005, 0.0081] | rtdetr_s2 better [0.0005, 0.0059] | rtdetr_s2 better [0.0003, 0.0060] | none |
| rtdetr vs yolo26 | rtdetr better [-0.0326, -0.0152] | cannot separate [-0.0168, 0.0004] | yolo26 better [0.0105, 0.0194] | rtdetr better [-0.0125, -0.0041] | ma_brier |
| rtdetr vs yolo26_ens | rtdetr better [-0.0434, -0.0259] | rtdetr better [-0.0173, -0.0004] | yolo26_ens better [0.0190, 0.0273] | cannot separate [-0.0061, 0.0021] | ma_brier |
| rtdetr vs yolo26_evid | rtdetr better [-0.0647, -0.0456] | rtdetr better [-0.0480, -0.0309] | yolo26_evid better [0.0003, 0.0080] | rtdetr better [-0.0279, -0.0209] | ma_brier |
| rtdetr vs yolo26_evid_s1 | rtdetr better [-0.0760, -0.0581] | rtdetr better [-0.0561, -0.0387] | yolo26_evid_s1 better [0.0008, 0.0096] | rtdetr better [-0.0298, -0.0220] | ma_brier |
| rtdetr vs yolo26_evid_s2 | rtdetr better [-0.0627, -0.0441] | rtdetr better [-0.0505, -0.0334] | cannot separate [-0.0036, 0.0042] | rtdetr better [-0.0318, -0.0244] | none |
| rtdetr vs yolo26_s1 | rtdetr better [-0.0372, -0.0209] | rtdetr better [-0.0212, -0.0043] | yolo26_s1 better [0.0104, 0.0191] | rtdetr better [-0.0138, -0.0061] | ma_brier |
| rtdetr vs yolo26_s2 | rtdetr better [-0.0407, -0.0245] | rtdetr better [-0.0218, -0.0051] | yolo26_s2 better [0.0129, 0.0215] | rtdetr better [-0.0125, -0.0045] | ma_brier |
| rtdetr vs yolov8 | rtdetr better [-0.0221, -0.0045] | cannot separate [-0.0116, 0.0058] | yolov8 better [0.0059, 0.0135] | rtdetr better [-0.0146, -0.0067] | ma_brier |
| rtdetr_ens vs rtdetr_s1 | cannot separate [-0.0059, 0.0032] | rtdetr_ens better [-0.0159, -0.0059] | rtdetr_ens better [-0.0132, -0.0086] | rtdetr_ens better [-0.0124, -0.0078] | none |
| rtdetr_ens vs rtdetr_s2 | cannot separate [-0.0066, 0.0033] | rtdetr_ens better [-0.0156, -0.0052] | rtdetr_ens better [-0.0114, -0.0073] | rtdetr_ens better [-0.0116, -0.0071] | none |
| rtdetr_ens vs yolo26 | rtdetr_ens better [-0.0338, -0.0202] | rtdetr_ens better [-0.0296, -0.0158] | cannot separate [-0.0013, 0.0061] | rtdetr_ens better [-0.0240, -0.0168] | none |
| rtdetr_ens vs yolo26_ens | rtdetr_ens better [-0.0445, -0.0307] | rtdetr_ens better [-0.0302, -0.0164] | yolo26_ens better [0.0069, 0.0142] | rtdetr_ens better [-0.0178, -0.0106] | ma_brier |
| rtdetr_ens vs yolo26_evid | rtdetr_ens better [-0.0656, -0.0504] | rtdetr_ens better [-0.0612, -0.0472] | rtdetr_ens better [-0.0118, -0.0049] | rtdetr_ens better [-0.0397, -0.0335] | none |
| rtdetr_ens vs yolo26_evid_s1 | rtdetr_ens better [-0.0771, -0.0632] | rtdetr_ens better [-0.0688, -0.0551] | rtdetr_ens better [-0.0110, -0.0037] | rtdetr_ens better [-0.0416, -0.0345] | none |
| rtdetr_ens vs yolo26_evid_s2 | rtdetr_ens better [-0.0634, -0.0487] | rtdetr_ens better [-0.0636, -0.0496] | rtdetr_ens better [-0.0156, -0.0089] | rtdetr_ens better [-0.0438, -0.0372] | none |
| rtdetr_ens vs yolo26_s1 | rtdetr_ens better [-0.0385, -0.0256] | rtdetr_ens better [-0.0351, -0.0201] | cannot separate [-0.0018, 0.0058] | rtdetr_ens better [-0.0259, -0.0186] | none |
| rtdetr_ens vs yolo26_s2 | rtdetr_ens better [-0.0423, -0.0289] | rtdetr_ens better [-0.0351, -0.0213] | yolo26_s2 better [0.0009, 0.0085] | rtdetr_ens better [-0.0245, -0.0168] | ma_brier |
| rtdetr_ens vs yolov8 | rtdetr_ens better [-0.0232, -0.0092] | rtdetr_ens better [-0.0249, -0.0109] | cannot separate [-0.0062, 0.0006] | rtdetr_ens better [-0.0265, -0.0195] | none |
| rtdetr_s1 vs rtdetr_s2 | cannot separate [-0.0040, 0.0031] | cannot separate [-0.0039, 0.0053] | cannot separate [-0.0013, 0.0047] | cannot separate [-0.0022, 0.0038] | none |
| rtdetr_s1 vs yolo26 | rtdetr_s1 better [-0.0341, -0.0165] | rtdetr_s1 better [-0.0198, -0.0026] | yolo26 better [0.0091, 0.0172] | rtdetr_s1 better [-0.0145, -0.0065] | ma_brier |
| rtdetr_s1 vs yolo26_ens | rtdetr_s1 better [-0.0449, -0.0271] | rtdetr_s1 better [-0.0203, -0.0032] | yolo26_ens better [0.0176, 0.0251] | rtdetr_s1 better [-0.0079, -0.0009] | ma_brier |
| rtdetr_s1 vs yolo26_evid | rtdetr_s1 better [-0.0656, -0.0467] | rtdetr_s1 better [-0.0514, -0.0337] | cannot separate [-0.0014, 0.0061] | rtdetr_s1 better [-0.0301, -0.0233] | none |
| rtdetr_s1 vs yolo26_evid_s1 | rtdetr_s1 better [-0.0777, -0.0588] | rtdetr_s1 better [-0.0596, -0.0418] | cannot separate [-0.0002, 0.0074] | rtdetr_s1 better [-0.0319, -0.0245] | none |
| rtdetr_s1 vs yolo26_evid_s2 | rtdetr_s1 better [-0.0640, -0.0452] | rtdetr_s1 better [-0.0542, -0.0363] | cannot separate [-0.0048, 0.0020] | rtdetr_s1 better [-0.0336, -0.0272] | none |
| rtdetr_s1 vs yolo26_s1 | rtdetr_s1 better [-0.0387, -0.0216] | rtdetr_s1 better [-0.0247, -0.0071] | yolo26_s1 better [0.0090, 0.0167] | rtdetr_s1 better [-0.0162, -0.0083] | ma_brier |
| rtdetr_s1 vs yolo26_s2 | rtdetr_s1 better [-0.0425, -0.0256] | rtdetr_s1 better [-0.0253, -0.0086] | yolo26_s2 better [0.0115, 0.0196] | rtdetr_s1 better [-0.0147, -0.0068] | ma_brier |
| rtdetr_s1 vs yolov8 | rtdetr_s1 better [-0.0236, -0.0051] | cannot separate [-0.0150, 0.0025] | yolov8 better [0.0044, 0.0117] | rtdetr_s1 better [-0.0167, -0.0096] | ma_brier |
| rtdetr_s2 vs yolo26 | rtdetr_s2 better [-0.0331, -0.0158] | rtdetr_s2 better [-0.0201, -0.0033] | yolo26 better [0.0080, 0.0160] | rtdetr_s2 better [-0.0152, -0.0071] | ma_brier |
| rtdetr_s2 vs yolo26_ens | rtdetr_s2 better [-0.0438, -0.0263] | rtdetr_s2 better [-0.0211, -0.0038] | yolo26_ens better [0.0163, 0.0238] | rtdetr_s2 better [-0.0084, -0.0011] | ma_brier |
| rtdetr_s2 vs yolo26_evid | rtdetr_s2 better [-0.0653, -0.0459] | rtdetr_s2 better [-0.0517, -0.0345] | cannot separate [-0.0026, 0.0045] | rtdetr_s2 better [-0.0307, -0.0239] | none |
| rtdetr_s2 vs yolo26_evid_s1 | rtdetr_s2 better [-0.0767, -0.0588] | rtdetr_s2 better [-0.0603, -0.0429] | cannot separate [-0.0016, 0.0060] | rtdetr_s2 better [-0.0324, -0.0251] | none |
| rtdetr_s2 vs yolo26_evid_s2 | rtdetr_s2 better [-0.0633, -0.0446] | rtdetr_s2 better [-0.0546, -0.0371] | cannot separate [-0.0064, 0.0003] | rtdetr_s2 better [-0.0347, -0.0276] | none |
| rtdetr_s2 vs yolo26_s1 | rtdetr_s2 better [-0.0382, -0.0213] | rtdetr_s2 better [-0.0252, -0.0078] | yolo26_s1 better [0.0077, 0.0156] | rtdetr_s2 better [-0.0166, -0.0087] | ma_brier |
| rtdetr_s2 vs yolo26_s2 | rtdetr_s2 better [-0.0414, -0.0250] | rtdetr_s2 better [-0.0259, -0.0086] | yolo26_s2 better [0.0100, 0.0181] | rtdetr_s2 better [-0.0152, -0.0073] | ma_brier |
| rtdetr_s2 vs yolov8 | rtdetr_s2 better [-0.0232, -0.0051] | cannot separate [-0.0159, 0.0020] | yolov8 better [0.0033, 0.0101] | rtdetr_s2 better [-0.0170, -0.0102] | ma_brier |
| yolo26 vs yolo26_ens | yolo26 better [-0.0125, -0.0087] | cannot separate [-0.0025, 0.0016] | yolo26_ens better [0.0065, 0.0095] | yolo26_ens better [0.0045, 0.0078] | ma_brier, oce_max |
| yolo26 vs yolo26_evid | yolo26 better [-0.0352, -0.0275] | yolo26 better [-0.0361, -0.0268] | yolo26 better [-0.0139, -0.0084] | yolo26 better [-0.0192, -0.0136] | none |
| yolo26 vs yolo26_evid_s1 | yolo26 better [-0.0457, -0.0403] | yolo26 better [-0.0429, -0.0365] | yolo26 better [-0.0121, -0.0078] | yolo26 better [-0.0201, -0.0154] | none |
| yolo26 vs yolo26_evid_s2 | yolo26 better [-0.0331, -0.0250] | yolo26 better [-0.0385, -0.0290] | yolo26 better [-0.0178, -0.0120] | yolo26 better [-0.0233, -0.0172] | none |
| yolo26 vs yolo26_s1 | yolo26 better [-0.0086, -0.0025] | yolo26 better [-0.0085, -0.0017] | cannot separate [-0.0027, 0.0013] | cannot separate [-0.0039, 0.0003] | none |
| yolo26 vs yolo26_s2 | yolo26 better [-0.0111, -0.0062] | yolo26 better [-0.0087, -0.0023] | cannot separate [-0.0001, 0.0039] | cannot separate [-0.0025, 0.0020] | none |
| yolo26 vs yolov8 | yolov8 better [0.0071, 0.0143] | yolov8 better [0.0007, 0.0086] | yolo26 better [-0.0083, -0.0026] | yolo26 better [-0.0057, -0.0001] | ma_brier, oce_max |
| yolo26_ens vs yolo26_evid | yolo26_ens better [-0.0243, -0.0169] | yolo26_ens better [-0.0349, -0.0265] | yolo26_ens better [-0.0215, -0.0166] | yolo26_ens better [-0.0252, -0.0196] | none |
| yolo26_ens vs yolo26_evid_s1 | yolo26_ens better [-0.0347, -0.0299] | yolo26_ens better [-0.0417, -0.0365] | yolo26_ens better [-0.0198, -0.0159] | yolo26_ens better [-0.0259, -0.0219] | none |
| yolo26_ens vs yolo26_evid_s2 | yolo26_ens better [-0.0221, -0.0149] | yolo26_ens better [-0.0376, -0.0288] | yolo26_ens better [-0.0256, -0.0202] | yolo26_ens better [-0.0292, -0.0236] | none |
| yolo26_ens vs yolo26_s1 | yolo26_s1 better [0.0024, 0.0078] | yolo26_ens better [-0.0070, -0.0017] | yolo26_ens better [-0.0101, -0.0071] | yolo26_ens better [-0.0097, -0.0063] | ma_ece, ma_brier, oce_max |
| yolo26_ens vs yolo26_s2 | cannot separate [-0.0006, 0.0039] | yolo26_ens better [-0.0079, -0.0027] | yolo26_ens better [-0.0075, -0.0047] | yolo26_ens better [-0.0080, -0.0048] | none |
| yolo26_ens vs yolov8 | yolov8 better [0.0178, 0.0246] | yolov8 better [0.0014, 0.0090] | yolo26_ens better [-0.0162, -0.0106] | yolo26_ens better [-0.0116, -0.0059] | ma_brier, oce_max |
| yolo26_evid vs yolo26_evid_s1 | yolo26_evid better [-0.0149, -0.0084] | yolo26_evid better [-0.0116, -0.0042] | cannot separate [-0.0011, 0.0037] | cannot separate [-0.0041, 0.0014] | none |
| yolo26_evid vs yolo26_evid_s2 | cannot separate [-0.0005, 0.0045] | cannot separate [-0.0052, 0.0002] | yolo26_evid better [-0.0061, -0.0015] | yolo26_evid better [-0.0063, -0.0013] | none |
| yolo26_evid vs yolo26_s1 | yolo26_s1 better [0.0221, 0.0299] | yolo26_s1 better [0.0221, 0.0315] | yolo26_s1 better [0.0080, 0.0129] | yolo26_s1 better [0.0116, 0.0174] | none |
| yolo26_evid vs yolo26_s2 | yolo26_s2 better [0.0185, 0.0266] | yolo26_s2 better [0.0213, 0.0303] | yolo26_s2 better [0.0104, 0.0158] | yolo26_s2 better [0.0132, 0.0192] | none |
| yolo26_evid vs yolov8 | yolov8 better [0.0389, 0.0451] | yolov8 better [0.0326, 0.0393] | yolov8 better [0.0029, 0.0085] | yolov8 better [0.0106, 0.0164] | none |
| yolo26_evid_s1 vs yolo26_evid_s2 | yolo26_evid_s2 better [0.0102, 0.0172] | yolo26_evid_s2 better [0.0017, 0.0095] | yolo26_evid_s1 better [-0.0077, -0.0025] | cannot separate [-0.0054, 0.0000] | ma_brier |
| yolo26_evid_s1 vs yolo26_s1 | yolo26_s1 better [0.0340, 0.0406] | yolo26_s1 better [0.0312, 0.0382] | yolo26_s1 better [0.0072, 0.0113] | yolo26_s1 better [0.0136, 0.0179] | none |
| yolo26_evid_s1 vs yolo26_s2 | yolo26_s2 better [0.0314, 0.0368] | yolo26_s2 better [0.0304, 0.0375] | yolo26_s2 better [0.0096, 0.0141] | yolo26_s2 better [0.0150, 0.0199] | none |
| yolo26_evid_s1 vs yolov8 | yolov8 better [0.0502, 0.0565] | yolov8 better [0.0406, 0.0477] | yolov8 better [0.0018, 0.0071] | yolov8 better [0.0122, 0.0177] | none |
| yolo26_evid_s2 vs yolo26_s1 | yolo26_s1 better [0.0197, 0.0280] | yolo26_s1 better [0.0244, 0.0337] | yolo26_s1 better [0.0119, 0.0171] | yolo26_s1 better [0.0155, 0.0215] | none |
| yolo26_evid_s2 vs yolo26_s2 | yolo26_s2 better [0.0163, 0.0242] | yolo26_s2 better [0.0235, 0.0329] | yolo26_s2 better [0.0140, 0.0201] | yolo26_s2 better [0.0169, 0.0233] | none |
| yolo26_evid_s2 vs yolov8 | yolov8 better [0.0367, 0.0430] | yolov8 better [0.0349, 0.0422] | yolov8 better [0.0064, 0.0122] | yolov8 better [0.0145, 0.0203] | none |
| yolo26_s1 vs yolo26_s2 | yolo26_s1 better [-0.0066, -0.0004] | cannot separate [-0.0044, 0.0028] | yolo26_s2 better [0.0005, 0.0046] | cannot separate [-0.0008, 0.0038] | ma_brier |
| yolo26_s1 vs yolov8 | yolov8 better [0.0122, 0.0196] | yolov8 better [0.0048, 0.0138] | yolo26_s1 better [-0.0074, -0.0021] | cannot separate [-0.0037, 0.0020] | ma_brier |
| yolo26_s2 vs yolov8 | yolov8 better [0.0158, 0.0231] | yolov8 better [0.0059, 0.0148] | yolo26_s2 better [-0.0103, -0.0046] | cannot separate [-0.0056, 0.0008] | ma_brier |

- D-ECE cannot separate rtdetr and rtdetr_ens (at matched false alarms per image).
- D-ECE cannot separate rtdetr and rtdetr_s1 (at matched false alarms per image).
- D-ECE cannot separate rtdetr and rtdetr_s2 (at matched false alarms per image).
- D-ECE cannot separate rtdetr_ens and rtdetr_s1 (at matched false alarms per image).
- D-ECE cannot separate rtdetr_ens and rtdetr_s2 (at matched false alarms per image).
- D-ECE cannot separate rtdetr_s1 and rtdetr_s2 (at matched false alarms per image).
- D-ECE cannot separate yolo26_ens and yolo26_s2 (at matched false alarms per image).
- D-ECE cannot separate yolo26_evid and yolo26_evid_s2 (at matched false alarms per image).

Ranking flip relative to D-ECE (both intervals exclude zero, opposite signs) survives matched cost: **True**

#### MA-Brier decomposition

Each term is its contribution to MA-Brier (share of MA-Brier in brackets); the three sum to MA-Brier. OCE averages over vehicles only, so it has no false-alarm term.

| detector | where | threshold | false alarms/image | MA-Brier | false-alarm term | matched-detection term | miss term | OCE |
|---|---|---|---|---|---|---|---|---|
| yolo26 | own threshold | 0.46 | 1.26 | 0.1785 | 0.0477 (0.27) | 0.0322 (0.18) | 0.0987 (0.55) | 0.1448 |
| rtdetr | own threshold | 0.69 | 1.35 | 0.1911 | 0.0731 (0.38) | 0.0117 (0.06) | 0.1063 (0.56) | 0.1311 |
| yolov8 | own threshold | 0.47 | 1.32 | 0.1812 | 0.0516 (0.29) | 0.0304 (0.17) | 0.0992 (0.55) | 0.1442 |
| yolo26_s1 | own threshold | 0.46 | 1.24 | 0.1796 | 0.0469 (0.26) | 0.0333 (0.19) | 0.0995 (0.55) | 0.1471 |
| yolo26_s2 | own threshold | 0.46 | 1.16 | 0.1810 | 0.0442 (0.24) | 0.0332 (0.18) | 0.1036 (0.57) | 0.1500 |
| rtdetr_s1 | own threshold | 0.70 | 1.27 | 0.1913 | 0.0709 (0.37) | 0.0114 (0.06) | 0.1090 (0.57) | 0.1331 |
| rtdetr_s2 | own threshold | 0.70 | 1.24 | 0.1917 | 0.0694 (0.36) | 0.0112 (0.06) | 0.1111 (0.58) | 0.1354 |
| yolo26_evid | own threshold | 0.45 | 1.23 | 0.1913 | 0.0440 (0.23) | 0.0422 (0.22) | 0.1050 (0.55) | 0.1630 |
| yolo26_evid_s1 | own threshold | 0.43 | 1.19 | 0.1916 | 0.0411 (0.21) | 0.0455 (0.24) | 0.1050 (0.55) | 0.1659 |
| yolo26_evid_s2 | own threshold | 0.45 | 1.26 | 0.1931 | 0.0446 (0.23) | 0.0418 (0.22) | 0.1067 (0.55) | 0.1646 |
| yolo26_ens | own threshold | 0.44 | 1.21 | 0.1733 | 0.0446 (0.26) | 0.0352 (0.20) | 0.0935 (0.54) | 0.1418 |
| rtdetr_ens | own threshold | 0.68 | 1.28 | 0.1811 | 0.0694 (0.38) | 0.0125 (0.07) | 0.0992 (0.55) | 0.1234 |
| yolo26 | matched cost | 0.46 | 1.25 | 0.1790 | 0.0476 (0.27) | 0.0321 (0.18) | 0.0993 (0.56) | 0.1452 |
| rtdetr | matched cost | 0.70 | 1.25 | 0.1942 | 0.0696 (0.36) | 0.0112 (0.06) | 0.1134 (0.58) | 0.1374 |
| yolov8 | matched cost | 0.48 | 1.25 | 0.1844 | 0.0506 (0.27) | 0.0292 (0.16) | 0.1046 (0.57) | 0.1480 |
| yolo26_s1 | matched cost | 0.46 | 1.25 | 0.1795 | 0.0470 (0.26) | 0.0333 (0.19) | 0.0993 (0.55) | 0.1470 |
| yolo26_s2 | matched cost | 0.44 | 1.25 | 0.1770 | 0.0454 (0.26) | 0.0348 (0.20) | 0.0969 (0.55) | 0.1456 |
| rtdetr_s1 | matched cost | 0.70 | 1.25 | 0.1924 | 0.0701 (0.36) | 0.0112 (0.06) | 0.1111 (0.58) | 0.1350 |
| rtdetr_s2 | matched cost | 0.70 | 1.25 | 0.1911 | 0.0696 (0.36) | 0.0113 (0.06) | 0.1101 (0.58) | 0.1344 |
| yolo26_evid | matched cost | 0.45 | 1.25 | 0.1900 | 0.0443 (0.23) | 0.0426 (0.22) | 0.1031 (0.54) | 0.1616 |
| yolo26_evid_s1 | matched cost | 0.42 | 1.25 | 0.1888 | 0.0418 (0.22) | 0.0466 (0.25) | 0.1004 (0.53) | 0.1630 |
| yolo26_evid_s2 | matched cost | 0.45 | 1.25 | 0.1938 | 0.0444 (0.23) | 0.0416 (0.21) | 0.1079 (0.56) | 0.1655 |
| yolo26_ens | matched cost | 0.43 | 1.25 | 0.1710 | 0.0451 (0.26) | 0.0362 (0.21) | 0.0896 (0.52) | 0.1391 |
| rtdetr_ens | matched cost | 0.68 | 1.25 | 0.1817 | 0.0683 (0.38) | 0.0123 (0.07) | 0.1011 (0.56) | 0.1250 |

## Matching: native

### Gates

- yolo26 / detector_usable_on_development: pass
- yolo26 / enough_occlusion: pass
- yolo26 / detector_usable_on_pals / coarse: pass
- rtdetr / detector_usable_on_development: pass
- rtdetr / enough_occlusion: pass
- rtdetr / detector_usable_on_pals / coarse: pass
- yolov8 / detector_usable_on_development: pass
- yolov8 / enough_occlusion: pass
- yolov8 / detector_usable_on_pals / coarse: pass
- yolo26_s1 / detector_usable_on_development: pass
- yolo26_s1 / enough_occlusion: pass
- yolo26_s1 / detector_usable_on_pals / coarse: pass
- yolo26_s2 / detector_usable_on_development: pass
- yolo26_s2 / enough_occlusion: pass
- yolo26_s2 / detector_usable_on_pals / coarse: pass
- rtdetr_s1 / detector_usable_on_development: pass
- rtdetr_s1 / enough_occlusion: pass
- rtdetr_s1 / detector_usable_on_pals / coarse: pass
- rtdetr_s2 / detector_usable_on_development: pass
- rtdetr_s2 / enough_occlusion: pass
- rtdetr_s2 / detector_usable_on_pals / coarse: pass
- yolo26_evid / detector_usable_on_development: pass
- yolo26_evid / enough_occlusion: pass
- yolo26_evid / detector_usable_on_pals / coarse: pass
- yolo26_evid_s1 / detector_usable_on_development: pass
- yolo26_evid_s1 / enough_occlusion: pass
- yolo26_evid_s1 / detector_usable_on_pals / coarse: pass
- yolo26_evid_s2 / detector_usable_on_development: pass
- yolo26_evid_s2 / enough_occlusion: pass
- yolo26_evid_s2 / detector_usable_on_pals / coarse: pass
- yolo26_ens / detector_usable_on_development: pass
- yolo26_ens / enough_occlusion: pass
- yolo26_ens / detector_usable_on_pals / coarse: pass
- rtdetr_ens / detector_usable_on_development: pass
- rtdetr_ens / enough_occlusion: pass
- rtdetr_ens / detector_usable_on_pals / coarse: pass

### Detection at each detector's operating threshold

| detector | threshold | AP50 | recall | precision | false alarms/image | recall at floor |
|---|---|---|---|---|---|---|
| yolo26 | 0.49 | 0.748 | 0.829 | 0.840 | 1.65 | 0.928 |
| rtdetr | 0.73 | 0.794 | 0.807 | 0.850 | 1.49 | 0.989 |
| yolov8 | 0.52 | 0.739 | 0.819 | 0.842 | 1.61 | 0.924 |
| yolo26_s1 | 0.51 | 0.745 | 0.820 | 0.849 | 1.52 | 0.928 |
| yolo26_s2 | 0.51 | 0.746 | 0.817 | 0.854 | 1.46 | 0.928 |
| rtdetr_s1 | 0.72 | 0.792 | 0.816 | 0.840 | 1.62 | 0.988 |
| rtdetr_s2 | 0.70 | 0.787 | 0.825 | 0.829 | 1.77 | 0.989 |
| yolo26_evid | 0.49 | 0.749 | 0.817 | 0.849 | 1.52 | 0.944 |
| yolo26_evid_s1 | 0.43 | 0.745 | 0.831 | 0.834 | 1.73 | 0.943 |
| yolo26_evid_s2 | 0.45 | 0.746 | 0.829 | 0.828 | 1.80 | 0.941 |
| yolo26_ens | 0.50 | 0.753 | 0.825 | 0.851 | 1.51 | 0.934 |
| rtdetr_ens | 0.70 | 0.782 | 0.826 | 0.838 | 1.67 | 0.983 |

### RQ1: miss rate by annotated occlusion severity

| detector | 0.0-0.2 (n=12021) | 0.2-0.4 (n=2152) | 0.4-0.6 (n=1169) | 0.6-0.8 (n=444) | 0.8-1.0 (n=307) | odds of a miss per +0.1 (size, class controlled) | partial Spearman | context vs annotated occlusion |
|---|---|---|---|---|---|---|---|---|
| yolo26 | 0.150 | 0.202 | 0.230 | 0.306 | 0.336 | 1.145 [1.124, 1.169] | 0.219 [0.201, 0.236] | 0.90 |
| rtdetr | 0.167 | 0.234 | 0.273 | 0.369 | 0.401 | 1.173 [1.151, 1.197] | 0.233 [0.214, 0.250] | 0.89 |
| yolov8 | 0.158 | 0.213 | 0.252 | 0.340 | 0.332 | 1.152 [1.134, 1.174] | 0.210 [0.191, 0.227] | 0.89 |
| yolo26_s1 | 0.158 | 0.211 | 0.252 | 0.331 | 0.349 | 1.153 [1.132, 1.176] | 0.220 [0.201, 0.236] | 0.89 |
| yolo26_s2 | 0.162 | 0.207 | 0.246 | 0.331 | 0.352 | 1.144 [1.126, 1.169] | 0.220 [0.201, 0.237] | 0.89 |
| rtdetr_s1 | 0.160 | 0.218 | 0.269 | 0.338 | 0.368 | 1.166 [1.143, 1.190] | 0.231 [0.213, 0.247] | 0.90 |
| rtdetr_s2 | 0.152 | 0.209 | 0.253 | 0.318 | 0.355 | 1.159 [1.137, 1.182] | 0.230 [0.212, 0.247] | 0.90 |
| yolo26_evid | 0.160 | 0.217 | 0.249 | 0.360 | 0.352 | 1.159 [1.139, 1.180] | 0.216 [0.197, 0.232] | 0.89 |
| yolo26_evid_s1 | 0.147 | 0.201 | 0.228 | 0.324 | 0.316 | 1.148 [1.127, 1.171] | 0.217 [0.197, 0.233] | 0.90 |
| yolo26_evid_s2 | 0.149 | 0.198 | 0.240 | 0.324 | 0.352 | 1.158 [1.139, 1.183] | 0.220 [0.202, 0.235] | 0.90 |
| yolo26_ens | 0.154 | 0.203 | 0.244 | 0.322 | 0.349 | 1.151 [1.130, 1.174] | 0.221 [0.202, 0.237] | 0.90 |
| rtdetr_ens | 0.151 | 0.206 | 0.246 | 0.322 | 0.342 | 1.156 [1.134, 1.183] | 0.232 [0.213, 0.248] | 0.90 |

`*` marks a bucket with fewer objects than `min_bucket_n`; do not read it.

#### RQ1 by object size (tertiles of box area relative to the image)

| detector | tertile edges (relative area) | small | medium | large |
|---|---|---|---|---|
| yolo26 | 4.14e-03 / 1.41e-02 | 0.221 [0.208, 0.233] (n=5362) | 0.164 [0.153, 0.176] (n=5366) | 0.127 [0.118, 0.137] (n=5365) |
| rtdetr | 4.14e-03 / 1.41e-02 | 0.266 [0.253, 0.279] (n=5362) | 0.184 [0.173, 0.197] (n=5366) | 0.130 [0.121, 0.140] (n=5365) |
| yolov8 | 4.14e-03 / 1.41e-02 | 0.236 [0.225, 0.249] (n=5362) | 0.173 [0.162, 0.185] (n=5366) | 0.133 [0.123, 0.143] (n=5365) |
| yolo26_s1 | 4.14e-03 / 1.41e-02 | 0.236 [0.224, 0.249] (n=5362) | 0.174 [0.162, 0.187] (n=5366) | 0.131 [0.122, 0.141] (n=5365) |
| yolo26_s2 | 4.14e-03 / 1.41e-02 | 0.240 [0.227, 0.252] (n=5362) | 0.174 [0.163, 0.187] (n=5366) | 0.133 [0.123, 0.144] (n=5365) |
| rtdetr_s1 | 4.14e-03 / 1.41e-02 | 0.250 [0.238, 0.263] (n=5362) | 0.176 [0.164, 0.188] (n=5366) | 0.127 [0.118, 0.138] (n=5365) |
| rtdetr_s2 | 4.14e-03 / 1.41e-02 | 0.234 [0.222, 0.247] (n=5362) | 0.169 [0.158, 0.182] (n=5366) | 0.123 [0.113, 0.133] (n=5365) |
| yolo26_evid | 4.14e-03 / 1.41e-02 | 0.246 [0.234, 0.259] (n=5362) | 0.174 [0.162, 0.186] (n=5366) | 0.129 [0.119, 0.139] (n=5365) |
| yolo26_evid_s1 | 4.14e-03 / 1.41e-02 | 0.225 [0.214, 0.237] (n=5362) | 0.159 [0.148, 0.171] (n=5366) | 0.122 [0.114, 0.132] (n=5365) |
| yolo26_evid_s2 | 4.14e-03 / 1.41e-02 | 0.223 [0.211, 0.235] (n=5362) | 0.164 [0.153, 0.176] (n=5366) | 0.126 [0.117, 0.135] (n=5365) |
| yolo26_ens | 4.14e-03 / 1.41e-02 | 0.231 [0.218, 0.244] (n=5362) | 0.167 [0.155, 0.178] (n=5366) | 0.129 [0.120, 0.139] (n=5365) |
| rtdetr_ens | 4.14e-03 / 1.41e-02 | 0.233 [0.220, 0.246] (n=5362) | 0.166 [0.155, 0.179] (n=5366) | 0.122 [0.113, 0.132] (n=5365) |

Miss rate by occlusion bucket within each size tertile (n in brackets; `*` = fewer than `min_bucket_n`):

| detector | size | 0.0-0.2 | 0.2-0.4 | 0.4-0.6 | 0.6-0.8 | 0.8-1.0 |
|---|---|---|---|---|---|---|
| yolo26 | small | 0.196 (4021) | 0.276 (733) | 0.272 (353) | 0.384 (138) | 0.359 (117) |
| yolo26 | medium | 0.138 (3730) | 0.182 (785) | 0.235 (520) | 0.308 (214) | 0.325 (117) |
| yolo26 | large | 0.117 (4270) | 0.140 (634) | 0.172 (296) | 0.185 (92) | 0.315 (73) |
| rtdetr | small | 0.231 (4021) | 0.344 (733) | 0.337 (353) | 0.486 (138) | 0.504 (117) |
| rtdetr | medium | 0.155 (3730) | 0.200 (785) | 0.265 (520) | 0.341 (214) | 0.350 (117) |
| rtdetr | large | 0.116 (4270) | 0.148 (634) | 0.209 (296) | 0.261 (92) | 0.315 (73) |
| yolov8 | small | 0.209 (4021) | 0.299 (733) | 0.292 (353) | 0.428 (138) | 0.393 (117) |
| yolov8 | medium | 0.144 (3730) | 0.190 (785) | 0.260 (520) | 0.322 (214) | 0.308 (117) |
| yolov8 | large | 0.122 (4270) | 0.144 (634) | 0.193 (296) | 0.250 (92) | 0.274 (73) |
| yolo26_s1 | small | 0.209 (4021) | 0.299 (733) | 0.286 (353) | 0.428 (138) | 0.393 (117) |
| yolo26_s1 | medium | 0.146 (3730) | 0.187 (785) | 0.263 (520) | 0.327 (214) | 0.325 (117) |
| yolo26_s1 | large | 0.121 (4270) | 0.137 (634) | 0.193 (296) | 0.196 (92) | 0.315 (73) |
| yolo26_s2 | small | 0.215 (4021) | 0.299 (733) | 0.289 (353) | 0.413 (138) | 0.410 (117) |
| yolo26_s2 | medium | 0.149 (3730) | 0.178 (785) | 0.258 (520) | 0.322 (214) | 0.316 (117) |
| yolo26_s2 | large | 0.124 (4270) | 0.137 (634) | 0.176 (296) | 0.228 (92) | 0.315 (73) |
| rtdetr_s1 | small | 0.221 (4021) | 0.310 (733) | 0.320 (353) | 0.428 (138) | 0.462 (117) |
| rtdetr_s1 | medium | 0.145 (3730) | 0.190 (785) | 0.273 (520) | 0.336 (214) | 0.333 (117) |
| rtdetr_s1 | large | 0.115 (4270) | 0.147 (634) | 0.203 (296) | 0.207 (92) | 0.274 (73) |
| rtdetr_s2 | small | 0.205 (4021) | 0.295 (733) | 0.303 (353) | 0.413 (138) | 0.419 (117) |
| rtdetr_s2 | medium | 0.143 (3730) | 0.185 (785) | 0.250 (520) | 0.285 (214) | 0.325 (117) |
| rtdetr_s2 | large | 0.109 (4270) | 0.139 (634) | 0.199 (296) | 0.250 (92) | 0.301 (73) |
| yolo26_evid | small | 0.216 (4021) | 0.323 (733) | 0.295 (353) | 0.464 (138) | 0.385 (117) |
| yolo26_evid | medium | 0.144 (3730) | 0.186 (785) | 0.258 (520) | 0.350 (214) | 0.350 (117) |
| yolo26_evid | large | 0.120 (4270) | 0.131 (634) | 0.179 (296) | 0.228 (92) | 0.301 (73) |
| yolo26_evid_s1 | small | 0.197 (4021) | 0.296 (733) | 0.272 (353) | 0.413 (138) | 0.376 (117) |
| yolo26_evid_s1 | medium | 0.136 (3730) | 0.162 (785) | 0.229 (520) | 0.313 (214) | 0.282 (117) |
| yolo26_evid_s1 | large | 0.111 (4270) | 0.139 (634) | 0.172 (296) | 0.217 (92) | 0.274 (73) |
| yolo26_evid_s2 | small | 0.195 (4021) | 0.286 (733) | 0.272 (353) | 0.413 (138) | 0.402 (117) |
| yolo26_evid_s2 | medium | 0.137 (3730) | 0.171 (785) | 0.248 (520) | 0.308 (214) | 0.325 (117) |
| yolo26_evid_s2 | large | 0.115 (4270) | 0.129 (634) | 0.186 (296) | 0.228 (92) | 0.315 (73) |
| yolo26_ens | small | 0.205 (4021) | 0.288 (733) | 0.283 (353) | 0.406 (138) | 0.402 (117) |
| yolo26_ens | medium | 0.140 (3730) | 0.176 (785) | 0.246 (520) | 0.327 (214) | 0.316 (117) |
| yolo26_ens | large | 0.119 (4270) | 0.137 (634) | 0.193 (296) | 0.185 (92) | 0.315 (73) |
| rtdetr_ens | small | 0.203 (4021) | 0.299 (733) | 0.300 (353) | 0.399 (138) | 0.419 (117) |
| rtdetr_ens | medium | 0.140 (3730) | 0.175 (785) | 0.242 (520) | 0.318 (214) | 0.316 (117) |
| rtdetr_ens | large | 0.111 (4270) | 0.137 (634) | 0.186 (296) | 0.217 (92) | 0.260 (73) |

### Conformal layer

| detector | rule | target | recall of rule alone | recall of system | system recall, most occluded bucket | flags/image | certifiable |
|---|---|---|---|---|---|---|---|
| yolo26 | standard_threshold | n/a | 0.829 [0.822, 0.836] | 0.829 | 0.664 | n/a |  |
| yolo26 | global_confidence | 0.90 | 0.896 [0.890, 0.902] | 0.896 | 0.808 | 3.10 | yes |
| yolo26 | mondrian_confidence | 0.90 | 0.896 [0.890, 0.902] | 0.896 | 0.837 | 3.40 | yes |
| yolo26 | global, duplicate- and context-aware score | 0.90 | 0.896 [0.890, 0.902] | 0.896 | 0.808 | 3.11 | yes |
| rtdetr | standard_threshold | n/a | 0.807 [0.799, 0.813] | 0.807 | 0.599 | n/a |  |
| rtdetr | global_confidence | 0.90 | 0.900 [0.893, 0.905] | 0.900 | 0.785 | 3.33 | yes |
| rtdetr | mondrian_confidence | 0.90 | 0.901 [0.894, 0.907] | 0.901 | 0.827 | 3.44 | yes |
| rtdetr | global, duplicate- and context-aware score | 0.90 | 0.902 [0.895, 0.908] | 0.902 | 0.782 | 3.36 | yes |
| yolov8 | standard_threshold | n/a | 0.819 [0.812, 0.827] | 0.819 | 0.668 | n/a |  |
| yolov8 | global_confidence | 0.90 | 0.896 [0.890, 0.902] | 0.896 | 0.821 | 3.61 | yes |
| yolov8 | mondrian_confidence | 0.90 | 0.897 [0.890, 0.903] | 0.897 | 0.857 | 4.24 | yes |
| yolov8 | global, duplicate- and context-aware score | 0.90 | 0.896 [0.890, 0.902] | 0.896 | 0.821 | 3.66 | yes |
| yolo26_s1 | standard_threshold | n/a | 0.820 [0.812, 0.827] | 0.820 | 0.651 | n/a |  |
| yolo26_s1 | global_confidence | 0.90 | 0.896 [0.889, 0.901] | 0.896 | 0.808 | 3.34 | yes |
| yolo26_s1 | mondrian_confidence | 0.90 | 0.895 [0.889, 0.901] | 0.895 | 0.837 | 3.62 | yes |
| yolo26_s1 | global, duplicate- and context-aware score | 0.90 | 0.896 [0.889, 0.901] | 0.896 | 0.818 | 3.32 | yes |
| yolo26_s2 | standard_threshold | n/a | 0.817 [0.810, 0.825] | 0.817 | 0.648 | n/a |  |
| yolo26_s2 | global_confidence | 0.90 | 0.896 [0.890, 0.902] | 0.896 | 0.808 | 3.57 | yes |
| yolo26_s2 | mondrian_confidence | 0.90 | 0.897 [0.890, 0.903] | 0.897 | 0.831 | 3.86 | yes |
| yolo26_s2 | global, duplicate- and context-aware score | 0.90 | 0.896 [0.889, 0.902] | 0.896 | 0.811 | 3.56 | yes |
| rtdetr_s1 | standard_threshold | n/a | 0.816 [0.808, 0.823] | 0.816 | 0.632 | n/a |  |
| rtdetr_s1 | global_confidence | 0.90 | 0.899 [0.892, 0.904] | 0.899 | 0.785 | 2.97 | yes |
| rtdetr_s1 | mondrian_confidence | 0.90 | 0.901 [0.894, 0.906] | 0.901 | 0.837 | 3.28 | yes |
| rtdetr_s1 | global, duplicate- and context-aware score | 0.90 | 0.899 [0.893, 0.905] | 0.899 | 0.785 | 2.98 | yes |
| rtdetr_s2 | standard_threshold | n/a | 0.825 [0.817, 0.832] | 0.825 | 0.645 | n/a |  |
| rtdetr_s2 | global_confidence | 0.90 | 0.898 [0.892, 0.903] | 0.898 | 0.788 | 2.69 | yes |
| rtdetr_s2 | mondrian_confidence | 0.90 | 0.896 [0.890, 0.902] | 0.896 | 0.834 | 2.85 | yes |
| rtdetr_s2 | global, duplicate- and context-aware score | 0.90 | 0.901 [0.894, 0.906] | 0.901 | 0.792 | 2.77 | yes |
| yolo26_evid | standard_threshold | n/a | 0.817 [0.809, 0.825] | 0.817 | 0.648 | n/a |  |
| yolo26_evid | global_confidence | 0.90 | 0.898 [0.892, 0.904] | 0.898 | 0.811 | 3.49 | yes |
| yolo26_evid | mondrian_confidence | 0.90 | 0.896 [0.890, 0.902] | 0.896 | 0.831 | 3.84 | yes |
| yolo26_evid | global, duplicate- and context-aware score | 0.90 | 0.897 [0.891, 0.903] | 0.897 | 0.818 | 3.55 | yes |
| yolo26_evid_s1 | standard_threshold | n/a | 0.831 [0.824, 0.838] | 0.831 | 0.684 | n/a |  |
| yolo26_evid_s1 | global_confidence | 0.90 | 0.895 [0.889, 0.901] | 0.895 | 0.814 | 3.06 | yes |
| yolo26_evid_s1 | mondrian_confidence | 0.90 | 0.897 [0.890, 0.903] | 0.897 | 0.850 | 3.96 | yes |
| yolo26_evid_s1 | global, duplicate- and context-aware score | 0.90 | 0.896 [0.890, 0.902] | 0.896 | 0.831 | 3.27 | yes |
| yolo26_evid_s2 | standard_threshold | n/a | 0.829 [0.821, 0.836] | 0.829 | 0.648 | n/a |  |
| yolo26_evid_s2 | global_confidence | 0.90 | 0.895 [0.889, 0.901] | 0.895 | 0.814 | 3.08 | yes |
| yolo26_evid_s2 | mondrian_confidence | 0.90 | 0.896 [0.890, 0.902] | 0.896 | 0.831 | 3.85 | yes |
| yolo26_evid_s2 | global, duplicate- and context-aware score | 0.90 | 0.895 [0.889, 0.901] | 0.895 | 0.814 | 3.17 | yes |
| yolo26_ens | standard_threshold | n/a | 0.825 [0.817, 0.832] | 0.825 | 0.651 | n/a |  |
| yolo26_ens | global_confidence | 0.90 | 0.897 [0.891, 0.902] | 0.897 | 0.801 | 3.44 | yes |
| yolo26_ens | mondrian_confidence | 0.90 | 0.897 [0.891, 0.903] | 0.897 | 0.837 | 3.80 | yes |
| yolo26_ens | global, duplicate- and context-aware score | 0.90 | 0.896 [0.890, 0.902] | 0.896 | 0.805 | 3.39 | yes |
| rtdetr_ens | standard_threshold | n/a | 0.826 [0.819, 0.833] | 0.826 | 0.658 | n/a |  |
| rtdetr_ens | global_confidence | 0.90 | 0.901 [0.894, 0.907] | 0.901 | 0.805 | 3.16 | yes |
| rtdetr_ens | mondrian_confidence | 0.90 | 0.902 [0.895, 0.908] | 0.902 | 0.850 | 3.48 | yes |
| rtdetr_ens | global, duplicate- and context-aware score | 0.90 | 0.902 [0.896, 0.908] | 0.902 | 0.805 | 3.23 | yes |

### RQ2: recovery of missed vehicles at matched extra false alarms

| detector | policy | misses | recoverable ceiling | mean recovery | difference from baseline, all misses | difference from baseline, occluded misses |
|---|---|---|---|---|---|---|
| yolo26 | confidence | 2744 | 0.580 | 0.150 [0.138, 0.163] | baseline | baseline |
| yolo26 | mondrian_rank | 2744 | 0.580 | 0.122 [0.109, 0.135] | -0.0285 [-0.0381, -0.0186] | 0.0879 [0.0731, 0.1036] |
| yolo26 | duplicate- and context-aware score | 2744 | 0.580 | 0.149 [0.136, 0.162] | -0.0013 [-0.0045, 0.0016] | -0.0103 [-0.0163, -0.0058] |
| rtdetr | confidence | 3111 | 0.942 | 0.167 [0.152, 0.181] | baseline | baseline |
| rtdetr | mondrian_rank | 3111 | 0.942 | 0.137 [0.123, 0.151] | -0.0298 [-0.0394, -0.0199] | 0.0929 [0.0772, 0.1076] |
| rtdetr | duplicate- and context-aware score | 3111 | 0.942 | 0.166 [0.152, 0.180] | -0.0013 [-0.0049, 0.0026] | -0.0015 [-0.0057, 0.0026] |
| yolov8 | confidence | 2905 | 0.578 | 0.154 [0.140, 0.170] | baseline | baseline |
| yolov8 | mondrian_rank | 2905 | 0.578 | 0.124 [0.111, 0.138] | -0.0297 [-0.0395, -0.0197] | 0.0817 [0.0663, 0.0986] |
| yolov8 | duplicate- and context-aware score | 2905 | 0.578 | 0.153 [0.138, 0.169] | -0.0011 [-0.0039, 0.0020] | -0.0092 [-0.0131, -0.0054] |
| yolo26_s1 | confidence | 2901 | 0.599 | 0.151 [0.137, 0.164] | baseline | baseline |
| yolo26_s1 | mondrian_rank | 2901 | 0.599 | 0.128 [0.113, 0.141] | -0.0235 [-0.0328, -0.0141] | 0.0868 [0.0715, 0.1006] |
| yolo26_s1 | duplicate- and context-aware score | 2901 | 0.599 | 0.151 [0.137, 0.164] | -0.0002 [-0.0042, 0.0038] | -0.0040 [-0.0095, 0.0011] |
| yolo26_s2 | confidence | 2939 | 0.605 | 0.153 [0.141, 0.167] | baseline | baseline |
| yolo26_s2 | mondrian_rank | 2939 | 0.605 | 0.132 [0.119, 0.145] | -0.0212 [-0.0306, -0.0113] | 0.0898 [0.0760, 0.1047] |
| yolo26_s2 | duplicate- and context-aware score | 2939 | 0.605 | 0.152 [0.140, 0.165] | -0.0017 [-0.0056, 0.0020] | -0.0029 [-0.0080, 0.0019] |
| rtdetr_s1 | confidence | 2967 | 0.937 | 0.157 [0.142, 0.171] | baseline | baseline |
| rtdetr_s1 | mondrian_rank | 2967 | 0.937 | 0.143 [0.130, 0.156] | -0.0146 [-0.0246, -0.0042] | 0.1121 [0.0979, 0.1273] |
| rtdetr_s1 | duplicate- and context-aware score | 2967 | 0.937 | 0.159 [0.145, 0.173] | 0.0017 [-0.0014, 0.0054] | 0.0017 [-0.0023, 0.0059] |
| rtdetr_s2 | confidence | 2820 | 0.935 | 0.158 [0.143, 0.173] | baseline | baseline |
| rtdetr_s2 | mondrian_rank | 2820 | 0.935 | 0.138 [0.123, 0.152] | -0.0203 [-0.0312, -0.0108] | 0.1037 [0.0872, 0.1195] |
| rtdetr_s2 | duplicate- and context-aware score | 2820 | 0.935 | 0.161 [0.146, 0.175] | 0.0024 [-0.0012, 0.0053] | 0.0030 [-0.0015, 0.0076] |
| yolo26_evid | confidence | 2943 | 0.695 | 0.162 [0.148, 0.177] | baseline | baseline |
| yolo26_evid | mondrian_rank | 2943 | 0.695 | 0.131 [0.118, 0.144] | -0.0312 [-0.0413, -0.0219] | 0.0905 [0.0763, 0.1039] |
| yolo26_evid | duplicate- and context-aware score | 2943 | 0.695 | 0.160 [0.147, 0.175] | -0.0016 [-0.0057, 0.0019] | -0.0027 [-0.0084, 0.0020] |
| yolo26_evid_s1 | confidence | 2712 | 0.661 | 0.138 [0.125, 0.151] | baseline | baseline |
| yolo26_evid_s1 | mondrian_rank | 2712 | 0.661 | 0.106 [0.094, 0.118] | -0.0317 [-0.0424, -0.0209] | 0.0786 [0.0640, 0.0929] |
| yolo26_evid_s1 | duplicate- and context-aware score | 2712 | 0.661 | 0.137 [0.124, 0.151] | -0.0008 [-0.0025, 0.0009] | -0.0032 [-0.0059, -0.0004] |
| yolo26_evid_s2 | confidence | 2747 | 0.653 | 0.144 [0.132, 0.159] | baseline | baseline |
| yolo26_evid_s2 | mondrian_rank | 2747 | 0.653 | 0.118 [0.105, 0.131] | -0.0261 [-0.0354, -0.0160] | 0.0950 [0.0797, 0.1102] |
| yolo26_evid_s2 | duplicate- and context-aware score | 2747 | 0.653 | 0.143 [0.130, 0.157] | -0.0015 [-0.0037, 0.0009] | -0.0099 [-0.0136, -0.0063] |
| yolo26_ens | confidence | 2824 | 0.625 | 0.144 [0.132, 0.156] | baseline | baseline |
| yolo26_ens | mondrian_rank | 2824 | 0.625 | 0.119 [0.107, 0.131] | -0.0254 [-0.0346, -0.0160] | 0.0866 [0.0725, 0.1004] |
| yolo26_ens | duplicate- and context-aware score | 2824 | 0.625 | 0.145 [0.133, 0.157] | 0.0005 [-0.0023, 0.0029] | -0.0020 [-0.0058, 0.0017] |
| rtdetr_ens | confidence | 2794 | 0.899 | 0.139 [0.125, 0.152] | baseline | baseline |
| rtdetr_ens | mondrian_rank | 2794 | 0.899 | 0.116 [0.103, 0.129] | -0.0234 [-0.0348, -0.0139] | 0.1010 [0.0835, 0.1169] |
| rtdetr_ens | duplicate- and context-aware score | 2794 | 0.899 | 0.140 [0.125, 0.153] | 0.0006 [-0.0004, 0.0014] | 0.0082 [0.0067, 0.0099] |

### RQ3: calibration metrics at the operating threshold

| detector | D-ECE | MA-ECE | miss mass | OCE | threshold D-ECE alone would choose | occluded vehicles: share of objects / share of D-ECE blind spot |
|---|---|---|---|---|---|---|
| yolo26 | 0.055 [0.050, 0.061] | 0.195 [0.188, 0.202] | 0.147 | 0.226 | 0.95 (recall 0.06) | 0.25 / 0.34 |
| rtdetr | 0.056 [0.051, 0.061] | 0.216 [0.209, 0.223] | 0.169 | 0.237 | 0.95 (recall 0.24) | 0.25 / 0.36 |
| yolov8 | 0.047 [0.041, 0.053] | 0.196 [0.189, 0.204] | 0.156 | 0.232 | 0.95 (recall 0.10) | 0.25 / 0.35 |
| yolo26_s1 | 0.051 [0.046, 0.056] | 0.201 [0.193, 0.208] | 0.157 | 0.234 | 0.95 (recall 0.05) | 0.25 / 0.35 |
| yolo26_s2 | 0.050 [0.045, 0.056] | 0.202 [0.195, 0.210] | 0.160 | 0.236 | 0.95 (recall 0.04) | 0.25 / 0.34 |
| rtdetr_s1 | 0.059 [0.053, 0.064] | 0.209 [0.202, 0.217] | 0.160 | 0.230 | 0.9 (recall 0.52) | 0.25 / 0.35 |
| rtdetr_s2 | 0.065 [0.060, 0.071] | 0.205 [0.198, 0.213] | 0.150 | 0.226 | 0.9 (recall 0.53) | 0.25 / 0.35 |
| yolo26_evid | 0.067 [0.061, 0.073] | 0.216 [0.209, 0.223] | 0.160 | 0.241 | 0.95 (recall 0.00) | 0.25 / 0.35 |
| yolo26_evid_s1 | 0.068 [0.064, 0.074] | 0.203 [0.197, 0.210] | 0.145 | 0.237 | 0.95 (recall 0.00) | 0.25 / 0.35 |
| yolo26_evid_s2 | 0.066 [0.060, 0.072] | 0.202 [0.196, 0.208] | 0.146 | 0.236 | 0.95 (recall 0.00) | 0.25 / 0.35 |
| yolo26_ens | 0.051 [0.047, 0.057] | 0.197 [0.190, 0.205] | 0.153 | 0.229 | 0.95 (recall 0.05) | 0.25 / 0.34 |
| rtdetr_ens | 0.061 [0.055, 0.066] | 0.201 [0.194, 0.209] | 0.150 | 0.221 | 0.9 (recall 0.52) | 0.25 / 0.35 |

### RQ5: why vehicles are missed (native classes, whatever the mode)

| detector | misses | confused | suppressed | unproposed | merged | visibility misses: odds per +0.1 occlusion | confusion misses: odds per +1 SD rarity | confusion misses: odds per +0.1 occlusion |
|---|---|---|---|---|---|---|---|---|
| yolo26 | 2744 | 0.29 | 0.55 | 0.06 | 0.00 | 1.178 [1.151, 1.205] | 2.490 [2.358, 2.646] | 1.012 [0.974, 1.054] |
| rtdetr | 3111 | 0.24 | 0.71 | 0.05 | 0.00 | 1.201 [1.177, 1.228] | 2.539 [2.416, 2.680] | 1.001 [0.959, 1.045] |
| yolov8 | 2905 | 0.27 | 0.56 | 0.06 | 0.00 | 1.175 [1.152, 1.202] | 2.493 [2.351, 2.639] | 1.009 [0.971, 1.048] |
| yolo26_s1 | 2901 | 0.27 | 0.57 | 0.06 | 0.00 | 1.188 [1.163, 1.215] | 2.492 [2.370, 2.652] | 1.001 [0.953, 1.041] |
| yolo26_s2 | 2939 | 0.26 | 0.57 | 0.06 | 0.00 | 1.173 [1.148, 1.200] | 2.480 [2.355, 2.629] | 0.994 [0.953, 1.036] |
| rtdetr_s1 | 2967 | 0.26 | 0.68 | 0.06 | 0.00 | 1.197 [1.173, 1.223] | 2.538 [2.387, 2.670] | 1.006 [0.963, 1.049] |
| rtdetr_s2 | 2820 | 0.29 | 0.65 | 0.06 | 0.00 | 1.195 [1.169, 1.221] | 2.564 [2.425, 2.709] | 1.008 [0.970, 1.047] |
| yolo26_evid | 2943 | 0.25 | 0.60 | 0.05 | 0.00 | 1.179 [1.156, 1.206] | 2.466 [2.307, 2.612] | 1.019 [0.980, 1.055] |
| yolo26_evid_s1 | 2712 | 0.31 | 0.55 | 0.06 | 0.00 | 1.174 [1.151, 1.205] | 2.588 [2.442, 2.740] | 1.033 [0.992, 1.069] |
| yolo26_evid_s2 | 2747 | 0.30 | 0.55 | 0.06 | 0.00 | 1.188 [1.164, 1.216] | 2.465 [2.330, 2.604] | 1.011 [0.970, 1.049] |
| yolo26_ens | 2824 | 0.29 | 0.57 | 0.05 | 0.00 | 1.189 [1.167, 1.218] | 2.498 [2.350, 2.646] | 1.003 [0.961, 1.045] |
| rtdetr_ens | 2794 | 0.30 | 0.63 | 0.05 | 0.00 | 1.196 [1.169, 1.224] | 2.569 [2.428, 2.706] | 1.001 [0.959, 1.045] |

### RQ4: transfer with the detector frozen (recall-side quantities only)

Each coverage is followed by (flags per image; certifiable). A recalibrated coverage that is not certifiable is the best the rule could do, not a guarantee.

| detector | dataset | matching | recall, development -> transfer | rule | target | coverage on development | coverage on transfer, frozen | coverage on transfer, recalibrated |
|---|---|---|---|---|---|---|---|---|
| yolo26 | pals | coarse | 0.864 -> 0.758 | global_confidence | 0.90 | 0.937 (3.05; yes) | 0.841 (1.88; yes) | 0.867 (13.96; NO) |
| yolo26 | pals | coarse | 0.864 -> 0.758 | mondrian_confidence | 0.90 | 0.937 (3.36; yes) | 0.838 (2.07; yes) | 0.867 (13.96; NO) |
| yolo26 | pals | coarse | 0.864 -> 0.758 | global_occlusion_aware | 0.90 | 0.937 (3.06; yes) | 0.841 (1.89; yes) | 0.867 (13.96; NO) |
| rtdetr | pals | coarse | 0.840 -> 0.567 | global_confidence | 0.90 | 0.927 (3.29; yes) | 0.730 (2.28; yes) | 0.916 (19.04; yes) |
| rtdetr | pals | coarse | 0.840 -> 0.567 | mondrian_confidence | 0.90 | 0.927 (3.41; yes) | 0.721 (2.15; yes) | 0.916 (19.04; yes) |
| rtdetr | pals | coarse | 0.840 -> 0.567 | global_occlusion_aware | 0.90 | 0.926 (3.32; yes) | 0.724 (2.25; yes) | 0.918 (17.28; yes) |
| yolov8 | pals | coarse | 0.853 -> 0.733 | global_confidence | 0.90 | 0.940 (3.55; yes) | 0.836 (2.05; yes) | 0.872 (14.38; NO) |
| yolov8 | pals | coarse | 0.853 -> 0.733 | mondrian_confidence | 0.90 | 0.941 (4.19; yes) | 0.829 (2.38; yes) | 0.872 (14.38; NO) |
| yolov8 | pals | coarse | 0.853 -> 0.733 | global_occlusion_aware | 0.90 | 0.940 (3.60; yes) | 0.836 (2.07; yes) | 0.872 (14.38; NO) |
| yolo26_s1 | pals | coarse | 0.855 -> 0.726 | global_confidence | 0.90 | 0.937 (3.29; yes) | 0.835 (2.06; yes) | 0.874 (14.06; NO) |
| yolo26_s1 | pals | coarse | 0.855 -> 0.726 | mondrian_confidence | 0.90 | 0.936 (3.57; yes) | 0.827 (2.18; yes) | 0.874 (14.06; NO) |
| yolo26_s1 | pals | coarse | 0.855 -> 0.726 | global_occlusion_aware | 0.90 | 0.937 (3.27; yes) | 0.834 (2.05; yes) | 0.874 (14.06; NO) |
| yolo26_s2 | pals | coarse | 0.852 -> 0.755 | global_confidence | 0.90 | 0.938 (3.51; yes) | 0.843 (2.07; yes) | 0.874 (12.73; NO) |
| yolo26_s2 | pals | coarse | 0.852 -> 0.755 | mondrian_confidence | 0.90 | 0.938 (3.81; yes) | 0.841 (2.28; yes) | 0.874 (12.73; NO) |
| yolo26_s2 | pals | coarse | 0.852 -> 0.755 | global_occlusion_aware | 0.90 | 0.938 (3.50; yes) | 0.843 (2.06; yes) | 0.874 (12.73; NO) |
| rtdetr_s1 | pals | coarse | 0.851 -> 0.549 | global_confidence | 0.90 | 0.928 (2.94; yes) | 0.680 (2.07; yes) | 0.916 (35.29; yes) |
| rtdetr_s1 | pals | coarse | 0.851 -> 0.549 | mondrian_confidence | 0.90 | 0.929 (3.25; yes) | 0.664 (2.01; yes) | 0.916 (35.29; yes) |
| rtdetr_s1 | pals | coarse | 0.851 -> 0.549 | global_occlusion_aware | 0.90 | 0.926 (2.95; yes) | 0.676 (2.07; yes) | 0.918 (35.28; yes) |
| rtdetr_s2 | pals | coarse | 0.861 -> 0.591 | global_confidence | 0.90 | 0.926 (2.66; yes) | 0.716 (1.93; yes) | 0.907 (32.14; yes) |
| rtdetr_s2 | pals | coarse | 0.861 -> 0.591 | mondrian_confidence | 0.90 | 0.925 (2.83; yes) | 0.700 (1.75; yes) | 0.907 (32.14; yes) |
| rtdetr_s2 | pals | coarse | 0.861 -> 0.591 | global_occlusion_aware | 0.90 | 0.926 (2.74; yes) | 0.715 (2.00; yes) | 0.901 (26.06; yes) |
| yolo26_evid | pals | coarse | 0.849 -> 0.721 | global_confidence | 0.90 | 0.937 (3.44; yes) | 0.839 (2.23; yes) | 0.941 (170.13; NO) |
| yolo26_evid | pals | coarse | 0.849 -> 0.721 | mondrian_confidence | 0.90 | 0.936 (3.80; yes) | 0.833 (2.26; yes) | 0.938 (96.30; NO) |
| yolo26_evid | pals | coarse | 0.849 -> 0.721 | global_occlusion_aware | 0.90 | 0.936 (3.50; yes) | 0.837 (2.22; yes) | 0.941 (170.13; NO) |
| yolo26_evid_s1 | pals | coarse | 0.868 -> 0.757 | global_confidence | 0.90 | 0.937 (3.02; yes) | 0.829 (1.84; yes) | 0.921 (182.34; NO) |
| yolo26_evid_s1 | pals | coarse | 0.868 -> 0.757 | mondrian_confidence | 0.90 | 0.939 (3.92; yes) | 0.822 (2.28; yes) | 0.904 (43.70; NO) |
| yolo26_evid_s1 | pals | coarse | 0.868 -> 0.757 | global_occlusion_aware | 0.90 | 0.937 (3.22; yes) | 0.827 (1.89; yes) | 0.921 (182.34; NO) |
| yolo26_evid_s2 | pals | coarse | 0.865 -> 0.746 | global_confidence | 0.90 | 0.936 (3.04; yes) | 0.824 (1.84; yes) | 0.920 (175.34; NO) |
| yolo26_evid_s2 | pals | coarse | 0.865 -> 0.746 | mondrian_confidence | 0.90 | 0.937 (3.82; yes) | 0.817 (2.16; yes) | 0.920 (175.34; NO) |
| yolo26_evid_s2 | pals | coarse | 0.865 -> 0.746 | global_occlusion_aware | 0.90 | 0.935 (3.13; yes) | 0.824 (1.84; yes) | 0.920 (175.34; NO) |
| yolo26_ens | pals | coarse | 0.860 -> 0.763 | global_confidence | 0.90 | 0.939 (3.38; yes) | 0.846 (1.99; yes) | 0.881 (19.80; NO) |
| yolo26_ens | pals | coarse | 0.860 -> 0.763 | mondrian_confidence | 0.90 | 0.939 (3.75; yes) | 0.844 (2.25; yes) | 0.881 (19.80; NO) |
| yolo26_ens | pals | coarse | 0.860 -> 0.763 | global_occlusion_aware | 0.90 | 0.938 (3.33; yes) | 0.844 (1.97; yes) | 0.881 (19.80; NO) |
| rtdetr_ens | pals | coarse | 0.863 -> 0.597 | global_confidence | 0.90 | 0.931 (3.12; yes) | 0.725 (2.19; yes) | 0.906 (35.30; yes) |
| rtdetr_ens | pals | coarse | 0.863 -> 0.597 | mondrian_confidence | 0.90 | 0.932 (3.45; yes) | 0.708 (2.18; yes) | 0.906 (35.30; yes) |
| rtdetr_ens | pals | coarse | 0.863 -> 0.597 | global_occlusion_aware | 0.90 | 0.931 (3.20; yes) | 0.728 (2.25; yes) | 0.899 (31.53; yes) |

### Detector ranking (best first)

| comparison | metric | ranking |
|---|---|---|
| each at its own threshold | dece | yolov8 < yolo26_s2 < yolo26_s1 < yolo26_ens < yolo26 < rtdetr < rtdetr_s1 < rtdetr_ens < rtdetr_s2 < yolo26_evid_s2 < yolo26_evid < yolo26_evid_s1 |
| each at its own threshold | ma_ece | yolo26 < yolov8 < yolo26_ens < yolo26_s1 < rtdetr_ens < yolo26_evid_s2 < yolo26_s2 < yolo26_evid_s1 < rtdetr_s2 < rtdetr_s1 < rtdetr < yolo26_evid |
| each at its own threshold | ma_brier | yolo26 < yolo26_ens < yolo26_evid_s1 < yolo26_evid_s2 < yolo26_s1 < yolo26_s2 < yolov8 < yolo26_evid < rtdetr_ens < rtdetr_s2 < rtdetr_s1 < rtdetr |
| each at its own threshold | oce_max | rtdetr_ens < yolo26 < rtdetr_s2 < yolo26_ens < rtdetr_s1 < yolov8 < yolo26_s1 < yolo26_s2 < yolo26_evid_s2 < rtdetr < yolo26_evid_s1 < yolo26_evid |
| at matched false alarms per image | dece | yolov8 < yolo26_s2 < yolo26_s1 < yolo26_ens < yolo26 < rtdetr_s1 < rtdetr_ens < rtdetr_s2 < rtdetr < yolo26_evid_s2 < yolo26_evid < yolo26_evid_s1 |
| at matched false alarms per image | ma_ece | yolo26_ens < yolo26_s2 < yolov8 < yolo26_s1 < yolo26 < rtdetr_ens < rtdetr_s1 < rtdetr_s2 < yolo26_evid < rtdetr < yolo26_evid_s1 < yolo26_evid_s2 |
| at matched false alarms per image | ma_brier | yolo26_ens < yolo26 < yolo26_s2 < yolo26_s1 < yolo26_evid < yolov8 < yolo26_evid_s1 < yolo26_evid_s2 < rtdetr_ens < rtdetr < rtdetr_s1 < rtdetr_s2 |
| at matched false alarms per image | oce_max | rtdetr_ens < yolo26_ens < yolo26 < yolo26_s1 < yolo26_s2 < rtdetr < rtdetr_s1 < yolov8 < rtdetr_s2 < yolo26_evid < yolo26_evid_s1 < yolo26_evid_s2 |

Point-estimate orders above can differ by noise. Below, a metric *separates* a pair only if its 95% paired-difference interval excludes zero; a *flip* needs both D-ECE and the other metric to separate the pair, in opposite directions.

**each at its own threshold**

| pair (difference = first minus second) | D-ECE | MA-ECE | MA-Brier | OCE | flips vs D-ECE |
|---|---|---|---|---|---|
| rtdetr vs rtdetr_ens | rtdetr better [-0.0087, -0.0013] | rtdetr_ens better [0.0102, 0.0182] | rtdetr_ens better [0.0103, 0.0154] | rtdetr_ens better [0.0122, 0.0186] | ma_ece, ma_brier, oce_max |
| rtdetr vs rtdetr_s1 | cannot separate [-0.0076, 0.0019] | rtdetr_s1 better [0.0017, 0.0118] | rtdetr_s1 better [0.0006, 0.0074] | rtdetr_s1 better [0.0026, 0.0107] | none |
| rtdetr vs rtdetr_s2 | rtdetr better [-0.0141, -0.0052] | rtdetr_s2 better [0.0049, 0.0152] | rtdetr_s2 better [0.0045, 0.0111] | rtdetr_s2 better [0.0063, 0.0147] | ma_ece, ma_brier, oce_max |
| rtdetr vs yolo26 | cannot separate [-0.0055, 0.0064] | yolo26 better [0.0139, 0.0276] | yolo26 better [0.0222, 0.0308] | yolo26 better [0.0063, 0.0155] | none |
| rtdetr vs yolo26_ens | cannot separate [-0.0014, 0.0105] | yolo26_ens better [0.0124, 0.0256] | yolo26_ens better [0.0204, 0.0284] | yolo26_ens better [0.0032, 0.0121] | none |
| rtdetr vs yolo26_evid | rtdetr better [-0.0186, -0.0047] | cannot separate [-0.0081, 0.0064] | yolo26_evid better [0.0099, 0.0181] | rtdetr better [-0.0095, -0.0003] | ma_brier |
| rtdetr vs yolo26_evid_s1 | rtdetr better [-0.0195, -0.0061] | yolo26_evid_s1 better [0.0044, 0.0188] | yolo26_evid_s1 better [0.0189, 0.0270] | cannot separate [-0.0049, 0.0036] | ma_ece, ma_brier |
| rtdetr vs yolo26_evid_s2 | rtdetr better [-0.0171, -0.0037] | yolo26_evid_s2 better [0.0063, 0.0199] | yolo26_evid_s2 better [0.0180, 0.0257] | cannot separate [-0.0040, 0.0040] | ma_ece, ma_brier |
| rtdetr vs yolo26_s1 | cannot separate [-0.0011, 0.0106] | yolo26_s1 better [0.0084, 0.0221] | yolo26_s1 better [0.0155, 0.0236] | cannot separate [-0.0013, 0.0076] | none |
| rtdetr vs yolo26_s2 | cannot separate [-0.0006, 0.0119] | yolo26_s2 better [0.0063, 0.0202] | yolo26_s2 better [0.0143, 0.0226] | cannot separate [-0.0045, 0.0051] | none |
| rtdetr vs yolov8 | yolov8 better [0.0027, 0.0152] | yolov8 better [0.0128, 0.0263] | yolov8 better [0.0129, 0.0211] | cannot separate [-0.0001, 0.0096] | none |
| rtdetr_ens vs rtdetr_s1 | cannot separate [-0.0022, 0.0055] | rtdetr_ens better [-0.0116, -0.0036] | rtdetr_ens better [-0.0113, -0.0063] | rtdetr_ens better [-0.0122, -0.0057] | none |
| rtdetr_ens vs rtdetr_s2 | rtdetr_ens better [-0.0083, -0.0011] | rtdetr_ens better [-0.0080, -0.0003] | rtdetr_ens better [-0.0072, -0.0026] | rtdetr_ens better [-0.0083, -0.0022] | none |
| rtdetr_ens vs yolo26 | cannot separate [-0.0005, 0.0113] | cannot separate [-0.0002, 0.0135] | yolo26 better [0.0094, 0.0175] | rtdetr_ens better [-0.0082, -0.0002] | none |
| rtdetr_ens vs yolo26_ens | yolo26_ens better [0.0035, 0.0150] | cannot separate [-0.0023, 0.0112] | yolo26_ens better [0.0077, 0.0156] | rtdetr_ens better [-0.0122, -0.0039] | oce_max |
| rtdetr_ens vs yolo26_evid | cannot separate [-0.0137, 0.0004] | rtdetr_ens better [-0.0221, -0.0077] | cannot separate [-0.0024, 0.0054] | rtdetr_ens better [-0.0241, -0.0158] | none |
| rtdetr_ens vs yolo26_evid_s1 | rtdetr_ens better [-0.0150, -0.0008] | cannot separate [-0.0095, 0.0050] | yolo26_evid_s1 better [0.0064, 0.0139] | rtdetr_ens better [-0.0200, -0.0120] | ma_brier |
| rtdetr_ens vs yolo26_evid_s2 | cannot separate [-0.0116, 0.0018] | cannot separate [-0.0078, 0.0059] | yolo26_evid_s2 better [0.0052, 0.0124] | rtdetr_ens better [-0.0194, -0.0111] | none |
| rtdetr_ens vs yolo26_s1 | yolo26_s1 better [0.0040, 0.0160] | cannot separate [-0.0055, 0.0076] | yolo26_s1 better [0.0029, 0.0106] | rtdetr_ens better [-0.0160, -0.0080] | oce_max |
| rtdetr_ens vs yolo26_s2 | yolo26_s2 better [0.0044, 0.0169] | cannot separate [-0.0079, 0.0059] | yolo26_s2 better [0.0017, 0.0096] | rtdetr_ens better [-0.0191, -0.0105] | oce_max |
| rtdetr_ens vs yolov8 | yolov8 better [0.0073, 0.0196] | cannot separate [-0.0018, 0.0120] | yolov8 better [0.0005, 0.0081] | rtdetr_ens better [-0.0149, -0.0061] | oce_max |
| rtdetr_s1 vs rtdetr_s2 | rtdetr_s1 better [-0.0111, -0.0020] | cannot separate [-0.0016, 0.0085] | rtdetr_s2 better [0.0010, 0.0068] | cannot separate [-0.0004, 0.0075] | ma_brier |
| rtdetr_s1 vs yolo26 | cannot separate [-0.0027, 0.0095] | yolo26 better [0.0079, 0.0213] | yolo26 better [0.0184, 0.0263] | yolo26 better [0.0004, 0.0092] | none |
| rtdetr_s1 vs yolo26_ens | yolo26_ens better [0.0012, 0.0136] | yolo26_ens better [0.0055, 0.0189] | yolo26_ens better [0.0165, 0.0244] | cannot separate [-0.0035, 0.0051] | none |
| rtdetr_s1 vs yolo26_evid | rtdetr_s1 better [-0.0153, -0.0005] | rtdetr_s1 better [-0.0146, -0.0000] | yolo26_evid better [0.0066, 0.0141] | rtdetr_s1 better [-0.0157, -0.0067] | ma_brier |
| rtdetr_s1 vs yolo26_evid_s1 | rtdetr_s1 better [-0.0168, -0.0020] | cannot separate [-0.0015, 0.0130] | yolo26_evid_s1 better [0.0150, 0.0228] | rtdetr_s1 better [-0.0114, -0.0030] | ma_brier |
| rtdetr_s1 vs yolo26_evid_s2 | rtdetr_s1 better [-0.0139, -0.0006] | cannot separate [-0.0001, 0.0132] | yolo26_evid_s2 better [0.0137, 0.0212] | rtdetr_s1 better [-0.0110, -0.0023] | ma_brier |
| rtdetr_s1 vs yolo26_s1 | yolo26_s1 better [0.0018, 0.0143] | yolo26_s1 better [0.0016, 0.0156] | yolo26_s1 better [0.0116, 0.0198] | cannot separate [-0.0076, 0.0009] | none |
| rtdetr_s1 vs yolo26_s2 | yolo26_s2 better [0.0017, 0.0145] | cannot separate [-0.0003, 0.0131] | yolo26_s2 better [0.0102, 0.0183] | rtdetr_s1 better [-0.0104, -0.0017] | oce_max |
| rtdetr_s1 vs yolov8 | yolov8 better [0.0052, 0.0178] | yolov8 better [0.0056, 0.0194] | yolov8 better [0.0093, 0.0171] | cannot separate [-0.0061, 0.0029] | none |
| rtdetr_s2 vs yolo26 | yolo26 better [0.0040, 0.0161] | yolo26 better [0.0034, 0.0178] | yolo26 better [0.0146, 0.0226] | cannot separate [-0.0035, 0.0053] | none |
| rtdetr_s2 vs yolo26_ens | yolo26_ens better [0.0082, 0.0199] | yolo26_ens better [0.0019, 0.0154] | yolo26_ens better [0.0124, 0.0202] | cannot separate [-0.0072, 0.0018] | none |
| rtdetr_s2 vs yolo26_evid | cannot separate [-0.0087, 0.0050] | rtdetr_s2 better [-0.0179, -0.0040] | yolo26_evid better [0.0027, 0.0105] | rtdetr_s2 better [-0.0189, -0.0103] | none |
| rtdetr_s2 vs yolo26_evid_s1 | cannot separate [-0.0101, 0.0044] | cannot separate [-0.0052, 0.0097] | yolo26_evid_s1 better [0.0110, 0.0188] | rtdetr_s2 better [-0.0152, -0.0064] | none |
| rtdetr_s2 vs yolo26_evid_s2 | cannot separate [-0.0075, 0.0064] | cannot separate [-0.0037, 0.0103] | yolo26_evid_s2 better [0.0102, 0.0171] | rtdetr_s2 better [-0.0144, -0.0055] | none |
| rtdetr_s2 vs yolo26_s1 | yolo26_s1 better [0.0080, 0.0209] | cannot separate [-0.0018, 0.0123] | yolo26_s1 better [0.0079, 0.0155] | rtdetr_s2 better [-0.0114, -0.0026] | oce_max |
| rtdetr_s2 vs yolo26_s2 | yolo26_s2 better [0.0086, 0.0213] | cannot separate [-0.0044, 0.0097] | yolo26_s2 better [0.0063, 0.0146] | rtdetr_s2 better [-0.0143, -0.0051] | oce_max |
| rtdetr_s2 vs yolov8 | yolov8 better [0.0120, 0.0244] | yolov8 better [0.0026, 0.0158] | yolov8 better [0.0054, 0.0129] | rtdetr_s2 better [-0.0101, -0.0005] | oce_max |
| yolo26 vs yolo26_ens | yolo26_ens better [0.0006, 0.0072] | cannot separate [-0.0062, 0.0016] | yolo26 better [-0.0039, -0.0005] | yolo26 better [-0.0059, -0.0016] | ma_brier, oce_max |
| yolo26 vs yolo26_evid | yolo26 better [-0.0178, -0.0063] | yolo26 better [-0.0276, -0.0152] | yolo26 better [-0.0149, -0.0095] | yolo26 better [-0.0190, -0.0126] | none |
| yolo26 vs yolo26_evid_s1 | yolo26 better [-0.0187, -0.0078] | yolo26 better [-0.0150, -0.0029] | yolo26 better [-0.0058, -0.0012] | yolo26 better [-0.0147, -0.0091] | none |
| yolo26 vs yolo26_evid_s2 | yolo26 better [-0.0167, -0.0052] | yolo26 better [-0.0146, -0.0010] | yolo26 better [-0.0076, -0.0019] | yolo26 better [-0.0144, -0.0075] | none |
| yolo26 vs yolo26_s1 | cannot separate [-0.0003, 0.0086] | yolo26 better [-0.0106, -0.0010] | yolo26 better [-0.0091, -0.0048] | yolo26 better [-0.0107, -0.0052] | none |
| yolo26 vs yolo26_s2 | yolo26_s2 better [0.0007, 0.0094] | yolo26 better [-0.0130, -0.0031] | yolo26 better [-0.0104, -0.0058] | yolo26 better [-0.0135, -0.0079] | ma_ece, ma_brier, oce_max |
| yolo26 vs yolov8 | yolov8 better [0.0034, 0.0131] | cannot separate [-0.0073, 0.0041] | yolo26 better [-0.0123, -0.0063] | yolo26 better [-0.0097, -0.0029] | ma_brier, oce_max |
| yolo26_ens vs yolo26_evid | yolo26_ens better [-0.0211, -0.0100] | yolo26_ens better [-0.0252, -0.0134] | yolo26_ens better [-0.0123, -0.0073] | yolo26_ens better [-0.0151, -0.0091] | none |
| yolo26_ens vs yolo26_evid_s1 | yolo26_ens better [-0.0222, -0.0125] | yolo26_ens better [-0.0119, -0.0012] | cannot separate [-0.0035, 0.0005] | yolo26_ens better [-0.0102, -0.0056] | none |
| yolo26_ens vs yolo26_evid_s2 | yolo26_ens better [-0.0201, -0.0091] | cannot separate [-0.0113, 0.0008] | cannot separate [-0.0053, 0.0004] | yolo26_ens better [-0.0105, -0.0039] | none |
| yolo26_ens vs yolo26_s1 | cannot separate [-0.0030, 0.0036] | cannot separate [-0.0074, 0.0002] | yolo26_ens better [-0.0067, -0.0033] | yolo26_ens better [-0.0064, -0.0023] | none |
| yolo26_ens vs yolo26_s2 | cannot separate [-0.0024, 0.0045] | yolo26_ens better [-0.0095, -0.0015] | yolo26_ens better [-0.0078, -0.0043] | yolo26_ens better [-0.0091, -0.0050] | none |
| yolo26_ens vs yolov8 | cannot separate [-0.0005, 0.0091] | cannot separate [-0.0048, 0.0061] | yolo26_ens better [-0.0100, -0.0043] | cannot separate [-0.0060, 0.0009] | none |
| yolo26_evid vs yolo26_evid_s1 | cannot separate [-0.0062, 0.0035] | yolo26_evid_s1 better [0.0069, 0.0189] | yolo26_evid_s1 better [0.0058, 0.0108] | yolo26_evid_s1 better [0.0009, 0.0069] | none |
| yolo26_evid vs yolo26_evid_s2 | cannot separate [-0.0037, 0.0059] | yolo26_evid_s2 better [0.0080, 0.0194] | yolo26_evid_s2 better [0.0052, 0.0097] | yolo26_evid_s2 better [0.0020, 0.0078] | none |
| yolo26_evid vs yolo26_s1 | yolo26_s1 better [0.0104, 0.0220] | yolo26_s1 better [0.0095, 0.0223] | yolo26_s1 better [0.0025, 0.0076] | yolo26_s1 better [0.0048, 0.0109] | none |
| yolo26_evid vs yolo26_s2 | yolo26_s2 better [0.0110, 0.0226] | yolo26_s2 better [0.0071, 0.0200] | yolo26_s2 better [0.0012, 0.0067] | yolo26_s2 better [0.0017, 0.0083] | none |
| yolo26_evid vs yolov8 | yolov8 better [0.0138, 0.0267] | yolov8 better [0.0131, 0.0269] | cannot separate [-0.0003, 0.0057] | yolov8 better [0.0057, 0.0131] | none |
| yolo26_evid_s1 vs yolo26_evid_s2 | cannot separate [-0.0027, 0.0069] | cannot separate [-0.0045, 0.0061] | cannot separate [-0.0037, 0.0013] | cannot separate [-0.0021, 0.0035] | none |
| yolo26_evid_s1 vs yolo26_s1 | yolo26_s1 better [0.0124, 0.0229] | cannot separate [-0.0030, 0.0090] | yolo26_evid_s1 better [-0.0056, -0.0010] | yolo26_s1 better [0.0012, 0.0067] | ma_brier |
| yolo26_evid_s1 vs yolo26_s2 | yolo26_s2 better [0.0134, 0.0233] | cannot separate [-0.0046, 0.0073] | yolo26_evid_s1 better [-0.0069, -0.0020] | cannot separate [-0.0017, 0.0039] | ma_brier |
| yolo26_evid_s1 vs yolov8 | yolov8 better [0.0161, 0.0272] | yolov8 better [0.0011, 0.0136] | yolo26_evid_s1 better [-0.0087, -0.0026] | yolov8 better [0.0021, 0.0092] | ma_brier |
| yolo26_evid_s2 vs yolo26_s1 | yolo26_s1 better [0.0097, 0.0202] | cannot separate [-0.0050, 0.0080] | cannot separate [-0.0051, 0.0005] | cannot separate [-0.0003, 0.0062] | none |
| yolo26_evid_s2 vs yolo26_s2 | yolo26_s2 better [0.0105, 0.0210] | cannot separate [-0.0058, 0.0053] | yolo26_evid_s2 better [-0.0061, -0.0007] | cannot separate [-0.0030, 0.0036] | ma_brier |
| yolo26_evid_s2 vs yolov8 | yolov8 better [0.0137, 0.0245] | cannot separate [-0.0007, 0.0123] | yolo26_evid_s2 better [-0.0076, -0.0015] | yolov8 better [0.0009, 0.0084] | ma_brier |
| yolo26_s1 vs yolo26_s2 | cannot separate [-0.0034, 0.0049] | cannot separate [-0.0068, 0.0027] | cannot separate [-0.0033, 0.0013] | cannot separate [-0.0054, 0.0001] | none |
| yolo26_s1 vs yolov8 | cannot separate [-0.0008, 0.0086] | cannot separate [-0.0014, 0.0095] | cannot separate [-0.0053, 0.0006] | cannot separate [-0.0022, 0.0051] | none |
| yolo26_s2 vs yolov8 | cannot separate [-0.0017, 0.0081] | yolov8 better [0.0008, 0.0120] | cannot separate [-0.0042, 0.0018] | yolov8 better [0.0009, 0.0083] | none |

- D-ECE cannot separate rtdetr and rtdetr_s1 (each at its own threshold).
- D-ECE cannot separate rtdetr and yolo26 (each at its own threshold).
- D-ECE cannot separate rtdetr and yolo26_ens (each at its own threshold).
- D-ECE cannot separate rtdetr and yolo26_s1 (each at its own threshold).
- D-ECE cannot separate rtdetr and yolo26_s2 (each at its own threshold).
- D-ECE cannot separate rtdetr_ens and rtdetr_s1 (each at its own threshold).
- D-ECE cannot separate rtdetr_ens and yolo26 (each at its own threshold).
- D-ECE cannot separate rtdetr_ens and yolo26_evid (each at its own threshold).
- D-ECE cannot separate rtdetr_ens and yolo26_evid_s2 (each at its own threshold).
- D-ECE cannot separate rtdetr_s1 and yolo26 (each at its own threshold).
- D-ECE cannot separate rtdetr_s2 and yolo26_evid (each at its own threshold).
- D-ECE cannot separate rtdetr_s2 and yolo26_evid_s1 (each at its own threshold).
- D-ECE cannot separate rtdetr_s2 and yolo26_evid_s2 (each at its own threshold).
- D-ECE cannot separate yolo26 and yolo26_s1 (each at its own threshold).
- D-ECE cannot separate yolo26_ens and yolo26_s1 (each at its own threshold).
- D-ECE cannot separate yolo26_ens and yolo26_s2 (each at its own threshold).
- D-ECE cannot separate yolo26_ens and yolov8 (each at its own threshold).
- D-ECE cannot separate yolo26_evid and yolo26_evid_s1 (each at its own threshold).
- D-ECE cannot separate yolo26_evid and yolo26_evid_s2 (each at its own threshold).
- D-ECE cannot separate yolo26_evid_s1 and yolo26_evid_s2 (each at its own threshold).
- D-ECE cannot separate yolo26_s1 and yolo26_s2 (each at its own threshold).
- D-ECE cannot separate yolo26_s1 and yolov8 (each at its own threshold).
- D-ECE cannot separate yolo26_s2 and yolov8 (each at its own threshold).

**at matched false alarms per image** (budget 1.62 false alarms/image)

| pair (difference = first minus second) | D-ECE | MA-ECE | MA-Brier | OCE | flips vs D-ECE |
|---|---|---|---|---|---|
| rtdetr vs rtdetr_ens | cannot separate [-0.0011, 0.0061] | rtdetr_ens better [0.0039, 0.0118] | rtdetr_ens better [0.0041, 0.0090] | rtdetr_ens better [0.0033, 0.0095] | none |
| rtdetr vs rtdetr_s1 | cannot separate [-0.0021, 0.0072] | cannot separate [-0.0034, 0.0065] | cannot separate [-0.0043, 0.0024] | cannot separate [-0.0044, 0.0034] | none |
| rtdetr vs rtdetr_s2 | cannot separate [-0.0027, 0.0064] | cannot separate [-0.0042, 0.0060] | cannot separate [-0.0048, 0.0020] | cannot separate [-0.0074, 0.0013] | none |
| rtdetr vs yolo26 | cannot separate [-0.0000, 0.0113] | yolo26 better [0.0080, 0.0216] | yolo26 better [0.0166, 0.0252] | cannot separate [-0.0006, 0.0085] | none |
| rtdetr vs yolo26_ens | yolo26_ens better [0.0033, 0.0146] | yolo26_ens better [0.0132, 0.0268] | yolo26_ens better [0.0199, 0.0280] | yolo26_ens better [0.0007, 0.0098] | none |
| rtdetr vs yolo26_evid | rtdetr better [-0.0134, -0.0001] | cannot separate [-0.0066, 0.0082] | yolo26_evid better [0.0098, 0.0175] | rtdetr better [-0.0120, -0.0029] | ma_brier |
| rtdetr vs yolo26_evid_s1 | rtdetr better [-0.0156, -0.0021] | cannot separate [-0.0084, 0.0067] | yolo26_evid_s1 better [0.0092, 0.0172] | rtdetr better [-0.0159, -0.0069] | ma_brier |
| rtdetr vs yolo26_evid_s2 | cannot separate [-0.0126, 0.0016] | cannot separate [-0.0106, 0.0037] | yolo26_evid_s2 better [0.0056, 0.0130] | rtdetr better [-0.0169, -0.0082] | none |
| rtdetr vs yolo26_s1 | yolo26_s1 better [0.0031, 0.0148] | yolo26_s1 better [0.0085, 0.0221] | yolo26_s1 better [0.0148, 0.0229] | cannot separate [-0.0037, 0.0053] | none |
| rtdetr vs yolo26_s2 | yolo26_s2 better [0.0037, 0.0155] | yolo26_s2 better [0.0111, 0.0252] | yolo26_s2 better [0.0160, 0.0250] | cannot separate [-0.0047, 0.0049] | none |
| rtdetr vs yolov8 | yolov8 better [0.0078, 0.0200] | yolov8 better [0.0091, 0.0223] | yolov8 better [0.0091, 0.0174] | cannot separate [-0.0058, 0.0043] | none |
| rtdetr_ens vs rtdetr_s1 | cannot separate [-0.0038, 0.0037] | rtdetr_ens better [-0.0101, -0.0023] | rtdetr_ens better [-0.0100, -0.0052] | rtdetr_ens better [-0.0098, -0.0035] | none |
| rtdetr_ens vs rtdetr_s2 | cannot separate [-0.0042, 0.0029] | rtdetr_ens better [-0.0113, -0.0026] | rtdetr_ens better [-0.0106, -0.0054] | rtdetr_ens better [-0.0126, -0.0060] | none |
| rtdetr_ens vs yolo26 | cannot separate [-0.0028, 0.0093] | yolo26 better [0.0004, 0.0136] | yolo26 better [0.0103, 0.0182] | cannot separate [-0.0064, 0.0017] | none |
| rtdetr_ens vs yolo26_ens | yolo26_ens better [0.0003, 0.0120] | yolo26_ens better [0.0053, 0.0188] | yolo26_ens better [0.0135, 0.0213] | cannot separate [-0.0056, 0.0031] | none |
| rtdetr_ens vs yolo26_evid | rtdetr_ens better [-0.0159, -0.0017] | rtdetr_ens better [-0.0140, -0.0002] | yolo26_evid better [0.0037, 0.0112] | rtdetr_ens better [-0.0176, -0.0093] | ma_brier |
| rtdetr_ens vs yolo26_evid_s1 | rtdetr_ens better [-0.0183, -0.0041] | rtdetr_ens better [-0.0163, -0.0016] | yolo26_evid_s1 better [0.0030, 0.0106] | rtdetr_ens better [-0.0216, -0.0140] | ma_brier |
| rtdetr_ens vs yolo26_evid_s2 | rtdetr_ens better [-0.0140, -0.0007] | rtdetr_ens better [-0.0175, -0.0046] | cannot separate [-0.0007, 0.0068] | rtdetr_ens better [-0.0229, -0.0146] | none |
| rtdetr_ens vs yolo26_s1 | yolo26_s1 better [0.0006, 0.0128] | yolo26_s1 better [0.0010, 0.0148] | yolo26_s1 better [0.0084, 0.0158] | rtdetr_ens better [-0.0098, -0.0016] | oce_max |
| rtdetr_ens vs yolo26_s2 | yolo26_s2 better [0.0008, 0.0134] | yolo26_s2 better [0.0030, 0.0174] | yolo26_s2 better [0.0100, 0.0182] | rtdetr_ens better [-0.0103, -0.0017] | oce_max |
| rtdetr_ens vs yolov8 | yolov8 better [0.0051, 0.0176] | yolov8 better [0.0007, 0.0144] | yolov8 better [0.0029, 0.0102] | rtdetr_ens better [-0.0115, -0.0032] | oce_max |
| rtdetr_s1 vs rtdetr_s2 | cannot separate [-0.0053, 0.0038] | cannot separate [-0.0059, 0.0044] | cannot separate [-0.0039, 0.0028] | cannot separate [-0.0069, 0.0010] | none |
| rtdetr_s1 vs yolo26 | cannot separate [-0.0028, 0.0092] | yolo26 better [0.0068, 0.0203] | yolo26 better [0.0179, 0.0259] | yolo26 better [0.0000, 0.0090] | none |
| rtdetr_s1 vs yolo26_ens | yolo26_ens better [0.0002, 0.0122] | yolo26_ens better [0.0112, 0.0255] | yolo26_ens better [0.0210, 0.0292] | yolo26_ens better [0.0011, 0.0100] | none |
| rtdetr_s1 vs yolo26_evid | rtdetr_s1 better [-0.0159, -0.0014] | cannot separate [-0.0077, 0.0068] | yolo26_evid better [0.0110, 0.0187] | rtdetr_s1 better [-0.0115, -0.0027] | ma_brier |
| rtdetr_s1 vs yolo26_evid_s1 | rtdetr_s1 better [-0.0179, -0.0034] | cannot separate [-0.0093, 0.0053] | yolo26_evid_s1 better [0.0102, 0.0184] | rtdetr_s1 better [-0.0154, -0.0069] | ma_brier |
| rtdetr_s1 vs yolo26_evid_s2 | rtdetr_s1 better [-0.0144, -0.0006] | cannot separate [-0.0112, 0.0025] | yolo26_evid_s2 better [0.0065, 0.0140] | rtdetr_s1 better [-0.0164, -0.0080] | ma_brier |
| rtdetr_s1 vs yolo26_s1 | yolo26_s1 better [0.0005, 0.0128] | yolo26_s1 better [0.0073, 0.0213] | yolo26_s1 better [0.0158, 0.0237] | cannot separate [-0.0034, 0.0052] | none |
| rtdetr_s1 vs yolo26_s2 | yolo26_s2 better [0.0008, 0.0131] | yolo26_s2 better [0.0100, 0.0238] | yolo26_s2 better [0.0175, 0.0258] | cannot separate [-0.0037, 0.0048] | none |
| rtdetr_s1 vs yolov8 | yolov8 better [0.0049, 0.0175] | yolov8 better [0.0070, 0.0206] | yolov8 better [0.0102, 0.0179] | cannot separate [-0.0051, 0.0041] | none |
| rtdetr_s2 vs yolo26 | cannot separate [-0.0021, 0.0098] | yolo26 better [0.0067, 0.0216] | yolo26 better [0.0184, 0.0265] | yolo26 better [0.0023, 0.0116] | none |
| rtdetr_s2 vs yolo26_ens | yolo26_ens better [0.0011, 0.0129] | yolo26_ens better [0.0121, 0.0265] | yolo26_ens better [0.0213, 0.0295] | yolo26_ens better [0.0037, 0.0134] | none |
| rtdetr_s2 vs yolo26_evid | rtdetr_s2 better [-0.0152, -0.0013] | cannot separate [-0.0071, 0.0074] | yolo26_evid better [0.0119, 0.0193] | cannot separate [-0.0085, 0.0004] | ma_brier |
| rtdetr_s2 vs yolo26_evid_s1 | rtdetr_s2 better [-0.0169, -0.0024] | cannot separate [-0.0093, 0.0061] | yolo26_evid_s1 better [0.0104, 0.0187] | rtdetr_s2 better [-0.0126, -0.0037] | ma_brier |
| rtdetr_s2 vs yolo26_evid_s2 | cannot separate [-0.0139, 0.0002] | cannot separate [-0.0109, 0.0030] | yolo26_evid_s2 better [0.0071, 0.0148] | rtdetr_s2 better [-0.0135, -0.0051] | none |
| rtdetr_s2 vs yolo26_s1 | yolo26_s1 better [0.0012, 0.0133] | yolo26_s1 better [0.0082, 0.0223] | yolo26_s1 better [0.0165, 0.0238] | cannot separate [-0.0009, 0.0081] | none |
| rtdetr_s2 vs yolo26_s2 | yolo26_s2 better [0.0016, 0.0145] | yolo26_s2 better [0.0100, 0.0248] | yolo26_s2 better [0.0177, 0.0263] | cannot separate [-0.0013, 0.0081] | none |
| rtdetr_s2 vs yolov8 | yolov8 better [0.0062, 0.0183] | yolov8 better [0.0075, 0.0214] | yolov8 better [0.0105, 0.0186] | cannot separate [-0.0027, 0.0073] | none |
| yolo26 vs yolo26_ens | cannot separate [-0.0004, 0.0061] | yolo26_ens better [0.0011, 0.0092] | yolo26_ens better [0.0013, 0.0046] | cannot separate [-0.0007, 0.0035] | none |
| yolo26 vs yolo26_evid | yolo26 better [-0.0181, -0.0064] | yolo26 better [-0.0204, -0.0076] | yolo26 better [-0.0098, -0.0045] | yolo26 better [-0.0145, -0.0082] | none |
| yolo26 vs yolo26_evid_s1 | yolo26 better [-0.0199, -0.0090] | yolo26 better [-0.0216, -0.0096] | yolo26 better [-0.0100, -0.0051] | yolo26 better [-0.0182, -0.0121] | none |
| yolo26 vs yolo26_evid_s2 | yolo26 better [-0.0167, -0.0052] | yolo26 better [-0.0246, -0.0115] | yolo26 better [-0.0141, -0.0084] | yolo26 better [-0.0198, -0.0126] | none |
| yolo26 vs yolo26_s1 | cannot separate [-0.0012, 0.0076] | cannot separate [-0.0042, 0.0052] | yolo26 better [-0.0044, -0.0002] | yolo26 better [-0.0061, -0.0006] | none |
| yolo26 vs yolo26_s2 | cannot separate [-0.0003, 0.0081] | cannot separate [-0.0018, 0.0081] | cannot separate [-0.0028, 0.0019] | yolo26 better [-0.0068, -0.0010] | none |
| yolo26 vs yolov8 | yolov8 better [0.0031, 0.0133] | cannot separate [-0.0047, 0.0065] | yolo26 better [-0.0108, -0.0047] | yolo26 better [-0.0083, -0.0016] | ma_brier, oce_max |
| yolo26_ens vs yolo26_evid | yolo26_ens better [-0.0204, -0.0095] | yolo26_ens better [-0.0247, -0.0129] | yolo26_ens better [-0.0123, -0.0075] | yolo26_ens better [-0.0154, -0.0095] | none |
| yolo26_ens vs yolo26_evid_s1 | yolo26_ens better [-0.0225, -0.0121] | yolo26_ens better [-0.0263, -0.0153] | yolo26_ens better [-0.0129, -0.0086] | yolo26_ens better [-0.0190, -0.0139] | none |
| yolo26_ens vs yolo26_evid_s2 | yolo26_ens better [-0.0195, -0.0079] | yolo26_ens better [-0.0287, -0.0162] | yolo26_ens better [-0.0170, -0.0115] | yolo26_ens better [-0.0210, -0.0145] | none |
| yolo26_ens vs yolo26_s1 | cannot separate [-0.0033, 0.0036] | yolo26_ens better [-0.0084, -0.0005] | yolo26_ens better [-0.0070, -0.0036] | yolo26_ens better [-0.0071, -0.0026] | none |
| yolo26_ens vs yolo26_s2 | cannot separate [-0.0024, 0.0043] | cannot separate [-0.0056, 0.0024] | yolo26_ens better [-0.0051, -0.0017] | yolo26_ens better [-0.0073, -0.0030] | none |
| yolo26_ens vs yolov8 | yolov8 better [0.0009, 0.0100] | cannot separate [-0.0099, 0.0015] | yolo26_ens better [-0.0135, -0.0077] | yolo26_ens better [-0.0095, -0.0025] | ma_brier, oce_max |
| yolo26_evid vs yolo26_evid_s1 | cannot separate [-0.0066, 0.0028] | cannot separate [-0.0071, 0.0038] | cannot separate [-0.0032, 0.0017] | yolo26_evid better [-0.0068, -0.0013] | none |
| yolo26_evid vs yolo26_evid_s2 | cannot separate [-0.0035, 0.0059] | cannot separate [-0.0089, 0.0009] | yolo26_evid better [-0.0067, -0.0021] | yolo26_evid better [-0.0080, -0.0022] | none |
| yolo26_evid vs yolo26_s1 | yolo26_s1 better [0.0096, 0.0211] | yolo26_s1 better [0.0084, 0.0208] | yolo26_s1 better [0.0022, 0.0072] | yolo26_s1 better [0.0048, 0.0107] | none |
| yolo26_evid vs yolo26_s2 | yolo26_s2 better [0.0104, 0.0219] | yolo26_s2 better [0.0112, 0.0237] | yolo26_s2 better [0.0039, 0.0093] | yolo26_s2 better [0.0043, 0.0105] | none |
| yolo26_evid vs yolov8 | yolov8 better [0.0140, 0.0270] | yolov8 better [0.0080, 0.0216] | cannot separate [-0.0037, 0.0020] | yolov8 better [0.0027, 0.0100] | none |
| yolo26_evid_s1 vs yolo26_evid_s2 | cannot separate [-0.0020, 0.0074] | cannot separate [-0.0077, 0.0027] | yolo26_evid_s1 better [-0.0060, -0.0011] | cannot separate [-0.0039, 0.0020] | none |
| yolo26_evid_s1 vs yolo26_s1 | yolo26_s1 better [0.0120, 0.0228] | yolo26_s1 better [0.0108, 0.0219] | yolo26_s1 better [0.0033, 0.0079] | yolo26_s1 better [0.0093, 0.0146] | none |
| yolo26_evid_s1 vs yolo26_s2 | yolo26_s2 better [0.0131, 0.0230] | yolo26_s2 better [0.0132, 0.0244] | yolo26_s2 better [0.0050, 0.0097] | yolo26_s2 better [0.0085, 0.0142] | none |
| yolo26_evid_s1 vs yolov8 | yolov8 better [0.0172, 0.0281] | yolov8 better [0.0105, 0.0227] | cannot separate [-0.0030, 0.0031] | yolov8 better [0.0070, 0.0142] | none |
| yolo26_evid_s2 vs yolo26_s1 | yolo26_s1 better [0.0087, 0.0193] | yolo26_s1 better [0.0123, 0.0247] | yolo26_s1 better [0.0066, 0.0118] | yolo26_s1 better [0.0097, 0.0163] | none |
| yolo26_evid_s2 vs yolo26_s2 | yolo26_s2 better [0.0093, 0.0206] | yolo26_s2 better [0.0152, 0.0272] | yolo26_s2 better [0.0083, 0.0138] | yolo26_s2 better [0.0091, 0.0158] | none |
| yolo26_evid_s2 vs yolov8 | yolov8 better [0.0137, 0.0250] | yolov8 better [0.0120, 0.0253] | yolov8 better [0.0006, 0.0067] | yolov8 better [0.0077, 0.0151] | none |
| yolo26_s1 vs yolo26_s2 | cannot separate [-0.0035, 0.0049] | cannot separate [-0.0024, 0.0075] | cannot separate [-0.0002, 0.0042] | cannot separate [-0.0031, 0.0026] | none |
| yolo26_s1 vs yolov8 | yolov8 better [0.0002, 0.0096] | cannot separate [-0.0053, 0.0050] | yolo26_s1 better [-0.0084, -0.0025] | cannot separate [-0.0052, 0.0020] | ma_brier |
| yolo26_s2 vs yolov8 | cannot separate [-0.0005, 0.0093] | cannot separate [-0.0081, 0.0037] | yolo26_s2 better [-0.0105, -0.0041] | cannot separate [-0.0046, 0.0030] | none |

- D-ECE cannot separate rtdetr and rtdetr_ens (at matched false alarms per image).
- D-ECE cannot separate rtdetr and rtdetr_s1 (at matched false alarms per image).
- D-ECE cannot separate rtdetr and rtdetr_s2 (at matched false alarms per image).
- D-ECE cannot separate rtdetr and yolo26 (at matched false alarms per image).
- D-ECE cannot separate rtdetr and yolo26_evid_s2 (at matched false alarms per image).
- D-ECE cannot separate rtdetr_ens and rtdetr_s1 (at matched false alarms per image).
- D-ECE cannot separate rtdetr_ens and rtdetr_s2 (at matched false alarms per image).
- D-ECE cannot separate rtdetr_ens and yolo26 (at matched false alarms per image).
- D-ECE cannot separate rtdetr_s1 and rtdetr_s2 (at matched false alarms per image).
- D-ECE cannot separate rtdetr_s1 and yolo26 (at matched false alarms per image).
- D-ECE cannot separate rtdetr_s2 and yolo26 (at matched false alarms per image).
- D-ECE cannot separate rtdetr_s2 and yolo26_evid_s2 (at matched false alarms per image).
- D-ECE cannot separate yolo26 and yolo26_ens (at matched false alarms per image).
- D-ECE cannot separate yolo26 and yolo26_s1 (at matched false alarms per image).
- D-ECE cannot separate yolo26 and yolo26_s2 (at matched false alarms per image).
- D-ECE cannot separate yolo26_ens and yolo26_s1 (at matched false alarms per image).
- D-ECE cannot separate yolo26_ens and yolo26_s2 (at matched false alarms per image).
- D-ECE cannot separate yolo26_evid and yolo26_evid_s1 (at matched false alarms per image).
- D-ECE cannot separate yolo26_evid and yolo26_evid_s2 (at matched false alarms per image).
- D-ECE cannot separate yolo26_evid_s1 and yolo26_evid_s2 (at matched false alarms per image).
- D-ECE cannot separate yolo26_s1 and yolo26_s2 (at matched false alarms per image).
- D-ECE cannot separate yolo26_s2 and yolov8 (at matched false alarms per image).

Ranking flip relative to D-ECE (both intervals exclude zero, opposite signs) survives matched cost: **True**

#### MA-Brier decomposition

Each term is its contribution to MA-Brier (share of MA-Brier in brackets); the three sum to MA-Brier. OCE averages over vehicles only, so it has no false-alarm term.

| detector | where | threshold | false alarms/image | MA-Brier | false-alarm term | matched-detection term | miss term | OCE |
|---|---|---|---|---|---|---|---|---|
| yolo26 | own threshold | 0.49 | 1.65 | 0.2386 | 0.0666 (0.28) | 0.0247 (0.10) | 0.1473 (0.62) | 0.2255 |
| rtdetr | own threshold | 0.73 | 1.49 | 0.2649 | 0.0873 (0.33) | 0.0085 (0.03) | 0.1692 (0.64) | 0.2366 |
| yolov8 | own threshold | 0.52 | 1.61 | 0.2478 | 0.0696 (0.28) | 0.0217 (0.09) | 0.1564 (0.63) | 0.2317 |
| yolo26_s1 | own threshold | 0.51 | 1.52 | 0.2456 | 0.0642 (0.26) | 0.0240 (0.10) | 0.1574 (0.64) | 0.2335 |
| yolo26_s2 | own threshold | 0.51 | 1.46 | 0.2467 | 0.0622 (0.25) | 0.0242 (0.10) | 0.1602 (0.65) | 0.2363 |
| rtdetr_s1 | own threshold | 0.72 | 1.62 | 0.2611 | 0.0927 (0.35) | 0.0089 (0.03) | 0.1596 (0.61) | 0.2303 |
| rtdetr_s2 | own threshold | 0.70 | 1.77 | 0.2571 | 0.0978 (0.38) | 0.0095 (0.04) | 0.1498 (0.58) | 0.2265 |
| yolo26_evid | own threshold | 0.49 | 1.52 | 0.2506 | 0.0591 (0.24) | 0.0319 (0.13) | 0.1597 (0.64) | 0.2413 |
| yolo26_evid_s1 | own threshold | 0.43 | 1.73 | 0.2421 | 0.0597 (0.25) | 0.0378 (0.16) | 0.1446 (0.60) | 0.2373 |
| yolo26_evid_s2 | own threshold | 0.45 | 1.80 | 0.2432 | 0.0631 (0.26) | 0.0345 (0.14) | 0.1456 (0.60) | 0.2365 |
| yolo26_ens | own threshold | 0.50 | 1.51 | 0.2407 | 0.0625 (0.26) | 0.0249 (0.10) | 0.1533 (0.64) | 0.2293 |
| rtdetr_ens | own threshold | 0.70 | 1.67 | 0.2521 | 0.0927 (0.37) | 0.0098 (0.04) | 0.1497 (0.59) | 0.2212 |
| yolo26 | matched cost | 0.50 | 1.62 | 0.2396 | 0.0662 (0.28) | 0.0244 (0.10) | 0.1491 (0.62) | 0.2264 |
| rtdetr | matched cost | 0.72 | 1.62 | 0.2605 | 0.0918 (0.35) | 0.0090 (0.03) | 0.1597 (0.61) | 0.2304 |
| yolov8 | matched cost | 0.52 | 1.62 | 0.2472 | 0.0698 (0.28) | 0.0218 (0.09) | 0.1556 (0.63) | 0.2312 |
| yolo26_s1 | matched cost | 0.49 | 1.62 | 0.2419 | 0.0657 (0.27) | 0.0250 (0.10) | 0.1512 (0.62) | 0.2299 |
| yolo26_s2 | matched cost | 0.48 | 1.62 | 0.2400 | 0.0646 (0.27) | 0.0261 (0.11) | 0.1493 (0.62) | 0.2303 |
| rtdetr_s1 | matched cost | 0.72 | 1.62 | 0.2616 | 0.0924 (0.35) | 0.0088 (0.03) | 0.1604 (0.61) | 0.2309 |
| rtdetr_s2 | matched cost | 0.72 | 1.62 | 0.2619 | 0.0925 (0.35) | 0.0089 (0.03) | 0.1606 (0.61) | 0.2335 |
| yolo26_evid | matched cost | 0.47 | 1.62 | 0.2465 | 0.0605 (0.25) | 0.0330 (0.13) | 0.1530 (0.62) | 0.2377 |
| yolo26_evid_s1 | matched cost | 0.45 | 1.62 | 0.2473 | 0.0584 (0.24) | 0.0360 (0.15) | 0.1528 (0.62) | 0.2417 |
| yolo26_evid_s2 | matched cost | 0.48 | 1.62 | 0.2510 | 0.0608 (0.24) | 0.0321 (0.13) | 0.1581 (0.63) | 0.2428 |
| yolo26_ens | matched cost | 0.48 | 1.62 | 0.2367 | 0.0640 (0.27) | 0.0261 (0.11) | 0.1465 (0.62) | 0.2252 |
| rtdetr_ens | matched cost | 0.71 | 1.62 | 0.2539 | 0.0908 (0.36) | 0.0095 (0.04) | 0.1535 (0.60) | 0.2240 |

## Training-seed variability (matching: class_agnostic)

Mean, SD across seeds (n-1 denominator) and [min, max]. With three seeds the SD is itself rough: read it as the size of run-to-run noise.

| quantity | yolo26 (yolo26, yolo26_s1, yolo26_s2) | rtdetr (rtdetr, rtdetr_s1, rtdetr_s2) | yolo26_evid (yolo26_evid, yolo26_evid_s1, yolo26_evid_s2) |
|---|---|---|---|
| operating threshold | 0.46 ± 0.00 [0.46, 0.46] (n=3) | 0.70 ± 0.01 [0.69, 0.70] (n=3) | 0.44 ± 0.01 [0.43, 0.45] (n=3) |
| AP50 (class-agnostic) | 0.9359 ± 0.0005 [0.9355, 0.9365] (n=3) | 0.9243 ± 0.0007 [0.9238, 0.9252] (n=3) | 0.9314 ± 0.0017 [0.9301, 0.9333] (n=3) |
| recall at threshold | 0.89 ± 0.00 [0.88, 0.89] (n=3) | 0.88 ± 0.00 [0.88, 0.88] (n=3) | 0.88 ± 0.00 [0.88, 0.88] (n=3) |
| false alarms/image at threshold | 1.22 ± 0.05 [1.16, 1.26] (n=3) | 1.29 ± 0.06 [1.24, 1.35] (n=3) | 1.23 ± 0.04 [1.19, 1.26] (n=3) |
| RQ1 odds of a miss per +0.1 occlusion | 1.1906 ± 0.0049 [1.1862, 1.1958] (n=3) | 1.2029 ± 0.0050 [1.1977, 1.2075] (n=3) | 1.1916 ± 0.0095 [1.1833, 1.2019] (n=3) |
| RQ1 partial Spearman | 0.2383 ± 0.0012 [0.2375, 0.2397] (n=3) | 0.2491 ± 0.0013 [0.2480, 0.2505] (n=3) | 0.2340 ± 0.0039 [0.2311, 0.2384] (n=3) |
| conformal global_confidence, target 0.95: coverage | 0.9487 ± 0.0008 [0.9477, 0.9492] (n=3) | 0.9485 ± 0.0014 [0.9472, 0.9500] (n=3) | 0.9484 ± 0.0006 [0.9480, 0.9491] (n=3) |
| conformal global_confidence, target 0.95: flags/image | 2.41 ± 0.06 [2.37, 2.48] (n=3) | 3.37 ± 0.12 [3.25, 3.49] (n=3) | 2.53 ± 0.06 [2.47, 2.58] (n=3) |
| conformal mondrian_confidence, target 0.95: coverage | 0.9488 ± 0.0013 [0.9473, 0.9497] (n=3) | 0.9502 ± 0.0005 [0.9497, 0.9506] (n=3) | 0.9490 ± 0.0003 [0.9487, 0.9493] (n=3) |
| conformal mondrian_confidence, target 0.95: flags/image | 2.68 ± 0.05 [2.64, 2.73] (n=3) | 3.98 ± 0.14 [3.90, 4.14] (n=3) | 3.06 ± 0.25 [2.86, 3.34] (n=3) |
| conformal global_occlusion_aware, target 0.95: coverage | 0.9484 ± 0.0011 [0.9472, 0.9493] (n=3) | 0.9485 ± 0.0014 [0.9472, 0.9500] (n=3) | 0.9484 ± 0.0008 [0.9477, 0.9492] (n=3) |
| conformal global_occlusion_aware, target 0.95: flags/image | 2.27 ± 0.04 [2.24, 2.31] (n=3) | 2.69 ± 0.09 [2.60, 2.78] (n=3) | 2.45 ± 0.10 [2.34, 2.53] (n=3) |
| conformal global_confidence, target 0.90: coverage | 0.9011 ± 0.0017 [0.8991, 0.9023] (n=3) | 0.9027 ± 0.0013 [0.9019, 0.9042] (n=3) | 0.9001 ± 0.0023 [0.8976, 0.9020] (n=3) |
| conformal global_confidence, target 0.90: flags/image | 0.36 ± 0.03 [0.32, 0.37] (n=3) | 0.71 ± 0.06 [0.63, 0.76] (n=3) | 0.46 ± 0.06 [0.39, 0.50] (n=3) |
| conformal mondrian_confidence, target 0.90: coverage | 0.9010 ± 0.0015 [0.8997, 0.9027] (n=3) | 0.9030 ± 0.0004 [0.9026, 0.9032] (n=3) | 0.9001 ± 0.0034 [0.8962, 0.9027] (n=3) |
| conformal mondrian_confidence, target 0.90: flags/image | 0.54 ± 0.04 [0.51, 0.58] (n=3) | 0.84 ± 0.03 [0.82, 0.87] (n=3) | 0.63 ± 0.09 [0.53, 0.69] (n=3) |
| conformal global_occlusion_aware, target 0.90: coverage | 0.9008 ± 0.0004 [0.9003, 0.9011] (n=3) | 0.9024 ± 0.0020 [0.9011, 0.9047] (n=3) | 0.8995 ± 0.0024 [0.8973, 0.9020] (n=3) |
| conformal global_occlusion_aware, target 0.90: flags/image | 0.41 ± 0.03 [0.39, 0.45] (n=3) | 0.71 ± 0.06 [0.64, 0.75] (n=3) | 0.52 ± 0.08 [0.44, 0.59] (n=3) |
| RQ2 mondrian_rank vs confidence, all misses | -0.0353 ± 0.0088 [-0.0414, -0.0252] (n=3) | -0.0328 ± 0.0049 [-0.0376, -0.0279] (n=3) | -0.0484 ± 0.0029 [-0.0517, -0.0465] (n=3) |
| RQ2 mondrian_rank vs confidence, occluded misses | 0.1169 ± 0.0022 [0.1144, 0.1186] (n=3) | 0.1260 ± 0.0094 [0.1175, 0.1360] (n=3) | 0.1077 ± 0.0083 [0.0994, 0.1160] (n=3) |
| RQ2 default score vs confidence, all misses | 0.0331 ± 0.0040 [0.0296, 0.0375] (n=3) | 0.0571 ± 0.0003 [0.0569, 0.0574] (n=3) | 0.0321 ± 0.0015 [0.0305, 0.0335] (n=3) |
| RQ2 default score vs confidence, occluded misses | 0.0078 ± 0.0060 [0.0026, 0.0143] (n=3) | 0.0525 ± 0.0019 [0.0505, 0.0542] (n=3) | 0.0039 ± 0.0054 [-0.0016, 0.0093] (n=3) |
| D-ECE | 0.0717 ± 0.0048 [0.0666, 0.0762] (n=3) | 0.0440 ± 0.0025 [0.0424, 0.0468] (n=3) | 0.1017 ± 0.0079 [0.0961, 0.1108] (n=3) |
| MA-ECE | 0.1650 ± 0.0066 [0.1587, 0.1719] (n=3) | 0.1480 ± 0.0008 [0.1472, 0.1488] (n=3) | 0.1966 ± 0.0065 [0.1925, 0.2041] (n=3) |
| MA-Brier | 0.1797 ± 0.0012 [0.1785, 0.1810] (n=3) | 0.1914 ± 0.0003 [0.1911, 0.1917] (n=3) | 0.1920 ± 0.0010 [0.1913, 0.1931] (n=3) |
| OCE | 0.1473 ± 0.0026 [0.1448, 0.1500] (n=3) | 0.1332 ± 0.0021 [0.1311, 0.1354] (n=3) | 0.1645 ± 0.0015 [0.1630, 0.1659] (n=3) |
