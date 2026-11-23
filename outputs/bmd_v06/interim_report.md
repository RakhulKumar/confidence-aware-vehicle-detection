# v0.6 interim report: experiment `bmd_v06` (2026-11-23 09:34)

Written automatically by scripts/21_interim_report.py when jobs a) to c) finished; numbers are copied from the result files named in the script. Interpretation is left to review.

## a) NMS pilot (yolo26 seed 0, BMD-45 test, class-agnostic)

Pre-NMS candidates: 251 per image (cap 1000). Operating threshold 0.46; 1779 missed vehicles.

**Why each missed vehicle is missed** (share of misses):

| annotated occlusion | misses | removed_by_nms | kept_below_threshold | suppressed_low_confidence | never_proposed |
|---|---|---|---|---|---|
| all | 1779 | 0.031 | 0.880 | 0.012 | 0.077 |
| o_j 0.0-0.2 | 1103 | 0.015 | 0.888 | 0.014 | 0.084 |
| o_j 0.2-0.4 | 271 | 0.018 | 0.923 | 0.011 | 0.048 |
| o_j 0.4-0.6 | 201 | 0.040 | 0.900 | 0.005 | 0.055 |
| o_j 0.6-0.8 | 119 | 0.092 | 0.815 | 0.008 | 0.084 |
| o_j 0.8-1.0 | 85 | 0.176 | 0.694 | 0.012 | 0.118 |

**Suppression variants at matched false alarms** (recall; recall by annotated occlusion bucket):

| variant | AP50 | false alarms/image | recall | recall by o_j bucket |
|---|---|---|---|---|
| standard_cached (Ultralytics NMS 0.7, max_det 300) | 0.9359 | 1.26 | 0.8895 | 0.908 / 0.874 / 0.828 / 0.732 / 0.723 |
| standard_cached (Ultralytics NMS 0.7, max_det 300) | 0.9359 | 1.76 | 0.9177 | 0.933 / 0.907 / 0.867 / 0.779 / 0.775 |
| standard_cached (Ultralytics NMS 0.7, max_det 300) | 0.9359 | 2.26 | 0.9333 | 0.945 / 0.929 / 0.896 / 0.818 / 0.808 |
| hard_nms_0.5 | 0.9364 | 1.26 | 0.8897 | 0.909 / 0.874 / 0.827 / 0.723 / 0.723 |
| hard_nms_0.5 | 0.9364 | 1.76 | 0.9183 | 0.934 / 0.907 / 0.867 / 0.777 / 0.779 |
| hard_nms_0.5 | 0.9364 | 2.26 | 0.9344 | 0.947 / 0.931 / 0.896 / 0.806 / 0.805 |
| hard_nms_0.7 | 0.9359 | 1.26 | 0.8895 | 0.908 / 0.874 / 0.828 / 0.732 / 0.723 |
| hard_nms_0.7 | 0.9359 | 1.76 | 0.9177 | 0.933 / 0.907 / 0.867 / 0.779 / 0.775 |
| hard_nms_0.7 | 0.9359 | 2.26 | 0.9333 | 0.945 / 0.929 / 0.896 / 0.818 / 0.808 |
| hard_nms_0.8 | 0.9347 | 1.26 | 0.8881 | 0.907 / 0.872 / 0.826 / 0.730 / 0.730 |
| hard_nms_0.8 | 0.9347 | 1.76 | 0.9159 | 0.932 / 0.904 / 0.865 / 0.773 / 0.782 |
| hard_nms_0.8 | 0.9347 | 2.26 | 0.9316 | 0.944 / 0.927 / 0.894 / 0.815 / 0.808 |
| soft_nms_gaussian_0.5 | 0.9287 | 1.26 | 0.8886 | 0.911 / 0.871 / 0.815 / 0.687 / 0.717 |
| soft_nms_gaussian_0.5 | 0.9287 | 1.76 | 0.9173 | 0.935 / 0.907 / 0.857 / 0.755 / 0.769 |
| soft_nms_gaussian_0.5 | 0.9287 | 2.26 | 0.9331 | 0.946 / 0.928 / 0.891 / 0.802 / 0.798 |

## b) Evidential pilot (yolo26_evid: YOLO26, Beta evidence head, current fit set)

AP50 class-agnostic 0.9307 vs yolo26 family mean 0.9359 (gap -0.0052); class-aware 0.7486. Threshold 0.45: recall 0.883, 1.23 false alarms/image. final v0.6 model, trained on Beocat (L40S) with the same batch and patience as the yolo26 family (configs/beocat_batch.yaml; patience 10).

Per vehicle (its matched candidate), Spearman and partial Spearman controlling box size, image-bootstrap 95% CI:

| association | Spearman | partial (size) |
|---|---|---|
| aleatoric vs annotated_occlusion | 0.187 [0.169, 0.202] | 0.178 [0.161, 0.193] |
| aleatoric vs class_rarity | -0.005 [-0.026, 0.014] | 0.163 [0.137, 0.184] |
| epistemic vs annotated_occlusion | 0.215 [0.200, 0.229] | 0.210 [0.194, 0.224] |
| epistemic vs class_rarity | 0.343 [0.325, 0.365] | 0.405 [0.384, 0.428] |

Uncertainty split: Spearman(aleatoric, epistemic) = 0.811; rho(epistemic, rarity) - rho(aleatoric, rarity) = 0.349 [0.339, 0.357]; rho(aleatoric, o_j) - rho(epistemic, o_j) = -0.028 [-0.037, -0.017].

These are the final v0.6 evidential models (one of three seeds); the pilot's gate and the RT-DETR feasibility note do not apply to them.

### evidential: rarity diagnostic (calib, yolo26_evid)

Calib role only, 15649 vehicles with a box, threshold 0.45. Hypothesis: evidence is per fine class, so the common, confusable car subtypes split it. Spearman with image-bootstrap 95% CI.

| group | n | vs fine-class rarity | vs coarse-class rarity |
|---|---|---|---|
| (a) epistemic, all vehicles | 15649 | 0.365 [0.349, 0.382] | 0.198 [0.181, 0.216] |
| (a) aleatoric, all vehicles | 15649 | 0.008 [-0.012, 0.023] | -0.106 [-0.126, -0.092] |
| (b) epistemic, confusion misses | 803 | -0.078 [-0.133, -0.011] | -0.152 [-0.210, -0.087] |
| (b) epistemic, visibility misses | 1519 | 0.318 [0.268, 0.372] | 0.235 [0.184, 0.286] |

(c) mean epistemic, car subtypes (Hatchback, Sedan, SUV) 0.00992 vs rare classes (fit share < 2%: Mini-bus, Tempo-traveller, Van) 0.00612; difference 0.00380 [0.00347, 0.00422].

| class | fit share | n | mean epistemic | mean aleatoric |
|---|---|---|---|---|
| Two-wheeler | 0.459 | 7260 | 0.00490 | 0.1416 |
| Three-wheeler | 0.176 | 2773 | 0.00492 | 0.1132 |
| Hatchback | 0.094 | 1539 | 0.01012 | 0.1593 |
| Sedan | 0.053 | 896 | 0.00892 | 0.1458 |
| LCV | 0.049 | 786 | 0.00775 | 0.1376 |
| SUV | 0.042 | 653 | 0.01083 | 0.1702 |
| Bus | 0.037 | 574 | 0.00627 | 0.1238 |
| Truck | 0.027 | 400 | 0.00764 | 0.1376 |
| MUV | 0.024 | 391 | 0.00918 | 0.1498 |
| Tempo-traveller | 0.011 | 148 | 0.00480 | 0.0970 |
| Van | 0.011 | 174 | 0.00635 | 0.1546 |
| Mini-bus | 0.004 | 55 | 0.00893 | 0.2011 |

## b) Evidential pilot (yolo26_evid_s1: YOLO26, Beta evidence head, current fit set)

AP50 class-agnostic 0.9333 vs yolo26 family mean 0.9359 (gap -0.0027); class-aware 0.7453. Threshold 0.43: recall 0.883, 1.19 false alarms/image. final v0.6 model, trained on Beocat (L40S) with the same batch and patience as the yolo26 family (configs/beocat_batch.yaml; patience 10).

Per vehicle (its matched candidate), Spearman and partial Spearman controlling box size, image-bootstrap 95% CI:

| association | Spearman | partial (size) |
|---|---|---|
| aleatoric vs annotated_occlusion | 0.178 [0.162, 0.194] | 0.169 [0.153, 0.184] |
| aleatoric vs class_rarity | 0.042 [0.023, 0.059] | 0.218 [0.198, 0.236] |
| epistemic vs annotated_occlusion | 0.207 [0.190, 0.221] | 0.199 [0.183, 0.213] |
| epistemic vs class_rarity | 0.358 [0.339, 0.377] | 0.450 [0.432, 0.467] |

Uncertainty split: Spearman(aleatoric, epistemic) = 0.844; rho(epistemic, rarity) - rho(aleatoric, rarity) = 0.316 [0.306, 0.324]; rho(aleatoric, o_j) - rho(epistemic, o_j) = -0.028 [-0.037, -0.019].

These are the final v0.6 evidential models (one of three seeds); the pilot's gate and the RT-DETR feasibility note do not apply to them.

### evidential: rarity diagnostic (calib, yolo26_evid_s1)

Calib role only, 15637 vehicles with a box, threshold 0.43. Hypothesis: evidence is per fine class, so the common, confusable car subtypes split it. Spearman with image-bootstrap 95% CI.

| group | n | vs fine-class rarity | vs coarse-class rarity |
|---|---|---|---|
| (a) epistemic, all vehicles | 15637 | 0.382 [0.366, 0.399] | 0.255 [0.240, 0.271] |
| (a) aleatoric, all vehicles | 15637 | 0.064 [0.046, 0.081] | -0.023 [-0.043, -0.007] |
| (b) epistemic, confusion misses | 825 | -0.048 [-0.111, 0.024] | -0.155 [-0.216, -0.092] |
| (b) epistemic, visibility misses | 1458 | 0.322 [0.266, 0.370] | 0.247 [0.197, 0.293] |

(c) mean epistemic, car subtypes (Hatchback, Sedan, SUV) 0.01005 vs rare classes (fit share < 2%: Mini-bus, Tempo-traveller, Van) 0.00652; difference 0.00353 [0.00316, 0.00392].

| class | fit share | n | mean epistemic | mean aleatoric |
|---|---|---|---|---|
| Two-wheeler | 0.459 | 7258 | 0.00527 | 0.1411 |
| Three-wheeler | 0.176 | 2772 | 0.00579 | 0.1240 |
| Hatchback | 0.094 | 1537 | 0.01037 | 0.1625 |
| Sedan | 0.053 | 896 | 0.00892 | 0.1449 |
| LCV | 0.049 | 781 | 0.00838 | 0.1468 |
| SUV | 0.042 | 652 | 0.01087 | 0.1702 |
| Bus | 0.037 | 573 | 0.00695 | 0.1368 |
| Truck | 0.027 | 401 | 0.00783 | 0.1420 |
| MUV | 0.024 | 390 | 0.00942 | 0.1526 |
| Tempo-traveller | 0.011 | 149 | 0.00495 | 0.0975 |
| Van | 0.011 | 173 | 0.00713 | 0.1746 |
| Mini-bus | 0.004 | 55 | 0.00887 | 0.2100 |

## b) Evidential pilot (yolo26_evid_s2: YOLO26, Beta evidence head, current fit set)

AP50 class-agnostic 0.9301 vs yolo26 family mean 0.9359 (gap -0.0058); class-aware 0.7462. Threshold 0.45: recall 0.880, 1.26 false alarms/image. final v0.6 model, trained on Beocat (L40S) with the same batch and patience as the yolo26 family (configs/beocat_batch.yaml; patience 10).

Per vehicle (its matched candidate), Spearman and partial Spearman controlling box size, image-bootstrap 95% CI:

| association | Spearman | partial (size) |
|---|---|---|
| aleatoric vs annotated_occlusion | 0.186 [0.169, 0.203] | 0.177 [0.160, 0.193] |
| aleatoric vs class_rarity | -0.005 [-0.022, 0.014] | 0.157 [0.135, 0.178] |
| epistemic vs annotated_occlusion | 0.204 [0.188, 0.220] | 0.199 [0.183, 0.215] |
| epistemic vs class_rarity | 0.330 [0.313, 0.351] | 0.386 [0.367, 0.409] |

Uncertainty split: Spearman(aleatoric, epistemic) = 0.812; rho(epistemic, rarity) - rho(aleatoric, rarity) = 0.334 [0.325, 0.343]; rho(aleatoric, o_j) - rho(epistemic, o_j) = -0.018 [-0.027, -0.007].

These are the final v0.6 evidential models (one of three seeds); the pilot's gate and the RT-DETR feasibility note do not apply to them.

### evidential: rarity diagnostic (calib, yolo26_evid_s2)

Calib role only, 15643 vehicles with a box, threshold 0.45. Hypothesis: evidence is per fine class, so the common, confusable car subtypes split it. Spearman with image-bootstrap 95% CI.

| group | n | vs fine-class rarity | vs coarse-class rarity |
|---|---|---|---|
| (a) epistemic, all vehicles | 15643 | 0.355 [0.341, 0.372] | 0.184 [0.168, 0.203] |
| (a) aleatoric, all vehicles | 15643 | 0.014 [-0.004, 0.030] | -0.101 [-0.120, -0.085] |
| (b) epistemic, confusion misses | 800 | -0.063 [-0.123, 0.008] | -0.127 [-0.194, -0.070] |
| (b) epistemic, visibility misses | 1510 | 0.349 [0.303, 0.395] | 0.271 [0.222, 0.323] |

(c) mean epistemic, car subtypes (Hatchback, Sedan, SUV) 0.00991 vs rare classes (fit share < 2%: Mini-bus, Tempo-traveller, Van) 0.00787; difference 0.00203 [0.00060, 0.00312].

| class | fit share | n | mean epistemic | mean aleatoric |
|---|---|---|---|---|
| Two-wheeler | 0.459 | 7258 | 0.00498 | 0.1414 |
| Three-wheeler | 0.176 | 2773 | 0.00490 | 0.1133 |
| Hatchback | 0.094 | 1539 | 0.01023 | 0.1604 |
| Sedan | 0.053 | 896 | 0.00889 | 0.1451 |
| LCV | 0.049 | 784 | 0.00798 | 0.1402 |
| SUV | 0.042 | 653 | 0.01055 | 0.1700 |
| Bus | 0.037 | 571 | 0.00704 | 0.1265 |
| Truck | 0.027 | 401 | 0.00779 | 0.1358 |
| MUV | 0.024 | 391 | 0.00945 | 0.1525 |
| Tempo-traveller | 0.011 | 148 | 0.00536 | 0.0984 |
| Van | 0.011 | 174 | 0.00620 | 0.1575 |
| Mini-bus | 0.004 | 55 | 0.01994 | 0.1964 |

## c) UA-DETRAC (new cameras; current models, zero-shot)

**Proxy validation** (66700 test vehicles, sequence-bootstrap CI): box-overlap proxy vs annotated occlusion = 0.844 [0.790, 0.883] (confirmed reading), 0.403 [0.269, 0.515] (naive reading). 0.247 of vehicles have annotated vehicle-occlusion >= 0.2; 0.025 have some non-vehicle occlusion; 0.138 are truncated.

| detector | o_hat vs annotated (confirmed) | o_hat vs annotated (naive) |
|---|---|---|
| yolo26 | 0.717 [0.645, 0.770] | 0.360 [0.238, 0.447] |
| yolov8 | 0.718 [0.644, 0.771] | 0.361 [0.241, 0.450] |
| rtdetr | 0.679 [0.602, 0.736] | 0.345 [0.228, 0.430] |

**yolo26**: 5625 test frames, 40 sequences, 66700 vehicles. Recall at the frozen threshold 0.46: 0.778 [0.723, 0.834] (7.25 false alarms/image); at the confidence floor 0.943.

| target | rule (setting:rule) | coverage | flags/image | certifiable | coverage by annotated occlusion bucket | lowest o_hat stratum | n in top bucket |
|---|---|---|---|---|---|---|---|
| 0.90 | frozen:global_confidence | 0.802 [0.750, 0.855] | 1.18 | yes | 0.902 / 0.741 / 0.486 / 0.280 / 0.055 | 0.544 | 3043 |
| 0.90 | frozen:mondrian_confidence | 0.800 [0.748, 0.850] | 0.99 | yes | 0.878 / 0.792 / 0.584 / 0.399 / 0.075 | 0.634 | 3043 |
| 0.90 | frozen:mondrian_gilg_iou_aware | 0.800 [0.748, 0.851] | 0.80 | yes | 0.878 / 0.794 / 0.585 / 0.391 / 0.086 | 0.639 | 3043 |
| 0.90 | recalibrated:global_confidence | 0.891 [0.853, 0.928] | 8.91 | yes | 0.968 / 0.888 / 0.707 / 0.499 / 0.121 | 0.685 | 3043 |
| 0.90 | recalibrated:mondrian_confidence | 0.895 [0.858, 0.933] | 15.81 | NO | 0.955 / 0.891 / 0.801 / 0.695 / 0.184 | 0.785 | 3043 |
| 0.90 | recalibrated:mondrian_gilg_iou_aware | 0.895 [0.857, 0.932] | 14.41 | NO | 0.955 / 0.888 / 0.795 / 0.686 / 0.181 | 0.785 | 3043 |
| 0.95 | frozen:global_confidence | 0.882 [0.842, 0.922] | 7.62 | yes | 0.963 / 0.871 / 0.678 / 0.472 / 0.109 | 0.671 | 3043 |
| 0.95 | frozen:mondrian_confidence | 0.885 [0.844, 0.925] | 7.30 | yes | 0.958 / 0.883 / 0.720 / 0.521 / 0.136 | 0.712 | 3043 |
| 0.95 | frozen:mondrian_gilg_iou_aware | 0.888 [0.847, 0.927] | 5.64 | yes | 0.959 / 0.890 / 0.738 / 0.527 / 0.142 | 0.727 | 3043 |
| 0.95 | recalibrated:global_confidence | 0.937 [0.911, 0.962] | 37.48 | yes | 0.991 / 0.965 / 0.886 / 0.697 / 0.174 | 0.770 | 3043 |
| 0.95 | recalibrated:mondrian_confidence | 0.929 [0.902, 0.957] | 33.62 | NO | 0.985 / 0.942 / 0.863 / 0.728 / 0.188 | 0.785 | 3043 |
| 0.95 | recalibrated:mondrian_gilg_iou_aware | 0.928 [0.900, 0.956] | 28.33 | NO | 0.984 / 0.941 / 0.857 / 0.714 / 0.184 | 0.785 | 3043 |

Why the recalibrated Mondrian rules are not certifiable (yolo26; UA-DETRAC calib, per o_hat group; a group is certifiable only if its smallest certifiable miss rate (never proposed + 1) / (n + 1) is at most the target alpha):

| target | rule | o_hat group | calib n | never proposed | min certifiable alpha | target alpha | certifiable |
|---|---|---|---|---|---|---|---|
| 0.90 | recalibrated:mondrian_confidence | 0.00-0.05 | 24667 | 907 (0.037) | 0.037 | 0.1 | yes |
| 0.90 | recalibrated:mondrian_confidence | 0.05-0.30 | 3544 | 51 (0.014) | 0.015 | 0.1 | yes |
| 0.90 | recalibrated:mondrian_confidence | 0.30-1.00 | 2151 | 300 (0.139) | 0.140 | 0.1 | NO |
| 0.90 | recalibrated:mondrian_gilg_iou_aware | 0.00-0.05 | 24667 | 907 (0.037) | 0.037 | 0.1 | yes |
| 0.90 | recalibrated:mondrian_gilg_iou_aware | 0.05-0.30 | 3544 | 51 (0.014) | 0.015 | 0.1 | yes |
| 0.90 | recalibrated:mondrian_gilg_iou_aware | 0.30-1.00 | 2151 | 300 (0.139) | 0.140 | 0.1 | NO |
| 0.95 | recalibrated:mondrian_confidence | 0.00-0.05 | 24667 | 907 (0.037) | 0.037 | 0.05 | yes |
| 0.95 | recalibrated:mondrian_confidence | 0.05-0.30 | 3544 | 51 (0.014) | 0.015 | 0.05 | yes |
| 0.95 | recalibrated:mondrian_confidence | 0.30-1.00 | 2151 | 300 (0.139) | 0.140 | 0.05 | NO |
| 0.95 | recalibrated:mondrian_gilg_iou_aware | 0.00-0.05 | 24667 | 907 (0.037) | 0.037 | 0.05 | yes |
| 0.95 | recalibrated:mondrian_gilg_iou_aware | 0.05-0.30 | 3544 | 51 (0.014) | 0.015 | 0.05 | yes |
| 0.95 | recalibrated:mondrian_gilg_iou_aware | 0.30-1.00 | 2151 | 300 (0.139) | 0.140 | 0.05 | NO |

**yolov8**: 5625 test frames, 40 sequences, 66700 vehicles. Recall at the frozen threshold 0.47: 0.768 [0.716, 0.819] (6.95 false alarms/image); at the confidence floor 0.941.

| target | rule (setting:rule) | coverage | flags/image | certifiable | coverage by annotated occlusion bucket | lowest o_hat stratum | n in top bucket |
|---|---|---|---|---|---|---|---|
| 0.90 | frozen:global_confidence | 0.792 [0.739, 0.841] | 0.98 | yes | 0.885 / 0.747 / 0.489 / 0.349 / 0.060 | 0.563 | 3043 |
| 0.90 | frozen:mondrian_confidence | 0.781 [0.728, 0.830] | 0.96 | yes | 0.852 / 0.787 / 0.578 / 0.406 / 0.101 | 0.642 | 3043 |
| 0.90 | frozen:mondrian_gilg_iou_aware | 0.785 [0.732, 0.833] | 0.85 | yes | 0.858 / 0.788 / 0.578 / 0.408 / 0.108 | 0.645 | 3043 |
| 0.90 | recalibrated:global_confidence | 0.887 [0.849, 0.926] | 8.61 | yes | 0.965 / 0.879 / 0.698 / 0.493 / 0.125 | 0.681 | 3043 |
| 0.90 | recalibrated:mondrian_confidence | 0.895 [0.856, 0.930] | 15.37 | NO | 0.955 / 0.887 / 0.801 / 0.675 / 0.180 | 0.783 | 3043 |
| 0.90 | recalibrated:mondrian_gilg_iou_aware | 0.894 [0.855, 0.929] | 14.29 | NO | 0.956 / 0.885 / 0.796 / 0.659 / 0.178 | 0.783 | 3043 |
| 0.95 | frozen:global_confidence | 0.873 [0.831, 0.914] | 6.72 | yes | 0.957 / 0.852 / 0.649 / 0.453 / 0.113 | 0.657 | 3043 |
| 0.95 | frozen:mondrian_confidence | 0.878 [0.835, 0.917] | 6.54 | yes | 0.949 / 0.867 / 0.728 / 0.536 / 0.145 | 0.721 | 3043 |
| 0.95 | frozen:mondrian_gilg_iou_aware | 0.879 [0.837, 0.917] | 5.33 | yes | 0.950 / 0.872 / 0.737 / 0.524 / 0.143 | 0.725 | 3043 |
| 0.95 | recalibrated:global_confidence | 0.937 [0.912, 0.962] | 44.15 | yes | 0.991 / 0.966 / 0.884 / 0.710 / 0.177 | 0.774 | 3043 |
| 0.95 | recalibrated:mondrian_confidence | 0.930 [0.903, 0.957] | 35.69 | NO | 0.986 / 0.947 / 0.863 / 0.718 / 0.186 | 0.783 | 3043 |
| 0.95 | recalibrated:mondrian_gilg_iou_aware | 0.929 [0.902, 0.956] | 29.94 | NO | 0.986 / 0.945 / 0.856 / 0.700 / 0.181 | 0.783 | 3043 |

Why the recalibrated Mondrian rules are not certifiable (yolov8; UA-DETRAC calib, per o_hat group; a group is certifiable only if its smallest certifiable miss rate (never proposed + 1) / (n + 1) is at most the target alpha):

| target | rule | o_hat group | calib n | never proposed | min certifiable alpha | target alpha | certifiable |
|---|---|---|---|---|---|---|---|
| 0.90 | recalibrated:mondrian_confidence | 0.00-0.05 | 24704 | 981 (0.040) | 0.040 | 0.1 | yes |
| 0.90 | recalibrated:mondrian_confidence | 0.05-0.30 | 3549 | 50 (0.014) | 0.014 | 0.1 | yes |
| 0.90 | recalibrated:mondrian_confidence | 0.30-1.00 | 2109 | 321 (0.152) | 0.153 | 0.1 | NO |
| 0.90 | recalibrated:mondrian_gilg_iou_aware | 0.00-0.05 | 24704 | 981 (0.040) | 0.040 | 0.1 | yes |
| 0.90 | recalibrated:mondrian_gilg_iou_aware | 0.05-0.30 | 3549 | 50 (0.014) | 0.014 | 0.1 | yes |
| 0.90 | recalibrated:mondrian_gilg_iou_aware | 0.30-1.00 | 2109 | 321 (0.152) | 0.153 | 0.1 | NO |
| 0.95 | recalibrated:mondrian_confidence | 0.00-0.05 | 24704 | 981 (0.040) | 0.040 | 0.05 | yes |
| 0.95 | recalibrated:mondrian_confidence | 0.05-0.30 | 3549 | 50 (0.014) | 0.014 | 0.05 | yes |
| 0.95 | recalibrated:mondrian_confidence | 0.30-1.00 | 2109 | 321 (0.152) | 0.153 | 0.05 | NO |
| 0.95 | recalibrated:mondrian_gilg_iou_aware | 0.00-0.05 | 24704 | 981 (0.040) | 0.040 | 0.05 | yes |
| 0.95 | recalibrated:mondrian_gilg_iou_aware | 0.05-0.30 | 3549 | 50 (0.014) | 0.014 | 0.05 | yes |
| 0.95 | recalibrated:mondrian_gilg_iou_aware | 0.30-1.00 | 2109 | 321 (0.152) | 0.153 | 0.05 | NO |

**rtdetr**: 5625 test frames, 40 sequences, 66700 vehicles. Recall at the frozen threshold 0.69: 0.708 [0.654, 0.764] (5.14 false alarms/image); at the confidence floor 0.941.

| target | rule (setting:rule) | coverage | flags/image | certifiable | coverage by annotated occlusion bucket | lowest o_hat stratum | n in top bucket |
|---|---|---|---|---|---|---|---|
| 0.90 | frozen:global_confidence | 0.753 [0.699, 0.806] | 1.90 | yes | 0.856 / 0.648 / 0.410 / 0.256 / 0.088 | 0.517 | 3043 |
| 0.90 | frozen:mondrian_confidence | 0.741 [0.686, 0.797] | 1.32 | yes | 0.827 / 0.685 / 0.460 / 0.341 / 0.105 | 0.588 | 3043 |
| 0.90 | frozen:mondrian_gilg_iou_aware | 0.743 [0.688, 0.799] | 0.93 | yes | 0.829 / 0.688 / 0.462 / 0.336 / 0.105 | 0.590 | 3043 |
| 0.90 | recalibrated:global_confidence | 0.869 [0.828, 0.909] | 16.46 | yes | 0.958 / 0.831 / 0.598 / 0.474 / 0.130 | 0.649 | 3043 |
| 0.90 | recalibrated:mondrian_confidence | 0.875 [0.834, 0.912] | 42.03 | NO | 0.945 / 0.824 / 0.711 / 0.632 / 0.212 | 0.767 | 3043 |
| 0.90 | recalibrated:mondrian_gilg_iou_aware | 0.874 [0.833, 0.911] | 36.62 | NO | 0.946 / 0.824 / 0.705 / 0.611 / 0.203 | 0.767 | 3043 |
| 0.95 | frozen:global_confidence | 0.847 [0.800, 0.890] | 11.35 | yes | 0.941 / 0.786 / 0.540 / 0.426 / 0.120 | 0.618 | 3043 |
| 0.95 | frozen:mondrian_confidence | 0.848 [0.801, 0.891] | 10.52 | yes | 0.933 / 0.796 / 0.604 / 0.485 / 0.137 | 0.678 | 3043 |
| 0.95 | frozen:mondrian_gilg_iou_aware | 0.847 [0.800, 0.891] | 5.50 | yes | 0.932 / 0.798 / 0.605 / 0.475 / 0.132 | 0.680 | 3043 |
| 0.95 | recalibrated:global_confidence | 0.930 [0.903, 0.956] | 89.65 | yes | 0.990 / 0.946 / 0.836 / 0.672 / 0.178 | 0.743 | 3043 |
| 0.95 | recalibrated:mondrian_confidence | 0.923 [0.895, 0.951] | 86.23 | NO | 0.983 / 0.916 / 0.819 / 0.697 / 0.217 | 0.767 | 3043 |
| 0.95 | recalibrated:mondrian_gilg_iou_aware | 0.920 [0.892, 0.948] | 49.99 | NO | 0.983 / 0.915 / 0.806 / 0.643 / 0.204 | 0.767 | 3043 |

Why the recalibrated Mondrian rules are not certifiable (rtdetr; UA-DETRAC calib, per o_hat group; a group is certifiable only if its smallest certifiable miss rate (never proposed + 1) / (n + 1) is at most the target alpha):

| target | rule | o_hat group | calib n | never proposed | min certifiable alpha | target alpha | certifiable |
|---|---|---|---|---|---|---|---|
| 0.90 | recalibrated:mondrian_confidence | 0.00-0.05 | 24888 | 778 (0.031) | 0.031 | 0.1 | yes |
| 0.90 | recalibrated:mondrian_confidence | 0.05-0.30 | 3485 | 38 (0.011) | 0.011 | 0.1 | yes |
| 0.90 | recalibrated:mondrian_confidence | 0.30-1.00 | 1989 | 276 (0.139) | 0.139 | 0.1 | NO |
| 0.90 | recalibrated:mondrian_gilg_iou_aware | 0.00-0.05 | 24888 | 778 (0.031) | 0.031 | 0.1 | yes |
| 0.90 | recalibrated:mondrian_gilg_iou_aware | 0.05-0.30 | 3485 | 38 (0.011) | 0.011 | 0.1 | yes |
| 0.90 | recalibrated:mondrian_gilg_iou_aware | 0.30-1.00 | 1989 | 276 (0.139) | 0.139 | 0.1 | NO |
| 0.95 | recalibrated:mondrian_confidence | 0.00-0.05 | 24888 | 778 (0.031) | 0.031 | 0.05 | yes |
| 0.95 | recalibrated:mondrian_confidence | 0.05-0.30 | 3485 | 38 (0.011) | 0.011 | 0.05 | yes |
| 0.95 | recalibrated:mondrian_confidence | 0.30-1.00 | 1989 | 276 (0.139) | 0.139 | 0.05 | NO |
| 0.95 | recalibrated:mondrian_gilg_iou_aware | 0.00-0.05 | 24888 | 778 (0.031) | 0.031 | 0.05 | yes |
| 0.95 | recalibrated:mondrian_gilg_iou_aware | 0.05-0.30 | 3485 | 38 (0.011) | 0.011 | 0.05 | yes |
| 0.95 | recalibrated:mondrian_gilg_iou_aware | 0.30-1.00 | 1989 | 276 (0.139) | 0.139 | 0.05 | NO |
