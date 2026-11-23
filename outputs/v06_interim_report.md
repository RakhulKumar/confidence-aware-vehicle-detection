# v0.6 interim report (2026-11-18 19:16)

Written automatically by scripts/21_interim_report.py when jobs a) to c) finished; numbers are copied from the result files named in the script. Interpretation is left to review.

## a) NMS pilot (yolo26 seed 0, BMD-45 test, class-agnostic)

Pre-NMS candidates: 228 per image (cap 1000). Operating threshold 0.46; 2107 missed vehicles.

**Why each missed vehicle is missed** (share of misses):

| annotated occlusion | misses | removed_by_nms | kept_below_threshold | suppressed_low_confidence | never_proposed |
|---|---|---|---|---|---|
| all | 2107 | 0.031 | 0.834 | 0.008 | 0.127 |
| o_j 0.0-0.2 | 1296 | 0.015 | 0.833 | 0.008 | 0.144 |
| o_j 0.2-0.4 | 327 | 0.015 | 0.881 | 0.006 | 0.098 |
| o_j 0.4-0.6 | 236 | 0.064 | 0.852 | 0.004 | 0.081 |
| o_j 0.6-0.8 | 146 | 0.103 | 0.760 | 0.021 | 0.116 |
| o_j 0.8-1.0 | 102 | 0.108 | 0.755 | 0.010 | 0.127 |

**Suppression variants at matched false alarms** (recall; recall by annotated occlusion bucket):

| variant | AP50 | false alarms/image | recall | recall by o_j bucket |
|---|---|---|---|---|
| standard_cached (Ultralytics NMS 0.7, max_det 300) | 0.9150 | 1.45 | 0.8691 | 0.892 / 0.849 / 0.798 / 0.671 / 0.668 |
| standard_cached (Ultralytics NMS 0.7, max_det 300) | 0.9150 | 1.95 | 0.8944 | 0.913 / 0.882 / 0.838 / 0.723 / 0.704 |
| standard_cached (Ultralytics NMS 0.7, max_det 300) | 0.9150 | 2.45 | 0.9120 | 0.928 / 0.904 / 0.867 / 0.766 / 0.739 |
| hard_nms_0.5 | 0.9127 | 1.45 | 0.8686 | 0.892 / 0.848 / 0.797 / 0.658 / 0.664 |
| hard_nms_0.5 | 0.9127 | 1.95 | 0.8950 | 0.914 / 0.884 / 0.839 / 0.714 / 0.704 |
| hard_nms_0.5 | 0.9127 | 2.45 | 0.9124 | 0.929 / 0.904 / 0.862 / 0.752 / 0.739 |
| hard_nms_0.7 | 0.9150 | 1.45 | 0.8691 | 0.892 / 0.849 / 0.798 / 0.671 / 0.668 |
| hard_nms_0.7 | 0.9150 | 1.95 | 0.8944 | 0.913 / 0.882 / 0.838 / 0.723 / 0.704 |
| hard_nms_0.7 | 0.9150 | 2.45 | 0.9120 | 0.928 / 0.904 / 0.867 / 0.766 / 0.739 |
| hard_nms_0.8 | 0.9130 | 1.45 | 0.8676 | 0.891 / 0.847 / 0.795 / 0.669 / 0.664 |
| hard_nms_0.8 | 0.9130 | 1.95 | 0.8925 | 0.912 / 0.878 / 0.837 / 0.718 / 0.697 |
| hard_nms_0.8 | 0.9130 | 2.45 | 0.9089 | 0.925 / 0.901 / 0.861 / 0.755 / 0.739 |
| soft_nms_gaussian_0.5 | 0.9021 | 1.45 | 0.8676 | 0.893 / 0.848 / 0.788 / 0.628 / 0.658 |
| soft_nms_gaussian_0.5 | 0.9021 | 1.95 | 0.8944 | 0.915 / 0.884 / 0.834 / 0.700 / 0.687 |
| soft_nms_gaussian_0.5 | 0.9021 | 2.45 | 0.9125 | 0.929 / 0.904 / 0.862 / 0.755 / 0.736 |

## b) Evidential pilot (YOLO26, Beta evidence head, current fit set)

AP50 class-agnostic 0.9155 vs yolo26 family mean 0.9157 (gap -0.0002); class-aware 0.6897. Threshold 0.39: recall 0.875, 1.52 false alarms/image. pilot trained with batch 4 / patience 10 under the GPU share; the yolo26 family used batch 8 / patience 25.

Per vehicle (its matched candidate), Spearman and partial Spearman controlling box size, image-bootstrap 95% CI:

| association | Spearman | partial (size) |
|---|---|---|
| aleatoric vs annotated_occlusion | 0.149 [0.129, 0.165] | 0.137 [0.118, 0.152] |
| aleatoric vs class_rarity | 0.006 [-0.013, 0.025] | 0.123 [0.103, 0.144] |
| epistemic vs annotated_occlusion | 0.073 [0.054, 0.093] | 0.068 [0.049, 0.088] |
| epistemic vs class_rarity | -0.169 [-0.186, -0.145] | -0.148 [-0.165, -0.126] |

Uncertainty split: Spearman(aleatoric, epistemic) = 0.777; rho(epistemic, rarity) - rho(aleatoric, rarity) = -0.176 [-0.188, -0.164]; rho(aleatoric, o_j) - rho(epistemic, o_j) = 0.075 [0.063, 0.089].

Gate for evidential x3 (decided 14 Nov): AP50 within 0.02 of baseline -> **met**; usable uncertainty split -> for review (the two parts must not be near-collinear and must differ in what they follow).

RT-DETR feasibility: its class scores come from the decoder's score head trained with a varifocal/focal loss inside DETRLoss; the same Beta head means doubling that head's outputs and replacing the varifocal term, with the Hungarian matcher kept on p = alpha/S. Feasible, more invasive than YOLO26; not attempted in this pilot.

### evidential: rarity diagnostic (calib)

Calib role only, 15605 vehicles with a box, threshold 0.39. Hypothesis: evidence is per fine class, so the common, confusable car subtypes split it. Spearman with image-bootstrap 95% CI.

| group | n | vs fine-class rarity | vs coarse-class rarity |
|---|---|---|---|
| (a) epistemic, all vehicles | 15605 | -0.132 [-0.149, -0.113] | -0.326 [-0.342, -0.311] |
| (a) aleatoric, all vehicles | 15605 | 0.023 [0.007, 0.040] | -0.096 [-0.115, -0.079] |
| (b) epistemic, confusion misses | 1077 | 0.061 [0.003, 0.126] | -0.089 [-0.147, -0.034] |
| (b) epistemic, visibility misses | 1482 | -0.035 [-0.084, 0.013] | -0.127 [-0.174, -0.072] |

(c) mean epistemic, car subtypes (Hatchback, Sedan, SUV) 0.00954 vs rare classes (fit share < 2%: Mini-bus, Tempo-traveller, Van) 0.00527; difference 0.00427 [0.00370, 0.00475].

| class | fit share | n | mean epistemic | mean aleatoric |
|---|---|---|---|---|
| Two-wheeler | 0.458 | 7242 | 0.00826 | 0.1366 |
| Three-wheeler | 0.175 | 2769 | 0.00284 | 0.1051 |
| Hatchback | 0.094 | 1534 | 0.00992 | 0.1601 |
| Sedan | 0.052 | 894 | 0.00886 | 0.1536 |
| LCV | 0.050 | 783 | 0.00731 | 0.1394 |
| SUV | 0.041 | 650 | 0.00958 | 0.1535 |
| Bus | 0.038 | 571 | 0.00489 | 0.1275 |
| Truck | 0.027 | 395 | 0.00697 | 0.1422 |
| MUV | 0.025 | 391 | 0.00849 | 0.1435 |
| Tempo-traveller | 0.012 | 148 | 0.00371 | 0.1173 |
| Van | 0.011 | 172 | 0.00510 | 0.1320 |
| Mini-bus | 0.004 | 56 | 0.00995 | 0.1793 |

## c) UA-DETRAC (new cameras; current models, zero-shot)

**Proxy validation** (66700 test vehicles, sequence-bootstrap CI): box-overlap proxy vs annotated occlusion = 0.844 [0.790, 0.883] (confirmed reading), 0.403 [0.269, 0.515] (naive reading). 0.247 of vehicles have annotated vehicle-occlusion >= 0.2; 0.025 have some non-vehicle occlusion; 0.138 are truncated.

| detector | o_hat vs annotated (confirmed) | o_hat vs annotated (naive) |
|---|---|---|
| yolo26 | 0.665 [0.581, 0.732] | 0.329 [0.208, 0.414] |
| yolov8 | 0.637 [0.532, 0.728] | 0.317 [0.205, 0.409] |
| rtdetr | 0.639 [0.556, 0.715] | 0.317 [0.208, 0.395] |

**yolo26**: 5625 test frames, 40 sequences, 66700 vehicles. Recall at the frozen threshold 0.46: 0.763 [0.708, 0.815] (7.85 false alarms/image); at the confidence floor 0.933.

| target | rule (setting:rule) | coverage | flags/image | certifiable | coverage by annotated occlusion bucket | lowest o_hat stratum | n in top bucket |
|---|---|---|---|---|---|---|---|
| 0.90 | frozen:global_confidence | 0.817 [0.768, 0.864] | 3.40 | yes | 0.912 / 0.775 / 0.537 / 0.305 / 0.052 | 0.581 | 3043 |
| 0.90 | frozen:mondrian_confidence | 0.816 [0.768, 0.859] | 2.86 | yes | 0.889 / 0.819 / 0.634 / 0.447 / 0.083 | 0.678 | 3043 |
| 0.90 | frozen:mondrian_gilg_iou_aware | 0.818 [0.770, 0.860] | 2.08 | yes | 0.891 / 0.822 / 0.637 / 0.420 / 0.098 | 0.682 | 3043 |
| 0.90 | recalibrated:global_confidence | 0.895 [0.860, 0.930] | 14.65 | yes | 0.970 / 0.899 / 0.725 / 0.527 / 0.107 | 0.706 | 3043 |
| 0.90 | recalibrated:mondrian_confidence | 0.896 [0.861, 0.928] | 21.40 | NO | 0.960 / 0.883 / 0.790 / 0.644 / 0.169 | 0.784 | 3043 |
| 0.90 | recalibrated:mondrian_gilg_iou_aware | 0.893 [0.858, 0.925] | 18.44 | NO | 0.959 / 0.878 / 0.778 / 0.626 / 0.168 | 0.784 | 3043 |
| 0.95 | frozen:global_confidence | 0.890 [0.854, 0.926] | 13.23 | yes | 0.967 / 0.891 / 0.706 / 0.507 / 0.100 | 0.695 | 3043 |
| 0.95 | frozen:mondrian_confidence | 0.894 [0.858, 0.928] | 13.79 | yes | 0.961 / 0.900 / 0.772 / 0.581 / 0.138 | 0.754 | 3043 |
| 0.95 | frozen:mondrian_gilg_iou_aware | 0.894 [0.858, 0.927] | 9.64 | yes | 0.962 / 0.900 / 0.759 / 0.559 / 0.141 | 0.753 | 3043 |
| 0.95 | recalibrated:global_confidence | 0.933 [0.907, 0.959] | 59.49 | NO | 0.988 / 0.966 / 0.874 / 0.691 / 0.178 | 0.784 | 3043 |
| 0.95 | recalibrated:mondrian_confidence | 0.924 [0.896, 0.950] | 52.35 | NO | 0.983 / 0.935 / 0.848 / 0.676 / 0.176 | 0.784 | 3043 |
| 0.95 | recalibrated:mondrian_gilg_iou_aware | 0.923 [0.895, 0.949] | 51.82 | NO | 0.983 / 0.934 / 0.844 / 0.674 / 0.176 | 0.784 | 3043 |

**yolov8**: 5625 test frames, 40 sequences, 66700 vehicles. Recall at the frozen threshold 0.47: 0.741 [0.686, 0.795] (7.72 false alarms/image); at the confidence floor 0.937.

| target | rule (setting:rule) | coverage | flags/image | certifiable | coverage by annotated occlusion bucket | lowest o_hat stratum | n in top bucket |
|---|---|---|---|---|---|---|---|
| 0.90 | frozen:global_confidence | 0.817 [0.767, 0.867] | 4.63 | yes | 0.913 / 0.780 / 0.504 / 0.299 / 0.090 | 0.593 | 3043 |
| 0.90 | frozen:mondrian_confidence | 0.819 [0.770, 0.865] | 3.93 | yes | 0.896 / 0.811 / 0.610 / 0.384 / 0.114 | 0.680 | 3043 |
| 0.90 | frozen:mondrian_gilg_iou_aware | 0.823 [0.775, 0.869] | 3.11 | yes | 0.900 / 0.819 / 0.621 / 0.375 / 0.112 | 0.689 | 3043 |
| 0.90 | recalibrated:global_confidence | 0.913 [0.883, 0.943] | 27.29 | yes | 0.979 / 0.935 / 0.797 / 0.543 / 0.139 | 0.748 | 3043 |
| 0.90 | recalibrated:mondrian_confidence | 0.908 [0.877, 0.936] | 36.29 | NO | 0.970 / 0.902 / 0.817 / 0.654 / 0.174 | 0.798 | 3043 |
| 0.90 | recalibrated:mondrian_gilg_iou_aware | 0.905 [0.874, 0.935] | 29.74 | NO | 0.969 / 0.898 / 0.806 / 0.640 / 0.173 | 0.798 | 3043 |
| 0.95 | frozen:global_confidence | 0.887 [0.849, 0.924] | 14.38 | yes | 0.965 / 0.889 / 0.699 / 0.427 / 0.118 | 0.699 | 3043 |
| 0.95 | frozen:mondrian_confidence | 0.897 [0.863, 0.931] | 16.46 | yes | 0.964 / 0.907 / 0.766 / 0.565 / 0.149 | 0.767 | 3043 |
| 0.95 | frozen:mondrian_gilg_iou_aware | 0.897 [0.863, 0.930] | 12.20 | yes | 0.966 / 0.909 / 0.762 / 0.542 / 0.145 | 0.766 | 3043 |
| 0.95 | recalibrated:global_confidence | 0.937 [0.912, 0.960] | 88.80 | NO | 0.990 / 0.970 / 0.885 / 0.700 / 0.180 | 0.798 | 3043 |
| 0.95 | recalibrated:mondrian_confidence | 0.927 [0.902, 0.953] | 79.29 | NO | 0.985 / 0.942 / 0.852 / 0.686 / 0.180 | 0.798 | 3043 |
| 0.95 | recalibrated:mondrian_gilg_iou_aware | 0.926 [0.901, 0.951] | 78.69 | NO | 0.985 / 0.940 / 0.848 / 0.682 / 0.180 | 0.798 | 3043 |

**rtdetr**: 5625 test frames, 40 sequences, 66700 vehicles. Recall at the frozen threshold 0.65: 0.717 [0.659, 0.774] (5.50 false alarms/image); at the confidence floor 0.930.

| target | rule (setting:rule) | coverage | flags/image | certifiable | coverage by annotated occlusion bucket | lowest o_hat stratum | n in top bucket |
|---|---|---|---|---|---|---|---|
| 0.90 | frozen:global_confidence | 0.786 [0.732, 0.838] | 3.75 | yes | 0.890 / 0.683 / 0.472 / 0.301 / 0.052 | 0.551 | 3043 |
| 0.90 | frozen:mondrian_confidence | 0.788 [0.735, 0.839] | 3.52 | yes | 0.876 / 0.718 / 0.543 / 0.413 / 0.093 | 0.645 | 3043 |
| 0.90 | frozen:mondrian_gilg_iou_aware | 0.788 [0.734, 0.839] | 2.03 | yes | 0.875 / 0.718 / 0.541 / 0.397 / 0.095 | 0.643 | 3043 |
| 0.90 | recalibrated:global_confidence | 0.896 [0.861, 0.930] | 31.80 | yes | 0.970 / 0.886 / 0.722 / 0.560 / 0.145 | 0.713 | 3043 |
| 0.90 | recalibrated:mondrian_confidence | 0.892 [0.857, 0.925] | 55.99 | NO | 0.961 / 0.858 / 0.750 / 0.647 / 0.177 | 0.767 | 3043 |
| 0.90 | recalibrated:mondrian_gilg_iou_aware | 0.889 [0.854, 0.923] | 42.58 | NO | 0.961 / 0.854 / 0.736 / 0.611 / 0.168 | 0.767 | 3043 |
| 0.95 | frozen:global_confidence | 0.867 [0.826, 0.908] | 16.80 | yes | 0.953 / 0.822 / 0.633 / 0.482 / 0.123 | 0.673 | 3043 |
| 0.95 | frozen:mondrian_confidence | 0.872 [0.831, 0.912] | 18.62 | yes | 0.948 / 0.830 / 0.698 / 0.557 / 0.148 | 0.734 | 3043 |
| 0.95 | frozen:mondrian_gilg_iou_aware | 0.872 [0.832, 0.912] | 8.22 | yes | 0.950 / 0.832 / 0.691 / 0.522 / 0.140 | 0.732 | 3043 |
| 0.95 | recalibrated:global_confidence | 0.930 [0.905, 0.957] | 275.90 | NO | 0.989 / 0.946 / 0.844 / 0.704 / 0.185 | 0.767 | 3043 |
| 0.95 | recalibrated:mondrian_confidence | 0.922 [0.895, 0.950] | 245.78 | NO | 0.984 / 0.920 / 0.819 / 0.690 / 0.184 | 0.767 | 3043 |
| 0.95 | recalibrated:mondrian_gilg_iou_aware | 0.921 [0.894, 0.949] | 243.92 | NO | 0.984 / 0.919 / 0.817 / 0.681 / 0.183 | 0.767 | 3043 |

## Queue status

```
done        3.0 h  download_bmd45_train  [started 14 Nov 08:34 outside the runner (resumable); runner waits for that PID]
done        0.5 h  build_bmd_v06
done        0.1 h  build_pals_rev
done        0.3 h  occlusion_bmd_v06
done        0.1 h  occlusion_pals_rev
done        0.6 h  nms_pilot_predict
done        0.2 h  nms_pilot_analyse
done        6.0 h  evid_pilot_train  [18 Nov: first attempt (yolo26_evid_pilot, evidence prior p ~ 0.15 by mistake) stopped after 2 epochs at val mAP50 0.069 / 0.067; folder kept. This job runs the corrected prior (p ~ 0.01).]
done        0.5 h  evid_pilot_predict
done        0.2 h  evid_pilot_analyse
done        0.3 h  detrac_convert
done        0.4 h  detrac_predict_yolo26
done        0.4 h  detrac_predict_yolov8
done        0.8 h  detrac_predict_rtdetr
done        0.5 h  detrac_transfer
done        0.1 h  interim_report
running    32.1 h  train_bmd_v06_yolo26
pending     0.5 h  predict_bmd_v06_yolo26
pending    80.5 h  train_bmd_v06_rtdetr
pending     1.0 h  predict_bmd_v06_rtdetr
pending    28.9 h  train_bmd_v06_yolov8
pending     0.5 h  predict_bmd_v06_yolov8
pending     1.5 h  train_pals_rev_yolo26_rev
pending     0.5 h  predict_pals_rev_yolo26_rev
pending     3.0 h  train_pals_rev_rtdetr_rev
pending     1.0 h  predict_pals_rev_rtdetr_rev
pending    32.1 h  train_bmd_v06_yolo26_s1
pending     0.5 h  predict_bmd_v06_yolo26_s1
pending    32.1 h  train_bmd_v06_yolo26_s2
pending     0.5 h  predict_bmd_v06_yolo26_s2
pending    80.5 h  train_bmd_v06_rtdetr_s1
pending     1.0 h  predict_bmd_v06_rtdetr_s1
pending    80.5 h  train_bmd_v06_rtdetr_s2
pending     1.0 h  predict_bmd_v06_rtdetr_s2
pending    33.7 h  train_bmd_v06_yolo26_evid  [released 18 Nov after the Phase 2 pilot met the AP gate; runs after Phase 4 core]
pending     0.5 h  predict_bmd_v06_yolo26_evid  [released 18 Nov after the Phase 2 pilot met the AP gate; runs after Phase 4 core]
pending    33.7 h  train_bmd_v06_yolo26_evid_s1  [released 18 Nov after the Phase 2 pilot met the AP gate; runs after Phase 4 core]
pending     0.5 h  predict_bmd_v06_yolo26_evid_s1  [released 18 Nov after the Phase 2 pilot met the AP gate; runs after Phase 4 core]
pending    33.7 h  train_bmd_v06_yolo26_evid_s2  [released 18 Nov after the Phase 2 pilot met the AP gate; runs after Phase 4 core]
pending     0.5 h  predict_bmd_v06_yolo26_evid_s2  [released 18 Nov after the Phase 2 pilot met the AP gate; runs after Phase 4 core]
held       84.6 h  train_bmd_v06_rtdetr_evid  [LAST and held separately: evidential RT-DETR, only if Phase 2 shows it is feasible and you release it]
held        1.0 h  predict_bmd_v06_rtdetr_evid  [LAST and held separately: evidential RT-DETR, only if Phase 2 shows it is feasible and you release it]
held       84.6 h  train_bmd_v06_rtdetr_evid_s1  [LAST and held separately: evidential RT-DETR, only if Phase 2 shows it is feasible and you release it]
held        1.0 h  predict_bmd_v06_rtdetr_evid_s1  [LAST and held separately: evidential RT-DETR, only if Phase 2 shows it is feasible and you release it]
held       84.6 h  train_bmd_v06_rtdetr_evid_s2  [LAST and held separately: evidential RT-DETR, only if Phase 2 shows it is feasible and you release it]
held        1.0 h  predict_bmd_v06_rtdetr_evid_s2  [LAST and held separately: evidential RT-DETR, only if Phase 2 shows it is feasible and you release it]
runner: ALIVE (last heartbeat 2026-11-18 19:16:14, 0.7 min ago, PID 62940, job train_bmd_v06_yolo26)

done 16 | running 1 | pending 23 | failed 0 | held 6 -> about 480.3 h left in released jobs (+256.8 h held)
```
