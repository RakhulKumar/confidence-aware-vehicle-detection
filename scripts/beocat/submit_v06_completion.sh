#!/usr/bin/env bash
# Run on the FINAL bmd_v06 models every analysis that run_all.sh --analysis-only does not cover
# (scripts 14-22), with the same scripts, defaults and pre-specified rules as on bmd_dev.
#
#   bash scripts/beocat/submit_v06_completion.sh [--dry-run]
#
# GPU (killable.q, L40S, like train_job.sbatch):
#   pred_detrac_<det>   05 with configs/experiment_v06_detrac.yaml (yolo26 yolov8 rtdetr; ua_detrac calib+test only,
#                       the bmd45 files are complete and skipped)
#   nms_predict         18 --predict (yolo26 seed 0, suppression off)
#   evid_predict_<det>  19 --predict (per-box evidence) for yolo26_evid, _s1, _s2
# CPU:
#   ens_seeds           15 ensembles -> 06/07 on yolo26_ens, rtdetr_ens -> 14 seed summary
#   baselines           16 (default detectors)                after ens_seeds
#   phase1b             17 (default detectors and families)   after ens_seeds
#   nms_analyse         18 --analyse                          after nms_predict
#   evid_analyse        19 --analyse + 22 for each evid seed  after evid_predict_*
#   detrac              20 (default detectors)                after pred_detrac_*
#   v06_finalize        08 + 12 digest + 21 report, commit + push   after ALL of the above (afterok)
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$CONFDET_REPO"
DRY=false; [ "${1:-}" = "--dry-run" ] && DRY=true

CFG=configs/experiment_v06.yaml
DCFG=configs/experiment_v06_detrac.yaml
LOG="$CONFDET_ROOT/logs"
PRE="cd '$CONFDET_REPO' && source scripts/beocat/env.sh && source '$CONFDET_VENV/bin/activate' && set -e"
GPU=(--partition=killable.q --gres=gpu:l40s:1,killable:1 --cpus-per-task=8 --mem=64G --requeue --time=06:00:00)
CPU=(--cpus-per-task=8 --mem=64G --time=12:00:00)

sub() {   # name dep-or-empty sbatch-args... ; prints job id
  local name="$1" dep="$2"; shift 2
  local args=(--parsable --job-name="$name" --output="$LOG/$name-%j.out" ${dep:+--dependency=afterok:$dep} "$@")
  if $DRY; then echo "sbatch ${args[*]}" >&2; echo "DRY_$name"; else sbatch "${args[@]}"; fi
}
wrap() { echo "--wrap=$PRE && $1"; }

# --- GPU ---
D=()
for det in yolo26 yolov8 rtdetr; do
  D+=("$(sub "pred_detrac_$det" "" "${GPU[@]}" --export=ALL,STAGE=predict,CONFIG=$DCFG,DETECTOR=$det scripts/beocat/train_job.sbatch)")
done
NMS_P=$(sub nms_predict "" "${GPU[@]}" "$(wrap "python scripts/18_nms_pilot.py --config $CFG --predict")")
E=()
for det in yolo26_evid yolo26_evid_s1 yolo26_evid_s2; do
  E+=("$(sub "evid_predict_$det" "" "${GPU[@]}" "$(wrap "python scripts/19_evidential_pilot.py --config $CFG --predict --detector $det")")")
done

# --- CPU ---
ENS=$(sub ens_seeds "" "${CPU[@]}" "$(wrap "python scripts/15_build_ensemble.py --config $CFG && \
for d in yolo26_ens rtdetr_ens; do python scripts/06_calibrate.py --config $CFG --detector \$d && python scripts/07_run_evaluation.py --config $CFG --detector \$d; done && \
python scripts/14_seed_summary.py --config $CFG --family yolo26=yolo26,yolo26_s1,yolo26_s2 --family rtdetr=rtdetr,rtdetr_s1,rtdetr_s2 \
--family yolo26_evid=yolo26_evid,yolo26_evid_s1,yolo26_evid_s2")")
BASE=$(sub baselines "$ENS" "${CPU[@]}" "$(wrap "python scripts/16_baselines.py --config $CFG")")
P1B=$(sub phase1b "$ENS" "${CPU[@]}" "$(wrap "python scripts/17_phase1b.py --config $CFG")")
NMS_A=$(sub nms_analyse "$NMS_P" "${CPU[@]}" "$(wrap "python scripts/18_nms_pilot.py --config $CFG --analyse")")
EV_A=$(sub evid_analyse "$(IFS=:; echo "${E[*]}")" "${CPU[@]}" "$(wrap "for d in yolo26_evid yolo26_evid_s1 yolo26_evid_s2; do \
python scripts/19_evidential_pilot.py --config $CFG --analyse --detector \$d && python scripts/22_evid_rarity_diag.py --config $CFG --detector \$d; done")")
DET_A=$(sub detrac "$(IFS=:; echo "${D[*]}")" "${CPU[@]}" "$(wrap "python scripts/20_detrac.py --config $DCFG")")
FIN=$(sub v06_finalize "$ENS:$BASE:$P1B:$NMS_A:$EV_A:$DET_A" "${CPU[@]}" --time=04:00:00 scripts/beocat/v06_finalize.sh)

cat <<EOF
GPU   pred_detrac yolo26/yolov8/rtdetr: ${D[*]}
      nms_predict: $NMS_P
      evid_predict yolo26_evid/_s1/_s2: ${E[*]}
CPU   ens_seeds: $ENS   baselines: $BASE   phase1b: $P1B
      nms_analyse: $NMS_A   evid_analyse: $EV_A   detrac: $DET_A
FINAL v06_finalize: $FIN
EOF
