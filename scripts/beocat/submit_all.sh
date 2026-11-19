#!/usr/bin/env bash
# Submit the v0.6 Phase 4 jobs to Slurm in the SAME ORDER as the Windows queue, chained with
# --dependency=afterok, then the analysis jobs. Evidential RT-DETR stays held (not submitted).
#
#   bash scripts/beocat/submit_all.sh             # strict chain: each job starts after the previous one succeeded
#   bash scripts/beocat/submit_all.sh --parallel  # only the queue's own dependencies (predict after its train,
#                                                 # evidential after Phase 4 core): independent models run at once
#   bash scripts/beocat/submit_all.sh --dry-run   # print the sbatch commands, submit nothing
#
#   TIME_TRAIN (default 24:00:00) and TIME_PREDICT (04:00:00): Slurm time limit per job. A training job
#   that reaches it requeues itself and resumes from last.pt (train_job.sbatch).
#
# Refuses to start while configs/beocat_batch.yaml still has a placeholder (null) for an architecture it
# needs. A failed job leaves its dependents pending with reason DependencyNeverSatisfied: fix, then
# resubmit from that job (the finished ones are skipped by 04 --resume / 05 themselves).
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$CONFDET_REPO"
source "$CONFDET_VENV/bin/activate"

MODE=chain; DRY=false
for a in "$@"; do
  case "$a" in
    --parallel) MODE=parallel ;;
    --dry-run) DRY=true ;;
    *) echo "unknown argument $a" >&2; exit 1 ;;
  esac
done
TIME_TRAIN="${TIME_TRAIN:-24:00:00}"
TIME_PREDICT="${TIME_PREDICT:-04:00:00}"
TIME_ANALYSIS="${TIME_ANALYSIS:-08:00:00}"
export BATCH_FILE="${BATCH_FILE:-configs/beocat_batch.yaml}"   # override only to test
V06=configs/experiment_v06.yaml
REV=configs/experiment_pals_rev.yaml

# order of the Windows queue (D:/confdet_project/queue/v06_jobs.yaml, 19 Nov): Phase 4 core, then evidential yolo26 x3
CORE=(
  "$V06 yolo26" "$V06 rtdetr" "$V06 yolov8"
  "$REV yolo26_rev" "$REV rtdetr_rev"
  "$V06 yolo26_s1" "$V06 yolo26_s2" "$V06 rtdetr_s1" "$V06 rtdetr_s2"
)
EVID=( "$V06 yolo26_evid" "$V06 yolo26_evid_s1" "$V06 yolo26_evid_s2" )

batch_of() {   # detector -> batch from configs/beocat_batch.yaml ("" if a placeholder)
  python - "$1" <<'EOF'
import sys, yaml
import os
arch = yaml.safe_load(open(os.environ["BATCH_FILE"], encoding="utf-8"))["architectures"]
hits = [a for a in arch.values() if sys.argv[1] in a["detectors"]]
if len(hits) != 1:
    sys.exit(f"{sys.argv[1]}: in {len(hits)} architectures of {os.environ['BATCH_FILE']}")
print("" if hits[0]["batch"] is None else int(hits[0]["batch"]))
EOF
}
run_dir_of() {   # config detector -> training run folder (logs go to <run>/logs)
  python - "$1" "$2" <<'EOF'
import sys
sys.path.insert(0, "src")
from confdet.utils.io_utils import load_config
from confdet.utils.paths import train_dir
print(train_dir(load_config(sys.argv[1]), sys.argv[2]))
EOF
}

# every batch must be set before anything is submitted
missing=()
for spec in "${CORE[@]}" "${EVID[@]}"; do
  set -- $spec
  [ -n "$(batch_of "$2")" ] || missing+=("$2")
done
if [ ${#missing[@]} -gt 0 ]; then
  echo "$BATCH_FILE has no batch yet for: ${missing[*]} -- run the batch trial first" >&2
  exit 1
fi

submit() {   # dependency-or-empty, then sbatch arguments; prints the job id
  local dep="$1"; shift
  if $DRY; then
    echo "sbatch ${dep:+--dependency=afterok:$dep} $*" >&2
    echo "DRY$RANDOM"
  else
    sbatch --parsable ${dep:+--dependency=afterok:$dep} "$@"
  fi
}

PREV=""            # last submitted job (strict chain)
declare -A PRED    # detector -> its predict job id
submit_model() {   # config detector after-dep -> submits train + predict
  local cfg="$1" det="$2" after="$3" run batch dep tid pid
  run=$(run_dir_of "$cfg" "$det"); batch=$(batch_of "$det")
  $DRY || mkdir -p "$run/logs"
  dep=$([ "$MODE" = chain ] && echo "$PREV" || echo "$after")
  tid=$(submit "$dep" --job-name="tr_$det" --time="$TIME_TRAIN" --output="$run/logs/slurm-%j.out" \
        --export=ALL,STAGE=train,CONFIG="$cfg",DETECTOR="$det",BATCH="$batch" scripts/beocat/train_job.sbatch)
  pid=$(submit "$tid" --job-name="pr_$det" --time="$TIME_PREDICT" --output="$run/logs/slurm-%j.out" \
        --export=ALL,STAGE=predict,CONFIG="$cfg",DETECTOR="$det" scripts/beocat/train_job.sbatch)
  PREV="$pid"; PRED[$det]="$pid"
  echo "$det: train $tid (batch $batch) -> predict $pid"
}

for spec in "${CORE[@]}"; do set -- $spec; submit_model "$1" "$2" ""; done
CORE_DONE=$(IFS=:; echo "${PRED[*]}")
# evidential yolo26 x3 after Phase 4 core, one after another (as in the queue)
EV_PREV="$CORE_DONE"
for spec in "${EVID[@]}"; do set -- $spec; submit_model "$1" "$2" "$EV_PREV"; EV_PREV="${PRED[$2]}"; done

# ---- analysis jobs (CPU only): post-hoc layer, RQ1-RQ5, detector comparison, digest ----------------
$DRY || mkdir -p "$CONFDET_ROOT/logs"
ana() {   # dependency name command...
  local dep="$1" name="$2"; shift 2
  submit "$dep" --job-name="$name" --time="$TIME_ANALYSIS" --cpus-per-task=8 --mem=64G \
         --output="$CONFDET_ROOT/logs/$name-%j.out" --wrap="cd '$CONFDET_REPO' && source scripts/beocat/env.sh && source '$CONFDET_VENV/bin/activate' && $*"
}
dep_of() { local ids=(); for d in "$@"; do ids+=("${PRED[$d]}"); done; (IFS=:; echo "${ids[*]}"); }
A1=$(ana "$(dep_of yolo26 rtdetr yolov8 yolo26_s1 yolo26_s2 rtdetr_s1 rtdetr_s2)" ana_v06_core \
     bash scripts/run_all.sh --analysis-only --config $V06 --detectors \"yolo26 rtdetr yolov8 yolo26_s1 yolo26_s2 rtdetr_s1 rtdetr_s2\")
A2=$(ana "$(dep_of yolo26_rev rtdetr_rev)" ana_pals_rev \
     bash scripts/run_all.sh --analysis-only --config $REV --detectors \"yolo26_rev rtdetr_rev\")
# all ten v0.6 detectors together, so the comparison (08) and the digest (12) cover core AND evidential models
A3=$(ana "$(dep_of yolo26_evid yolo26_evid_s1 yolo26_evid_s2):$A1" ana_v06_all \
     bash scripts/run_all.sh --analysis-only --config $V06 --detectors \"yolo26 rtdetr yolov8 yolo26_s1 yolo26_s2 rtdetr_s1 rtdetr_s2 yolo26_evid yolo26_evid_s1 yolo26_evid_s2\")
echo "analysis: v06 core $A1, pals_rev $A2, v06 core + evidential $A3"
echo "submitted ($MODE). Watch: squeue -u \$USER ; logs in <run>/logs and $CONFDET_ROOT/logs"
