#!/bin/bash
# One Slurm job: train (scripts/04 --resume) or predict (scripts/05) one detector on one L40S.
#
#   sbatch --time=24:00:00 --export=ALL,STAGE=train,CONFIG=configs/experiment_v06.yaml,DETECTOR=yolo26,BATCH=16 \
#          scripts/beocat/train_job.sbatch
#   (scripts/beocat/submit_all.sh does this for every Phase 4 job, with the log in the run folder)
#
# Restart-safe: 15 minutes before the time limit Slurm sends USR1; the job stops training and
# requeues itself (same job id, so --dependency=afterok chains keep waiting). The requeued job runs
# `04 --resume` again, which continues from <run>/train/weights/last.pt exactly as on the Windows PC
# (Ultralytics resume=True: same arguments, seed and epoch counter). A node failure or preemption is
# requeued by Slurm itself (--requeue). After MAX_RESTARTS restarts the job fails instead of looping.
#
# Variables:
#   STAGE         train | predict
#   CONFIG        e.g. configs/experiment_v06.yaml
#   DETECTOR      an entry of detectors.models in CONFIG
#   BATCH         train only: the architecture's batch from configs/beocat_batch.yaml
#   MAX_RESTARTS  default 40
#SBATCH --job-name=confdet
#SBATCH --partition=killable.q
#SBATCH --gres=gpu:l40s:1,killable:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@900
#SBATCH --output=slurm-%x-%j.out

set -uo pipefail
: "${STAGE:?set STAGE=train|predict}" "${CONFIG:?set CONFIG}" "${DETECTOR:?set DETECTOR}"
cd "${CONFDET_REPO:-${SLURM_SUBMIT_DIR:-.}}"     # sbatch from the repo folder (submit_all.sh does)
source scripts/beocat/env.sh          # CONFDET_SITE=beocat, CONFDET_GPU_SHARE=off
source "$CONFDET_VENV/bin/activate"
export CONFDET_GPU_SHARE=off          # the GPU is ours: no VRAM cap, no duty cycle

RUN_DIR=$(python - "$CONFIG" "$DETECTOR" <<'EOF'
import sys
sys.path.insert(0, "src")
from confdet.utils.io_utils import load_config
from confdet.utils.paths import train_dir
print(train_dir(load_config(sys.argv[1]), sys.argv[2]))
EOF
)
mkdir -p "$RUN_DIR/logs"
LOG="$RUN_DIR/logs/${STAGE}_${SLURM_JOB_ID}.log"
exec >>"$LOG" 2>&1
RESTARTS="${SLURM_RESTART_COUNT:-0}"
echo "===== $(date '+%F %T') $STAGE $DETECTOR ($CONFIG) job $SLURM_JOB_ID restart $RESTARTS on $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
if [ "$RESTARTS" -gt "${MAX_RESTARTS:-40}" ]; then
  echo "restarted $RESTARTS times (> MAX_RESTARTS): failing instead of looping"; exit 1
fi

case "$STAGE" in
  train)
    : "${BATCH:?set BATCH (configs/beocat_batch.yaml)}"
    python scripts/04_train_detector.py --config "$CONFIG" --detector "$DETECTOR" --resume --batch "$BATCH" &
    ;;
  predict)
    python scripts/05_predict.py --config "$CONFIG" --detector "$DETECTOR" &
    ;;
  *) echo "unknown STAGE=$STAGE"; exit 1 ;;
esac
PID=$!

requeue() {
  echo "===== $(date '+%F %T') time limit near: stopping and requeueing (resumes from last.pt)"
  kill -TERM "$PID" 2>/dev/null
  wait "$PID" 2>/dev/null
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap requeue USR1

wait "$PID"
CODE=$?
echo "===== $(date '+%F %T') EXIT CODE $CODE"
exit "$CODE"
