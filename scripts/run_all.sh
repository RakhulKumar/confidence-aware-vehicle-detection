#!/usr/bin/env bash
# Run the whole pipeline, stopping at the first failure.
#
#   bash scripts/run_all.sh                              # active detector, configs/experiment.yaml
#   bash scripts/run_all.sh --all-detectors              # yolo26, yolov8, rtdetr + cross-detector ranking
#   bash scripts/run_all.sh --detectors "yolo26 yolov8"
#   bash scripts/run_all.sh --skip-training              # reuse existing checkpoints
#   bash scripts/run_all.sh --skip-data                  # keep the current split (do NOT re-split)
#   bash scripts/run_all.sh --analysis-only --all-detectors
#                                                        # no GPU: re-fit the post-hoc layer and re-run every
#                                                        # analysis from the cached predictions (minutes)
#   bash scripts/run_all.sh --config configs/experiment_pals_dev.yaml
#
# Stages read and write files only, so any stage can be re-run on its own.
# NOTE: re-running stage 02 re-splits the data; checkpoints trained on an
# older split must then be retrained (the test images may have changed).

set -euo pipefail
cd "$(dirname "$0")/.."

# On Windows `python3` can be the Microsoft Store stub that only prints a
# message; fall back to `python` (the activated venv).
if command -v python3 >/dev/null 2>&1 && python3 --version >/dev/null 2>&1; then
  PYTHON=python3
else
  PYTHON=python
fi

CONFIG="configs/experiment.yaml"
SKIP_TRAINING=false
SKIP_DATA=false
SKIP_PREDICT=false
DETECTORS=""
ALL=false
while [ $# -gt 0 ]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --detectors) DETECTORS="$2"; shift 2 ;;
    --all-detectors) ALL=true; shift ;;
    --skip-training) SKIP_TRAINING=true; shift ;;
    --skip-data) SKIP_DATA=true; shift ;;
    --analysis-only) SKIP_DATA=true; SKIP_TRAINING=true; SKIP_PREDICT=true; shift ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

if [ "$ALL" = true ]; then
  DETECTORS=$($PYTHON -c "import yaml; print(' '.join(yaml.safe_load(open('$CONFIG', encoding='utf-8'))['detectors']['models']))")
elif [ -z "$DETECTORS" ]; then
  DETECTORS=$($PYTHON -c "import yaml; print(yaml.safe_load(open('$CONFIG', encoding='utf-8'))['detectors']['active'])")
fi

banner() { echo "=================================================================="; echo "$1"; echo "=================================================================="; }

if [ "$SKIP_DATA" = false ]; then
  banner "STAGE 0: validate datasets, build leak-free roles"
  $PYTHON scripts/02_prepare_data.py --config "$CONFIG"
fi

banner "STAGE 2: occlusion report and power gate"
$PYTHON scripts/03_compute_occlusion.py --config "$CONFIG"

for DET in $DETECTORS; do
  if [ "$SKIP_TRAINING" = false ]; then
    banner "STAGE 1: train $DET"
    $PYTHON scripts/04_train_detector.py --config "$CONFIG" --detector "$DET"
  fi
  if [ "$SKIP_PREDICT" = false ]; then
    banner "PREDICT: cache raw candidates ($DET)"
    $PYTHON scripts/05_predict.py --config "$CONFIG" --detector "$DET"
  fi
  banner "STAGE 3: fit the post-hoc layer ($DET)"
  $PYTHON scripts/06_calibrate.py --config "$CONFIG" --detector "$DET"
  banner "STAGE 4: RQ1-RQ5 ($DET)"
  $PYTHON scripts/07_run_evaluation.py --config "$CONFIG" --detector "$DET"
done

banner "CROSS-DETECTOR: reliability ranking under each metric"
$PYTHON scripts/08_compare_detectors.py --config "$CONFIG" --detectors $DETECTORS

banner "DIGEST: all headline numbers in one file"
$PYTHON scripts/12_summarize.py --config "$CONFIG" --detectors $DETECTORS > /dev/null

banner "ALL STAGES COMPLETE"
echo "Read first:    outputs/<experiment>/digest.md"
echo "Per detector:  outputs/<experiment>/<detector>/<mode>/results/  and  .../figures/"
echo "Ranking:       outputs/<experiment>/comparison/<mode>/ranking.json"
echo "Optional:      $PYTHON scripts/09_synthetic_occlusion_probe.py --config $CONFIG --detector <name>"
