#!/usr/bin/env bash
# Last job of submit_v06_completion.sh (runs only after every other completion job ended COMPLETED 0:0):
# cross-detector ranking + digest for bmd_v06 with the ensembles included, the v06 interim report (21),
# then commit and push the text outputs (digests, result jsons; never .pt / .npz / calibration.json).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"   # sbatch runs a spool copy: $0 is not in the repo
source scripts/beocat/env.sh
source "$CONFDET_VENV/bin/activate"

CFG=configs/experiment_v06.yaml
DETS="yolo26 rtdetr yolov8 yolo26_s1 yolo26_s2 rtdetr_s1 rtdetr_s2 yolo26_evid yolo26_evid_s1 yolo26_evid_s2 yolo26_ens rtdetr_ens"

echo "== $(date '+%F %T') 08 ranking"
python scripts/08_compare_detectors.py --config "$CFG" --detectors $DETS
echo "== $(date '+%F %T') 12 digest"
python scripts/12_summarize.py --config "$CFG" --detectors $DETS > /dev/null
echo "== $(date '+%F %T') 21 interim report (v06)"
python scripts/21_interim_report.py --experiment bmd_v06 --evid yolo26_evid,yolo26_evid_s1,yolo26_evid_s2 --queue none

echo "== $(date '+%F %T') commit + push text outputs"
# outputs/* is gitignored: add the reporting files explicitly (-f)
find outputs -type f \( -name '*.md' \
  -o \( -name '*.json' \( -path '*/results/*' -o -path '*/comparison/*' -o -path '*/seeds/*' -o -path '*/derive/*' \
        -o -name 'ensemble.json' -o -name 'occlusion_report.json' -o -name 'data_report.json' \) \) \) \
  -not -name 'calibration.json' -print0 | xargs -0 git add -f
git add configs/experiment_v06_detrac.yaml scripts/19_evidential_pilot.py scripts/21_interim_report.py \
        scripts/beocat/submit_v06_completion.sh scripts/beocat/v06_finalize.sh
git status --short | sed -n "1,50p" || true   # head would SIGPIPE git under pipefail
if git diff --cached --quiet; then
  echo "nothing to commit"
else
  git commit -q -F - <<'EOF'
v06 results: digest, interim report and result jsons for the final models

Analyses on the final bmd_v06 models: RQ1-RQ5 + PALS transfer (07), seed summary (14),
ensembles (15, then 06/07), literature baselines (16), Phase 1b incl. the pre-specified
Mondrian on the Gilg score (17), NMS pilot (18), evidential analysis (19) and calib
rarity diagnostic (22) for the three evidential seeds, UA-DETRAC proxy validation and
transfer (20). Text outputs only (no weights, prediction caches or calibration.json).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
fi
git push origin HEAD
git log -1 --format='%h %s'
echo "== $(date '+%F %T') done"
