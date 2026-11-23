#!/usr/bin/env bash
# Results freeze for bmd_v06 (runs after the 13 ablation and 20b certifiability jobs ended COMPLETED 0:0):
# digest (12) incl. the RQ2 ablation section, interim report (21), commit + push the text outputs
# (never .pt / .npz / calibration.json), then tag that commit v0.6-final and push the tag.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"   # sbatch runs a spool copy: $0 is not in the repo
source scripts/beocat/env.sh
source "$CONFDET_VENV/bin/activate"

CFG=configs/experiment_v06.yaml
DETS="yolo26 rtdetr yolov8 yolo26_s1 yolo26_s2 rtdetr_s1 rtdetr_s2 yolo26_evid yolo26_evid_s1 yolo26_evid_s2 yolo26_ens rtdetr_ens"

echo "== $(date '+%F %T') 12 digest"
python scripts/12_summarize.py --config "$CFG" --detectors $DETS > /dev/null
grep -c "RQ2 ablation" outputs/bmd_v06/digest.md || { echo "digest has no RQ2 ablation section"; exit 1; }
echo "== $(date '+%F %T') 21 interim report (v06)"
python scripts/21_interim_report.py --experiment bmd_v06 --evid yolo26_evid,yolo26_evid_s1,yolo26_evid_s2 --queue none

echo "== $(date '+%F %T') commit + push + tag"
find outputs/bmd_v06 -type f \( -name '*.md' \
  -o \( -name '*.json' \( -path '*/results/*' -o -path '*/comparison/*' -o -path '*/seeds/*' -o -name 'ablation_*.json' \) \) \) \
  -not -name 'calibration.json' -print0 | xargs -0 git add -f
git add src/confdet/uq/conformal.py tests/test_conformal.py scripts/20b_detrac_certifiability.py \
        scripts/21_interim_report.py scripts/beocat/v06_freeze.sh
git status --short | sed -n "1,60p" || true
if ! git diff --cached --quiet; then
  git commit -q -F - <<'MSG'
v06 freeze: RQ2 ablation, UA-DETRAC certifiability per o_hat group, final evidential note

- 13 (RQ2 ablation) on the bmd_v06 models, same settings as bmd_dev (5 folds, class_agnostic,
  config detectors); the digest now has its "RQ2 ablation" section.
- 20b + conformal.certifiability_by_group (with test): per o_hat group of the Mondrian rules,
  calib n, objects never proposed, minimum certifiable alpha vs target. Thresholds refitted
  exactly as scripts/20 fits them; analysis only, no rule changed. Table in the v06 interim report.
- 21: the final evidential models get a one-line note instead of the pilot's gate line and
  RT-DETR feasibility paragraph (numbers unchanged; the bmd_dev report is unchanged).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
MSG
fi
git push origin HEAD
git tag -a v0.6-final -m "v0.6 results freeze (bmd_v06 final models)"
git push origin v0.6-final
git log -1 --format='%H %s'
echo "== $(date '+%F %T') done"
