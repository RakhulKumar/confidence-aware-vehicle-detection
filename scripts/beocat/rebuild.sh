#!/usr/bin/env bash
# Rebuild every processed dataset on Beocat from the TRANSFERRED manifests (no re-splitting), then
# check identity. Stops at the FIRST difference: one missing, extra or changed file, one image whose
# leakage group or object count differs, or one role assignment that 02b --verify cannot reproduce.
#
#   bash scripts/beocat/rebuild.sh
#
# Needs: get_data.sh bundle, bmd45 and detrac done. Writes $CONFDET_ROOT/confdet_data/<experiment>/<dataset>/
# and outputs/<experiment>/derive/{rebuild,verify}_*.json. Refuses to overwrite a folder that exists.
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$CONFDET_REPO"
source "$CONFDET_VENV/bin/activate"
BUNDLE="$CONFDET_TRANSFER/manifests"
[ -d "$BUNDLE" ] || { echo "no $BUNDLE: run get_data.sh bundle first" >&2; exit 1; }

rebuild() {   # <config> <dataset>
  echo "== rebuild $1 $2"
  python scripts/02d_rebuild_from_manifest.py --config "configs/$1.yaml" --bundle "$BUNDLE" --dataset "$2"
}

# split experiments first, then the experiments derived from them (02b --build is NOT used: the
# derived manifests are rebuilt directly, and compared file by file)
rebuild experiment bmd45
rebuild experiment pals
rebuild experiment_v06 bmd45
rebuild experiment_v06 pals
rebuild experiment_pals_rev pals
rebuild experiment_pals_rev bmd45
rebuild experiment_pals_dev pals
rebuild experiment_pals_dev bmd45

# UA-DETRAC: converted again from the raw XML (deterministic), then compared file by file
echo "== convert + verify UA-DETRAC"
python scripts/02c_convert_detrac.py --config configs/experiment_detrac.yaml --every 10
python scripts/02d_rebuild_from_manifest.py --config configs/experiment_detrac.yaml --bundle "$BUNDLE" \
  --dataset ua_detrac --verify-only

# role identity: re-derive every `roles` rule from the raw data with scripts/02's code and seed
echo "== 02b --verify"
python scripts/02b_derive_experiment.py --verify --config configs/experiment.yaml --dataset bmd45 --rule 0
python scripts/02b_derive_experiment.py --verify --config configs/experiment.yaml --dataset pals --rule 0
python scripts/02b_derive_experiment.py --verify --config configs/experiment_pals_dev.yaml --dataset pals --rule 0
python scripts/02b_derive_experiment.py --verify --config configs/experiment_pals_dev.yaml --dataset bmd45 --rule 0

echo "REBUILD IDENTICAL: every dataset matches the Windows PC file by file"
