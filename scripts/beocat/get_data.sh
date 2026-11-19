#!/usr/bin/env bash
# Get every input on Beocat: the transfer bundle from the Windows PC, BMD-45 from Hugging Face,
# UA-DETRAC from its Google Drive links. Stops at the first error.
#
#   bash scripts/beocat/get_data.sh bundle    # verify SHA256 of the 4 archives, then unpack them
#   bash scripts/beocat/get_data.sh bmd45     # BMD-45 train + val (needs `hf auth login` once, interactively)
#   bash scripts/beocat/get_data.sh detrac    # UA-DETRAC images + XML annotations (needs the Drive file ids below)
#   bash scripts/beocat/get_data.sh all
#
# The bundle (built on the Windows PC in D:\confdet_project\transfer\) must first be copied to
# $CONFDET_ROOT/transfer/ (scp / rsync / Globus): 1_manifests, 2_pals_raw, 3_weights, 4_outputs,
# each as .tar.gz plus .tar.gz.sha256.
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$CONFDET_REPO"
WHAT="${1:-all}"
RAW="$CONFDET_ROOT/raw"
TRANSFER="$CONFDET_ROOT/transfer"
mkdir -p "$RAW" "$TRANSFER"

# ---- UA-DETRAC Google Drive file ids: FILL IN before running `detrac` ------------------------------
# The official UA-DETRAC release links (images and XML annotations, train and test). Each id is the
# part after /d/ in https://drive.google.com/file/d/<id>/view .
DETRAC_IDS=(
  # "<id of DETRAC-train-data.zip>"
  # "<id of DETRAC-test-data.zip>"
  # "<id of DETRAC-Train-Annotations-XML.zip>"
  # "<id of DETRAC-Test-Annotations-XML.zip>"
)

bundle() {
  cd "$TRANSFER"
  for part in 1_manifests 2_pals_raw 3_weights 4_outputs; do
    [ -f "$part.tar.gz" ] || { echo "missing $TRANSFER/$part.tar.gz: copy the bundle from the Windows PC first" >&2; exit 1; }
    sha256sum -c "$part.tar.gz.sha256"          # exits non-zero on a mismatch
  done
  mkdir -p "$TRANSFER/manifests"
  tar -xzf 1_manifests.tar.gz -C "$TRANSFER/manifests"
  tar -xzf 2_pals_raw.tar.gz -C "$RAW"                                  # -> raw/AU-INV-PALS/
  # weights: repo/outputs/... (models trained before v0.6) and root/runs/... (pilots), see the archive's README
  tar -xzf 3_weights.tar.gz -C "$CONFDET_REPO" --strip-components=1 repo
  tar -xzf 3_weights.tar.gz -C "$CONFDET_ROOT" --strip-components=1 root
  tar -xzf 4_outputs.tar.gz -C "$CONFDET_REPO"                           # -> outputs/
  cd "$CONFDET_REPO"
  echo "bundle unpacked: manifests in $TRANSFER/manifests, PALS in $RAW/AU-INV-PALS, outputs/ and weights in place"
}

bmd45() {
  source "$CONFDET_VENV/bin/activate"
  python -c "from huggingface_hub import whoami; whoami()" >/dev/null 2>&1 || {
    echo "not logged in to Hugging Face: run \`hf auth login\` once, interactively (the token stays in \$HF_HOME, never in the repo)" >&2; exit 1; }
  python scripts/download_bmd45.py --splits train val --out "$RAW/bmd45" --raw-download-dir "$RAW/hf_stage"
  echo "BMD-45 in $RAW/bmd45 (identity is checked image by image by rebuild.sh)"
}

detrac() {
  source "$CONFDET_VENV/bin/activate"
  if [ ${#DETRAC_IDS[@]} -eq 0 ]; then
    echo "fill in DETRAC_IDS in $0 (the official Google Drive file ids) first" >&2; exit 1
  fi
  local D="$RAW/ua_detrac"
  mkdir -p "$D/_zips" "$D/_unzipped" "$D/DETRAC-Images" "$D/DETRAC-Annos"
  for id in "${DETRAC_IDS[@]}"; do
    (cd "$D/_zips" && gdown --continue "$id")
  done
  for z in "$D"/_zips/*.zip; do unzip -q -n "$z" -d "$D/_unzipped"; done
  # bring the unpacked folders into the layout the converter reads (configs/paths_local.yaml raw_ua_detrac):
  #   DETRAC-Images/MVI_xxxxx/img00001.jpg   DETRAC-Annos/DETRAC-{Train,Test}-Annotations-XML/MVI_xxxxx.xml
  find "$D/_unzipped" -type d -name 'MVI_*' -prune -exec mv -n {} "$D/DETRAC-Images/" \;
  find "$D/_unzipped" -type d -name 'DETRAC-*-Annotations-XML' -prune -exec mv -n {} "$D/DETRAC-Annos/" \;
  local n_seq n_tr n_te
  n_seq=$(find "$D/DETRAC-Images" -mindepth 1 -maxdepth 1 -type d | wc -l)
  n_tr=$(find "$D/DETRAC-Annos/DETRAC-Train-Annotations-XML" -name '*.xml' | wc -l)
  n_te=$(find "$D/DETRAC-Annos/DETRAC-Test-Annotations-XML" -name '*.xml' | wc -l)
  echo "UA-DETRAC: $n_seq image sequences, $n_tr train XML, $n_te test XML (the Windows copy had 60 train + 40 test)"
  [ "$n_tr" -eq 60 ] && [ "$n_te" -eq 40 ] || { echo "unexpected sequence counts: stop" >&2; exit 1; }
}

case "$WHAT" in
  bundle) bundle ;;
  bmd45) bmd45 ;;
  detrac) detrac ;;
  all) bundle; bmd45; detrac ;;
  *) echo "usage: $0 bundle|bmd45|detrac|all" >&2; exit 1 ;;
esac
