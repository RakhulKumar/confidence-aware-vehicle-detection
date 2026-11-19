#!/usr/bin/env bash
# Get every input on Beocat: the transfer bundle from the Windows PC (including UA-DETRAC, copied as
# it was on the PC's disk) and BMD-45 from Hugging Face. Stops at the first error.
#
#   bash scripts/beocat/get_data.sh bundle    # verify SHA256 of the 4 archives, then unpack them
#   bash scripts/beocat/get_data.sh bmd45     # BMD-45 train + val (needs `hf auth login` once, interactively)
#   bash scripts/beocat/get_data.sh detrac    # UA-DETRAC: verify SHA256 of 5_ua_detrac_raw.tar.gz, then unpack it
#   bash scripts/beocat/get_data.sh all
#
# The bundle (built on the Windows PC in D:\confdet_project\transfer\) must first be copied to
# $CONFDET_TRANSFER (default ~/confdet_transfer; scp / rsync / Globus): 1_manifests, 2_pals_raw, 3_weights, 4_outputs,
# 5_ua_detrac_raw, each as .tar.gz plus .tar.gz.sha256. (UA-DETRAC is not downloaded with gdown:
# Google Drive's download limits make that unreliable.)
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$CONFDET_REPO"
WHAT="${1:-all}"
RAW="$CONFDET_ROOT/raw"
TRANSFER="$CONFDET_TRANSFER"
mkdir -p "$RAW" "$TRANSFER"

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
  cd "$TRANSFER"
  [ -f 5_ua_detrac_raw.tar.gz ] || { echo "missing $TRANSFER/5_ua_detrac_raw.tar.gz: copy it from the Windows PC first" >&2; exit 1; }
  sha256sum -c 5_ua_detrac_raw.tar.gz.sha256
  [ ! -e "$RAW/ua_detrac" ] || { echo "$RAW/ua_detrac exists: refusing to unpack over it" >&2; exit 1; }
  tar -xzf 5_ua_detrac_raw.tar.gz -C "$RAW"                                # -> raw/ua_detrac/{DETRAC-Images,DETRAC-Annos}
  cd "$CONFDET_REPO"
  local n_tr n_te
  n_tr=$(find "$RAW/ua_detrac/DETRAC-Annos/DETRAC-Train-Annotations-XML" -name '*.xml' | wc -l)
  n_te=$(find "$RAW/ua_detrac/DETRAC-Annos/DETRAC-Test-Annotations-XML" -name '*.xml' | wc -l)
  echo "UA-DETRAC in $RAW/ua_detrac: $n_tr train XML, $n_te test XML (the Windows copy: 60 + 40)"
  [ "$n_tr" -eq 60 ] && [ "$n_te" -eq 40 ] || { echo "unexpected sequence counts: stop" >&2; exit 1; }
}

case "$WHAT" in
  bundle) bundle ;;
  bmd45) bmd45 ;;
  detrac) detrac ;;
  all) bundle; bmd45; detrac ;;
  *) echo "usage: $0 bundle|bmd45|detrac|all" >&2; exit 1 ;;
esac
