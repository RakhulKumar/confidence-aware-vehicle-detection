#!/usr/bin/env bash
# Shared environment for every Beocat script (source it: `source scripts/beocat/env.sh`).
# Override any variable by exporting it before sourcing.
#
#   CONFDET_SITE        beocat: configs/paths_local.yaml's beocat block supplies every path
#   CONFDET_ROOT        data, runs, raw downloads (~250 GB). Home on Beocat: 1 TB, not purged, free;
#                       /fastscratch purges files unused for 30 days, /bulk is paid ($45/TB/yr).
#   CONFDET_REPO        the git checkout of this repository
#   CONFDET_TRANSFER    the copied transfer bundle (*.tar.gz + *.sha256)
#   CONFDET_VENV        the Python virtual environment
#   CONFDET_GPU_SHARE   off: the GPU is ours alone (no VRAM cap, no duty cycle)
#   CONFDET_PYTHON_MODULE  environment module that provides Python 3.10 (check: `module spider Python`)

export CONFDET_SITE=beocat
export CONFDET_ROOT="${CONFDET_ROOT:-/homes/$USER/confdet}"
export CONFDET_REPO="${CONFDET_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export CONFDET_TRANSFER="${CONFDET_TRANSFER:-$HOME/confdet_transfer}"   # where the bundle archives were copied
export CONFDET_VENV="${CONFDET_VENV:-$CONFDET_ROOT/venv}"
export CONFDET_GPU_SHARE=off
export CONFDET_PYTHON_MODULE="${CONFDET_PYTHON_MODULE:-Python/3.10.4-GCCcore-11.3.0}"   # PC: 3.10.21
export HF_HOME="${HF_HOME:-$CONFDET_ROOT/hf_home}"          # Hugging Face cache and login token, next to the data
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONUNBUFFERED=1

if [ -n "$CONFDET_PYTHON_MODULE" ] && command -v module >/dev/null 2>&1; then
  module load "$CONFDET_PYTHON_MODULE"
fi
