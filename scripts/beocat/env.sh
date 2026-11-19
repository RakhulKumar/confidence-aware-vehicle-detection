#!/usr/bin/env bash
# Shared environment for every Beocat script (source it: `source scripts/beocat/env.sh`).
# Override any variable by exporting it before sourcing.
#
#   CONFDET_SITE        beocat: configs/paths_local.yaml's beocat block supplies every path
#   CONFDET_ROOT        data, runs, raw downloads, the transfer bundle (large: NOT your home folder)
#   CONFDET_REPO        the git checkout of this repository
#   CONFDET_VENV        the Python virtual environment
#   CONFDET_GPU_SHARE   off: the GPU is ours alone (no VRAM cap, no duty cycle)
#   CONFDET_PYTHON_MODULE  environment module that provides Python 3.10 (check: `module spider Python`)

export CONFDET_SITE=beocat
export CONFDET_ROOT="${CONFDET_ROOT:-/bulk/$USER/confdet_project}"
export CONFDET_REPO="${CONFDET_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export CONFDET_VENV="${CONFDET_VENV:-$CONFDET_ROOT/venv}"
export CONFDET_GPU_SHARE=off
export CONFDET_PYTHON_MODULE="${CONFDET_PYTHON_MODULE:-}"
export HF_HOME="${HF_HOME:-$CONFDET_ROOT/hf_home}"          # Hugging Face cache off the home quota
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONUNBUFFERED=1

if [ -n "$CONFDET_PYTHON_MODULE" ] && command -v module >/dev/null 2>&1; then
  module load "$CONFDET_PYTHON_MODULE"
fi
