#!/usr/bin/env bash
# Create the virtual environment on Beocat, install the pinned packages, run the self-test and the tests.
#
#   bash scripts/beocat/setup.sh                 # torch build chosen from the GPU node's driver (run on a GPU node)
#   TORCH_CUDA=cu128 bash scripts/beocat/setup.sh   # or name the build yourself (cu130 | cu128 | cu126 ...)
#
# Run it inside an interactive GPU session so the driver can be read and CUDA tested, e.g.
#   srun --gres=gpu:l40s:1 --cpus-per-task=4 --mem=16G --time=01:00:00 --pty bash
# Stops at the first error. Re-running it is safe (pip skips what is installed).
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$CONFDET_REPO"

TORCH_VERSION="${TORCH_VERSION:-2.14.0}"           # as on the Windows PC (requirements-lock.txt)
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.29.0}"

PY="${PYTHON:-python3.10}"
command -v "$PY" >/dev/null 2>&1 || PY=python3
"$PY" - <<'EOF'
import sys
if sys.version_info[:2] != (3, 10):
    sys.exit(f"Python {sys.version.split()[0]} found; the Windows environment is 3.10.21. "
             "Load a Python 3.10 module (CONFDET_PYTHON_MODULE in scripts/beocat/env.sh; `module spider Python`).")
EOF

mkdir -p "$CONFDET_ROOT"
if [ ! -x "$CONFDET_VENV/bin/python" ]; then
  "$PY" -m venv "$CONFDET_VENV"
fi
source "$CONFDET_VENV/bin/activate"
python -m pip install --upgrade pip

# ---- torch / torchvision: the build must match the node's CUDA driver ---------------------------
if [ -z "${TORCH_CUDA:-}" ]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "No nvidia-smi here (login node?). Run this inside a GPU session, or set TORCH_CUDA (e.g. cu128)." >&2
    exit 1
  fi
  DRIVER_CUDA=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9]*\)\.\([0-9]*\).*/\1\2/p' | head -1)
  echo "driver supports CUDA ${DRIVER_CUDA:0:2}.${DRIVER_CUDA:2}"
  if   [ "$DRIVER_CUDA" -ge 130 ]; then TORCH_CUDA=cu130
  elif [ "$DRIVER_CUDA" -ge 128 ]; then TORCH_CUDA=cu128
  elif [ "$DRIVER_CUDA" -ge 126 ]; then TORCH_CUDA=cu126
  else echo "driver CUDA $DRIVER_CUDA is older than 12.6: pick TORCH_CUDA and a torch version by hand" >&2; exit 1
  fi
fi
echo "installing torch==$TORCH_VERSION torchvision==$TORCHVISION_VERSION from the $TORCH_CUDA index"
python -m pip install "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
  --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"

# ---- everything else, exact versions ------------------------------------------------------------
python -m pip install -r requirements-beocat.txt
python -m pip freeze > "$CONFDET_ROOT/requirements-installed-$(date +%Y%m%d).txt"

python - <<'EOF'
import torch
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU visible")
EOF

# ---- self-test and tests ------------------------------------------------------------------------
python scripts/00_selftest.py
python -m pytest tests -q
echo "SETUP DONE: venv $CONFDET_VENV"
