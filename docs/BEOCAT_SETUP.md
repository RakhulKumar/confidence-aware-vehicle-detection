# Running the project on Beocat

Beocat is K-State's HPC cluster (Slurm; L40S 48 GB GPUs). This page covers moving the v0.6 work
from the Windows PC to Beocat: code from GitHub, data rebuilt from the transferred manifests, and
Phase 4 training as Slurm jobs. Nothing is re-split, and every rebuilt file is checked against the
Windows copy.

## 0. What moves, and how

| What | From | To on Beocat |
|---|---|---|
| Code | GitHub (private repo) | `git clone` → `$CONFDET_REPO` |
| Split manifests and per-file SHA256 lists | bundle `1_manifests.tar.gz` | `$CONFDET_ROOT/transfer/manifests/` |
| AU-INV-PALS raw export | bundle `2_pals_raw.tar.gz` | `$CONFDET_ROOT/raw/AU-INV-PALS/` |
| Weights of the trained models and the pilots | bundle `3_weights.tar.gz` | `outputs/…` and `$CONFDET_ROOT/runs/…` |
| `outputs/` (cached predictions, results, digest) | bundle `4_outputs.tar.gz` | `$CONFDET_REPO/outputs/` |
| BMD-45 images | Hugging Face `iisc-aim/BMD-45` | `$CONFDET_ROOT/raw/bmd45/` |
| UA-DETRAC images and XML | official Google Drive links | `$CONFDET_ROOT/raw/ua_detrac/` |

Every machine-specific path comes from `configs/paths_local.yaml`. The block is chosen by
`CONFDET_SITE`: `windows` when unset, and `beocat` as set by `scripts/beocat/env.sh`.
`CONFDET_ROOT` defaults to `/bulk/$USER/confdet_project`. Check Beocat's current storage guide and
quota for the right large, non-home filesystem; the data needs about 250 GB.

## 1. Copy the bundle from the Windows PC

The bundle is the four archives in `D:\confdet_project\transfer\`, plus their `.sha256` files.
Copy them to `$CONFDET_ROOT/transfer/` with Globus, or with `scp`/`rsync` through the Beocat
login node, for example:

```bash
# on Beocat
mkdir -p /bulk/$USER/confdet_project/transfer
# on the PC (Git Bash):
scp /d/confdet_project/transfer/*.tar.gz* <eid>@headnode.beocat.ksu.edu:/bulk/<eid>/confdet_project/transfer/
```

## 2. Code

```bash
cd ~
git clone https://github.com/RakhulKumar/confidence-aware-vehicle-detection.git
cd confidence-aware-vehicle-detection
```

Authenticate with `gh auth login`, or with an SSH key added to GitHub. Never write a token into a
file inside the repository.

## 3. Python environment (venv)

The Windows environment is Python 3.10.21, recorded in `requirements-lock.txt`. Beocat installs the
same versions from `requirements-beocat.txt`. The exception is torch/torchvision, whose CUDA build
must match the GPU nodes' driver.

```bash
module spider Python                       # find a Python 3.10 module
export CONFDET_PYTHON_MODULE=Python/3.10.<x>   # the exact name it lists (or put it in scripts/beocat/env.sh)

# on a GPU node, so the driver's CUDA version can be read and CUDA tested:
srun --gres=gpu:l40s:1 --cpus-per-task=4 --mem=16G --time=01:00:00 --pty bash
bash scripts/beocat/setup.sh
```

What `setup.sh` does:

1. Creates `$CONFDET_VENV` (default `$CONFDET_ROOT/venv`) with `python3.10 -m venv`.
2. Reads the CUDA version from `nvidia-smi` and installs `torch==2.14.0` and
   `torchvision==0.29.0` from the newest matching PyTorch index (cu130 for driver CUDA ≥ 13.0,
   cu128 for ≥ 12.8, cu126 for ≥ 12.6). You can force one with `TORCH_CUDA=cu128`. If that
   torch version has no build for the driver, `pip` stops. Pick the nearest version and record it.
3. Installs `requirements-beocat.txt` and writes what was installed to
   `$CONFDET_ROOT/requirements-installed-<date>.txt`.
4. Runs `scripts/00_selftest.py`, which must print SELF-TEST PASSED, then `pytest tests`.

To use the environment later:

```bash
source scripts/beocat/env.sh && source "$CONFDET_VENV/bin/activate"
```

## 4. Data

```bash
hf auth login                              # once, interactive; the token stays in $HF_HOME
bash scripts/beocat/get_data.sh bundle     # SHA256 of the 4 archives, then unpack
bash scripts/beocat/get_data.sh bmd45      # BMD-45 train + val from Hugging Face
# edit DETRAC_IDS in scripts/beocat/get_data.sh (official Google Drive file ids), then:
bash scripts/beocat/get_data.sh detrac
```

## 5. Rebuild the processed datasets and check identity

```bash
bash scripts/beocat/rebuild.sh
```

For every dataset of every experiment (`bmd_dev`, `bmd_v06`, `pals_rev`, `pals_dev`) the script:

1. Puts each image of the transferred `manifest.json` into its recorded role.
2. Regenerates the image's label from the raw export.
3. Compares every file with the Windows hash list `files.sha256`:
   - images are compared byte for byte;
   - text files are compared with line endings normalised, because Windows wrote CRLF.

UA-DETRAC is converted again with `02c` and compared the same way. Then `02b --verify` re-derives
every role assignment from the raw data with the original code and seed. **The script stops at the
first difference.** A changed Hugging Face release or a different UA-DETRAC download shows up here.
Do not train until it prints `REBUILD IDENTICAL`.

This check was dry-run on the Windows PC on 19 Nov. All checked datasets had zero problems:
labels regenerated from the raw exports were identical to the processed ones. `pals_dev/bmd45` was
not checked, because its raw folder no longer exists on the PC; it is checked on Beocat.

## 6. Batch size on the L40S (once per architecture)

`configs/beocat_batch.yaml` has a `null` placeholder per architecture: yolo26, rtdetr, yolov8 and
yolo26_evid. Choose each with a short measured trial: a few hundred iterations at increasing
batch, keeping the largest that trains without `VRAMOverflowError` and leaves headroom. Then write
the value in, with the trial's date, node and reserved memory. The value is fixed for every seed of
that architecture. `submit_all.sh` refuses to run while a needed value is `null`.

## 7. Submit Phase 4

```bash
bash scripts/beocat/submit_all.sh --dry-run    # print the sbatch commands
bash scripts/beocat/submit_all.sh              # same order as the Windows queue, chained afterok
bash scripts/beocat/submit_all.sh --parallel   # or: only the queue's own dependencies
```

- **Template:** every job uses `scripts/beocat/train_job.sbatch`.
- **Resources:** one L40S (`--gres=gpu:l40s:1`) and `--requeue`.
- **Time limits:** set with `TIME_TRAIN` (default `24:00:00`) and `TIME_PREDICT` (`04:00:00`).
- **GPU share:** off (`CONFDET_GPU_SHARE=off`).
- **Resuming:** training runs `04 --resume`, as on the Windows PC. Fifteen minutes before the time
  limit, the job stops and requeues itself under the same job id, so the chain keeps waiting. The
  requeued job continues from `last.pt`.
- **Logs:** go to `<run folder>/logs/`, for example
  `$CONFDET_ROOT/runs/bmd_v06/yolo26/logs/train_<jobid>.log`.
- **Analysis:** CPU jobs run `run_all.sh --analysis-only` for the v0.6 core models, for PALS-reverse,
  and for all ten v0.6 models. Their logs go to `$CONFDET_ROOT/logs/`.
- **Not submitted:** the evidential RT-DETR models stay held.

The YOLO26 run that was in progress on the PC, stopped at epoch 5, is **not** resumed on Beocat.
Its checkpoint has Windows paths and batch 4, and all seeds of an architecture must share one
batch. It starts again with the others. A copy of that checkpoint is kept for the record in the
weights archive (`root/windows_partial/`).

Watch the jobs with `squeue -u $USER` and `sacct -j <id> --format=JobID,State,Elapsed,ExitCode`.
If a job fails, its dependents stay pending with reason `DependencyNeverSatisfied`. Fix the
problem, then resubmit. Finished models are skipped by `04 --resume` (nothing to do) and by `05`
(files that are already complete).
