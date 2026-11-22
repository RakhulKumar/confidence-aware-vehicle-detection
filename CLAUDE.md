# CLAUDE.md — working in this repository

## What this is

A post-hoc reliability layer around a frozen vehicle detector, and the evaluation of five research questions about occlusion-driven missed detections. Read `docs/AUDIT.md` (what went wrong in v0.1 and why the code is shaped this way) and `docs/theoretical_foundations.md` (every formula) before changing anything under `src/confdet/uq`, `calibration` or `evaluation`.

## This machine (the Windows PC)

- **Training, prediction and dataset building run on Beocat** (`docs/BEOCAT_SETUP.md`, `CONFDET_SITE=beocat`). Since 21 Nov 2026 the PC has no BMD-45 or UA-DETRAC data, no processed roles, no GPU environment and no job queue or scheduled tasks; do not run `02`/`02b`/`02c`/`04`/`05` here.
- Use the Anaconda environment `C:\Users\rakhul007\Anaconda3\envs\its\python.exe` (Python 3.10, torch 2.14.0+cpu, ultralytics 8.4.154): CPU only. The test suite and the self-test pass on it; analysis that reads `outputs/` (`06`, `07`, `08`, `12`, `run_all.sh --analysis-only`) runs on it. The v1 folder's `.venv` (CUDA torch) was deleted.
- `python3` can be the Microsoft Store stub; `scripts/run_all.sh` falls back to `python`.
- Kept on the PC: `outputs/` (cached predictions, results, best.pt of the bmd_dev models), `data/raw/AU-INV-PALS`, and `D:\confdet_project\transfer\` archives 1–4 (manifests, PALS raw, weights, outputs; SHA256 files beside them). Cleanup log: `D:\confdet_project\logs\cleanup_2026-11-21.log`.
- Paths of this machine are the `windows` block of `configs/paths_local.yaml` (the default site). Raw-file access inside this package goes through `utils.paths.long_path` (OneDrive's 260-character path limit).

## First commands of every session

```bash
python scripts/00_selftest.py        # must print SELF-TEST PASSED
python tests/run_tests.py            # or pytest tests/ -v
```

## Pipeline

`02` data roles → `03` occlusion report and power gate → `04` train → `05` cache predictions → `06` fit post-hoc layer → `07` RQ1–RQ5 → `08` compare detectors → `12` digest. Stages communicate through files only. `06`, `07`, `08` and `12` need no GPU: `bash scripts/run_all.sh --analysis-only --all-detectors` re-runs all of them from the cached predictions in minutes.

`06`–`08` run once per matching mode in `evaluation.matching_modes` (first = primary). Outputs: `outputs/<experiment>/<detector>/<mode>/…`; cached predictions and weights are shared by all modes.

## Rules

1. **Never report a number that fails a gate as a result.** `summary.json → gates` comes first. An underpowered occlusion bucket or a dataset where the detector proposes nothing is a finding about the data, not a measurement.
2. **Roles are sacred.** The detector sees fit and val. The post-hoc layer sees calib. Reported numbers come from test. Re-running `02` re-splits; every checkpoint trained before that must be retrained.
3. **Decision rules use only detector outputs.** Ground-truth occlusion (`g_occ`) is for analysis. Anything that conditions a decision on it is a bug. Use occlusion context (`c_ctx`).
4. **Class indices of two datasets are never compared.** Go through `class_map` and `evaluation.common_classes`.
5. **Do not re-match at thresholds.** Labels are assigned once on raw candidates (`labeling.build_tables`); every operating point is a subset.
6. **Do not reintroduce "outcome = survived the threshold"** in any calibration metric (`docs/AUDIT.md` item 4). `tests/test_evaluation.py::test_silently_dropping_hard_objects…` guards this.
7. **Label format is declared per dataset** (`label_format`), never inferred for a whole dataset from line length.
8. **Do not invent data or results.** Simulated data lives in tests and the self-test only.
9. New analysis → new pure function with a test on the simulator (`evaluation/simulate.py`) where the right answer is known, then wire it into a script.
10. New dependency → `requirements.txt` with a reason.
11. **The primary matching mode is class-agnostic** ("was any box put on this vehicle"). Native mode counts a misnamed vehicle as a miss; keep it for RQ5 and for comparison, never as the headline for silent misses.
12. **Compare detectors at matched cost** before calling one more reliable; a difference that exists only at each detector's own threshold is an operating-point difference.

## Long runs

Training takes hours. Run it in the background with output redirected to a log. If it stops with `VRAMOverflowError`, lower that detector's `batch` in the config; do not disable the guard.

## State after the first real run (read `docs/FIRST_RUN_REVIEW.md`)

- Weights and cached predictions for yolo26, yolov8 and rtdetr exist under `outputs/bmd_dev/`. Do not retrain or re-predict unless `02` is re-run.
- Next action: `bash scripts/run_all.sh --analysis-only --all-detectors`, then read `outputs/bmd_dev/digest.md`.
- The occlusion-aware score showed no gain over confidence under native matching. Report whatever the class-agnostic re-analysis shows; do not tune features until it wins.

## Known open items

- Evidential ablation: `scripts/experimental/train_evidential_TEMPLATE.py`, never run.
- BMD-45 `group_regex`: unset. If file names expose a camera id, set it and re-run from `02`.
- Class maps between AU-INV-PALS and BMD-45: reviewed by nobody yet.
- v0.3's analysis changes have not yet been run on the real cached predictions.
- One seed per detector.
- The val-vs-test AP gap for the YOLO models (0.75 vs 0.67) is unexplained until `scripts/check_ultralytics_val.py` is run.
