# Confidence-aware vehicle detection (v0.3)

Calibration and uncertainty quantification for occlusion-driven missed detections in dense traffic: a post-hoc layer around a frozen detector, and the evaluation that tests it.

- `docs/FIRST_RUN_REVIEW.md`: what the first real run showed, what it did not, and what changed in v0.3 because of it. Read this first.
- `docs/AUDIT.md`: what was wrong in version 0.1.
- `docs/theoretical_foundations.md`: every formula.

## Upgrading from v0.2 after the first real run

Replace the code (`src/`, `scripts/`, `tests/`, `configs/`, `docs/`, `tasks/`, `README.md`, `CLAUDE.md`). Keep `data/`, `outputs/` and `logs/`: the trained weights and cached predictions from the first run are reused as they are. Then, with no GPU work at all:

```bash
python scripts/00_selftest.py
python tests/run_tests.py
bash scripts/run_all.sh --analysis-only --all-detectors     # minutes; writes outputs/bmd_dev/digest.md
```

The old `outputs/<experiment>/<detector>/results/` folders are superseded by `<detector>/<mode>/results/`.

## Before anything else (fresh install)

```bash
pip install -r requirements.txt          # torch separately, see the file
python scripts/00_selftest.py            # ~1 minute, no GPU, no data
python tests/run_tests.py                # or: pytest tests/ -v
```

The self-test runs the real scripts on synthetic data with a simulated detector whose behaviour is known, and checks the pipeline's answers against it. If it does not print `SELF-TEST PASSED`, stop and fix that first.

## Data

| Dataset | Role (recommended config) | Where |
|---|---|---|
| BMD-45 val (10,194 images) | development: train, calibrate, test | `data/raw/bmd45` (already downloaded) |
| AU-INV-PALS, non-augmented export | transfer target for RQ4 | set `datasets.pals.root` in `configs/experiment.yaml` |

See `data/README.md`. Old checkpoints and old results must not be reused: they come from a different split.

## Run

```bash
python scripts/02_prepare_data.py        # validate, split by leakage group, materialise roles
python scripts/03_compute_occlusion.py   # occlusion histogram + power gate  <- read this output
bash scripts/run_all.sh --skip-data --all-detectors
```

or stage by stage:

| Script | Does | Needs GPU |
|---|---|---|
| `02_prepare_data.py` | validate raw data; fit / val / calib / test roles with no group in two roles | no |
| `03_compute_occlusion.py` | occlusion per role; can this data answer the questions? | no |
| `04_train_detector.py --detector X` | fine-tune on fit, select on val | yes |
| `05_predict.py --detector X` | cache raw candidates for calib and test | yes |
| `06_calibrate.py --detector X` | per matching mode: operating threshold, occlusion-aware score, conformal thresholds | no |
| `07_run_evaluation.py --detector X` | per matching mode: RQ1 to RQ5, figures, `summary.json` | no |
| `08_compare_detectors.py` | reliability ranking under each metric, at own thresholds and at matched cost | no |
| `12_summarize.py` | every headline number in one file, `digest.md` | no |
| `09_synthetic_occlusion_probe.py` | controlled within-vehicle occlusion probe | yes |
| `10_run_inference_images.py`, `11_run_inference_video.py` | annotated accepted / flagged output on your own media | yes |
| `check_ultralytics_val.py` | Ultralytics' own validator on the test role, as a cross-check | yes |

Everything lands in `outputs/<experiment>/<detector>/<mode>/`, where `<mode>` is the matching mode: `class_agnostic` (primary: was any box put on the vehicle), `native` (fine-grained classes) and optionally `coarse`. Start with `outputs/<experiment>/digest.md`; its gates say whether the detector works on each dataset and whether there is enough occlusion to interpret the stratified results.

`configs/experiment_pals_dev.yaml` runs the original arrangement (your dataset as development). Expect its power gate to fail; it is there so you can show that, get clean accuracy numbers on your own data, and run the controlled probe on it.

## Honest limits

- The v0.3 changes (matching modes, per-detector thresholds, matched-cost ranking) are verified on the simulated detector and on a stand-in for Ultralytics, not yet on the real cached predictions.
- One training seed per detector so far.
- The evidential-learning ablation is not implemented (`scripts/experimental/`). RQ5 is answered from the frozen detector's raw output instead.
- In the first real run the occlusion-aware score did not beat plain confidence at matched cost. Nothing in v0.3 is designed to change that answer; it is designed to ask the question about the right kind of miss.
