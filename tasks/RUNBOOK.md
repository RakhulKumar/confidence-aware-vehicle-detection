# Runbook: after the first real run (v0.3)

Nothing here retrains or re-predicts. Steps 1-4 take minutes and need no GPU.

1. **Install v0.3 over v0.2.** Replace `src/`, `scripts/`, `tests/`, `configs/`, `docs/`, `tasks/`, `README.md`, `CLAUDE.md`. Keep `data/`, `outputs/`, `logs/`. Use the same Python environment as the first run (the one with CUDA torch).
2. **Check.** `python scripts/00_selftest.py` → `SELF-TEST PASSED`; `python tests/run_tests.py` → 76 passed.
3. **Config.** Both configs already carry the first run's settings (`processed_dir`, now `D:/confdet_project/confdet_data`, PALS root `data/raw/AU-INV-PALS`). Review the two `class_map` blocks; they matter for `coarse` mode and for RQ4.
4. **Re-analyse from the cached predictions.**
   `bash scripts/run_all.sh --analysis-only --all-detectors`
   This re-fits the post-hoc layer and re-runs RQ1-RQ5 and the detector comparison in two matching modes (`class_agnostic` primary, `native`), each detector at its own F1-optimal threshold.
5. **Read `outputs/bmd_dev/digest.md`**, class-agnostic section first:
   - gates;
   - each detector's operating threshold and false alarms per image (they should now be of the same order);
   - RQ1: is the occlusion gradient steeper than under native matching?
   - conformal: is the 0.95 target certifiable, and at how many flags per image? Is the heavily occluded context group certifiable now?
   - RQ2: does `occlusion_aware` differ from `confidence`? Report the interval whatever it says;
   - ranking: does any flip relative to D-ECE survive matched cost?
6. **Cross-check the detector's accuracy** (GPU, about a minute per detector):
   `python scripts/check_ultralytics_val.py --detector yolo26` (and `yolov8`, `rtdetr`).
   Compare its mAP50 with `class_aware.ap50` in `outputs/bmd_dev/<detector>/native/results/detection.json`.
7. **Re-run the controlled probe** under the primary mode: `python scripts/09_synthetic_occlusion_probe.py --detector yolo26`.
8. **Optional:** `bash scripts/run_all.sh --analysis-only --config configs/experiment_pals_dev.yaml --detectors yolo26`.

Bring back: `outputs/bmd_dev/digest.md`, the `<detector>/<mode>/results/` folders, `comparison/`, and the three `ultralytics_val_test.json` files.

## After that (needs GPU time; decide with your professor first)

- **Seeds.** Two more training seeds per detector (change `project.seed`, use a new `experiment.name`), so that differences between detectors can be told apart from run-to-run noise.
- **More training data.** Download the BMD-45 train split and switch the `roles` rule (see the comment in `configs/experiment.yaml`): about 6x more training images, and the whole val split left for calib and test.
