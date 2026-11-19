# Data

Nothing in `data/` is committed. Two datasets are expected.

## BMD-45 (development dataset in the recommended config)

Already downloaded in the first run to `data/raw/bmd45/` (`images/val`, `labels/val`, `data.yaml`; 10,194 images, 13 classes, plain boxes). If it is missing:

```bash
hf auth login
python scripts/download_bmd45.py --splits val
```

Optional, stronger: `--splits train` adds the 35,792 training images (most of the dataset's 153 GB). Then switch the `roles` rule for `bmd45` in `configs/experiment.yaml` to the commented alternative and re-run from `scripts/02`.

## AU-INV-PALS, non-augmented export (transfer dataset)

Export from Roboflow in YOLO format with **Auto-Orient only** and **no augmentation**, unzip so that `data.yaml` sits directly in the folder, and point `datasets.pals.root` at it (default `data/raw/AU-INV-PALS-clean`). Polygon and box lines may be mixed; `label_format: polygon` handles both.

Roboflow's own train / valid / test assignment is ignored: images are pooled and re-split so that frames of the same video stay together.

## What `scripts/02_prepare_data.py` produces

`data/processed/<experiment>/<dataset>/images/<role>/` (hard links, no extra disk space), matching `labels/<role>/` in plain box format, `data.yaml` for Ultralytics, and `manifest.json` recording each image's role, source split and leakage group.

If Windows reports a path that is too long, set `project.processed_dir` in the config to a short folder on the same drive.
