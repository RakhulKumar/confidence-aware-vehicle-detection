"""Where everything lives. One experiment (configs/*.yaml -> experiment.name),
one detector and one matching mode never overwrite another's files:

  outputs/<experiment>/<detector>/weights/best.pt
  outputs/<experiment>/<detector>/predictions/<dataset>__<role>.npz      (shared by all modes)
  outputs/<experiment>/<detector>/<mode>/calibration.json
  outputs/<experiment>/<detector>/<mode>/results/*.json
  outputs/<experiment>/<detector>/<mode>/figures/*.png
  outputs/<experiment>/comparison/<mode>/...                              (cross-detector)
  <processed_dir>/<experiment>/<dataset>/                                 materialised roles

<mode> is the matching mode: class_agnostic | coarse | native
(docs/theoretical_foundations.md section 3).
"""
from __future__ import annotations

import os
from pathlib import Path

MATCHING_MODES = ("class_agnostic", "coarse", "native")

_WIN_LONG_PATH_FROM = 240          # MAX_PATH is 260 including the terminator; leave headroom


def extended_path_str(abs_path: str) -> str:
    """The Win32 extended-length form of an absolute Windows path (`\\\\?\\C:\\...`),
    which lifts the 260-character limit. Pure string function, so it is testable anywhere.
    """
    if abs_path.startswith("\\\\?\\"):
        return abs_path
    if abs_path.startswith("\\\\"):                       # UNC share
        return "\\\\?\\UNC\\" + abs_path[2:]
    return "\\\\?\\" + abs_path


def long_path(p) -> Path:
    """`p` in a form that open / exists / os.link accept even when it is 260+ characters
    on Windows (Roboflow file names are ~85 characters and the repository sits deep in a
    OneDrive folder). Unchanged elsewhere and for short paths.

    Found in the first real run: label files at 260+ characters were read as "no label
    file", so their vehicles silently vanished while validation still printed [OK].
    Use this only for raw-file access done by this package; external tools (Ultralytics,
    OpenCV) still need short paths, which is what project.processed_dir is for.
    """
    p = Path(p)
    if os.name != "nt":
        return p
    s = os.path.abspath(str(p))
    return Path(extended_path_str(s)) if len(s) >= _WIN_LONG_PATH_FROM else p


def processed_root(cfg: dict) -> Path:
    return Path(cfg["project"].get("processed_dir", "data/processed")) / cfg["experiment"]["name"]


def experiment_dir(cfg: dict) -> Path:
    return Path(cfg["project"].get("output_dir", "outputs")) / cfg["experiment"]["name"]


def detector_dir(cfg: dict, detector: str) -> Path:
    return experiment_dir(cfg) / detector


def train_dir(cfg: dict, detector: str) -> Path:
    """Training run folder (Ultralytics output, last.pt, best.pt). Under
    project.runs_dir/<experiment>/<detector> when the config sets runs_dir
    (v0.6: on D:, outside OneDrive), else next to the other outputs."""
    runs = cfg["project"].get("runs_dir")
    if not runs:
        return detector_dir(cfg, detector)
    new = Path(runs) / cfg["experiment"]["name"] / detector
    legacy = detector_dir(cfg, detector)
    # detectors trained before runs_dir existed keep their weights where they are
    if not new.exists() and (legacy / "weights" / "best.pt").exists():
        return legacy
    return new


def weights_path(cfg: dict, detector: str) -> Path:
    return train_dir(cfg, detector) / "weights" / "best.pt"


def predictions_path(cfg: dict, detector: str, dataset: str, role: str) -> Path:
    return detector_dir(cfg, detector) / "predictions" / f"{dataset}__{role}.npz"


def _check_mode(mode: str) -> str:
    if mode not in MATCHING_MODES:
        raise ValueError(f"matching mode must be one of {MATCHING_MODES}, got {mode!r}")
    return mode


def calibration_path(cfg: dict, detector: str, mode: str) -> Path:
    return detector_dir(cfg, detector) / _check_mode(mode) / "calibration.json"


def results_dir(cfg: dict, detector: str, mode: str) -> Path:
    return detector_dir(cfg, detector) / _check_mode(mode) / "results"


def figures_dir(cfg: dict, detector: str, mode: str) -> Path:
    return detector_dir(cfg, detector) / _check_mode(mode) / "figures"


def comparison_dir(cfg: dict, mode: str) -> Path:
    return experiment_dir(cfg) / "comparison" / _check_mode(mode)
