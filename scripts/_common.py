"""Shared CLI plumbing for scripts/*.py."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from confdet.utils.io_utils import load_config  # noqa: E402
from confdet.utils.logging_utils import get_logger  # noqa: E402
from confdet.utils.seed import set_seed  # noqa: E402


def setup(name: str, description: str, detector: bool = False, extra=None):
    """Parse --config (and --detector), load + validate the config, seed
    everything. Returns (args, cfg, logger).
    """
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/experiment.yaml")
    if detector:
        parser.add_argument("--detector", default=None,
                            help="entry of detectors.models (default: detectors.active)")
    if extra:
        extra(parser)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if detector and args.detector is None:
        args.detector = cfg["detectors"]["active"]
    set_seed(int(cfg["project"].get("seed", 42)))
    return args, cfg, get_logger(name)
