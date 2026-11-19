"""Machine-specific paths come only from configs/paths_local.yaml, selected by CONFDET_SITE."""
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from confdet.utils import io_utils  # noqa: E402

@contextmanager
def env(**values):
    """Set (str) or unset (None) environment variables, restoring them afterwards."""
    old = {k: os.environ.get(k) for k in values}
    try:
        for k, v in values.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


WINDOWS = {
    "processed_dir": "D:/confdet_project/confdet_data",
    "runs_dir": "D:/confdet_project/runs",
    "queue": "D:/confdet_project/queue/v06_jobs.yaml",
    "raw_bmd45_train": "D:/confdet_project/raw/bmd45",
    "raw_pals": "data/raw/AU-INV-PALS",
    "raw_ua_detrac": "D:/confdet_project/raw/ua_detrac",
}


def test_default_site_is_windows_and_reproduces_the_windows_paths():
    with env(CONFDET_SITE=None):
        assert io_utils.site_name() == "windows"
        p = io_utils.site_paths()
        cfg = io_utils.load_config(REPO / "configs" / "experiment_v06.yaml")
    for k, v in WINDOWS.items():
        assert p[k] == v
    assert cfg["project"]["processed_dir"] == WINDOWS["processed_dir"]
    assert cfg["derive"]["bmd45"]["source_processed"] == WINDOWS["processed_dir"] + "/bmd_dev/bmd45"
    assert cfg["derive"]["bmd45"]["append"]["root"] == WINDOWS["raw_bmd45_train"]


def test_beocat_site_expands_its_root_and_refuses_an_unset_variable():
    with env(CONFDET_SITE="beocat", CONFDET_ROOT="/bulk/someone/confdet_project"):
        cfg = io_utils.load_config(REPO / "configs" / "experiment_v06.yaml")
    assert cfg["project"]["processed_dir"] == "/bulk/someone/confdet_project/confdet_data"
    assert cfg["datasets"]["pals"]["root"] == "/bulk/someone/confdet_project/raw/AU-INV-PALS"
    with env(CONFDET_SITE="beocat", CONFDET_ROOT=None):
        with pytest.raises(KeyError, match="not set"):
            io_utils.load_config(REPO / "configs" / "experiment_v06.yaml")
    with env(CONFDET_SITE="nowhere"):
        with pytest.raises(KeyError, match="not a site"):
            io_utils.site_paths()


def test_no_config_carries_a_machine_path_and_every_placeholder_exists_for_every_site():
    sites = yaml.safe_load((REPO / "configs" / "paths_local.yaml").read_text(encoding="utf-8"))
    assert set(sites["windows"]) == set(sites["beocat"])
    for p in (REPO / "configs").glob("experiment*.yaml"):
        text = "\n".join(line.split("#")[0] for line in p.read_text(encoding="utf-8").splitlines())
        assert "D:/" not in text and "C:/" not in text and "OneDrive" not in text, p.name
        for key in io_utils._PLACEHOLDER.findall(text):
            assert key in sites["windows"], (p.name, key)


def test_unknown_placeholder_is_an_error():
    with pytest.raises(KeyError, match="no such key"):
        io_utils.resolve_site_paths({"a": "${paths.nope}/x"}, {"processed_dir": "/p"})
    assert io_utils.resolve_site_paths({"a": ["${paths.k}/x", 3]}, {"k": "/p"}) == {"a": ["/p/x", 3]}


def test_gpu_share_switch_turns_the_share_off_and_leaves_it_alone_when_unset():
    cfg_path = REPO / "configs" / "experiment_v06.yaml"
    with env(CONFDET_GPU_SHARE=None, CONFDET_SITE=None):
        assert io_utils.load_config(cfg_path)["detectors"]["gpu_share"] == {"memory_fraction": 0.45, "duty_cycle": 0.8}
    with env(CONFDET_GPU_SHARE="off", CONFDET_SITE=None):
        assert io_utils.load_config(cfg_path)["detectors"]["gpu_share"] is None
