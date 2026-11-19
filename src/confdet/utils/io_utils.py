"""Config loading + results persistence, shared across scripts/*.py so every
script reads configs/experiment.yaml the same way and writes results to
outputs/results/ in the same JSON shape (making it trivial to later load
every stage's results back in for the paper's tables/figures without
re-running anything).
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml


SITES_FILE = Path(__file__).resolve().parents[3] / "configs" / "paths_local.yaml"
DEFAULT_SITE = "windows"
_PLACEHOLDER = re.compile(r"\$\{paths\.([A-Za-z0-9_]+)\}")


def site_name() -> str:
    """The machine whose paths are used: environment variable CONFDET_SITE (unset = windows)."""
    return os.environ.get("CONFDET_SITE", "").strip() or DEFAULT_SITE


def site_paths(site: str | None = None, path: str | Path = SITES_FILE) -> dict[str, str]:
    """Machine-specific paths of one site from configs/paths_local.yaml, environment variables
    expanded. An unset variable is an error (never silently an empty path)."""
    site = site or site_name()
    sites = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if site not in sites:
        raise KeyError(f"CONFDET_SITE={site!r} is not a site in {path} (have: {sorted(sites)})")
    out = {}
    for k, v in (sites[site] or {}).items():
        s = os.path.expandvars(str(v))
        if "$" in s:
            raise KeyError(f"{path}: {site}.{k} = {v!r} uses an environment variable that is not set")
        out[k] = s
    return out


def resolve_site_paths(obj: Any, paths: dict[str, str] | None = None) -> Any:
    """Replace every "${paths.<key>}" in the strings of a loaded config (recursively). The site's
    paths are read only if a placeholder is present."""
    cache = {"paths": paths}

    def one(m):
        if cache["paths"] is None:
            cache["paths"] = site_paths()
        key = m.group(1)
        if key not in cache["paths"]:
            raise KeyError(f"${{paths.{key}}}: no such key for site {site_name()!r} in {SITES_FILE}")
        return cache["paths"][key]

    def walk(o):
        if isinstance(o, dict):
            return {k: walk(v) for k, v in o.items()}
        if isinstance(o, list):
            return [walk(v) for v in o]
        if isinstance(o, str) and "${paths." in o:
            return _PLACEHOLDER.sub(one, o)
        return o
    return walk(obj)


def load_config(path: str | Path = "configs/experiment.yaml") -> dict:
    """Read and sanity-check an experiment config. Fails early, with the
    offending key named, instead of deep inside a later stage.
    Machine-specific paths ("${paths.<key>}") come from configs/paths_local.yaml (site CONFDET_SITE).
    """
    with open(path, encoding="utf-8") as f:
        cfg = resolve_site_paths(yaml.safe_load(f))
    # A GPU of one's own (Beocat): CONFDET_GPU_SHARE=off drops the VRAM cap and duty cycle that
    # shared the Windows PC's GPU with another project. Unset = the config's gpu_share, unchanged.
    if os.environ.get("CONFDET_GPU_SHARE", "").strip().lower() in ("off", "0", "false", "none")             and isinstance(cfg.get("detectors"), dict):
        cfg["detectors"]["gpu_share"] = None
    for section in ("project", "experiment", "datasets", "detectors", "occlusion", "posthoc", "evaluation"):
        if section not in cfg:
            raise KeyError(f"{path}: missing top-level section {section!r}")
    exp = cfg["experiment"]
    for key in ("name", "development"):
        if key not in exp:
            raise KeyError(f"{path}: experiment.{key} is required")
    names = [exp["development"]] + list(exp.get("transfer") or [])
    for n in names:
        if n not in cfg["datasets"]:
            raise KeyError(f"{path}: experiment refers to dataset {n!r}, which is not defined under datasets "
                           f"(have: {sorted(cfg['datasets'])})")
    if cfg["detectors"]["active"] not in cfg["detectors"]["models"]:
        raise KeyError(f"{path}: detectors.active={cfg['detectors']['active']!r} is not in detectors.models")
    spec = cfg["evaluation"].get("standard_operating_threshold")
    if not (spec == "max_f1" or (isinstance(spec, (int, float)) and 0 < spec < 1)):
        raise ValueError(f"{path}: evaluation.standard_operating_threshold must be 'max_f1' or a number in "
                         f"(0, 1), got {spec!r}")
    matching_modes(cfg)
    edges = cfg["occlusion"]["bucket_edges"]
    if edges[0] != 0.0 or edges[-1] != 1.0 or any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError(f"{path}: occlusion.bucket_edges must increase from 0.0 to 1.0, got {edges}")
    return cfg


def matching_modes(cfg: dict) -> list[str]:
    """Matching modes to analyse; the FIRST is the primary one (used for the
    deployed rule and the headline numbers).
    """
    from confdet.utils.paths import MATCHING_MODES

    modes = list(cfg["evaluation"].get("matching_modes") or ["class_agnostic", "native"])
    bad = [m for m in modes if m not in MATCHING_MODES]
    if bad or not modes:
        raise ValueError(f"evaluation.matching_modes must be a non-empty subset of {MATCHING_MODES}, got {modes}")
    return modes


def experiment_datasets(cfg: dict) -> list[str]:
    """Development dataset first, then transfer datasets."""
    return [cfg["experiment"]["development"]] + list(cfg["experiment"].get("transfer") or [])


def _json_default(obj: Any):
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if hasattr(obj, "tolist"):  # numpy arrays/scalars
        return obj.tolist()
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)  # last resort: never let a results-write crash a run


def _clean(obj: Any) -> Any:
    """NaN/inf are not valid JSON; write them as null. Keys become strings."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        obj = dataclasses.asdict(obj)
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if hasattr(obj, "tolist") and not isinstance(obj, (str, bytes)):
        return _clean(obj.tolist())
    if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))):
        return None
    return obj


def save_results(results: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # atomic: write a temporary file, then rename over the target (a crash or restart
    # mid-write never leaves a truncated results file behind)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_clean(results), f, indent=2, default=_json_default)
    os.replace(tmp, path)


def load_results(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
