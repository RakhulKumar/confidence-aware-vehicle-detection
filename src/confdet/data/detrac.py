"""UA-DETRAC (v0.6 Phase 3): parse the per-sequence XML and compute per-vehicle annotated occlusion.

XML (one file per sequence): <ignored_region><box .../></ignored_region>, then per frame
<target id><box left top width height/><attribute vehicle_type truncation_ratio .../>
[<occlusion><region_overlap left top width height occlusion_id occlusion_status/></occlusion>].

occlusion_status, inferred from the data and confirmed 18 Nov 2026:
   0  the carrying vehicle is hidden by vehicle `occlusion_id` (which stands in front: 93%)
   1  the carrying vehicle HIDES vehicle `occlusion_id` (which stands behind: 97.5%); the region is
      the overlap seen on the vehicle behind
  -1  hidden by a non-vehicle (occlusion_id 0)
Each overlapping pair is recorded once.

Per vehicle v (all fractions of v's box area, regions clipped to the box, overlaps counted once):
  occ_vehicle     union of v's own code-0 regions and the code-1 regions on other vehicles naming v
                  (the headline annotated occlusion o_j)
  occ_naive       union of v's own code-0 AND code-1 regions (sensitivity check only)
  occ_nonvehicle  union of v's own code -1 regions (reported separately)
  truncation      the annotated truncation_ratio (reported separately)
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from confdet.data.occlusion import _union_area_clipped, box_areas

CLASSES = ("car", "van", "bus", "others", "ignore")


@dataclass
class Vehicle:
    track_id: str
    xyxy: tuple
    vehicle_type: str
    truncation: float
    entries: list = field(default_factory=list)       # (region xyxy, occlusion_id, status)


def _xyxy(el) -> tuple:
    l, t, w, h = (float(el.get(k)) for k in ("left", "top", "width", "height"))
    return (l, t, l + w, t + h)


def parse_sequence(path: str | Path) -> dict:
    """{"name", "ignored": [xyxy], "frames": {num: [Vehicle]}}"""
    root = ET.parse(path).getroot()
    ign = root.find("ignored_region")
    ignored = [_xyxy(b) for b in ign.findall("box")] if ign is not None else []
    frames = {}
    for fr in root.iter("frame"):
        tl = fr.find("target_list")
        vs = []
        for t in (list(tl) if tl is not None else []):
            a = t.find("attribute")
            v = Vehicle(t.get("id"), _xyxy(t.find("box")), a.get("vehicle_type"), float(a.get("truncation_ratio", 0)))
            oc = t.find("occlusion")
            for r in (list(oc) if oc is not None else []):
                v.entries.append((_xyxy(r), r.get("occlusion_id"), int(r.get("occlusion_status"))))
            vs.append(v)
        frames[int(fr.get("num"))] = vs
    return {"name": root.get("name"), "ignored": ignored, "frames": frames}


def frame_occlusion(vehicles: list[Vehicle]) -> list[dict]:
    """Annotated occlusion of every vehicle of one frame (see module docstring)."""
    def frac(box, regions):
        if not regions:
            return 0.0
        area = float(box_areas(np.array([box]))[0])
        return float(min(1.0, _union_area_clipped(np.array(box, float), np.array(regions, float)) / area)) if area > 0 else 0.0

    by_id_from_others: dict[str, list] = {}
    for v in vehicles:
        for reg, oid, st in v.entries:
            if st == 1:
                by_id_from_others.setdefault(oid, []).append(reg)
    out = []
    for v in vehicles:
        own0 = [r for r, _, st in v.entries if st == 0]
        own1 = [r for r, _, st in v.entries if st == 1]
        ownm1 = [r for r, _, st in v.entries if st == -1]
        out.append({"track_id": v.track_id, "vehicle_type": v.vehicle_type, "truncation": v.truncation,
                    "occ_vehicle": frac(v.xyxy, own0 + by_id_from_others.get(v.track_id, [])),
                    "occ_naive": frac(v.xyxy, own0 + own1),
                    "occ_nonvehicle": frac(v.xyxy, ownm1)})
    return out


def sampled_frames(frames: dict, every: int = 10) -> list[int]:
    """Every `every`-th frame, starting with the first (frame numbers start at 1)."""
    nums = sorted(frames)
    return [n for n in nums if (n - nums[0]) % every == 0] if nums else []


def yolo_label_lines(vehicles: list[Vehicle], ignored: list[tuple], width: int, height: int
                     ) -> tuple[list[str], list[int]]:
    """One line per vehicle (class index in CLASSES), then one per ignored region (class 'ignore').
    Boxes with no area inside the image are dropped. Returns (lines, indices of the vehicles kept,
    in line order) so per-vehicle annotations stay aligned with the label lines."""
    lines, kept = [], []
    items = [(v.xyxy, CLASSES.index(v.vehicle_type if v.vehicle_type in CLASSES else "others"), i)
             for i, v in enumerate(vehicles)] + [(r, CLASSES.index("ignore"), None) for r in ignored]
    for xyxy, c, vi in items:
        x1, y1, x2, y2 = (max(0.0, min(float(xyxy[0]), width)), max(0.0, min(float(xyxy[1]), height)),
                          max(0.0, min(float(xyxy[2]), width)), max(0.0, min(float(xyxy[3]), height)))
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            continue
        lines.append(f"{c} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
                     f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")
        if vi is not None:
            kept.append(vi)
    return lines, kept


VEHICLE_CLASS_IDS = (0, 1, 2)          # car, van, bus: the classes that are kept as objects after the class map


def annotations_for_tables(ann: dict, records, tables, key: str) -> np.ndarray:
    """Per object row of `tables` (class-map filtered: car, van, bus), the annotated value `key`
    from annotations_<role>.json (one entry per vehicle label line, 'others' included)."""
    out = []
    for r in records:
        entries = ann[r.image_id]
        vehicle_cls = [c for c in np.asarray(r.gt_cls).tolist() if c != CLASSES.index("ignore")]
        if len(entries) != len(vehicle_cls):
            raise ValueError(f"{r.image_id}: {len(entries)} annotations for {len(vehicle_cls)} vehicle labels")
        out += [e[key] for e, c in zip(entries, vehicle_cls) if c in VEHICLE_CLASS_IDS]
    out = np.asarray(out, float)
    if len(out) != tables.n_gt:
        raise ValueError(f"{len(out)} annotations for {tables.n_gt} objects in the tables")
    return out


def group_bootstrap_image_weights(groups: list[str], n_boot: int, seed: int = 0) -> np.ndarray:
    """(n_boot, n_images) weights from resampling GROUPS (sequences) with replacement: an image's
    weight is how often its sequence was drawn."""
    uniq, inv = np.unique(np.asarray(groups), return_inverse=True)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(uniq), np.full(len(uniq), 1.0 / len(uniq)), size=n_boot).astype(float)
    return counts[:, inv]
