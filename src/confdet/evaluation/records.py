"""Cached detector outputs.

scripts/05_predict.py runs the frozen detector ONCE per image at a near-zero
confidence floor and stores, for every image of a role, the ground truth
(with occlusion severity) and every raw candidate. Everything downstream --
calibration, RQ1-RQ5, detector comparison -- is a pure function of these
files, so analyses can be re-run without a GPU and always see the same
predictions.

On disk: one compressed .npz per (detector, dataset, role), flat arrays plus
per-image offsets.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ImageRecord:
    image_id: str
    width: int
    height: int
    gt_xyxy: np.ndarray            # (n, 4) pixels
    gt_cls: np.ndarray             # (n,) int
    gt_occ: np.ndarray             # (n,) occlusion severity in [0, 1]
    cand_xyxy: np.ndarray          # (m, 4) pixels
    cand_conf: np.ndarray          # (m,)
    cand_cls: np.ndarray           # (m,) int

    def __post_init__(self):
        self.gt_xyxy = np.asarray(self.gt_xyxy, dtype=np.float32).reshape(-1, 4)
        self.gt_cls = np.asarray(self.gt_cls, dtype=np.int32).reshape(-1)
        self.gt_occ = np.asarray(self.gt_occ, dtype=np.float32).reshape(-1)
        self.cand_xyxy = np.asarray(self.cand_xyxy, dtype=np.float32).reshape(-1, 4)
        self.cand_conf = np.asarray(self.cand_conf, dtype=np.float32).reshape(-1)
        self.cand_cls = np.asarray(self.cand_cls, dtype=np.int32).reshape(-1)
        if not (len(self.gt_xyxy) == len(self.gt_cls) == len(self.gt_occ)):
            raise ValueError(f"{self.image_id}: ground-truth arrays disagree in length")
        if not (len(self.cand_xyxy) == len(self.cand_conf) == len(self.cand_cls)):
            raise ValueError(f"{self.image_id}: candidate arrays disagree in length")


@dataclass
class RecordSet:
    records: list[ImageRecord]
    meta: dict = field(default_factory=dict)   # detector, dataset, role, class names, conf floor ...

    def __len__(self) -> int:
        return len(self.records)


def save_records(path: str | Path, rs: RecordSet) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    recs = rs.records
    empty4 = np.zeros((0, 4), np.float32)
    # atomic: write <name>.tmp.npz, then rename over the target
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        tmp,
        image_ids=np.array([r.image_id for r in recs], dtype=str),
        sizes=np.array([[r.width, r.height] for r in recs], dtype=np.int32).reshape(-1, 2),
        gt_ptr=np.cumsum([0] + [len(r.gt_cls) for r in recs]).astype(np.int64),
        gt_xyxy=np.concatenate([r.gt_xyxy for r in recs]) if recs else empty4,
        gt_cls=np.concatenate([r.gt_cls for r in recs]) if recs else np.zeros(0, np.int32),
        gt_occ=np.concatenate([r.gt_occ for r in recs]) if recs else np.zeros(0, np.float32),
        cand_ptr=np.cumsum([0] + [len(r.cand_cls) for r in recs]).astype(np.int64),
        cand_xyxy=np.concatenate([r.cand_xyxy for r in recs]) if recs else empty4,
        cand_conf=np.concatenate([r.cand_conf for r in recs]) if recs else np.zeros(0, np.float32),
        cand_cls=np.concatenate([r.cand_cls for r in recs]) if recs else np.zeros(0, np.int32),
        meta=np.array(json.dumps(rs.meta)),
    )
    os.replace(tmp, path)


def load_records(path: str | Path) -> RecordSet:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run scripts/05_predict.py first.")
    # Decompress each array ONCE. Indexing the NpzFile (z["cand_xyxy"]) inside the per-image loop
    # re-reads the whole array every time, and each record then pins its own full copy: memory
    # grew as images x file size (44 GB for three RT-DETR calib files in v0.5). Slicing one
    # array per field gives views that all share it.
    with np.load(path, allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    gp, cp, sizes = a["gt_ptr"], a["cand_ptr"], a["sizes"]
    recs = []
    for i, image_id in enumerate(a["image_ids"].tolist()):
        recs.append(ImageRecord(
            image_id=str(image_id), width=int(sizes[i, 0]), height=int(sizes[i, 1]),
            gt_xyxy=a["gt_xyxy"][gp[i]:gp[i + 1]], gt_cls=a["gt_cls"][gp[i]:gp[i + 1]],
            gt_occ=a["gt_occ"][gp[i]:gp[i + 1]],
            cand_xyxy=a["cand_xyxy"][cp[i]:cp[i + 1]], cand_conf=a["cand_conf"][cp[i]:cp[i + 1]],
            cand_cls=a["cand_cls"][cp[i]:cp[i + 1]],
        ))
    return RecordSet(records=recs, meta=json.loads(str(a["meta"])))
