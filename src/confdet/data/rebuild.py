"""Rebuild materialised datasets on another machine from their MANIFESTS, never re-splitting (v0.6, Beocat).

  hash_tree              SHA256 of every file of a processed dataset folder (Ultralytics' *.cache
                         files excluded: they are regenerated), relative POSIX paths. Text files
                         (.txt labels, .json, .yaml) are hashed with line endings normalised to LF:
                         Windows' text mode wrote CRLF, Linux writes LF, the content is the same.
                         Images are hashed byte for byte.
  write_hash_list / read_hash_list
                         a text file of "<hex>  <path>" lines (sha256sum layout; text files carry
                         the LF-normalised hash, so use verify_tree, not `sha256sum -c`).
  rebuild_from_manifest  every image of a transferred manifest.json: found again in the raw export
                         (by source split and file name), checked against the manifest (leakage
                         group, object count), then hard-linked / copied to images/<role>/<file> with
                         its label regenerated exactly as scripts/02 and 02b wrote it. With
                         check_only=True nothing is written: the regenerated label bytes (and, if
                         asked, the raw image bytes) are compared with the hash list instead.
  verify_tree            a processed folder against a hash list: missing, mismatched, extra files.

The role of every image comes from the manifest, so the split is carried over, not recomputed.
"""
from __future__ import annotations

import hashlib
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from confdet.data.splits import group_key
from confdet.data.yolo_dataset import list_split_images, load_dataset_info, read_label_file, rows_to_yolo_lines
from confdet.utils.paths import long_path

EXCLUDE_SUFFIXES = (".cache",)
TEXT_SUFFIXES = (".txt", ".json", ".yaml", ".yml")
CRLF, LF = bytes([13, 10]), bytes([10])


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """Byte hash; for text files (TEXT_SUFFIXES) the hash of the content with CRLF -> LF."""
    if Path(path).suffix.lower() in TEXT_SUFFIXES:
        return sha256_bytes(Path(long_path(Path(path))).read_bytes().replace(CRLF, LF))
    h = hashlib.sha256()
    with open(long_path(Path(path)), "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix not in EXCLUDE_SUFFIXES)


def hash_tree(root: str | Path, workers: int = 8, cache: dict | None = None) -> dict[str, str]:
    """{relative POSIX path: sha256}. `cache` (keyed by file identity: device, inode, size, mtime)
    lets hard-linked copies in several datasets be read once."""
    root = Path(root)
    files = _files(root)
    cache = {} if cache is None else cache

    def one(p: Path) -> tuple[str, str]:
        st = p.stat()
        key = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns) if st.st_ino else None
        if key is not None and key in cache:
            return p.relative_to(root).as_posix(), cache[key]
        h = sha256_file(p)
        if key is not None:
            cache[key] = h
        return p.relative_to(root).as_posix(), h

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return dict(ex.map(one, files))


def write_hash_list(hashes: dict[str, str], path: str | Path) -> None:
    Path(path).write_text("".join(f"{h}  {p}\n" for p, h in sorted(hashes.items())), encoding="utf-8")


def read_hash_list(path: str | Path) -> dict[str, str]:
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            h, p = line.split("  ", 1)
            out[p.lstrip("*")] = h
    return out


def verify_tree(root: str | Path, expected: dict[str, str], workers: int = 8) -> dict:
    got = hash_tree(root, workers)
    missing = sorted(set(expected) - set(got))
    extra = sorted(set(got) - set(expected))
    mismatched = sorted(p for p in set(expected) & set(got) if expected[p] != got[p])
    return {"n_expected": len(expected), "n_found": len(got), "missing": missing, "extra": extra,
            "mismatched": mismatched, "identical": not (missing or extra or mismatched)}


def _label_text(label_path: Path, label_format: str) -> tuple[str, int]:
    rows, _ = read_label_file(label_path, label_format, strict=True)
    return "\n".join(rows_to_yolo_lines(rows)), len(rows)


def _link_or_copy(src: Path, dst: Path) -> None:
    src, dst = long_path(src), long_path(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def rebuild_from_manifest(manifest: dict, sources: dict[str, dict], out_root: str | Path, check_only: bool = False,
                          expected: dict[str, str] | None = None, hash_images: bool = False) -> dict:
    """sources: {source split: {"root", "label_format", "group_regex", "video_segment_frames"}}, with the
    key "*" as the default for splits not listed. Returns counts and every problem found; writes the
    images and labels unless check_only (the caller writes data.yaml / manifest.json)."""
    out_root = Path(out_root)
    if not check_only and out_root.exists() and any(out_root.iterdir()):
        raise FileExistsError(f"{out_root} exists and is not empty; refusing to overwrite it")
    splits = sorted({im["source_split"] for im in manifest["images"]})
    lookup, spec_of = {}, {}
    for s in splits:
        spec = sources.get(s, sources.get("*"))
        if spec is None:
            raise KeyError(f"no raw source given for split {s!r}")
        spec_of[s] = spec
        info = load_dataset_info(spec["root"], spec.get("label_format", "bbox"))
        lookup[s] = {img.name: (img, lbl) for img, lbl in list_split_images(info, s)}
    problems = {"not_in_raw": [], "group_differs": [], "n_objects_differs": [], "label_differs": [],
                "image_differs": [], "not_in_hash_list": []}
    n = 0
    for im in manifest["images"]:
        s, name, role = im["source_split"], im["file"], im["role"]
        raw_name = name if name in lookup[s] else (name[len(s) + 2:] if name.startswith(f"{s}__") else name)
        if raw_name not in lookup[s]:
            problems["not_in_raw"].append(f"{s}/{name}")
            continue
        img, lbl = lookup[s][raw_name]
        spec = spec_of[s]
        g = group_key(raw_name, spec.get("group_regex"), int(spec.get("video_segment_frames", 30)))
        if g != im["group"]:
            problems["group_differs"].append(f"{name}: {g} != {im['group']}")
        text, n_obj = _label_text(lbl, spec.get("label_format", "bbox"))
        if n_obj != im["n_objects"]:
            problems["n_objects_differs"].append(f"{name}: {n_obj} != {im['n_objects']}")
        rel_img, rel_lbl = f"images/{role}/{name}", f"labels/{role}/{Path(name).stem}.txt"
        if expected is not None:
            for rel in (rel_img, rel_lbl):
                if rel not in expected:
                    problems["not_in_hash_list"].append(rel)
            if rel_lbl in expected and sha256_bytes(text.encode("utf-8")) != expected[rel_lbl]:
                problems["label_differs"].append(rel_lbl)
            if hash_images and rel_img in expected and sha256_file(img) != expected[rel_img]:
                problems["image_differs"].append(rel_img)
        if not check_only:
            (out_root / "images" / role).mkdir(parents=True, exist_ok=True)
            (out_root / "labels" / role).mkdir(parents=True, exist_ok=True)
            _link_or_copy(img, out_root / "images" / role / name)
            (out_root / "labels" / role / f"{Path(name).stem}.txt").write_text(text, encoding="utf-8")
        n += 1
    return {"n_manifest": len(manifest["images"]), "n_rebuilt" if not check_only else "n_checked": n,
            "check_only": check_only, "images_hashed": bool(hash_images and expected is not None),
            "n_problems": sum(len(v) for v in problems.values()),
            "problems": {k: v[:20] for k, v in problems.items()},
            "problem_counts": {k: len(v) for k, v in problems.items()}}
