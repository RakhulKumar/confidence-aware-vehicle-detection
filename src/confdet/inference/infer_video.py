"""Video inference: run the three-tier rule on every Nth frame and
optionally write an annotated copy of the video. Frames are passed to the
detector in memory (BGR arrays), so nothing is written to a temp directory.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from confdet.inference.tiers import TieredDetection, assign_tiers


@dataclass
class FrameResult:
    frame_index: int
    timestamp_sec: float
    detections: list[TieredDetection]


def run_inference_on_video(detector, video_path: str | Path, std_threshold: float,
                           frame_stride: int = 5, max_frames: int | None = None,
                           **tier_kwargs) -> list[FrameResult]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    results: list[FrameResult] = []
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % frame_stride == 0:
                xyxy, conf, cls = detector.predict(frame)
                results.append(FrameResult(index, index / fps,
                                           assign_tiers(xyxy, conf, cls, std_threshold, **tier_kwargs)))
                if max_frames is not None and len(results) >= max_frames:
                    break
            index += 1
    finally:
        cap.release()
    return results


def write_annotated_video(video_path: str | Path, frame_results: list[FrameResult],
                          class_names: list[str], out_path: str | Path) -> None:
    import cv2
    import numpy as np
    from PIL import Image

    from confdet.visualization.annotate import annotate_image

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    by_index = {r.frame_index: r for r in frame_results}
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index in by_index:
                pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                annotated = annotate_image(pil, by_index[index].detections, class_names)
                frame = cv2.cvtColor(np.array(annotated), cv2.COLOR_RGB2BGR)
            writer.write(frame)
            index += 1
    finally:
        cap.release()
        writer.release()
