"""Pick the N most useful frames of a processed video and save them as
annotated stills. Pure selection logic (no model I/O), unit-tested on
synthetic FrameResults.
"""
from __future__ import annotations

from pathlib import Path

from confdet.inference.infer_video import FrameResult


def score_frame_high_confidence(frame: FrameResult) -> float:
    """Number of accepted detections x their mean confidence."""
    acc = [d for d in frame.detections if d.tier == "accepted"]
    return len(acc) * (sum(d.confidence for d in acc) / len(acc)) if acc else 0.0


def score_frame_flagged_showcase(frame: FrameResult) -> float:
    """Frames that show the method at work: several confident detections AND
    flagged candidates sitting behind them (high occlusion context).
    """
    acc = sum(d.tier == "accepted" for d in frame.detections)
    flagged = [d for d in frame.detections if d.tier == "flagged"]
    if not acc or not flagged:
        return 0.0
    return min(acc, 10) * sum(0.25 + d.occlusion_context for d in flagged)


def score_frame_most_detections(frame: FrameResult) -> float:
    return float(sum(d.tier == "accepted" for d in frame.detections))


SCORERS = {
    "high_confidence": score_frame_high_confidence,
    "flagged_showcase": score_frame_flagged_showcase,
    "most_detections": score_frame_most_detections,
}


def select_best_frames(frame_results: list[FrameResult], n: int = 10, scorer: str = "high_confidence",
                       min_gap_frames: int = 15) -> list[FrameResult]:
    """Top-scoring frames, at least `min_gap_frames` apart so the selection
    is not N near-identical frames from the same few seconds.
    """
    if scorer not in SCORERS:
        raise ValueError(f"Unknown scorer {scorer!r}; choose from {list(SCORERS)}")
    ranked = sorted(frame_results, key=SCORERS[scorer], reverse=True)
    selected: list[FrameResult] = []
    for frame in ranked:
        if all(abs(frame.frame_index - s.frame_index) >= min_gap_frames for s in selected):
            selected.append(frame)
        if len(selected) >= n:
            break
    return sorted(selected, key=lambda f: f.frame_index)


def save_best_frames(video_path: str | Path, frame_results: list[FrameResult], class_names: list[str],
                     output_dir: str | Path, n: int = 10, scorer: str = "high_confidence",
                     min_gap_frames: int = 15) -> list[Path]:
    import cv2
    from PIL import Image

    from confdet.visualization.annotate import annotate_image

    best = select_best_frames(frame_results, n=n, scorer=scorer, min_gap_frames=min_gap_frames)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    saved = []
    try:
        for fr in best:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fr.frame_index)
            ok, frame = cap.read()
            if not ok:
                continue
            out_path = output_dir / f"best_frame_{fr.frame_index:08d}.jpg"
            annotate_image(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), fr.detections,
                           class_names, out_path=out_path)
            saved.append(out_path)
    finally:
        cap.release()
    return saved
