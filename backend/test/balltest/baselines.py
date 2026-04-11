from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

from .dataset import iter_clip_frames, load_labels_manifest, load_split, prepare
from .metrics import aggregate_metric_rows, compute_tracker_metrics, public_metric_dict
from .schema import MetricBundle
from .utils import lazy_import


def _detect_classical_positions(frame_paths: list[Path]) -> list[Optional[list[float]]]:
    cv2 = lazy_import("cv2")
    positions: list[Optional[list[float]]] = []
    prev_gray = None
    last_pos = None
    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            positions.append(None)
            prev_gray = gray
            continue
        diff = cv2.absdiff(gray, prev_gray)
        _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
        thresh = cv2.medianBlur(thresh, 5)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 4 or area > 250:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            cx = x + w / 2.0
            cy = y + h / 2.0
            score = float(area)
            if last_pos is not None:
                score -= math.hypot(cx - last_pos[0], cy - last_pos[1]) * 0.2
            if best is None or score > best[2]:
                best = (cx, cy, score)
        if best is None:
            positions.append(None)
        else:
            last_pos = (best[0], best[1])
            positions.append([best[0], best[1]])
        prev_gray = gray
    return positions


def evaluate_classical_motion(dist_threshold: float, max_clips: int = 0) -> MetricBundle:
    prepare()
    clips = load_split("test")
    if max_clips > 0:
        clips = clips[:max_clips]
    labels_by_clip = load_labels_manifest()

    metric_rows = []
    total_runtime = 0.0
    per_clip = []

    for clip in clips:
        frame_paths = iter_clip_frames(clip)
        t0 = time.perf_counter()
        positions = _detect_classical_positions(frame_paths)
        elapsed = time.perf_counter() - t0
        total_runtime += elapsed
        metrics = compute_tracker_metrics(
            labels_by_clip[clip.clip_id],
            raw_positions=positions,
            final_positions=positions,
            dist_threshold=dist_threshold,
        )
        metric_rows.append(metrics)
        per_clip.append({"clip": clip.clip_id, **public_metric_dict(metrics)})

    aggregate = aggregate_metric_rows(metric_rows)
    frames_total = int(aggregate["frames_total"])
    return MetricBundle(
        method="classical_motion",
        suite="test",
        model_path="classical-motion",
        runtime_env="balltest-runner",
        device="cpu_or_gpu",
        frames_total=frames_total,
        frames_eval_visible_12=int(aggregate["frames_eval_visible_12"]),
        frames_ignored_visibility_3=int(aggregate["frames_ignored_visibility_3"]),
        tp=int(aggregate["tp"]),
        fp=int(aggregate["fp"]),
        fn=int(aggregate["fn"]),
        precision=float(aggregate["precision"]),
        recall=float(aggregate["recall"]),
        f1=float(aggregate["f1"]),
        mean_error_px=float(aggregate["mean_error_px"]),
        median_error_px=float(aggregate["median_error_px"]),
        runtime_sec=total_runtime,
        avg_ms_per_frame=(total_runtime / frames_total * 1000.0) if frames_total else 0.0,
        throughput_fps=(frames_total / total_runtime) if total_runtime else 0.0,
        extras=dict(aggregate["extras"]),
        per_clip=per_clip,
    )
