"""Compute per-keypoint and aggregate metrics for court detection."""
from __future__ import annotations

import math
from typing import Optional

from .schema import (
    NUM_KP, KP_NAMES, CORNER_IDX,
    EvalResult, ImageLabel, ImagePrediction, PerKeyPointMetrics,
)


def _dist(x1: Optional[float], y1: Optional[float],
          x2: Optional[float], y2: Optional[float]) -> Optional[float]:
    if x1 is None or y1 is None or x2 is None or y2 is None:
        return None
    return math.hypot(x2 - x1, y2 - y1)


def compute_eval_result(
    method: str,
    labels: list[ImageLabel],
    predictions: list[ImagePrediction],
    fps: float,
) -> EvalResult:
    """Compute PCK / RMSE / detection-rate across all test images."""
    assert len(labels) == len(predictions)

    detected_count = sum(1 for p in predictions if p.detected)
    detection_rate = detected_count / len(predictions) if predictions else 0.0

    per_kp: list[PerKeyPointMetrics] = []

    for k in range(NUM_KP):
        n_gt = 0
        n_detected = 0
        tp_5 = tp_10 = tp_20 = 0
        sq_errors: list[float] = []
        abs_errors: list[float] = []

        for label, pred in zip(labels, predictions):
            gt_kp = label.keypoints[k]
            if gt_kp.x is None:
                continue
            n_gt += 1

            pred_kp = pred.keypoints[k]
            if pred_kp.x is None or not pred.detected:
                continue
            n_detected += 1

            d = _dist(gt_kp.x, gt_kp.y, pred_kp.x, pred_kp.y)
            if d is None:
                continue
            abs_errors.append(d)
            sq_errors.append(d * d)
            if d <= 5:
                tp_5 += 1
            if d <= 10:
                tp_10 += 1
            if d <= 20:
                tp_20 += 1

        pck_5 = tp_5 / n_gt if n_gt else 0.0
        pck_10 = tp_10 / n_gt if n_gt else 0.0
        pck_20 = tp_20 / n_gt if n_gt else 0.0
        rmse = math.sqrt(sum(sq_errors) / len(sq_errors)) if sq_errors else 0.0
        mean_error = sum(abs_errors) / len(abs_errors) if abs_errors else 0.0

        per_kp.append(PerKeyPointMetrics(
            idx=k, name=KP_NAMES[k],
            n_gt=n_gt, n_detected=n_detected,
            pck_5=pck_5, pck_10=pck_10, pck_20=pck_20,
            rmse=rmse, mean_error=mean_error,
        ))

    def _mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    mpck_5  = _mean([m.pck_5  for m in per_kp])
    mpck_10 = _mean([m.pck_10 for m in per_kp])
    mpck_20 = _mean([m.pck_20 for m in per_kp])
    mean_rmse = _mean([m.rmse for m in per_kp])

    corner_pck_5  = _mean([per_kp[i].pck_5  for i in CORNER_IDX])
    corner_pck_10 = _mean([per_kp[i].pck_10 for i in CORNER_IDX])
    corner_pck_20 = _mean([per_kp[i].pck_20 for i in CORNER_IDX])

    return EvalResult(
        method=method,
        n_images=len(labels),
        detection_rate=detection_rate,
        mpck_5=mpck_5,
        mpck_10=mpck_10,
        mpck_20=mpck_20,
        corner_pck_5=corner_pck_5,
        corner_pck_10=corner_pck_10,
        corner_pck_20=corner_pck_20,
        mean_rmse=mean_rmse,
        fps=fps,
        per_kp=per_kp,
    )
