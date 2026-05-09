"""Evaluate YOLO-pose court keypoint models on the test split."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .dataset import load_test_split
from .metrics import compute_eval_result
from .paths import WEIGHTS
from .schema import (
    NUM_KP, EvalResult,
    ImagePrediction, KeypointPred,
)


def _run_yolo_on_split(
    model_path: Path,
    imgsz: int,
    labels,
    device: str,
    warmup: int = 5,
) -> tuple[list[ImagePrediction], float]:
    """Run YOLO-pose inference on every labeled test image.

    Returns (predictions, fps).
    """
    from ultralytics import YOLO  # type: ignore

    model = YOLO(str(model_path))

    # Warm up
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(warmup):
        model(dummy, imgsz=imgsz, device=device, verbose=False)

    predictions: list[ImagePrediction] = []
    t_start = time.perf_counter()

    for label in labels:
        img = cv2.imread(label.image_path)
        results = model(img, imgsz=imgsz, device=device, verbose=False)
        result = results[0]

        if result.keypoints is None or len(result.keypoints.xy) == 0:
            kps = [KeypointPred(idx=i, x=None, y=None) for i in range(NUM_KP)]
            predictions.append(ImagePrediction(
                image_path=label.image_path, detected=False, keypoints=kps))
            continue

        # Pick the highest-confidence detection
        boxes = result.boxes
        best_idx = int(boxes.conf.argmax()) if boxes is not None and len(boxes) > 0 else 0
        kp_xy = result.keypoints.xy[best_idx].cpu().numpy()   # (14, 2)
        kp_conf = result.keypoints.conf
        confs = kp_conf[best_idx].cpu().numpy() if kp_conf is not None else np.ones(NUM_KP)

        kps: list[KeypointPred] = []
        for i in range(NUM_KP):
            x, y = float(kp_xy[i, 0]), float(kp_xy[i, 1])
            # YOLO returns (0, 0) for invisible keypoints
            if x == 0.0 and y == 0.0:
                kps.append(KeypointPred(idx=i, x=None, y=None, conf=0.0))
            else:
                kps.append(KeypointPred(idx=i, x=x, y=y, conf=float(confs[i])))

        predictions.append(ImagePrediction(
            image_path=label.image_path, detected=True, keypoints=kps))

    t_elapsed = time.perf_counter() - t_start
    fps = len(labels) / t_elapsed if t_elapsed > 0 else 0.0

    return predictions, fps


def evaluate_yolo(
    variant: str,
    imgsz: Optional[int] = None,
    device: str = "",
) -> EvalResult:
    """Evaluate one pre-trained YOLO variant on the test split.

    variant: key in paths.WEIGHTS, e.g. 'yolo26n_640_100ep'
    imgsz:   inference resolution; defaults to the value embedded in the variant name
    """
    model_path = WEIGHTS[variant]
    if not model_path.exists():
        raise FileNotFoundError(f"No weights at {model_path}")

    if imgsz is None:
        # infer from variant name: yolo26n_640_... → 640
        for part in variant.split("_"):
            if part.isdigit():
                imgsz = int(part)
                break
        if imgsz is None:
            imgsz = 640

    labels = load_test_split()
    predictions, fps = _run_yolo_on_split(model_path, imgsz, labels, device)

    return compute_eval_result(
        method=variant,
        labels=labels,
        predictions=predictions,
        fps=fps,
    )


def evaluate_yolo_custom(
    method_name: str,
    model_path: Path,
    imgsz: int = 640,
    device: str = "",
) -> EvalResult:
    """Evaluate any YOLO-pose weights file on the test split."""
    if not model_path.exists():
        raise FileNotFoundError(f"No weights at {model_path}")
    labels = load_test_split()
    predictions, fps = _run_yolo_on_split(model_path, imgsz, labels, device)
    return compute_eval_result(method=method_name, labels=labels,
                               predictions=predictions, fps=fps)
