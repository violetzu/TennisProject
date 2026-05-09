"""Path constants for courttest."""
from __future__ import annotations

import os
from pathlib import Path

# Project root: TennisProject/
_THIS = Path(__file__).resolve()
ROOT = _THIS.parents[3]  # TennisProject/
BACKEND = ROOT / "backend"

# Dataset
COURT_DATASET = BACKEND / "models" / "court_dataset"
DATASET_YAML = COURT_DATASET / "data.yaml"
IMAGES_TEST = COURT_DATASET / "images" / "test"
LABELS_TEST = COURT_DATASET / "labels" / "test"

# Model weights — pre-existing YOLO runs
RUNS_DIR = BACKEND / "models" / "runs" / "pose" / "court"

WEIGHTS = {
    "yolo26n_640_250ep": RUNS_DIR / "yolo26n-pose-epochs250-imgsz640-mAP50@0.994-5.6it" / "weights" / "best.pt",
}

# Newly trained YOLO variants (via train-yolo command)
def _yolo_run_best(variant: str, imgsz: int = 640) -> Path:
    return COURTTEST_DIR / "artifacts" / "yolo_runs" / f"{variant}-imgsz{imgsz}-ep500-p30" / "weights" / "best.pt"

YOLO_TRAINED_VARIANTS = ["yolo26n", "yolo26s", "yolov8n", "yolov8s", "yolov8m"]

# Artifacts
COURTTEST_DIR = _THIS.parent
ARTIFACTS = COURTTEST_DIR / "artifacts"
RESULTS_DIR = ARTIFACTS / "results"
CHECKPOINTS_DIR = ARTIFACTS / "checkpoints"

# DL model checkpoints
DL_WEIGHTS = {
    "heatmap_mobilenet": CHECKPOINTS_DIR / "heatmap_mobilenet.pt",
    "resnet50_regressor": CHECKPOINTS_DIR / "resnet50_regressor.pt",
}


def ensure_layout() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
