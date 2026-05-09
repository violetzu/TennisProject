from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
TEST_ROOT = PACKAGE_ROOT.parent
BACKEND_ROOT = TEST_ROOT.parent
PROJECT_ROOT = BACKEND_ROOT.parent

MODELS_ROOT = BACKEND_ROOT / "models"
DATASET_ROOT = MODELS_ROOT / "tracknet_dataset"

ARTIFACT_ROOT = PACKAGE_ROOT / "artifacts"
MANIFEST_ROOT = ARTIFACT_ROOT / "manifests"
DATASETS_ROOT = ARTIFACT_ROOT / "datasets"
CACHE_ROOT = ARTIFACT_ROOT / "cache"
CHECKPOINT_ROOT = ARTIFACT_ROOT / "checkpoints"
RESULT_ROOT = ARTIFACT_ROOT / "results"

YOLO_DATASET_ROOT = DATASETS_ROOT / "yolo"
TRACKNET_DATASET_ROOT = DATASETS_ROOT / "tracknet"

YOLO_CACHE_ROOT = CACHE_ROOT / "yolo"
METHOD_RESULT_ROOT = RESULT_ROOT / "methods"
METHODS_RESULT_PATH = RESULT_ROOT / "methods.json"
ABLATION_RESULT_PATH = RESULT_ROOT / "ablation.json"
TRACKERS_RESULT_ROOT = RESULT_ROOT / "trackers"
TRACKERS_RESULT_PATH = RESULT_ROOT / "trackers.json"

REPORT_PATH = PACKAGE_ROOT / "實驗報告.md"
DOCKERFILE_PATH = PACKAGE_ROOT / "Dockerfile"


def ensure_layout() -> None:
    for path in (
        ARTIFACT_ROOT,
        MANIFEST_ROOT,
        DATASETS_ROOT,
        CACHE_ROOT,
        CHECKPOINT_ROOT,
        RESULT_ROOT,
        YOLO_DATASET_ROOT,
        TRACKNET_DATASET_ROOT,
        YOLO_CACHE_ROOT,
        METHOD_RESULT_ROOT,
        TRACKERS_RESULT_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)
