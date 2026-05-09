"""Train YOLO-pose models on the court keypoint dataset."""
from __future__ import annotations

from pathlib import Path


YOLO_VARIANTS = {
    "yolo26n": "yolo26n-pose.pt",
    "yolo26s": "yolo26s-pose.pt",
    "yolo26m": "yolo26m-pose.pt",
    "yolov8n": "yolov8n-pose.pt",
    "yolov8s": "yolov8s-pose.pt",
    "yolov8m": "yolov8m-pose.pt",
}


def train_yolo_court(
    variant: str,
    imgsz: int = 640,
    epochs: int = 500,
    patience: int = 30,
    batch: int = 48,
    device: str = "",
    save_dir: Path | None = None,
) -> Path:
    """Train a YOLO-pose model on the court dataset with early stopping.

    Returns path to best.pt weights.
    """
    from ultralytics import YOLO
    from .paths import ARTIFACTS, COURT_DATASET

    if variant not in YOLO_VARIANTS:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(YOLO_VARIANTS)}")

    model_name = YOLO_VARIANTS[variant]
    model = YOLO(model_name)

    run_name = f"{variant}-imgsz{imgsz}-ep{epochs}-p{patience}"
    out_dir = save_dir or (ARTIFACTS / "yolo_runs")

    results = model.train(
        data=str(COURT_DATASET / "data.yaml"),
        epochs=epochs,
        patience=patience,
        imgsz=imgsz,
        batch=batch,
        device=device or "",
        project=str(out_dir),
        name=run_name,
        exist_ok=True,
        save=True,
        plots=False,
        val=True,
        augment=True,
        mosaic=1.0,
        mixup=0.0,
        degrees=10.0,
        fliplr=0.5,
        workers=2,
    )

    best_pt = out_dir / run_name / "weights" / "best.pt"
    print(f"[train-yolo] done. best.pt → {best_pt}")
    return best_pt
