"""Data types for courttest evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# Keypoint index → description (YOLO physical order, see data.yaml)
KP_NAMES = [
    "TL",          # 0  左雙打×遠底線
    "FAR_SL",      # 1  左單打×遠底線
    "FAR_SR",      # 2  右單打×遠底線
    "TR",          # 3  右雙打×遠底線
    "BL",          # 4  左雙打×近底線
    "NEAR_SL",     # 5  左單打×近底線
    "NEAR_SR",     # 6  右單打×近底線
    "BR",          # 7  右雙打×近底線
    "SVC_FAR_L",   # 8  左單打×遠發球線
    "SVC_FAR_C",   # 9  中線×遠發球線
    "SVC_FAR_R",   # 10 右單打×遠發球線
    "SVC_NR_R",    # 11 右單打×近發球線
    "SVC_NR_C",    # 12 中線×近發球線
    "SVC_NR_L",    # 13 左單打×近發球線
]
NUM_KP = len(KP_NAMES)
# Indices of the 4 Homography-critical corners
CORNER_IDX = [0, 3, 4, 7]  # TL, TR, BL, BR


@dataclass
class KeypointGT:
    """Ground-truth for one keypoint in one image (pixel coords)."""
    idx: int                    # keypoint index 0..13
    x: Optional[float]          # pixel x (None if not labeled / invisible)
    y: Optional[float]          # pixel y
    visible: bool               # visibility flag (v==2 in YOLO format)


@dataclass
class ImageLabel:
    """Ground-truth for one image."""
    image_path: str
    width: int
    height: int
    keypoints: list[KeypointGT]  # length == NUM_KP


@dataclass
class KeypointPred:
    """Predicted position for one keypoint."""
    idx: int
    x: Optional[float]
    y: Optional[float]
    conf: float = 0.0


@dataclass
class ImagePrediction:
    """Model output for one image."""
    image_path: str
    detected: bool               # True if model returned any result
    keypoints: list[KeypointPred]  # length == NUM_KP; x/y may be None


@dataclass
class PerKeyPointMetrics:
    """Metrics for a single keypoint across all evaluated images."""
    idx: int
    name: str
    n_gt: int                    # number of images where GT is labeled
    n_detected: int              # number where model returned a prediction
    pck_5: float                 # PCK @ 5 px
    pck_10: float                # PCK @ 10 px
    pck_20: float                # PCK @ 20 px
    rmse: float                  # RMSE over TP detections (px)
    mean_error: float            # mean |error| over TP detections (px)


@dataclass
class EvalResult:
    """Full evaluation result for one method."""
    method: str
    n_images: int
    detection_rate: float        # fraction of images with a valid detection
    mpck_5: float                # mean PCK@5 across all 14 keypoints
    mpck_10: float
    mpck_20: float
    corner_pck_5: float          # mPCK@5 for corners only
    corner_pck_10: float
    corner_pck_20: float
    mean_rmse: float             # mean RMSE across keypoints (px)
    fps: float
    per_kp: list[PerKeyPointMetrics] = field(default_factory=list)
