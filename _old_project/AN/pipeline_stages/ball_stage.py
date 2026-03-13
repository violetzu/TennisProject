from __future__ import annotations

from typing import Any, List, Optional, Tuple
import numpy as np

from ..pipeline_core.trackers.ball_tracker import BallTracker


BallBox = Optional[List[float]]  # [x1,y1,x2,y2] or None


def detect_ball_from_model(
    video_frames: List[np.ndarray],
    *,
    ball_model_path: str,
    conf: float = 0.15,
    stub_path: Optional[str] = None,
    read_from_stub: bool = False,
) -> Tuple[List[Any], List[Any], List[bool]]:
    """Run ball detection using BallTracker + interpolate.

    對應原本 pipeline/main.py：
      ball_tracker = BallTracker(model_path=BALL_MODEL_PTAH, conf=0.15)
      raw_ball_detections, detection_mask = ball_tracker.detect_frames(...)
      smooth_ball_positions = ball_tracker.interpolate_ball_positions(raw_ball_detections)
    """
    tracker = BallTracker(model_path=ball_model_path, conf=conf)
    raw_ball_detections, detection_mask = tracker.detect_frames(
        video_frames,
        read_from_stub=read_from_stub,
        stub_path=stub_path,
    )
    smooth_ball_positions = tracker.interpolate_ball_positions(raw_ball_detections)
    return raw_ball_detections, smooth_ball_positions, detection_mask


def use_external_ball_tracks(
    ball_boxes_per_frame: List[BallBox],
    *,
    ball_model_path_for_interpolator: str,
    conf: float = 0.15,
) -> Tuple[List[Any], List[Any], List[bool]]:
    """Use external ball boxes (e.g., from AN ROI+Kalman tracker) then interpolate.

    這不是原本 pipeline/main.py 的流程，但它是你要把 AN 當主 pipeline
    時，讓 pipeline 的後半段（world/speed/events）能吃 AN 產出的球軌跡。

    Notes
    -----
    - 這裡仍然沿用 BallTracker.interpolate_ball_positions 的邏輯，
      避免你有兩套不同插值導致行為不一致。
    """
    detection_mask = [b is not None for b in ball_boxes_per_frame]

    # BallTracker.interpolate_ball_positions 是 method，因此建一個 tracker 來用它。
    tracker = BallTracker(model_path=ball_model_path_for_interpolator, conf=conf)

    # pipeline 的 raw_ball_detections 格式通常是每幀一個 bbox/center 結構，
    # 但你目前外部只給 bbox。若你的 pipeline 期待更複雜的 dict，
    # 請在這裡做 mapping。
    raw_ball_detections = ball_boxes_per_frame

    smooth_ball_positions = tracker.interpolate_ball_positions(raw_ball_detections)
    return raw_ball_detections, smooth_ball_positions, detection_mask
