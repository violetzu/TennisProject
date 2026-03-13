from __future__ import annotations

import os
from typing import Any, List

from ..pipeline_core.detectors.detect_ball import save_video_info_json


def write_video_json(
    *,
    output_path: str,
    input_video_path: str,
    fps: float,
    player_frames_payload: Any,
    raw_ball_detections: Any,
    smooth_ball_positions: Any,
    detection_mask: List[bool],
) -> str:
    """Write video_info_*.json (image-space data).

    對應原本 pipeline/main.py 的 save_video_info_json(...) 段落。
    """
    save_video_info_json(
        output_path=output_path,
        video_name=os.path.basename(input_video_path),
        fps=fps,
        player_frames=player_frames_payload,
        detected_balls=raw_ball_detections,
        interpolated_balls=smooth_ball_positions,
        detection_mask=detection_mask,
    )
    return output_path
