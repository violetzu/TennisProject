from __future__ import annotations

import os
from typing import Any

from ..pipeline_core.court_homography import save_world_coordinate_json


def write_world_json(
    *,
    output_path: str,
    input_video_path: str,
    fps: float,
    world_frames: Any,
    homography_matrix: Any,
) -> str:
    """Write world_info_*.json (world-space data).

    對應原本 pipeline/main.py 的 save_world_coordinate_json(...) 段落。
    """
    save_world_coordinate_json(
        output_path=output_path,
        video_name=os.path.basename(input_video_path),
        fps=fps,
        frames=world_frames,
        homography_matrix=homography_matrix,
    )
    return output_path
