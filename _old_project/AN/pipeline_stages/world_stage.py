from __future__ import annotations

from typing import Any, List, Tuple

from ..pipeline_core.court_homography import (
    compute_homography,
    project_player_positions,
    project_ball_positions,
    build_world_frames,
)


def build_world_coordinates(
    *,
    fps: float,
    court_keypoints: Any,
    player_detections: Any,
    player_frames_payload: Any,
    smooth_ball_positions: Any,
    detection_mask: List[bool],
):
    """Compute homography + project players/ball into world coordinates.

    對應原本 pipeline/main.py（Step 4 同質變換 + 世界座標）
    """
    H_img_to_world, _ = compute_homography(court_keypoints)
    player_world_positions = project_player_positions(H_img_to_world, player_detections)
    ball_world_positions = project_ball_positions(H_img_to_world, smooth_ball_positions)

    world_frames = build_world_frames(
        fps=fps,
        player_frames_image=player_detections,
        player_frames_world=player_world_positions,
        player_serialized=player_frames_payload,
        ball_frames_image=smooth_ball_positions,
        ball_frames_world=ball_world_positions,
        detection_mask=detection_mask,
    )

    return H_img_to_world, world_frames
