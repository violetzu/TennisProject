from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..pipeline_io.paths import prepare_paths
from ..pipeline_stages.video_loader import load_video_frames
from ..pipeline_stages.court_stage import detect_court_keypoints
from ..pipeline_stages.players_stage import track_players, choose_and_serialize_players
from ..pipeline_stages.ball_stage import detect_ball_from_model, use_external_ball_tracks
from ..pipeline_io.video_json_writer import write_video_json
from ..pipeline_stages.world_stage import build_world_coordinates
from ..pipeline_io.world_json_writer import write_world_json
from ..pipeline_stages.analysis_stage import run_post_analysis


def run_pipeline_from_video(
    *,
    input_path: str,
    output_name: Optional[str] = None,
    data_dir: str,
    video_dir: str,
    court_model_path: str,
    ball_model_path: str,
    player_model_path: str = "yolo11s.pt",
    ball_source: str = "model",  # 'model' or 'external'
    external_ball_boxes: Optional[List] = None,
    ball_stub_path: Optional[str] = None,
) -> Dict[str, str]:
    """AN 作為主 pipeline 的入口（把原 pipeline/main.py 拆分後串起來）。

    Parameters
    ----------
    ball_source:
      - 'model': 使用 pipeline 原本 BallTracker.detect_frames
      - 'external': 使用外部 ball_boxes（通常由 AN 的 ROI+Kalman tracker 提供）

    Returns
    -------
    dict: {'video_json': ..., 'world_json': ...}
    """

    paths = prepare_paths(input_path, output_name, data_dir=data_dir, video_dir=video_dir)

    frames, fps = load_video_frames(paths.input_path)

    # Court
    court_keypoints = detect_court_keypoints(frames[0], court_model_path)

    # Players
    player_tracker, player_detections = track_players(frames, model_path=player_model_path, conf=0.25)
    player_detections, player_frames_payload = choose_and_serialize_players(player_tracker, court_keypoints, player_detections)

    # Ball
    if ball_source == "external":
        if external_ball_boxes is None:
            raise ValueError("ball_source='external' 需要提供 external_ball_boxes")
        raw_ball, smooth_ball, mask = use_external_ball_tracks(
            external_ball_boxes,
            ball_model_path_for_interpolator=ball_model_path,
            conf=0.15,
        )
    else:
        raw_ball, smooth_ball, mask = detect_ball_from_model(
            frames,
            ball_model_path=ball_model_path,
            conf=0.15,
            stub_path=ball_stub_path,
            read_from_stub=False,
        )

    # Write video json
    write_video_json(
        output_path=paths.video_json_path,
        input_video_path=paths.input_path,
        fps=fps,
        player_frames_payload=player_frames_payload,
        raw_ball_detections=raw_ball,
        smooth_ball_positions=smooth_ball,
        detection_mask=mask,
    )

    # World
    H, world_frames = build_world_coordinates(
        fps=fps,
        court_keypoints=court_keypoints,
        player_detections=player_detections,
        player_frames_payload=player_frames_payload,
        smooth_ball_positions=smooth_ball,
        detection_mask=mask,
    )

    write_world_json(
        output_path=paths.world_json_path,
        input_video_path=paths.input_path,
        fps=fps,
        world_frames=world_frames,
        homography_matrix=H,
    )

    # Analysis
    run_post_analysis(video_json_path=paths.video_json_path, world_json_path=paths.world_json_path, fps=fps)

    return {"video_json": paths.video_json_path, "world_json": paths.world_json_path}
