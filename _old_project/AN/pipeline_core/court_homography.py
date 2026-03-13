"""Utilities for mapping court keypoints to world coordinates and validation renders."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Tennis court dimensions in meters
COURT_LENGTH = 23.77
DOUBLES_WIDTH = 10.97
SINGLES_WIDTH = 8.23
HALF_WIDTH = DOUBLES_WIDTH / 2.0
SINGLES_OFFSET = (DOUBLES_WIDTH - SINGLES_WIDTH) / 2.0
SERVICE_LINE_FROM_BASELINE = COURT_LENGTH / 2.0 - 6.40  # 5.485 m
SERVICE_LINE_FAR = COURT_LENGTH - SERVICE_LINE_FROM_BASELINE
CENTER_WIDTH = DOUBLES_WIDTH / 2.0

# Canonical world coordinates for the 14 keypoints (indices 0~13)
CANONICAL_WORLD_POINTS = np.array(
    [
        # far baseline (towards opponent)
        [0.0, COURT_LENGTH],  # 0 - far left doubles corner
        [DOUBLES_WIDTH, COURT_LENGTH],  # 1 - far right doubles corner
        [0.0, 0.0],  # 2 - near left doubles corner
        [DOUBLES_WIDTH, 0.0],  # 3 - near right doubles corner
        [SINGLES_OFFSET, COURT_LENGTH],  # 4 - far left singles corner
        [SINGLES_OFFSET, 0.0],  # 5 - near left singles corner
        [DOUBLES_WIDTH - SINGLES_OFFSET, COURT_LENGTH],  # 6 - far right singles corner
        [DOUBLES_WIDTH - SINGLES_OFFSET, 0.0],  # 7 - near right singles corner
        [SINGLES_OFFSET, SERVICE_LINE_FAR],  # 8 - far left service corner
        [DOUBLES_WIDTH - SINGLES_OFFSET, SERVICE_LINE_FAR],  # 9 - far right service corner
        [SINGLES_OFFSET, SERVICE_LINE_FROM_BASELINE],  # 10 - near left service corner
        [DOUBLES_WIDTH - SINGLES_OFFSET, SERVICE_LINE_FROM_BASELINE],  # 11 - near right service corner
        [CENTER_WIDTH, SERVICE_LINE_FAR],  # 12 - far service center
        [CENTER_WIDTH, SERVICE_LINE_FROM_BASELINE],  # 13 - near service center
    ],
    dtype=np.float32,
)

KEYPOINT_COUNT = CANONICAL_WORLD_POINTS.shape[0]


def reshape_keypoints(flat_keypoints: Sequence[float]) -> np.ndarray:
    """Convert flat list [x0,y0,...] into Nx2 float32 array."""
    arr = np.asarray(flat_keypoints, dtype=np.float32)
    if arr.size != KEYPOINT_COUNT * 2:
        raise ValueError(f"Expected {KEYPOINT_COUNT * 2} values, got {arr.size}")
    points = np.column_stack((arr[0::2], arr[1::2])).astype(np.float32)
    return points


def compute_homography(image_keypoints: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (image_to_world, world_to_image) homography matrices."""
    img_pts = reshape_keypoints(image_keypoints)
    H, mask = cv2.findHomography(img_pts, CANONICAL_WORLD_POINTS, method=cv2.RANSAC, ransacReprojThreshold=8.0)
    if H is None:
        raise RuntimeError("Failed to compute homography from keypoints")
    H_world_to_img = np.linalg.inv(H)
    return H, H_world_to_img


def project_points(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply homography H (3x3) to Nx2 points."""
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    pts = points.reshape(-1, 1, 2).astype(np.float32)
    projected = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return projected


def project_player_positions(H_img_to_world: np.ndarray, player_frames: List[Dict[str, List[float]]]) -> List[Dict[str, Optional[List[float]]]]:
    """Project player foot points into world coordinates for each frame."""
    world_frames: List[Dict[str, Optional[List[float]]]] = []
    for frame in player_frames:
        frame_world: Dict[str, Optional[List[float]]] = {}
        for player_id, bbox in frame.items():
            if not bbox or len(bbox) != 4:
                frame_world[player_id] = None
                continue
            foot_x = (bbox[0] + bbox[2]) / 2.0
            foot_y = bbox[3]
            world_pt = project_points(H_img_to_world, np.array([[foot_x, foot_y]], dtype=np.float32))
            frame_world[player_id] = world_pt[0].tolist()
        world_frames.append(frame_world)
    return world_frames


def project_ball_positions(H_img_to_world: np.ndarray, ball_frames: List[Dict[int, List[float]]]) -> List[Optional[List[float]]]:
    """Project ball centers (if available) into world coordinates."""
    projected: List[Optional[List[float]]] = []
    for frame in ball_frames:
        bbox = frame.get(1) if isinstance(frame, dict) else None
        if not bbox or len(bbox) != 4:
            projected.append(None)
            continue
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        world_pt = project_points(H_img_to_world, np.array([[cx, cy]], dtype=np.float32))
        projected.append(world_pt[0].tolist())
    return projected


def compute_ball_speed(world_points: List[Optional[List[float]]], fps: float) -> List[Optional[float]]:
    speeds: List[Optional[float]] = [None]
    for i in range(1, len(world_points)):
        prev = world_points[i - 1]
        curr = world_points[i]
        if prev is None or curr is None or fps <= 0:
            speeds.append(None)
            continue
        dist = np.linalg.norm(np.array(curr) - np.array(prev))
        speeds.append(dist * fps)
    return speeds


def build_world_frames(
    fps: float,
    player_frames_image: List[Dict[str, List[float]]],
    player_frames_world: List[Dict[str, Optional[List[float]]]],
    player_serialized: List[List[Dict[str, object]]],
    ball_frames_image: List[Dict[int, List[float]]],
    ball_frames_world: List[Optional[List[float]]],
    detection_mask: List[bool],
) -> List[Dict[str, object]]:
    """Combine image + world data into per-frame records."""
    max_len = max(len(player_frames_image), len(ball_frames_image), len(ball_frames_world), len(detection_mask))

    frames: List[Dict[str, object]] = []
    for idx in range(max_len):
        time_sec = round(idx / fps, 3) if fps > 0 else float(idx)

        players_payload: List[Dict[str, object]] = []
        if idx < len(player_serialized):
            for entry in player_serialized[idx]:
                player_id = entry.get("id")
                world_coord = None
                if idx < len(player_frames_world):
                    frame_world = player_frames_world[idx] or {}
                    world_coord = frame_world.get(player_id)
                players_payload.append(
                    {
                        "id": player_id,
                        "team": entry.get("team"),
                        "bbox": entry.get("bbox"),  # keep original bbox for reference
                        "world": world_coord,
                    }
                )

        detected = detection_mask[idx] if idx < len(detection_mask) else False
        image_bbox = None
        if idx < len(ball_frames_image):
            image_bbox = ball_frames_image[idx].get(1)
        world_coord = ball_frames_world[idx] if idx < len(ball_frames_world) else None
        speed_val = None  # 暫時先給 null，之後再計算實際速度

        status = "missing"
        if detected:
            status = "detected"
        elif world_coord is not None:
            status = "interpolated"

        frames.append(
            {
                "time": time_sec,
                "players": players_payload,
                "ball": {
                    "bbox": image_bbox,
                    "world": world_coord,
                    "speed": speed_val,
                    "status": status,
                },
                "ball_conf": bool(detected),
                "events": [],
            }
        )

    return frames


def save_world_coordinate_json(
    output_path: str,
    video_name: str,
    fps: float,
    frames: List[Dict[str, object]],
    homography_matrix: np.ndarray,
) -> None:
    duration = round(len(frames) / fps, 3) if fps > 0 else len(frames)
    payload = {
        "metadata": {
            "video_name": video_name,
            "fps": fps,
            "duration": duration,
            "frame_count": len(frames),
            "created_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "homography": homography_matrix.flatten().tolist(),
        },
        "frames": frames,
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        import json

        json.dump(payload, f, indent=2)


@dataclass
class MiniCourtConfig:
    scale: float = 20.0  # pixels per meter
    margin: int = 30     # 邊距
    # 改進配色 - 現代深綠色球場風格
    court_bg_color: Tuple[int, int, int] = (26, 71, 42)        # 深綠草地
    court_line_color: Tuple[int, int, int] = (255, 255, 255)   # 白線
    net_color: Tuple[int, int, int] = (180, 180, 180)          # 灰色網
    player_top_color: Tuple[int, int, int] = (79, 195, 247)    # 淺藍色 (上方球員)
    player_bottom_color: Tuple[int, int, int] = (255, 183, 77) # 橙色 (下方球員)
    ball_color: Tuple[int, int, int] = (76, 175, 80)           # 綠色
    serve_marker_color: Tuple[int, int, int] = (255, 235, 59)  # 黃色 (發球標記)
    winner_marker_color: Tuple[int, int, int] = (244, 67, 54)  # 紅色 (勝利球)


def _world_to_canvas(points: np.ndarray, cfg: MiniCourtConfig) -> np.ndarray:
    width_px = int(DOUBLES_WIDTH * cfg.scale)
    height_px = int(COURT_LENGTH * cfg.scale)
    translated = np.empty_like(points)
    translated[:, 0] = cfg.margin + points[:, 0] * cfg.scale
    translated[:, 1] = cfg.margin + height_px - points[:, 1] * cfg.scale
    return translated


def draw_minicourt_background(cfg: MiniCourtConfig) -> np.ndarray:
    width_px = int(DOUBLES_WIDTH * cfg.scale) + cfg.margin * 2
    height_px = int(COURT_LENGTH * cfg.scale) + cfg.margin * 2
    canvas = np.zeros((height_px, width_px, 3), dtype=np.uint8)
    # 深色背景
    canvas[:] = (20, 20, 25)
    
    # 球場區域填充
    keypoints = CANONICAL_WORLD_POINTS
    court_poly = _world_to_canvas(keypoints[[0, 1, 3, 2]], cfg).astype(int)
    cv2.fillPoly(canvas, [court_poly], cfg.court_bg_color)

    # 球場外框
    cv2.polylines(canvas, [court_poly], isClosed=True, color=cfg.court_line_color, thickness=2)

    # 單打線
    singles_poly = _world_to_canvas(keypoints[[4, 6, 7, 5]], cfg).astype(int)
    cv2.polylines(canvas, [singles_poly], True, cfg.court_line_color, 1)

    # 發球區垂直線
    for pair in [(8, 10), (9, 11)]:
        pts = _world_to_canvas(keypoints[list(pair)], cfg).astype(int)
        cv2.line(canvas, tuple(pts[0]), tuple(pts[1]), cfg.court_line_color, 1)

    # 發球線
    service_line_far = _world_to_canvas(keypoints[[8, 9]], cfg).astype(int)
    service_line_near = _world_to_canvas(keypoints[[10, 11]], cfg).astype(int)
    cv2.line(canvas, tuple(service_line_far[0]), tuple(service_line_far[1]), cfg.court_line_color, 1)
    cv2.line(canvas, tuple(service_line_near[0]), tuple(service_line_near[1]), cfg.court_line_color, 1)

    # 中線
    center_line = _world_to_canvas(keypoints[[12, 13]], cfg).astype(int)
    cv2.line(canvas, tuple(center_line[0]), tuple(center_line[1]), cfg.court_line_color, 1)

    # 網子 (粗一點)
    net_y = COURT_LENGTH / 2.0
    net_pts = _world_to_canvas(np.array([[0, net_y], [DOUBLES_WIDTH, net_y]], dtype=np.float32), cfg).astype(int)
    cv2.line(canvas, tuple(net_pts[0]), tuple(net_pts[1]), cfg.net_color, 3)

    return canvas


def render_minicourt_video(
    frames_world: List[Dict[str, object]],
    output_path: str,
    fps: float,
    cfg: Optional[MiniCourtConfig] = None,
) -> str:
    if cfg is None:
        cfg = MiniCourtConfig()

    background = draw_minicourt_background(cfg)
    height_px, width_px = background.shape[:2]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    codecs_to_try = [
        ('avc1', '.mp4'),
        ('vp09', '.webm'),
        ('vp80', '.webm'),
        ('mp4v', '.mp4')
    ]
    writer = None
    final_out_path = output_path
    
    for codec, ext in codecs_to_try:
        current_out_path = os.path.splitext(output_path)[0] + ext
        fourcc = cv2.VideoWriter_fourcc(*codec)
        temp_writer = cv2.VideoWriter(current_out_path, fourcc, max(fps, 1.0), (width_px, height_px))
        if temp_writer.isOpened():
            writer = temp_writer
            final_out_path = current_out_path
            print(f"MiniCourt VideoWriter initialized with codec: {codec}, output: {final_out_path}")
            break
    
    if writer is None or not writer.isOpened():
        print(f"Warning: Failed to initialize MiniCourt VideoWriter with codecs {codecs_to_try}")
        return output_path

    for frame in frames_world:
        canvas = background.copy()
        players = frame.get("players", [])
        for entry in players:
            world = entry.get("world")
            if world is None:
                continue
            pt = _world_to_canvas(np.array([world], dtype=np.float32), cfg)[0].astype(int)
            color = cfg.player_top_color if entry.get("team") == "top" else cfg.player_bottom_color
            cv2.circle(canvas, tuple(pt), 8, color, -1)
            cv2.putText(canvas, str(entry.get("id")), (pt[0] + 5, pt[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        ball = frame.get("ball", {})
        world = ball.get("world")
        if world is not None:
            pt = _world_to_canvas(np.array([world], dtype=np.float32), cfg)[0].astype(int)
            cv2.circle(canvas, tuple(pt), 6, cfg.ball_color, -1)

        writer.write(canvas)
    
    writer.release()
    return final_out_path

    writer.release()