"""3D pose rendering utilities for already-tracked players."""

from __future__ import annotations

from collections import deque
from typing import List, Optional, Sequence, Tuple
import math

import cv2
import numpy as np

from .person_detector import PlayerDetection

BLAZEPOSE_CONNECTIONS = [
    (11, 12), (23, 24), (11, 23), (12, 24),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
    (27, 29), (28, 30),
    (29, 31), (30, 32),
]


def bbox_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    denom = area_a + area_b - inter_area
    if denom <= 0:
        return 0.0
    return inter_area / denom


def convert_to_standing_coordinates(
    pose_3d: np.ndarray,
    scale_factor: float = 1000.0,
) -> np.ndarray:
    pose_converted = pose_3d.copy()
    temp_x = pose_converted[:, 0] * scale_factor
    temp_y = -pose_converted[:, 2] * scale_factor
    temp_z = -pose_converted[:, 1] * scale_factor
    pose_converted[:, 0] = temp_x
    pose_converted[:, 1] = temp_y
    pose_converted[:, 2] = temp_z

    if len(pose_converted) > 0:
        left_ankle = 27 if len(pose_converted) > 27 else -1
        right_ankle = 28 if len(pose_converted) > 28 else -1
        if left_ankle != -1 and right_ankle != -1:
            ground_level = min(pose_converted[left_ankle, 2], pose_converted[right_ankle, 2])
            pose_converted[:, 2] -= ground_level
        center_x = np.mean(pose_converted[:, 0])
        center_y = np.mean(pose_converted[:, 1])
        pose_converted[:, 0] -= center_x
        pose_converted[:, 1] -= center_y
    return pose_converted


def draw_coordinate_axes(canvas: np.ndarray, view_w: int, view_h: int, rotation_angle: float, axis_length: float = 100):
    origin_x = int(view_w * 0.15)
    origin_y = int(view_h * 0.85)
    angle_rad = math.radians(rotation_angle)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    elevation = math.radians(30)
    cos_elev = math.cos(elevation)
    sin_elev = math.sin(elevation)
    axes = {
        "X": (axis_length, 0, 0),
        "Y": (0, axis_length, 0),
        "Z": (0, 0, axis_length),
    }
    colors = {
        "X": (0, 0, 255),
        "Y": (0, 255, 0),
        "Z": (255, 0, 0),
    }
    for axis_name, (x, y, z) in axes.items():
        rot_x = x * cos_a - y * sin_a
        rot_y = x * sin_a + y * cos_a
        rot_z = z
        y_proj = rot_y * cos_elev - rot_z * sin_elev
        z_proj = rot_y * sin_elev + rot_z * cos_elev
        end_x = int(origin_x + rot_x)
        end_y = int(origin_y - z_proj)
        cv2.arrowedLine(canvas, (origin_x, origin_y), (end_x, end_y), colors[axis_name], 3, tipLength=0.1)
        cv2.putText(canvas, axis_name, (end_x + 10, end_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[axis_name], 2)


def render_rotating_pose(
    view_w: int,
    view_h: int,
    keypoints3d: Optional[np.ndarray],
    frame_idx: int,
) -> np.ndarray:
    canvas = np.zeros((view_h, view_w, 3), dtype=np.uint8)
    if keypoints3d is None or keypoints3d.shape[0] == 0:
        cv2.putText(canvas, "No pose detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
        return canvas

    pose_standing = convert_to_standing_coordinates(keypoints3d)
    rotation_angle = (frame_idx * 0.5) % 360
    angle_rad = math.radians(rotation_angle)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    rotated_pose = pose_standing.copy()
    rotated_pose[:, 0] = pose_standing[:, 0] * cos_a - pose_standing[:, 1] * sin_a
    rotated_pose[:, 1] = pose_standing[:, 0] * sin_a + pose_standing[:, 1] * cos_a

    elevation = math.radians(30)
    cos_elev = math.cos(elevation)
    sin_elev = math.sin(elevation)
    proj_x, proj_y, depths = [], [], []
    for x, y, z in rotated_pose:
        y_rot = y * cos_elev - z * sin_elev
        z_rot = y * sin_elev + z * cos_elev
        proj_x.append(x)
        proj_y.append(-z_rot)
        depths.append(y_rot)

    min_x, max_x = min(proj_x), max(proj_x)
    min_y, max_y = min(proj_y), max(proj_y)
    range_x = max(max_x - min_x, 1.0)
    range_y = max(max_y - min_y, 1.0)
    scale_x = (view_w * 0.6) / range_x
    scale_y = (view_h * 0.6) / range_y
    scale = min(scale_x, scale_y)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    screen_x = [((x - center_x) * scale) + (view_w / 2) for x in proj_x]
    screen_y = [((y - center_y) * scale) + (view_h / 2) for y in proj_y]
    if max(depths) != min(depths):
        norm_depths = [(d - min(depths)) / (max(depths) - min(depths)) for d in depths]
    else:
        norm_depths = [0.5] * len(depths)

    draw_coordinate_axes(canvas, view_w, view_h, rotation_angle, scale * 100)
    skeleton_color = (255, 255, 255)
    for i, j in BLAZEPOSE_CONNECTIONS:
        if i < len(screen_x) and j < len(screen_x):
            depth_factor = (norm_depths[i] + norm_depths[j]) / 2
            thickness = max(2, int(4 * (1 - depth_factor * 0.3)))
            p1 = (int(screen_x[i]), int(screen_y[i]))
            p2 = (int(screen_x[j]), int(screen_y[j]))
            cv2.line(canvas, p1, p2, skeleton_color, thickness)
    for idx in range(len(screen_x)):
        depth_factor = norm_depths[idx]
        radius = max(3, int(6 * (1 - depth_factor * 0.2)))
        brightness = 0.7 + 0.3 * (1 - depth_factor)
        point_color = tuple(int(255 * brightness) for _ in range(3))
        cv2.circle(canvas, (int(screen_x[idx]), int(screen_y[idx])), radius, point_color, -1)

    cv2.putText(canvas, f"3D Pose (Angle: {rotation_angle:.1f}\u00b0)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(canvas, "Standing coordinate system", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    return canvas


class Pose3DVisualizer:
    def __init__(self, enable_smoothing: bool = True, smoothing_window: int = 5) -> None:
        self.enable_smoothing = enable_smoothing
        self.pose_history: deque[np.ndarray] = deque(maxlen=max(smoothing_window, 1))
        self.last_bbox: Optional[Tuple[int, int, int, int]] = None

    def render(
        self,
        player: Optional[PlayerDetection],
        frame_idx: int,
        frame_shape: Tuple[int, int, int],
    ) -> np.ndarray:
        if player is None or player.world_landmarks is None:
            self.pose_history.clear()
            self.last_bbox = None
            return render_rotating_pose(frame_shape[1], frame_shape[0], None, frame_idx)

        world = np.array(player.world_landmarks, dtype=np.float32)
        if self.enable_smoothing:
            if self.last_bbox is None or bbox_iou(self.last_bbox, player.bbox) < 0.1:
                self.pose_history.clear()
            self.pose_history.append(world)
            world = np.mean(np.stack(self.pose_history, axis=0), axis=0)
        else:
            self.pose_history.clear()

        self.last_bbox = player.bbox
        return render_rotating_pose(frame_shape[1], frame_shape[0], world, frame_idx)

    @staticmethod
    def draw_2d_skeleton(frame: np.ndarray, landmarks: Sequence[Tuple[float, float, float]], visibility_thr: float = 0.5) -> None:
        color_skeleton = (0, 255, 0)
        color_points = (0, 180, 255)
        for i, j in BLAZEPOSE_CONNECTIONS:
            if 0 <= i < len(landmarks) and 0 <= j < len(landmarks):
                xi, yi, vi = landmarks[i]
                xj, yj, vj = landmarks[j]
                if vi >= visibility_thr and vj >= visibility_thr:
                    cv2.line(frame, (int(xi), int(yi)), (int(xj), int(yj)), color_skeleton, 2)
        for x, y, v in landmarks:
            if v >= visibility_thr:
                cv2.circle(frame, (int(x), int(y)), 3, color_points, -1)

    @staticmethod
    def draw_annotations(
        frame: np.ndarray,
        players_top: Sequence[PlayerDetection],
        players_bottom: Sequence[PlayerDetection],
        target_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        for player in list(players_top) + list(players_bottom):
            color = (0, 180, 255) if player.side == "top" else (255, 120, 0)
            if target_bbox is not None and bbox_iou(player.bbox, target_bbox) > 0.5:
                color = (0, 255, 0)
            x1, y1, x2, y2 = player.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = "Top player" if player.side == "top" else "Bottom player"
            if target_bbox is not None and bbox_iou(player.bbox, target_bbox) > 0.5:
                label += " (3D)"
            label_y = y1 - 6 if y1 - 6 > 10 else y1 + 18
            cv2.putText(frame, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            Pose3DVisualizer.draw_2d_skeleton(frame, player.landmarks)

        if not players_top and not players_bottom:
            cv2.putText(frame, "No person detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)


__all__ = ["Pose3DVisualizer", "PlayerDetection"]
