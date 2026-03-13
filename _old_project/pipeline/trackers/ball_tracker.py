# backend/trackers/ball_tracker.py
"""Simple ball detector + interpolation helper."""

from __future__ import annotations

import pickle
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

class BallTracker:
    def __init__(self, model_path: str, conf: float = 0.15) -> None:
        self.model = YOLO(model_path)
        self.conf = float(conf)

    # ---------- Detection ----------
    def detect_frames(
        self,
        frames: List[np.ndarray],
        read_from_stub: bool = False,
        stub_path: Optional[str] = None,
    ) -> Tuple[List[Dict[int, List[float]]], List[bool]]:
        """Run YOLO on all frames and return detections with a mask of confirmed frames."""
        if read_from_stub and stub_path is not None:
            with open(stub_path, "rb") as f:
                cached = pickle.load(f)
            if isinstance(cached, dict) and "detections" in cached and "mask" in cached:
                return cached["detections"], cached["mask"]
            if isinstance(cached, list):
                fallback_mask: List[bool] = []
                for entry in cached:
                    if isinstance(entry, dict):
                        fallback_mask.append(bool(entry.get(1)))
                    else:
                        fallback_mask.append(False)
                return cached, fallback_mask
            raise ValueError("Unsupported ball detection stub format")

        detections: List[Dict[int, List[float]]] = []
        detection_mask: List[bool] = []
        for frame in frames:
            detection, has_ball = self.detect_frame(frame)
            detections.append(detection)
            detection_mask.append(has_ball)

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump({"detections": detections, "mask": detection_mask}, f)

        return detections, detection_mask

    def detect_frame(self, frame: np.ndarray) -> Tuple[Dict[int, List[float]], bool]:
        """Return the highest confidence ball bbox for a single frame."""
        results = self.model.predict(frame, conf=self.conf, verbose=False)[0]
        best_bbox: Optional[List[float]] = None
        best_conf = -1.0
        for box in results.boxes:
            conf_val = float(box.conf[0]) if hasattr(box.conf, "__len__") else float(box.conf)
            if conf_val > best_conf:
                best_conf = conf_val
                best_bbox = box.xyxy.tolist()[0]

        if best_bbox is not None:
            return {1: best_bbox}, True
        return {}, False

    # ---------- Interpolation ----------
    def interpolate_ball_positions(
        self, ball_positions: List[Dict[int, List[float]]]
    ) -> List[Dict[int, List[float]]]:
        """Interpolate missing frames with linear fill + forward/backward fill."""
        rows: List[List[float]] = []
        for entry in ball_positions:
            bbox = entry.get(1)
            if bbox and len(bbox) == 4:
                rows.append([float(v) for v in bbox])
            else:
                rows.append([np.nan, np.nan, np.nan, np.nan])

        df = pd.DataFrame(rows, columns=["x1", "y1", "x2", "y2"], dtype=float)
        df = df.interpolate(limit_direction="both", axis=0).bfill().ffill()

        interpolated: List[Dict[int, List[float]]] = []
        for row in df.itertuples(index=False):
            values = list(row)
            if any(np.isnan(v) for v in values):
                interpolated.append({})
            else:
                interpolated.append({1: values})
        return interpolated

    # ---------- Analytics ----------
    def get_ball_shot_frames(self, ball_positions: List[Dict[int, List[float]]]):
        rows: List[List[float]] = []
        for entry in ball_positions:
            bbox = entry.get(1)
            if bbox and len(bbox) == 4:
                rows.append([float(v) for v in bbox])
            else:
                rows.append([np.nan, np.nan, np.nan, np.nan])

        df_ball_positions = pd.DataFrame(rows, columns=["x1", "y1", "x2", "y2"])

        df_ball_positions["ball_hit"] = 0

        df_ball_positions["mid_y"] = (df_ball_positions["y1"] + df_ball_positions["y2"]) / 2
        df_ball_positions["mid_y_rolling_mean"] = (
            df_ball_positions["mid_y"].rolling(window=5, min_periods=1, center=False).mean()
        )
        df_ball_positions["delta_y"] = df_ball_positions["mid_y_rolling_mean"].diff()
        minimum_change_frames_for_hit = 25
        for i in range(1, len(df_ball_positions) - int(minimum_change_frames_for_hit * 1.2)):
            negative_position_change = df_ball_positions["delta_y"].iloc[i] > 0 and df_ball_positions["delta_y"].iloc[i + 1] < 0
            positive_position_change = df_ball_positions["delta_y"].iloc[i] < 0 and df_ball_positions["delta_y"].iloc[i + 1] > 0

            if negative_position_change or positive_position_change:
                change_count = 0
                for change_frame in range(i + 1, i + int(minimum_change_frames_for_hit * 1.2) + 1):
                    negative_follow = df_ball_positions["delta_y"].iloc[i] > 0 and df_ball_positions["delta_y"].iloc[change_frame] < 0
                    positive_follow = df_ball_positions["delta_y"].iloc[i] < 0 and df_ball_positions["delta_y"].iloc[change_frame] > 0

                    if negative_position_change and negative_follow:
                        change_count += 1
                    elif positive_position_change and positive_follow:
                        change_count += 1

                if change_count > minimum_change_frames_for_hit - 1:
                    df_ball_positions.loc[i, "ball_hit"] = 1

        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions["ball_hit"] == 1].index.tolist()

        return frame_nums_with_ball_hits

    # ---------- Drawing & Export ----------
    def draw_bboxes(
        self,
        video_frames: List[np.ndarray],
        ball_detections: List[Dict[int, List[float]]],
        detection_mask: Optional[List[bool]] = None,
        detected_color: Tuple[int, int, int] = (0, 255, 0),
        interpolated_color: Tuple[int, int, int] = (0, 165, 255),
    ) -> List[np.ndarray]:
        output_video_frames: List[np.ndarray] = []
        total = len(video_frames)
        for idx in range(total):
            frame = video_frames[idx]
            ball_dict = ball_detections[idx] if idx < len(ball_detections) else {}
            detected = False
            if detection_mask is not None and idx < len(detection_mask):
                detected = detection_mask[idx]
            elif 1 in ball_dict:
                detected = True

            bbox = ball_dict.get(1)
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                color = detected_color if detected else interpolated_color
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = "Ball" if detected else "Ball (interp)"
                cv2.putText(
                    frame,
                    label,
                    (int(x1), max(0, int(y1) - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )
            output_video_frames.append(frame)

        return output_video_frames

