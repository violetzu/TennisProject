# backend/trackers/player_tracker.py
"""
PlayerTracker using YOLOv11 + Supervision + ByteTrack (2025 最新、最穩版)
修復：BoxAnnotator 無 text_thickness + 之前 TypeError
完全相容舊 API
"""
from ultralytics import YOLO
import cv2
import pickle
from typing import List, Dict
import numpy as np

from supervision import Detections, BoxAnnotator, ByteTrack

from services.pipeline.utils import measure_distance, get_center_of_bbox

# 更穩的 track_id 轉 int：逐層展開直到找到第一個純量
def safe_track_id(value):
    def _extract_scalar(v):
        if isinstance(v, (int, np.integer)):
            return int(v)
        if isinstance(v, (float, np.floating)):
            return int(v)
        if hasattr(v, "item"):
            try:
                return _extract_scalar(v.item())
            except (TypeError, ValueError):
                pass
        if isinstance(v, np.ndarray):
            if v.size == 0:
                raise ValueError("empty array")
            for item in v.flat:
                try:
                    return _extract_scalar(item)
                except (TypeError, ValueError):
                    continue
            raise ValueError("no scalar in array")
        if isinstance(v, (list, tuple)):
            if len(v) == 0:
                raise ValueError("empty sequence")
            for item in v:
                try:
                    return _extract_scalar(item)
                except (TypeError, ValueError):
                    continue
            raise ValueError("no scalar in sequence")
        return int(v)

    try:
        return _extract_scalar(value)
    except Exception:
        return 0

class PlayerTracker:
    def __init__(self, model_path: str = "yolo11s.pt", conf: float = 0.10, keepalive_frames: int = 45):
        self.model = YOLO(model_path)
        self.conf = float(conf)
        self.keepalive_frames = int(keepalive_frames)
        
        # SV ByteTrack 新參數名（無警告）
        self.tracker = ByteTrack(
            track_activation_threshold=self.conf,
            lost_track_buffer=300,
            minimum_matching_threshold=0.8,
            frame_rate=30
        )
        
        # 修復：BoxAnnotator 只用 thickness（無 text_thickness）
        self.box_annotator = BoxAnnotator(thickness=2)

    def detect_and_track_frames(self, frames: List, read_from_stub: bool = False, stub_path: str = None) -> List[Dict[str, list]]:
        player_detections: List[Dict[str, list]] = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        for frame in frames:
            results = self.model(frame, conf=self.conf, classes=[0], verbose=False)[0]
            detections = self._yolo_to_sv(results)

            tracks = self.tracker.update_with_detections(detections)

            frame_dict: Dict[str, list] = {}
            if tracks is not None and len(tracks) > 0 and tracks.tracker_id is not None:
                for box, track_id in zip(tracks.xyxy, tracks.tracker_id):
                    x1, y1, x2, y2 = box.tolist()
                    track_id_int = safe_track_id(track_id)
                    frame_dict[str(track_id_int)] = [x1, y1, x2, y2]

            player_detections.append(frame_dict)

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(player_detections, f)

        return player_detections

    def _yolo_to_sv(self, yolo_results):
        boxes = yolo_results.boxes.xyxy.cpu().numpy() if yolo_results.boxes is not None else np.empty((0, 4))
        confidences = yolo_results.boxes.conf.cpu().numpy() if yolo_results.boxes is not None else np.empty(0)
        class_ids = yolo_results.boxes.cls.cpu().numpy() if yolo_results.boxes is not None else np.empty(0)
        return Detections(xyxy=boxes, confidence=confidences, class_id=class_ids)

    # ---------- 以下完全沿用你的舊函數 ----------
    def choose_and_filter_players(self, court_keypoints, player_detections: List[Dict[str, list]]):
        if not player_detections:
            return player_detections
        if court_keypoints is None:
            return player_detections
        if isinstance(court_keypoints, np.ndarray):
            court_keypoints = court_keypoints.reshape(-1).tolist()
        if isinstance(court_keypoints, (list, tuple)):
            if len(court_keypoints) == 0:
                return player_detections
        else:
            return player_detections

        court_ys = court_keypoints[1::2]
        top_line_y = min(court_ys)
        bottom_line_y = max(court_ys)
        mid_y = (top_line_y + bottom_line_y) / 2

        top_points = [(court_keypoints[i], court_keypoints[i+1]) for i in range(0, len(court_keypoints), 2) if court_keypoints[i+1] <= mid_y]
        bottom_points = [(court_keypoints[i], court_keypoints[i+1]) for i in range(0, len(court_keypoints), 2) if court_keypoints[i+1] > mid_y]
        if not top_points:
            top_points = [(court_keypoints[i], court_keypoints[i+1]) for i in range(0, len(court_keypoints), 2)]
        if not bottom_points:
            bottom_points = top_points

        def min_distance_to_points(bbox, points):
            center = get_center_of_bbox(bbox)
            return min(measure_distance(center, p) for p in points) if points else float("inf")

        state = {
            "top": {
                "track_id": None,
                "alias": "PLAYER_TOP",
                "last_seen": -10**9,
                "last_bbox": None,
            },
            "bottom": {
                "track_id": None,
                "alias": "PLAYER_BOTTOM",
                "last_seen": -10**9,
                "last_bbox": None,
            },
        }

        def candidate_score(bbox, points, last_bbox):
            center = get_center_of_bbox(bbox)
            anchor_dist = min(measure_distance(center, p) for p in points) if points else 0.0
            if last_bbox is None:
                return anchor_dist
            last_center = get_center_of_bbox(last_bbox)
            move_dist = measure_distance(center, last_center)
            return anchor_dist + 0.5 * move_dist

        filtered: List[Dict[str, list]] = []
        for frame_idx, frame_dict in enumerate(player_detections):
            frame_out: Dict[str, list] = {}
            candidates = {"top": [], "bottom": []}

            for tid, bbox in frame_dict.items():
                foot_y = bbox[3]
                side = "top" if foot_y <= mid_y else "bottom"
                pts = top_points if side == "top" else bottom_points
                score = candidate_score(bbox, pts, state[side]["last_bbox"])
                candidates[side].append((str(tid), bbox, score))

            used_ids = set()
            for side in ("top", "bottom"):
                record = state[side]
                alias = record["alias"]
                current_id = record["track_id"]
                if current_id is not None and current_id in frame_dict:
                    bbox = frame_dict[current_id]
                    record["last_bbox"] = bbox
                    record["last_seen"] = frame_idx
                    frame_out[alias] = bbox
                    used_ids.add(current_id)
                    continue

                if current_id is not None and frame_idx - record["last_seen"] <= self.keepalive_frames and record["last_bbox"] is not None:
                    frame_out[alias] = record["last_bbox"]
                    continue

                available = [cand for cand in candidates[side] if cand[0] not in used_ids]
                if available:
                    available.sort(key=lambda x: x[2])
                    new_id, bbox, _ = available[0]
                    record["track_id"] = new_id
                    record["last_bbox"] = bbox
                    record["last_seen"] = frame_idx
                    frame_out[alias] = bbox
                    used_ids.add(new_id)
                else:
                    record["track_id"] = None
                    record["last_bbox"] = None

            filtered.append(frame_out)

        return filtered

    def choose_players(self, court_keypoints, player_dict: Dict[str, list]):
        player_positions = []
        for track_id, bbox in player_dict.items():
            x1, y1, x2, y2 = bbox
            foot_x = (x1 + x2) / 2
            foot_y = y2
            player_positions.append((track_id, foot_x, foot_y))

        player_positions.sort(key=lambda x: x[2], reverse=True)
        chosen_ids = [p[0] for p in player_positions[:2]]
        return chosen_ids

    def draw_bboxes(self, video_frames: List, player_detections: List[Dict[str, list]]):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = map(int, bbox)
                label = "Player" if track_id not in ("PLAYER_TOP", "PLAYER_BOTTOM") else ("Player Top" if track_id == "PLAYER_TOP" else "Player Bottom")
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            output_video_frames.append(frame)
        return output_video_frames
