# tennis_analysis_v1.0/backend/detectors/detect_players.py

# 任務：利用 YOLO 或其他模型偵測所有人（bounding boxes）

# 這裡輸入是 影格 (frame)，輸出應是：

# [
#     {"bbox": [x1, y1, x2, y2], "conf": 0.9, "class": "person"},
#     ...
# ]


# detectors/detect_players.py
from ultralytics import YOLO
import numpy as np
from typing import List, Dict, Any

class PlayerDetector:
    def __init__(self, model_path: str = "yolov8s.pt"):
        """
        初始化 YOLO 模型
        """
        self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        偵測影格中的所有人
        輸入:
            frame: np.ndarray (單張畫面)
        輸出:
            list of dicts -> [{'bbox': [x, y, w, h], 'conf': 置信度}]
        """
        results = self.model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            if results.names[cls_id] != "person":
                continue  # 只取人

            x1, y1, x2, y2 = map(float, box.xyxy[0])
            w, h = x2 - x1, y2 - y1

            detections.append({
                "bbox": [x1, y1, w, h],
                "conf": conf
            })

        return detections


def serialize_player_frames(player_detections: List[Dict[str, List[float]]]) -> List[List[Dict[str, Any]]]:
    """Convert tracker output into a JSON-friendly list per frame."""
    serialized_frames: List[List[Dict[str, Any]]] = []
    for frame_dict in player_detections:
        players_payload: List[Dict[str, Any]] = []
        for track_id, bbox in frame_dict.items():
            team = None
            if track_id == "PLAYER_TOP":
                team = "top"
            elif track_id == "PLAYER_BOTTOM":
                team = "bottom"
            else:
                team = "unknown"

            players_payload.append(
                {
                    "id": track_id,
                    "bbox": [float(v) for v in bbox],
                    "team": team,
                }
            )
        serialized_frames.append(players_payload)

    return serialized_frames
