# ball_detector_yolo.py
from ultralytics import YOLO
import numpy as np
from tqdm import tqdm
from scipy.spatial import distance

class BallDetector:
    def __init__(self, device="cpu", path_model = "./model/ball/yolov8_ball_09250900_best.pt", conf: float = 0.25, max_dist: float = 80.0):
        """
        以 YOLOv8 偵測網球位置（不縮放輸入影像）。
        :param path_model: 你重訓後的 YOLOv8 權重（.pt）
        :param device: 'cuda' 或 'cpu'
        :param conf: 置信度閾值
        :param max_dist: 與前一偵球中心的最大允許位移（像素），過大則視為新起點
        """
        self.model = YOLO(path_model)
        self.model.to(device)
        self.device = device
        self.conf = conf
        self.max_dist = max_dist
        self.prev_xy = None  # (x, y) of previous frame
        # 嘗試找出「ball」類別 id；若沒標籤名稱就預設 0
        self.names = self.model.names if hasattr(self.model, 'names') else {}
        self.ball_cls = None
        for k, v in self.names.items():
            if str(v).lower() in ['ball', 'tennis_ball', 'tennis-ball', 'tennisball']:
                self.ball_cls = int(k)
                break
        if self.ball_cls is None:
            # 若你的資料只有單一類別(球)，通常就是 0
            self.ball_cls = 0

    def infer_model(self, frames):
        """
        :param frames: List[np.ndarray] BGR 影格（任意解析度，直接丟原圖）
        :return: List[(x, y)] 每一偵的球中心座標（像素；找不到時為 (None, None)）
        """
        ball_track = []
        self.prev_xy = None

        for img in tqdm(frames, unit="frame"):
            x, y = self._detect_single(img)
            ball_track.append((x, y))
            self.prev_xy = (x, y) if x is not None and y is not None else self.prev_xy

        return ball_track

    def _detect_single(self, img):
        """
        回傳該影格的球中心 (x, y)。若有多顆候選，優先挑：1) 距離前一偵最近且小於 max_dist；否則 2) 置信度最高。
        """
        # Ultralytics 會自動 letterbox 並把輸出座標映回原圖尺度；不需手動縮放
        results = self.model.predict(img, conf=self.conf, verbose=False, device=self.device)[0]

        # 取出屬於球類別的框
        bxs = []
        if results.boxes is not None and len(results.boxes) > 0:
            for b in results.boxes:
                cls_id = int(b.cls.item()) if b.cls is not None else -1
                if cls_id == self.ball_cls:
                    xyxy = b.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]（已是原圖座標）
                    conf = float(b.conf.item()) if b.conf is not None else 0.0
                    cx = 0.5 * (xyxy[0] + xyxy[2])
                    cy = 0.5 * (xyxy[1] + xyxy[3])
                    bxs.append((cx, cy, conf))

        if not bxs:
            return (None, None)

        # 若上一偵有點，先用「最近且距離 < max_dist」做優先選取
        if self.prev_xy is not None and all(v is not None for v in self.prev_xy):
            px, py = self.prev_xy
            bxs_sorted = sorted(
                bxs, key=lambda t: distance.euclidean((t[0], t[1]), (px, py))
            )
            nearest = bxs_sorted[0]
            if distance.euclidean((nearest[0], nearest[1]), (px, py)) <= self.max_dist:
                return (float(nearest[0]), float(nearest[1]))
            # 否則當作新一拍起點，退回「最高 conf」
            best = max(bxs, key=lambda t: t[2])
            return (float(best[0]), float(best[1]))
        else:
            # 第一偵／上一偵遺失：選最高置信度
            best = max(bxs, key=lambda t: t[2])
            return (float(best[0]), float(best[1]))
