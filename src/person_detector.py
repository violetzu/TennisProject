"""選手偵測與追蹤：利用 YOLO 姿態模型抓取場上選手。"""

import cv2
import numpy as np
import torch
from scipy.spatial import distance
from tqdm import tqdm
from ultralytics import YOLO

from .court_reference import CourtReference


class PersonDetector():
    def __init__(self, device="cpu", model_name="../model/yolo/yolov8n-pose.pt"):
        self.device = device
        self.detection_model = YOLO(model_name)
        self.detection_model.to(self.device)

        self.court_ref = CourtReference()
        self.ref_top_court = self.court_ref.get_court_mask(2)     # 1/0 mask
        self.ref_bottom_court = self.court_ref.get_court_mask(1)  # 1/0 mask

        # 召回友善的預設
        self.default_imgsz = 1280
        self.default_conf = 0.2

        self.point_person_top = None
        self.point_person_bottom = None
        self.counter_top = 0
        self.counter_bottom = 0

    def detect(self, image, person_min_score=None, imgsz=None):
        """Run YOLO pose inference on the provided image."""
        conf_thr = self.default_conf if person_min_score is None else person_min_score
        imgsz = self.default_imgsz if imgsz is None else imgsz

        with torch.no_grad():
            results = self.detection_model.predict(
                image,
                conf=conf_thr,
                imgsz=imgsz,
                device=self.device,
                classes=[0],          # 只抓 person
                agnostic_nms=True,
                verbose=False
            )

        persons_boxes, probs, keypoints = [], [], []
        if results:
            result = results[0]
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
            else:
                boxes, scores = [], []
            if getattr(result, "keypoints", None) is not None and result.keypoints is not None:
                kps = result.keypoints.xy.cpu().numpy()
            else:
                kps = []

            for idx in range(len(boxes)):
                persons_boxes.append(boxes[idx])
                probs.append(float(scores[idx]))
                keypoints.append(kps[idx] if len(kps) > idx else None)
        return persons_boxes, probs, keypoints

    @staticmethod
    def _min_h_by_y(y, h_img):
        # 視角自適應最小高度：遠端允許更小
        # 你可微調係數以符合鏡頭視角
        return max(12, int(0.03 * h_img + (y / h_img) * 40))

    def detect_top_and_bottom_players(self, image, inv_matrix, filter_players=False):
        # 取得從小地圖回到影像座標的單應矩陣
        matrix = cv2.invert(inv_matrix)[1]

        # 生成對應到影像座標系的上下半場 ROI mask（值為 0 或 1）
        H, W = image.shape[:2]
        mask_top_court = cv2.warpPerspective(self.ref_top_court, matrix, (W, H), flags=cv2.INTER_NEAREST)
        mask_bottom_court = cv2.warpPerspective(self.ref_bottom_court, matrix, (W, H), flags=cv2.INTER_NEAREST)

        # 先做寬鬆偵測（低 conf、高解析度）
        bboxes, probs, keypoints = self.detect(image, person_min_score=self.default_conf, imgsz=self.default_imgsz)

        # 依 ROI 與「視角自適應最小高度」過濾
        cand_top, cand_bottom = [], []
        for bbox, score, kps in zip(bboxes, probs, keypoints):
            x1, y1, x2, y2 = bbox.astype(int)
            cx = (x1 + x2) // 2
            cy = y2  # 以腳底中心當作定位
            cx = int(np.clip(cx, 0, W - 1))
            cy = int(np.clip(cy - 1, 0, H - 1))
            h_box = y2 - y1

            # 視角自適應最小高度：遠端別誤刪
            if h_box < self._min_h_by_y(cy, H):
                continue

            # ROI：只保留落在球場區域內的框，排除觀眾席
            in_top = mask_top_court[cy, cx] == 1
            in_bottom = mask_bottom_court[cy, cx] == 1
            if not (in_top or in_bottom):
                continue

            record = (bbox, (cx, cy), score, kps)
            if in_top:
                cand_top.append(record)
            if in_bottom:
                cand_bottom.append(record)

        # 各半場各挑一個「最佳球員」
        def pick_one(cands, which="top"):
            if not cands:
                return []
            # 兩種策略綜合：先以分數排序，若多個則挑離該半場中心最近的
            cands = sorted(cands, key=lambda x: x[2], reverse=True)
            # 半場「幾何中心」：用 CourtReference 提供的 key_points 做一次透視映射
            refer_kps = np.array(self.court_ref.key_points[12:], dtype=np.float32).reshape((-1, 1, 2))
            trans_kps = cv2.perspectiveTransform(refer_kps, matrix)
            center_top = trans_kps[0][0]
            center_bottom = trans_kps[1][0]
            center = center_top if which == "top" else center_bottom

            # 從分數前幾名中，挑距離中心最近的一個
            topk = cands[:3] if len(cands) > 3 else cands
            dists = [np.hypot(c[1][0] - center[0], c[1][1] - center[1]) for c in topk]
            chosen = topk[int(np.argmin(dists))]
            # 回傳格式 (bbox, point, kps)
            return [ (chosen[0], chosen[1], chosen[3]) ]

        person_bboxes_top = pick_one(cand_top, "top")
        person_bboxes_bottom = pick_one(cand_bottom, "bottom")

        if filter_players:
            person_bboxes_top, person_bboxes_bottom = self.filter_players(
                person_bboxes_top, person_bboxes_bottom, matrix
            )
        return person_bboxes_top, person_bboxes_bottom


    def filter_players(self, person_bboxes_top, person_bboxes_bottom, matrix):
        """
        Leave one person at the top and bottom of the tennis court
        """
        refer_kps = np.array(self.court_ref.key_points[12:], dtype=np.float32).reshape((-1, 1, 2))
        trans_kps = cv2.perspectiveTransform(refer_kps, matrix)
        center_top_court = trans_kps[0][0]
        center_bottom_court = trans_kps[1][0]
        if len(person_bboxes_top) > 1:
            dists = [distance.euclidean(x[1], center_top_court) for x in person_bboxes_top]
            ind = dists.index(min(dists))
            person_bboxes_top = [person_bboxes_top[ind]]
        if len(person_bboxes_bottom) > 1:
            dists = [distance.euclidean(x[1], center_bottom_court) for x in person_bboxes_bottom]
            ind = dists.index(min(dists))
            person_bboxes_bottom = [person_bboxes_bottom[ind]]
        return person_bboxes_top, person_bboxes_bottom
    
    def track_players(self, frames, matrix_all, filter_players=False):
        persons_top = []
        persons_bottom = []
        min_len = min(len(frames), len(matrix_all))
        for num_frame in tqdm(range(min_len)):
            img = frames[num_frame]
            if matrix_all[num_frame] is not None:
                inv_matrix = matrix_all[num_frame]
                # 將偵測結果依單應矩陣映射到上下場區
                person_top, person_bottom = self.detect_top_and_bottom_players(img, inv_matrix, filter_players)
            else:
                person_top, person_bottom = [], []
            persons_top.append(person_top)
            persons_bottom.append(person_bottom)
        return persons_top, persons_bottom    

