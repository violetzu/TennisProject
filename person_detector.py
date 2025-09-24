"""選手偵測與追蹤：利用 YOLO 姿態模型抓取場上選手。"""

import cv2
import numpy as np
import torch
from scipy.spatial import distance
from tqdm import tqdm
from ultralytics import YOLO

from court_reference import CourtReference


class PersonDetector():
    def __init__(self, device="cpu", model_name="yolov8n-pose.pt"):
        """Detect tennis players using YOLO pose models.

        Args:
            device (str): Torch device string to run inference on.
            model_name (str): Name or path to the YOLO pose model weights.
        """
        self.device = device
        self.detection_model = YOLO(model_name)
        # Ensure the model runs on the desired device.
        self.detection_model.to(self.device)
        self.court_ref = CourtReference()
        self.ref_top_court = self.court_ref.get_court_mask(2)
        self.ref_bottom_court = self.court_ref.get_court_mask(1)
        self.point_person_top = None
        self.point_person_bottom = None
        self.counter_top = 0
        self.counter_bottom = 0

        
    def detect(self, image, person_min_score=0.5):
        """Run YOLO pose inference on the provided image.

        透過 YOLO 模型推論影像中的人物框與關節點。

        Returns:
            tuple: bounding boxes, probabilities and keypoints for detected persons.
        """
        with torch.no_grad():
            results = self.detection_model.predict(
                image,
                conf=person_min_score,
                device=self.device,
                classes=[0],  # person class
                verbose=False
            )

        persons_boxes = []
        probs = []
        keypoints = []
        if results:
            result = results[0]
            boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
            scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else []
            kps = result.keypoints.xy.cpu().numpy() if result.keypoints is not None else []

            for idx in range(len(boxes)):
                persons_boxes.append(boxes[idx])
                probs.append(scores[idx])
                if len(kps) > idx:
                    keypoints.append(kps[idx])
                else:
                    keypoints.append(None)
        return persons_boxes, probs, keypoints

    def detect_top_and_bottom_players(self, image, inv_matrix, filter_players=False):
        matrix = cv2.invert(inv_matrix)[1]
        mask_top_court = cv2.warpPerspective(self.ref_top_court, matrix, image.shape[1::-1])
        mask_bottom_court = cv2.warpPerspective(self.ref_bottom_court, matrix, image.shape[1::-1])
        person_bboxes_top, person_bboxes_bottom = [], []

        bboxes, probs, keypoints = self.detect(image, person_min_score=0.5)
        if len(bboxes) > 0:
            detections = []
            for bbox, score, kps in zip(bboxes, probs, keypoints):
                person_point = [int((bbox[2] + bbox[0]) / 2), int(bbox[3])]
                # 以腳底中心當作小地圖上的選手位置
                detections.append((bbox, person_point, score, kps))

            # keep only two most confident detections overall
            detections = sorted(detections, key=lambda x: x[2], reverse=True)[:2]

            for detection in detections:
                bbox, point, _, kps = detection
                x_idx = np.clip(point[0], 0, mask_top_court.shape[1] - 1)
                y_idx = np.clip(point[1] - 1, 0, mask_top_court.shape[0] - 1)
                top_mask_val = mask_top_court[y_idx, x_idx]
                bottom_mask_val = mask_bottom_court[y_idx, x_idx]
                record = (bbox, point, kps)
                if top_mask_val == 1:
                    person_bboxes_top.append(record)
                if bottom_mask_val == 1:
                    person_bboxes_bottom.append(record)

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

