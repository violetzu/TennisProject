# src_llm/court_manager.py
import cv2
import numpy as np
import torch
from torchvision import trans

class CourtManager:
    def __init__(self):
        # 標準網球場尺寸 (單位：公尺)
        # 以網子中心在地面的投影點為原點 (0, 0)
        # X 軸：左右寬度，Y 軸：上下長度
        self.court_length = 23.77
        self.court_width = 10.97
        self.half_court_length = self.court_length / 2
        self.half_court_width = self.court_width / 2
        self.singles_half_width = 8.23 / 2
        self.service_line_dist = 6.40

        # 定義 14 個標準關鍵點 (對應 yastrebksv 專案的輸出順序)
        # 這裡的順序必須與你的 Court Detection 模型輸出的 Class ID 嚴格對應
        self.std_points = np.array([
            [-self.half_court_width, -self.half_court_length],    # 0: Bottom Left Corner
            [self.half_court_width, -self.half_court_length],     # 1: Bottom Right Corner
            [-self.singles_half_width, -self.half_court_length],  # 2: Bottom Singles Left
            [self.singles_half_width, -self.half_court_length],   # 3: Bottom Singles Right
            [-self.singles_half_width, -self.service_line_dist],  # 4: Bottom Service Left
            [self.singles_half_width, -self.service_line_dist],   # 5: Bottom Service Right
            [-self.singles_half_width, 0],                        # 6: Net Left (Singles)
            [self.singles_half_width, 0],                         # 7: Net Right (Singles)
            [-self.singles_half_width, self.service_line_dist],   # 8: Top Service Left
            [self.singles_half_width, self.service_line_dist],    # 9: Top Service Right
            [-self.singles_half_width, self.half_court_length],   # 10: Top Singles Left
            [self.singles_half_width, self.half_court_length],    # 11: Top Singles Right
            [-self.half_court_width, self.half_court_length],     # 12: Top Left Corner
            [self.half_court_width, self.half_court_length],      # 13: Top Right Corner
        ], dtype=np.float32)

        self.homography_matrix = None

    def detect_court_and_compute_homography(self, frame_img):
        """
        傳入一張 OpenCV 圖片 (Frame)，計算並儲存 Homography Matrix。
        注意：這需要實際的球場偵測模型。
        """
        # TODO: 在此處整合你的球場關鍵點偵測模型
        # 範例邏輯 (偽代碼):
        # keypoints = court_model(frame_img) # 預測出 14 個點的 (x, y)
        # valid_points_src = []
        # valid_points_dst = []
        # for i, pt in enumerate(keypoints):
        #     if pt.confidence > 0.5:
        #         valid_points_src.append(pt.xy)
        #         valid_points_dst.append(self.std_points[i])
        
        # 目前回傳 None，表示尚未實作模型，避免程式崩潰
        # 若你有手動標註的 4 個點，也可以在這裡寫死
        detected_points = None 

        if detected_points is None or len(detected_points) < 4:
            print("[CourtManager] 無法偵測足夠的球場關鍵點，略過座標映射。")
            self.homography_matrix = None
            return None

        src_pts = np.array(detected_points, dtype=np.float32)
        dst_pts = np.array([self.std_points[i] for i in detected_indices], dtype=np.float32)

        H, status = cv2.findHomography(src_pts, dst_pts)
        self.homography_matrix = H
        return H

    def transform_point(self, pixel_pt):
        """
        將 (image_x, image_y) 轉換為真實世界 (real_x, real_y)
        """
        if self.homography_matrix is None or pixel_pt is None:
            return None, None
            
        src_pt = np.array([pixel_pt], dtype=np.float32).reshape(-1, 1, 2)
        try:
            dst_pt = cv2.perspectiveTransform(src_pt, self.homography_matrix)
            return float(dst_pt[0][0][0]), float(dst_pt[0][0][1])
        except Exception:
            return None, None

    def get_bbox_foot_position(self, bbox):
        """
        取得 Bounding Box 的著地點 (底部中心)
        bbox: [x1, y1, x2, y2]
        """
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, y2)