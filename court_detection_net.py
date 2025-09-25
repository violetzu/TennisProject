"""球場偵測模組：輸出球場關鍵點與單應矩陣。"""

import cv2
import numpy as np
import torch
from tracknet import BallTrackerNet
import torch.nn.functional as F
from tqdm import tqdm
from postprocess import refine_kps
from homography import get_trans_matrix, refer_kps

class CourtDetectorNet():
    def __init__(self, path_model=None,  device='cuda'):
        """載入訓練好的網路並設定推論裝置。"""
        self.model = BallTrackerNet(out_channels=15)
        self.device = device
        if path_model:
            self.model.load_state_dict(torch.load(path_model, map_location=device))
            self.model = self.model.to(device)
            self.model.eval()
            
    def infer_model(self, frames):
        """逐格推論球場關鍵點並估計單應矩陣。"""
        output_width = 640
        output_height = 360

        
        kps_res = []
        matrixes_res = []
        for num_frame, image in enumerate(tqdm(frames)):
            h, w = image.shape[:2]   #依影像解析度變化
            scale_x = w / output_width
            scale_y = h / output_height

            img = cv2.resize(image, (output_width, output_height))
            # 正規化影像並整理成模型輸入格式
            inp = (img.astype(np.float32) / 255.)
            inp = torch.tensor(np.rollaxis(inp, 2, 0))
            inp = inp.unsqueeze(0)

            out = self.model(inp.float().to(self.device))[0]
            pred = F.sigmoid(out).detach().cpu().numpy()

            points = []
            for kps_num in range(14):
                heatmap = (pred[kps_num]*255).astype(np.uint8)
                ret, heatmap = cv2.threshold(heatmap, 170, 255, cv2.THRESH_BINARY)
                # 以霍夫圓找出各個球場關鍵點
                circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=2,
                                           minRadius=10, maxRadius=25)
                if circles is not None:
                    x_pred = circles[0][0][0]*scale_x
                    y_pred = circles[0][0][1]*scale_y
                    if kps_num not in [8, 12, 9]:
                        # 大部分點再透過局部影像進行線段交會微調
                        x_pred, y_pred = refine_kps(image, int(y_pred), int(x_pred), crop_size=40)
                    points.append((x_pred, y_pred))                
                else:
                    points.append(None)

            matrix_trans = get_trans_matrix(points) 
            points = None
            if matrix_trans is not None:
                points = cv2.perspectiveTransform(refer_kps, matrix_trans)
                matrix_trans = cv2.invert(matrix_trans)[1]
            kps_res.append(points)
            matrixes_res.append(matrix_trans)
            
        return matrixes_res, kps_res    
