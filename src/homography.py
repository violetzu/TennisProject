"""單應矩陣估計：根據球場關鍵點建立投影轉換。"""

from .court_reference import CourtReference
import numpy as np
import cv2
from scipy.spatial import distance

court_ref = CourtReference()
refer_kps = np.array(court_ref.key_points, dtype=np.float32).reshape((-1, 1, 2))

court_conf_ind = {}
for i in range(len(court_ref.court_conf)):
    conf = court_ref.court_conf[i+1]
    inds = []
    for j in range(4):
        inds.append(court_ref.key_points.index(conf[j]))
    court_conf_ind[i+1] = inds
# 預先記錄每組配置在 key_points 中的索引，以加速比對

def get_trans_matrix(points):
    """
    Determine the best homography matrix from court points

    從偵測到的關鍵點組合挑選最合適的單應矩陣。
    """
    matrix_trans = None
    dist_max = np.inf
    for conf_ind in range(1, 13):
        conf = court_ref.court_conf[conf_ind]

        inds = court_conf_ind[conf_ind]
        inters = [points[inds[0]], points[inds[1]], points[inds[2]], points[inds[3]]]
        if None not in inters:
            # 以四點組合計算單應矩陣，並比較外插點的對齊誤差
            matrix, _ = cv2.findHomography(np.float32(conf), np.float32(inters), method=0)
            trans_kps = cv2.perspectiveTransform(refer_kps, matrix).squeeze(1)
            dists = []
            for i in range(12):
                if i not in inds and points[i] is not None:
                    dists.append(distance.euclidean(points[i], trans_kps[i]))
            dist_median = np.mean(dists)
            if dist_median < dist_max:
                matrix_trans = matrix
                dist_max = dist_median
    return matrix_trans 
