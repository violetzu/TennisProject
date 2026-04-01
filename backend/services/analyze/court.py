"""
球場偵測 (Court Detection)

對外 API：
  - CourtPoints                   一幀的球場關鍵點集合（像素座標 + Homography）
      .corners                    四個雙打角點 [TL, TR, BL, BR]，shape (4, 2)
      .net_y                      球網中心的影像 y 像素座標
      .center_line_px             中線兩端像素座標，若未偵測到回傳 None
  - CourtDetectior.detect()       對單幀執行場地偵測，回傳 CourtPoints 或 None
  - project_to_world()            像素座標 → 世界公尺座標（使用 CourtPoints.H）
  - court_side_x_at_y()           給定影像 y，插值左右雙打邊線 x，供球員邊界篩選用
  - draw_court()                  在影像上繪製完整球場線

偵測策略：
  - 使用 YOLO Pose 模型偵測球場 14 個關鍵點
  - 取信心度最高的結果，以 4 個雙打角點（TL/TR/BL/BR）計算 Homography
  - 球網端點無 YOLO 關鍵點，由 H 逆投影世界座標補算
  - 偵測失敗時記錄場景切換，last_valid_H 保留供後續幀投影使用
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# 球場實際尺寸（公尺）
# ─────────────────────────────────────────────────────────────────────────────
COURT_LENGTH_M   = 23.77   # 球場全長（底線到底線）
DOUBLES_WIDTH_M  = 10.97   # 雙打場寬

# Homography 目標世界座標（TL, TR, BL, BR）
# 座標系：BL 為原點，x 向右，y 向遠端底線
WORLD_CORNERS = np.array([
    [0.0,            COURT_LENGTH_M],   # TL
    [DOUBLES_WIDTH_M, COURT_LENGTH_M],  # TR
    [0.0,            0.0],              # BL
    [DOUBLES_WIDTH_M, 0.0],             # BR
], dtype=np.float32)

_NET_Y_M         = COURT_LENGTH_M / 2                         # 11.885 m  球網 y 座標
SINGLES_WIDTH_M  = 8.23
_SINGLES_L_X     = (DOUBLES_WIDTH_M - SINGLES_WIDTH_M) / 2   # 1.37 m   左單打邊線 x
_SINGLES_R_X     = DOUBLES_WIDTH_M - _SINGLES_L_X             # 9.60 m   右單打邊線 x
_SERVICE_DIST_M  = 6.40                                       # 球網到發球線距離
_SERVICE_Y_NEAR  = _NET_Y_M - _SERVICE_DIST_M                 # 5.485 m  近端發球線 y
_SERVICE_Y_FAR   = _NET_Y_M + _SERVICE_DIST_M                 # 18.285 m 遠端發球線 y
_CENTER_X        = DOUBLES_WIDTH_M / 2.0                      # 5.485 m  中線 x

COURT_CONF_TH = 0.9  # 場地偵測最低信心值（球場靜止時 0.95 足夠）


# ─────────────────────────────────────────────────────────────────────────────
# 資料結構
# ─────────────────────────────────────────────────────────────────────────────
# YOLO Pose 模型輸出的物理索引（模型實際輸出順序，≠ 標註者的邏輯編號）：
#   遠端底線:   [0]=TL  [1]=單打左  [2]=單打右  [3]=TR
#   近端底線:   [4]=BL  [5]=單打左  [6]=單打右  [7]=BR
#   遠端發球線: [8]=左  [9]=中      [10]=右
#   近端發球線: [11]=右 [12]=中     [13]=左   ← 注意近端 R/C/L 反序
#   球網：模型無此點，由 detect() 用 H 投影補算

@dataclass
class CourtPoints:
    """一幀的球場關鍵點集合（像素座標）。
    每個欄位都是 (2,) float32 的像素座標 [x, y]。
    H 為本幀的 Homography（影像 → 世界公尺座標）。
    """
    # ── YOLO 偵測點（遠端底線） ─────────────────────────────────────────────
    tl:             np.ndarray  # 0  左雙打×遠底線
    far_singles_l:  np.ndarray  # 1  左單打×遠底線
    far_singles_r:  np.ndarray  # 2  右單打×遠底線
    tr:             np.ndarray  # 3  右雙打×遠底線
    # ── YOLO 偵測點（近端底線） ─────────────────────────────────────────────
    bl:             np.ndarray  # 4  左雙打×近底線
    near_singles_l: np.ndarray  # 5  左單打×近底線
    near_singles_r: np.ndarray  # 6  右單打×近底線
    br:             np.ndarray  # 7  右雙打×近底線
    # ── YOLO 偵測點（發球線） ──────────────────────────────────────────────
    service_far_l:  np.ndarray  # 8  左單打×遠發球線
    service_far_c:  np.ndarray  # 9  中線×遠發球線
    service_far_r:  np.ndarray  # 10 右單打×遠發球線
    service_near_r: np.ndarray  # 11 右單打×近發球線
    service_near_c: np.ndarray  # 12 中線×近發球線
    service_near_l: np.ndarray  # 13 左單打×近發球線
    # ── 補算點（球網，由 H 投影） ──────────────────────────────────────────
    net_l:          np.ndarray  # 14 球網左端（左雙打邊線）
    net_r:          np.ndarray  # 15 球網右端（右雙打邊線）
    # ── Homography ─────────────────────────────────────────────────────────
    H:              np.ndarray  # (3,3) float32，影像像素 → 世界公尺座標

    @property
    def corners(self) -> np.ndarray:
        """四個雙打角點 [TL, TR, BL, BR]，shape (4, 2) float32。
        可直接傳入 cv2.findHomography / perspectiveTransform。"""
        return np.array([self.tl, self.tr, self.bl, self.br], dtype=np.float32)

    @property
    def net_y(self) -> float:
        """球網中心在影像上的 y 像素座標（左右端 y 的平均值）。"""
        return float((self.net_l[1] + self.net_r[1]) / 2)

    @property
    def center_line_px(self) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """中線兩端像素座標 ((x1,y1), (x2,y2))。
        若兩端皆為未標注的 (0,0) 則回傳 None。"""
        p1 = (float(self.service_far_c[0]),  float(self.service_far_c[1]))
        p2 = (float(self.service_near_c[0]), float(self.service_near_c[1]))
        if p1 == (0.0, 0.0) and p2 == (0.0, 0.0):
            return None
        return (p1, p2)


# ─────────────────────────────────────────────────────────────────────────────
# 投影 / 幾何工具
# ─────────────────────────────────────────────────────────────────────────────

def project_to_world(
    pos: Tuple[float, float],
    H: np.ndarray,
) -> Optional[Tuple[float, float]]:
    """像素座標 → 世界公尺座標（使用 CourtPoints.H 或 last_valid_H）。
    世界系統：BL=(0,0)，BR=(10.97,0)，TL=(0,23.77)，TR=(10.97,23.77)，網 Y=11.885m。
    """
    pt = np.array([[[pos[0], pos[1]]]], dtype=np.float32)
    wpt = cv2.perspectiveTransform(pt, H)
    return (float(wpt[0, 0, 0]), float(wpt[0, 0, 1]))


def court_side_x_at_y(pts: CourtPoints, y: float) -> Tuple[float, float]:
    """給定影像 y 座標，線性插值左、右雙打邊線的 x 座標。
    用於判斷球員是否在球場範圍內。
    Returns: (left_x, right_x)
    """
    def _x_at(p_top: np.ndarray, p_bot: np.ndarray) -> float:
        """沿邊線在 y 處插值 x。p_top 為遠端點，p_bot 為近端點。"""
        dy = float(p_bot[1]) - float(p_top[1])
        if abs(dy) < 1e-6:
            return float(p_top[0])
        t = (y - float(p_top[1])) / dy
        return float(p_top[0]) + t * (float(p_bot[0]) - float(p_top[0]))

    return _x_at(pts.tl, pts.bl), _x_at(pts.tr, pts.br)



def _inv_H(H: np.ndarray) -> np.ndarray:
    """計算 Homography 逆矩陣：世界公尺座標 → 影像像素。"""
    return np.linalg.inv(H.astype(np.float64)).astype(np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# 繪製
# ─────────────────────────────────────────────────────────────────────────────

def draw_court(frame: np.ndarray, court_pts: CourtPoints) -> None:
    """在影像上繪製完整球場線（底線、邊線、單打邊線、發球線、中線、球網）。
    所有線段定義來自 CourtDetectior.LINE_PAIRS。
    """
    color, thickness = (0, 0, 255), 2
    for a, b in CourtDetectior.LINE_PAIRS:
        p1 = tuple(getattr(court_pts, a).astype(int).tolist())
        p2 = tuple(getattr(court_pts, b).astype(int).tolist())
        cv2.line(frame, p1, p2, color, thickness)


# ─────────────────────────────────────────────────────────────────────────────
# 場地偵測器
# ─────────────────────────────────────────────────────────────────────────────

class CourtDetectior:
    """封裝場地偵測狀態 + 場景切換追蹤。

    使用方式::

        court = CourtDetectior(model)
        for idx, frame in enumerate(frames):
            court_pts = court.detect(frame, idx)
            if court_pts is not None:
                # 本幀 H：court_pts.H
                # 投影球員位置：project_to_world(pos, court_pts.H)
                # 球場分析失效時取最後已知 H：court.last_valid_H
    """

    # 球場線段定義：每個 tuple 為 CourtPoints 的兩個屬性名稱
    # draw_court() 迭代此清單繪製所有線段
    LINE_PAIRS: List[Tuple[str, str]] = [
        ('tl',            'tr'),              # 遠端雙打底線
        ('bl',            'br'),              # 近端雙打底線
        ('tl',            'bl'),              # 左雙打邊線
        ('tr',            'br'),              # 右雙打邊線
        ('far_singles_l', 'near_singles_l'),  # 左單打邊線
        ('far_singles_r', 'near_singles_r'),  # 右單打邊線
        ('service_far_l', 'service_far_r'),   # 遠端發球線
        ('service_near_l','service_near_r'),  # 近端發球線
        ('service_far_c', 'service_near_c'),  # 中線
        ('net_l',         'net_r'),           # 球網
    ]

    def __init__(self, model):
        self.model = model
        # H 在偵測失敗時清為 None，用於判斷場景切換
        self.H: Optional[np.ndarray] = None
        # last_valid_H 在偵測失敗時保留，供 analysis / aggregate 投影落點用
        self.last_valid_H: Optional[np.ndarray] = None
        self.scene_cuts: List[int] = []       # 場景切換的 frame index 列表
        self.scene_cut_set: set = set()       # 同上，set 版本供 O(1) 查詢

    def detect(self, frame: np.ndarray, frame_idx: int) -> Optional[CourtPoints]:
        """對單幀執行場地偵測。

        成功 → 回傳 CourtPoints（含 H）並更新 last_valid_H。
        失敗且前一幀有效 → 記錄場景切換，self.H 清為 None。
        Returns: CourtPoints 或 None。
        """
        results = self.model.predict(
            source=frame, imgsz=320, conf=COURT_CONF_TH, verbose=False, half=True,
        )
        court_pts = self._build(results)

        if court_pts is None:
            if self.H is not None:           # 前一幀有效 → 這幀斷了 = 場景切換
                self.scene_cuts.append(frame_idx)
                self.scene_cut_set.add(frame_idx)
            self.H = None
            return None

        return court_pts

    def _build(self, results) -> Optional[CourtPoints]:
        """從 YOLO 推論結果建構 CourtPoints；任何步驟失敗回傳 None。"""
        if not results:
            return None
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return None
        kps = r.keypoints
        if kps is None:
            return None

        # 取信心度最高的偵測結果
        best_idx = int(r.boxes.conf.argmax())
        xy = kps.xy[best_idx].cpu().numpy().astype(np.float32)  # (14, 2)

        # 四個雙打角點（TL=0, TR=3, BL=4, BR=7）
        corners = xy[[0, 3, 4, 7]]
        if np.any(np.all(corners == 0, axis=1)):   # (0,0) = 未標注點，無法建立 H
            return None

        H, _ = cv2.findHomography(corners, WORLD_CORNERS, cv2.RANSAC, 5.0)
        if H is None:
            return None
        H = H.astype(np.float32)
        self.H = H
        self.last_valid_H = H

        # 球網端點：YOLO 無此點，由 H 逆投影世界座標補算
        H_inv = _inv_H(H)
        world_net = np.array([[[0.0, _NET_Y_M], [DOUBLES_WIDTH_M, _NET_Y_M]]], dtype=np.float32)
        net_pts = cv2.perspectiveTransform(world_net, H_inv)[0]  # (2, 2)

        return CourtPoints(
            tl=xy[0],  far_singles_l=xy[1],  far_singles_r=xy[2],  tr=xy[3],
            bl=xy[4],  near_singles_l=xy[5], near_singles_r=xy[6], br=xy[7],
            service_far_l=xy[8],  service_far_c=xy[9],  service_far_r=xy[10],
            service_near_r=xy[11], service_near_c=xy[12], service_near_l=xy[13],
            net_l=net_pts[0], net_r=net_pts[1],
            H=H,
        )
