"""
後處理分析輔助 (Post-processing Analysis Helpers)

包含：
  - full_interpolate()          全域球位置線性插值
  - smooth()                    滑動平均平滑
  - compute_frame_speeds_world() 逐幀球速（世界座標，km/h）
  - detect_events()             速度方向變化 → contact / bounce
  - segment_rallies()           依時間間隔 + 切鏡切割回合
  - find_winner_after()         勝利球落點
  - assign_player()             歸屬球員（top / bottom）
  - project_to_world()          像素座標 → 世界公尺座標
  - bounce_zone()               落地區域分類（發球區/底線區/網前/出界）
  - player_court_zone()         球員站位區域（net/service/baseline）
  - classify_shot_type()        擊球類型（serve/overhead/swing/unknown）

所有函式為純函式，不依賴模型或影像處理。
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .court import COURT_LENGTH_M, DOUBLES_WIDTH_M

# ── 球速範圍（ITF：最慢約 20 km/h 吊球；最快發球紀錄 263 km/h，280 留緩衝）──
MIN_BALL_SPEED_KMH = 20.0
MAX_BALL_SPEED_KMH = 280.0

# ── 事件偵測參數 ───────────────────────────────────────────────────────────────
COOLDOWN_FRAMES = 20           # 高fps下給更長冷卻（60fps ≈ 0.33s）
MIN_VEL_CHANGE_RATIO = 0.018   # 相對圖高（像素/幀）速度變化閾值，過低會把雜訊誤判
PROXIMITY_RATIO = 0.18         # 球員接近閾值（相對圖寬）；1920px × 0.18 ≈ 346px
MIN_CONTACT_VY = 2.0           # 接觸事件 Y 速度下限（近端水平球路 vy 很小，不能設太高）

# ── 回合切割 ──────────────────────────────────────────────────────────────────
RALLY_GAP_SEC = 2.5

# ── 世界座標球場幾何常數（ITF 標準，公尺）──────────────────────────────────────
_NET_Y_M        = COURT_LENGTH_M / 2      # 11.885m，球網 Y
_SERVICE_DIST_M = 6.40                    # 球網到發球線距離
_SERVICE_Y_NEAR = _NET_Y_M - _SERVICE_DIST_M  # 5.485m，近端發球線
_SERVICE_Y_FAR  = _NET_Y_M + _SERVICE_DIST_M  # 18.285m，遠端發球線
_NET_ZONE_M     = 2.0                     # 距球網 2m 以內算「網前」
_COURT_TOL_M    = 0.5                     # 邊界容差（避免邊線稍微出界就算 out）


# ─────────────────────────────────────────────────────────────────────────────
# 插值 / 平滑
# ─────────────────────────────────────────────────────────────────────────────

def full_interpolate(
    positions: List[Optional[Tuple[float, float]]],
    max_gap: int = 30,
) -> List[Optional[Tuple[float, float]]]:
    """
    對全幀球位置做線性插值，填補連續缺失（gap ≤ max_gap）。
    超過 max_gap 的缺失段不插值，保留 None。
    """
    out: List[Optional[Tuple[float, float]]] = list(positions)
    n = len(out)
    i = 0
    while i < n:
        if out[i] is not None:
            i += 1
            continue
        prev_i = i - 1
        while prev_i >= 0 and out[prev_i] is None:
            prev_i -= 1
        next_i = i
        while next_i < n and out[next_i] is None:
            next_i += 1
        if prev_i < 0 or next_i >= n:
            i += 1
            continue
        gap = next_i - prev_i
        if gap > max_gap:
            i = next_i
            continue
        p0 = np.array(out[prev_i], dtype=np.float64)
        p1 = np.array(out[next_i], dtype=np.float64)
        for j in range(prev_i + 1, next_i):
            t = (j - prev_i) / gap
            out[j] = tuple((p0 + t * (p1 - p0)).tolist())
        i = next_i
    return out


def smooth(values: List[Optional[float]], window: int = 3) -> List[Optional[float]]:
    """對數值序列做滑動平均（跳過 None）。"""
    out: List[Optional[float]] = []
    for i, v in enumerate(values):
        if v is None:
            out.append(None)
            continue
        start = max(0, i - window // 2)
        end = min(len(values), i + window // 2 + 1)
        bucket = [values[j] for j in range(start, end) if values[j] is not None]
        out.append(sum(bucket) / len(bucket) if bucket else None)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 球速換算（僅世界座標，無 fallback）
# ─────────────────────────────────────────────────────────────────────────────

def compute_frame_speeds_world(
    positions: List[Optional[Tuple[float, float]]],
    fps: float,
    H: np.ndarray,
) -> List[Optional[float]]:
    """
    逐幀球速（km/h），使用 H_img_to_world 投影到真實公尺座標後計算。

    Args:
        positions: 插值後球心像素座標列表
        fps:       影片幀率
        H:         img→world 單應矩陣（3×3）

    Returns:
        同長度的 Optional[float] 列表，無法計算則為 None。
    """
    speeds: List[Optional[float]] = [None] * len(positions)
    for i in range(1, len(positions)):
        p0, p1 = positions[i - 1], positions[i]
        if p0 is None or p1 is None:
            continue
        try:
            pts = np.array([[list(p0)], [list(p1)]], dtype=np.float32)
            world = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
            dist_m = float(np.linalg.norm(world[1] - world[0]))
            kmh = round(dist_m * fps * 3.6, 1)
            if MIN_BALL_SPEED_KMH <= kmh <= MAX_BALL_SPEED_KMH:
                speeds[i] = kmh
        except Exception:
            pass
    return speeds


# ─────────────────────────────────────────────────────────────────────────────
# 事件偵測
# ─────────────────────────────────────────────────────────────────────────────

def detect_events(
    ball_positions: List[Optional[Tuple[float, float]]],
    player_top: List[Optional[Tuple[float, float]]],
    player_bottom: List[Optional[Tuple[float, float]]],
    img_w: int,
    img_h: int,
    fps: float,
    scene_cut_frames: List[int],
) -> Tuple[List[int], List[int]]:
    """
    速度 Y 方向變化偵測 → 分類為 contact（擊球）或 bounce（落地）。

    切鏡幀強制重置冷卻計數，使切鏡後可立刻偵測新事件。

    Returns:
        (contact_frames, bounce_frames)
    """
    pos = full_interpolate(ball_positions, max_gap=30)

    vy: List[Optional[float]] = [None] * len(pos)
    for i in range(1, len(pos)):
        p0, p1 = pos[i - 1], pos[i]
        if p0 is not None and p1 is not None:
            vy[i] = p1[1] - p0[1]

    vy_s = smooth(vy, 3)
    vel_threshold = img_h * MIN_VEL_CHANGE_RATIO
    proximity_px = img_w * PROXIMITY_RATIO
    cut_set = set(scene_cut_frames)

    contacts: List[int] = []
    bounces: List[int] = []
    last_event = -100

    for i in range(2, len(vy_s) - 1):
        if i in cut_set:
            last_event = -100
            continue
        v_prev = vy_s[i - 1]
        v_curr = vy_s[i]
        if v_prev is None or v_curr is None:
            continue
        if i - last_event < COOLDOWN_FRAMES:
            continue

        reversed_y = (v_prev * v_curr) < 0
        if not reversed_y and abs(v_curr - v_prev) < vel_threshold:
            continue

        # 球的實際速度太小（靜止雜訊），跳過
        if max(abs(v_prev), abs(v_curr)) < MIN_CONTACT_VY:
            continue

        ball_pos = pos[i]
        if ball_pos is None:
            continue

        bx, by = ball_pos

        def dist_to(p: Optional[Tuple[float, float]]) -> float:
            return (math.sqrt((bx - p[0]) ** 2 + (by - p[1]) ** 2)
                    if p is not None else float("inf"))

        tp = player_top[i] if i < len(player_top) else None
        bp = player_bottom[i] if i < len(player_bottom) else None

        if min(dist_to(tp), dist_to(bp)) < proximity_px:
            contacts.append(i)
        else:
            bounces.append(i)

        last_event = i

    return contacts, bounces


# ─────────────────────────────────────────────────────────────────────────────
# 回合切割
# ─────────────────────────────────────────────────────────────────────────────

def segment_rallies(
    contacts: List[int],
    fps: float,
    scene_cut_frames: List[int],
) -> List[List[int]]:
    """
    依時間間隔（> RALLY_GAP_SEC）或切鏡邊界切割回合。

    Returns:
        回合列表，每個回合為 contact 幀索引列表。
    """
    if not contacts:
        return []
    gap_frames = int(RALLY_GAP_SEC * fps)
    cut_set = set(scene_cut_frames)

    rallies: List[List[int]] = []
    current = [contacts[0]]

    for c in contacts[1:]:
        cut_between = any(current[-1] < sc <= c for sc in cut_set)
        if (c - current[-1]) > gap_frames or cut_between:
            rallies.append(current)
            current = []
        current.append(c)

    if current:
        rallies.append(current)
    return rallies


def find_winner_after(
    bounce_frames: List[int],
    last_contact: int,
    next_rally_start: Optional[int],
    fps: float,
) -> Optional[int]:
    """在最後一拍之後找第一個 bounce（勝利球落點），不超出下一回合開始前。"""
    search_end = (next_rally_start if next_rally_start is not None
                  else last_contact + int(fps * 4))
    for bf in bounce_frames:
        if last_contact < bf <= search_end:
            return bf
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 擊球歸屬
# ─────────────────────────────────────────────────────────────────────────────

def assign_player(
    frame_idx: int,
    ball_pos: Tuple[float, float],
    player_top: List[Optional[Tuple[float, float]]],
    player_bottom: List[Optional[Tuple[float, float]]],
) -> str:
    """依球距上下球員的距離判斷擊球歸屬，回傳 'top' 或 'bottom'。"""
    bx, by = ball_pos

    def dist_to(p: Optional[Tuple[float, float]]) -> float:
        return (math.sqrt((bx - p[0]) ** 2 + (by - p[1]) ** 2)
                if p is not None else float("inf"))

    tp = player_top[frame_idx] if frame_idx < len(player_top) else None
    bp = player_bottom[frame_idx] if frame_idx < len(player_bottom) else None
    return "top" if dist_to(tp) <= dist_to(bp) else "bottom"


# ─────────────────────────────────────────────────────────────────────────────
# 世界座標轉換與場地分區
# ─────────────────────────────────────────────────────────────────────────────

def project_to_world(
    pos: Tuple[float, float],
    H: np.ndarray,
) -> Optional[Tuple[float, float]]:
    """
    將像素座標投影到世界公尺座標（H: img→world）。
    世界系統：BL=(0,0), BR=(10.97,0), TL=(0,23.77), TR=(10.97,23.77)，網 Y=11.885m。
    """
    try:
        pt = np.array([[[pos[0], pos[1]]]], dtype=np.float32)
        wpt = cv2.perspectiveTransform(pt, H)
        return (float(wpt[0, 0, 0]), float(wpt[0, 0, 1]))
    except Exception:
        return None


def bounce_zone(world_x: float, world_y: float) -> str:
    """
    根據世界座標判斷落地區域。

    Returns:
        'top_service'    遠端發球區（網到遠發球線之間）
        'top_back'       遠端底線區（遠發球線後方）
        'bottom_service' 近端發球區（網到近發球線之間）
        'bottom_back'    近端底線區（近發球線後方）
        'net_area'       網前 1m 以內
        'out'            球場範圍外
    """
    if (world_x < -_COURT_TOL_M or world_x > DOUBLES_WIDTH_M + _COURT_TOL_M or
            world_y < -_COURT_TOL_M or world_y > COURT_LENGTH_M + _COURT_TOL_M):
        return "out"

    if abs(world_y - _NET_Y_M) < 1.0:
        return "net_area"

    if world_y > _NET_Y_M:  # 遠端（top）半場
        return "top_service" if world_y <= _SERVICE_Y_FAR else "top_back"
    else:                    # 近端（bottom）半場
        return "bottom_service" if world_y >= _SERVICE_Y_NEAR else "bottom_back"


def player_court_zone(world_y: float) -> str:
    """
    根據球員世界 Y 座標判斷其站位區域（以距球網距離衡量）。

    Returns:
        'net'      距網 < 2m（截擊位置）
        'service'  距網 2–6.4m（發球區深度）
        'baseline' 距網 > 6.4m（底線附近）
    """
    dist_from_net = abs(world_y - _NET_Y_M)
    if dist_from_net < _NET_ZONE_M:
        return "net"
    if dist_from_net < _SERVICE_DIST_M:
        return "service"
    return "baseline"


def classify_shot_type(
    is_serve: bool,
    action_events: Dict[int, str],
    frame_idx: int,
    window: int = 10,
) -> str:
    """
    根據動作辨識結果分類擊球類型。

    is_serve:      該拍是否為回合第一拍（發球）
    action_events: {幀索引: 動作名稱} 字典（由 ActionRecognizer 產生）
    frame_idx:     接觸幀索引
    window:        搜尋窗格（前後各 window 幀）

    Returns:
        'serve'    發球
        'overhead' 高壓球
        'swing'    正常揮拍（正手/反手）
        'unknown'  無法判斷
    """
    if is_serve:
        return "serve"
    for f in range(max(0, frame_idx - window), frame_idx + window + 1):
        act = action_events.get(f)
        if act == "overhead_hit":
            return "overhead"
        if act == "swing":
            return "swing"
    return "unknown"
