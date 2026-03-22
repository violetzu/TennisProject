"""
後處理分析輔助 (Post-processing Analysis Helpers)

包含：
  - full_interpolate()          全域位置線性插值（球/球員/手腕）
  - smooth()                    滑動平均平滑
  - compute_frame_speeds_world() 逐幀球速（世界座標，km/h）
  - detect_events()             手腕距離法擊球 + vy 反轉觸地偵測
  - segment_rallies()           依時間間隔 + 切鏡切割回合
  - find_winner_after()         勝利球落點
  - assign_player()             歸屬球員（top / bottom）
  - project_to_world()          像素座標 → 世界公尺座標
  - bounce_zone()               落地區域分類（發球區/底線區/網前/出界）
  - player_court_zone()         球員站位區域（net/service/baseline）

所有函式為純函式，不依賴模型或影像處理。
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .court import COURT_LENGTH_M, DOUBLES_WIDTH_M


def _dist2d(
    origin: Tuple[float, float],
    target: Optional[Tuple[float, float]],
) -> float:
    """二維歐氏距離；target 為 None 時回傳 inf。"""
    if target is None:
        return float("inf")
    return math.hypot(origin[0] - target[0], origin[1] - target[1])

# ── 球速範圍（ITF：最慢約 20 km/h 吊球；最快發球紀錄 263 km/h，280 留緩衝）──
MIN_BALL_SPEED_KMH = 20.0
MAX_BALL_SPEED_KMH = 280.0

# ── 事件偵測參數 ───────────────────────────────────────────────────────────────
WRIST_HIT_RADIUS = 0.08        # 球距手腕 < 畫面高度 8% → 擊球候選
WRIST_SEARCH_WINDOW = 5        # 局部最小值搜尋窗口（前後各 N 幀）
COOLDOWN_FRAMES = 8            # 擊球冷卻（同一球員連續擊球不可能太快）
BOUNCE_COOLDOWN = 4            # bounce 後接 hit 允許更短冷卻

# ── 回合切割 ──────────────────────────────────────────────────────────────────
RALLY_GAP_SEC = 3.5            # 回合間隔閾值（秒）；2.5 太短，底線長回合中球追蹤中斷可能超過 2s

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
    scene_cut_frames: Optional[List[int]] = None,
) -> List[Optional[Tuple[float, float]]]:
    """
    對全幀球位置做線性插值，填補連續缺失（gap ≤ max_gap）。
    超過 max_gap 的缺失段不插值，保留 None。
    scene_cut_frames：切鏡幀列表；若缺失段跨越切鏡則不插值，避免產生假軌跡。
    """
    cut_set: set = set(scene_cut_frames) if scene_cut_frames else set()
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
        # 若缺失區間內有切鏡幀則不插值
        if any(prev_i < sc <= next_i for sc in cut_set):
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
    wrist_top: List[Optional[Tuple[float, float]]],
    wrist_bottom: List[Optional[Tuple[float, float]]],
    player_top: List[Optional[Tuple[float, float]]],
    player_bottom: List[Optional[Tuple[float, float]]],
    img_w: int,
    img_h: int,
    fps: float,
    scene_cut_frames: List[int],
) -> Tuple[List[int], List[int], Dict[int, float]]:
    """
    第一性原理事件偵測：球靠近手腕 = 擊球，軌跡反轉且遠離所有人 = 觸地。

    擊球偵測：
      1. 逐幀算球到每個球員手腕的距離
      2. 找距離的局部最小值（前後 WRIST_SEARCH_WINDOW 幀內最小）
      3. 最小值 < WRIST_HIT_RADIUS（畫面高度 8%）→ 確認為擊球

    觸地偵測：
      - 球軌跡 vy 由正→負（下→上）反轉且遠離所有球員 → bounce

    Returns:
        (contact_frames, bounce_frames, confidence)
    """
    pos = ball_positions  # 已在呼叫端插值
    n = len(pos)
    hit_radius = img_h * WRIST_HIT_RADIUS
    W = WRIST_SEARCH_WINDOW
    cut_set = set(scene_cut_frames)

    # ── 逐幀球到最近手腕的距離 ──────────────────────────────────────────
    ball_wrist_dist: List[Optional[float]] = [None] * n
    ball_wrist_player: List[Optional[str]] = [None] * n  # 'top' or 'bottom'

    for i in range(n):
        bp = pos[i]
        if bp is None:
            continue
        wt = wrist_top[i] if i < len(wrist_top) else None
        wb = wrist_bottom[i] if i < len(wrist_bottom) else None
        d_top = _dist2d(bp, wt)
        d_bot = _dist2d(bp, wb)

        # 手腕漏偵測補償：球在該球員半場 → 用身體中心 fallback
        if wt is None and i < len(player_top) and player_top[i] is not None:
            d_top = _dist2d(bp, player_top[i])
        if wb is None and i < len(player_bottom) and player_bottom[i] is not None:
            d_bot = _dist2d(bp, player_bottom[i])

        if d_top <= d_bot:
            ball_wrist_dist[i] = d_top
            ball_wrist_player[i] = "top"
        else:
            ball_wrist_dist[i] = d_bot
            ball_wrist_player[i] = "bottom"

    # ── vy 計算（用於 bounce 偵測）──────────────────────────────────────
    vy: List[Optional[float]] = [None] * n
    for i in range(1, n):
        p0, p1 = pos[i - 1], pos[i]
        if p0 is not None and p1 is not None:
            vy[i] = p1[1] - p0[1]
    vy_s = smooth(vy, 3)

    # ── 擊球：球-手腕距離的局部最小值 ──────────────────────────────────
    contacts: List[int] = []
    bounces: List[int] = []
    confidence: Dict[int, float] = {}
    last_event = -100
    last_event_type = ""

    for i in range(W, n - W):
        if i in cut_set:
            last_event = -100
            last_event_type = ""
            continue

        d = ball_wrist_dist[i]
        if d is None or d > hit_radius:
            continue

        # 冷卻
        eff_cd = BOUNCE_COOLDOWN if last_event_type == "bounce" else COOLDOWN_FRAMES
        if i - last_event < eff_cd:
            continue

        # 局部最小值檢查：前後 W 幀內 d[i] 是否最小
        is_local_min = True
        for j in range(i - W, i + W + 1):
            if j == i:
                continue
            dj = ball_wrist_dist[j]
            if dj is not None and dj < d:
                is_local_min = False
                break

        if not is_local_min:
            continue

        conf = min(0.95, 0.5 + 0.5 * (1.0 - d / hit_radius))
        contacts.append(i)
        confidence[i] = conf
        last_event_type = "contact"
        last_event = i

        player = ball_wrist_player[i] or "?"
        print(f"  [hit] f={i} t={i/fps:.2f}s player={player} "
              f"d_wrist={d:.0f} radius={hit_radius:.0f}")

    # ── 觸地：vy 反轉（下→上）且遠離所有球員 ──────────────────────────
    for i in range(2, n - 1):
        if i in cut_set:
            continue
        bp = pos[i]
        if bp is None:
            continue
        vy_prev, vy_curr = vy_s[i - 1], vy_s[i]
        if vy_prev is None or vy_curr is None:
            continue
        # 下→上反轉（image Y: vy 正→負）
        if not (vy_prev > 0.5 and vy_curr < -0.5):
            continue
        # 不能太靠近已偵測的擊球
        if any(abs(i - c) < COOLDOWN_FRAMES for c in contacts):
            continue
        # 不能太靠近其他 bounce
        if any(abs(i - b) < BOUNCE_COOLDOWN for b in bounces):
            continue
        # 遠離所有球員
        d = ball_wrist_dist[i]
        if d is not None and d < hit_radius * 2:
            continue

        bounces.append(i)
        confidence[i] = 0.85
        print(f"  [bounce] f={i} t={i/fps:.2f}s "
              f"d_nearest={f'{d:.0f}' if d is not None else 'inf'}")

    print(f"[detect_events] {len(contacts)} hits, {len(bounces)} bounces")
    return contacts, bounces, confidence


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
    tp = player_top[frame_idx] if frame_idx < len(player_top) else None
    bp = player_bottom[frame_idx] if frame_idx < len(player_bottom) else None
    return "top" if _dist2d(ball_pos, tp) <= _dist2d(ball_pos, bp) else "bottom"


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


