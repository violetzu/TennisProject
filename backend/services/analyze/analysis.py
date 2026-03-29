"""
分析邏輯判斷 (Analysis Logic)

負責：使用偵測 + 滑動窗口處理好的資料進行邏輯判斷。

包含：
  - smooth()                    滑動平均平滑
  - compute_frame_speeds_world() 逐幀球速（世界座標，km/h）
  - detect_events()             手腕距離法擊球 + vy 反轉觸地偵測
  - bounce_zone()               落地區域分類
  - player_court_zone()         球員站位區域分類
  - assign_court_side()         半場歸屬判定
  - find_serve_index()          發球偵測（越網判定）
  - find_winner_landing()       勝利球落點
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .court import (
    COURT_LENGTH_M, DOUBLES_WIDTH_M, _NET_Y_M,
    _SERVICE_DIST_M, _SERVICE_Y_NEAR, _SERVICE_Y_FAR,
    project_to_world,
)


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
WRIST_SEARCH_SEC = 0.17        # 局部最小值搜尋窗口（前後各 N 秒）
COOLDOWN_SEC = 0.27            # 擊球冷卻（不同球員交替）
SAME_PLAYER_COOLDOWN_SEC = 0.33  # 同一球員連擊冷卻（防止同次揮拍重複觸發）
BOUNCE_COOLDOWN_SEC = 0.13     # bounce 後接 hit 允許更短冷卻
SWING_MIN_WRIST_RATIO = 0.05   # 手腕位移 / 球員縮放閾值（乘以球員 y 位置比例）
SWING_CHECK_SEC = 0.13         # 揮拍動作檢查窗口（前後各 N 秒）
FORWARD_COURT_SEC = 0.67       # 擊球後前看秒數，確認球回到場地
SERVE_TOSS_LOOKBACK_SEC = 0.7  # 發球拋球偵測回看秒數
SERVE_CROSS_SEC = 3.0          # 擊球後球進入對方發球區的最大等待秒數

# ── 回合切割 ──────────────────────────────────────────────────────────────────
RALLY_GAP_SEC = 3.5            # 回合間隔閾值（秒）；2.5 太短，底線長回合中球追蹤中斷可能超過 2s

# ── 世界座標球場幾何常數（ITF 標準，公尺）──────────────────────────────────────
# _SERVICE_DIST_M, _SERVICE_Y_NEAR, _SERVICE_Y_FAR → 從 court.py import
_NET_ZONE_M     = 2.0                     # 距球網 2m 以內算「網前」
_COURT_TOL_M    = 0.5                     # 邊界容差（避免邊線稍微出界就算 out）


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
    frame_offset: int = 0,
) -> Tuple[List[int], List[int]]:
    """
    第一性原理事件偵測：球靠近手腕 = 擊球，軌跡反轉且遠離所有人 = 觸地。

    擊球偵測：
      1. 逐幀算球到每個球員手腕的距離
      2. 找距離的局部最小值（前後 WRIST_SEARCH_SEC 秒內最小）
      3. 最小值 < WRIST_HIT_RADIUS（畫面高度 8%）→ 確認為擊球

    觸地偵測：
      - 球軌跡 vy 由正→負（下→上）反轉且遠離所有球員 → bounce

    Returns:
        (contact_frames, bounce_frames)
    """
    pos = ball_positions  # 已在呼叫端插值
    n = len(pos)
    hit_radius = img_h * WRIST_HIT_RADIUS
    cut_set = set(scene_cut_frames)

    # 以 fps 動態換算幀數
    W = max(1, int(WRIST_SEARCH_SEC * fps))
    cd_frames = max(1, int(COOLDOWN_SEC * fps))
    cd_same = max(1, int(SAME_PLAYER_COOLDOWN_SEC * fps))
    cd_bounce = max(1, int(BOUNCE_COOLDOWN_SEC * fps))
    SW = max(1, int(SWING_CHECK_SEC * fps))
    fwd_court = max(1, int(FORWARD_COURT_SEC * fps))
    serve_lookback = max(1, int(SERVE_TOSS_LOOKBACK_SEC * fps))

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

    # ── 擊球：球-手腕距離的局部最小值（延遲確認）─────────────────────
    contacts: List[int] = []
    bounces: List[int] = []
    last_event = -100
    last_event_type = ""
    last_event_player = ""

    # 延遲確認：球在手腕範圍內時持續更新候選，離開後才正式記錄
    _pending: Optional[Tuple[int, float, str]] = None  # (frame, dist, player)

    def _wrist_move(frame_idx: int, player: str) -> float:
        """計算該球員手腕在 ±SW 幀內的最大位移（像素）。"""
        wl = wrist_top if player == "top" else wrist_bottom
        pts = []
        for j in range(max(0, frame_idx - SW), min(n, frame_idx + SW + 1)):
            w = wl[j] if j < len(wl) else None
            if w is not None:
                pts.append(w)
        if len(pts) < 2:
            return float("inf")  # 資料不足，不過濾
        max_d = 0.0
        for a in range(len(pts)):
            for b in range(a + 1, len(pts)):
                d = math.hypot(pts[a][0] - pts[b][0], pts[a][1] - pts[b][1])
                if d > max_d:
                    max_d = d
        return max_d

    def _swing_threshold(frame_idx: int, player: str) -> float:
        """根據球員在畫面中的 y 位置縮放閾值（遠端球員小、近端球員大）。"""
        pl = player_top if player == "top" else player_bottom
        pp = pl[frame_idx] if frame_idx < len(pl) else None
        scale = pp[1] / img_h if pp is not None else 0.5
        return img_h * SWING_MIN_WRIST_RATIO * scale

    def _is_serve_toss(fi: int, player: str) -> bool:
        """偵測發球拋球模式：球在頭頂 + vy 反轉 + 球在同側。"""
        bp = pos[fi] if fi < n else None
        if bp is None:
            return False
        pl = player_top if player == "top" else player_bottom
        pp_pos = pl[fi] if fi < len(pl) else None
        if pp_pos is None:
            return False
        # A: 球在球員頭頂上方，垂直距離合理，且水平接近
        vert_dist = pp_pos[1] - bp[1]
        if vert_dist < img_h * 0.02 or vert_dist > img_h * 0.20:
            return False
        if abs(bp[0] - pp_pos[0]) > img_w * 0.25:
            return False
        # B: 擊球前 vy 反轉（拋球：先上後下）
        start_look = max(0, fi - serve_lookback)
        up_count = 0
        down_after_up = 0
        found_up = False
        for j in range(start_look, fi):
            v = vy_s[j] if j < len(vy_s) else None
            if v is None:
                continue
            if v < -0.5:
                up_count += 1
                found_up = True
            elif found_up and v > 0.5:
                down_after_up += 1
        min_phase = max(2, int(0.1 * fps))
        if up_count < min_phase or down_after_up < min_phase:
            return False
        # C: 球在發球方半場（非從對面飛來）
        half_y = img_h * 0.5
        same_side = 0
        total = 0
        for j in range(start_look, fi):
            p = pos[j] if j < n else None
            if p is None:
                continue
            total += 1
            if player == "top" and p[1] < half_y:
                same_side += 1
            elif player == "bottom" and p[1] > half_y:
                same_side += 1
        if total < 3 or same_side / total < 0.8:
            return False
        return True

    def _in_opp_court(fi: int, player: str, window: int) -> bool:
        """fi 後 window 幀內球是否在對方場地形成軌跡（至少 min_pts 個連續偵測點）。
        單一誤判點不計算，需要連續命中才確認。"""
        min_pts = max(2, int(0.1 * fps))
        count = 0
        for j in range(fi, min(n, fi + window)):
            p = pos[j]
            if p is None:
                continue  # 跳過缺幀，不重置連續計數
            in_opp = (player == "top" and p[1] > img_h * 0.25) or \
                     (player == "bottom" and p[1] < img_h * 0.75)
            if in_opp:
                count += 1
                if count >= min_pts:
                    return True
            else:
                count = 0  # 球離開對方場地，重置連續計數
        return False

    def _commit_pending() -> None:
        nonlocal last_event, last_event_type, last_event_player, _pending
        if _pending is None:
            return
        pi, pd, pp = _pending
        _pending = None
        # 手腕位移過小 → 沒有揮拍動作，否定擊球
        wm = _wrist_move(pi, pp)
        thr = _swing_threshold(pi, pp)
        if wm < thr:
            print(f"  [hit-rejected] f={pi + frame_offset} t={(pi + frame_offset)/fps:.2f}s player={pp} "
                  f"wrist_move={wm:.0f} < {thr:.0f}")
            return
        # 發球拋球偵測：豁免前向軌跡檢查
        serve_cross = max(1, int(SERVE_CROSS_SEC * fps))
        if _is_serve_toss(pi, pp):
            pre_trail = [pos[j] for j in range(max(0, pi - serve_lookback), pi)
                         if pos[j] is not None]
            pre_disp = sum(
                math.hypot(pre_trail[k][0] - pre_trail[k - 1][0],
                           pre_trail[k][1] - pre_trail[k - 1][1])
                for k in range(1, len(pre_trail))) if len(pre_trail) >= 2 else 0
            if pre_disp < img_h * 0.05:
                print(f"  [hit-rejected] f={pi + frame_offset} t={(pi + frame_offset)/fps:.2f}s player={pp} "
                      f"serve toss but no displacement (pre_disp={pre_disp:.0f})")
                return
            # 驗證球確實進入對方發球區（過發球線）
            if not _in_opp_court(pi, pp, serve_cross):
                print(f"  [hit-rejected] f={pi + frame_offset} t={(pi + frame_offset)/fps:.2f}s player={pp} "
                      f"serve toss but ball never reached opponent court")
                return
            contacts.append(pi)
            last_event_type = "contact"
            last_event = pi
            last_event_player = pp
            print(f"  [hit-serve] f={pi + frame_offset} t={(pi + frame_offset)/fps:.2f}s player={pp} "
                  f"d_wrist={pd:.0f} wrist_move={wm:.0f} pre_disp={pre_disp:.0f}")
            return
        # 前向軌跡檢查：擊球後球應有移動軌跡且進入場地
        # 靜態假偵測（場地標記）→ 無位移 → 拒絕
        # 球僮回收球 → 有位移但不在場內 → 拒絕
        # 快球追丟 → 無足夠偵測點 → 無法驗證，拒絕
        fwd_skip = max(1, int(0.1 * fps))
        fwd_trail = []
        for j in range(pi + fwd_skip, min(n, pi + fwd_court)):
            fp = pos[j]
            if fp is not None:
                fwd_trail.append(fp)
        # 條件 1：需有足夠偵測點形成軌跡
        fwd_min_pts = max(2, int(0.1 * fps))
        if len(fwd_trail) < fwd_min_pts:
            # 備援：快速擊球可能追丟，但球確實進入對方場地
            if _in_opp_court(pi, pp, serve_cross):
                contacts.append(pi)
                last_event_type = "contact"
                last_event = pi
                last_event_player = pp
                print(f"  [hit-fast] f={pi + frame_offset} t={(pi + frame_offset)/fps:.2f}s player={pp} "
                      f"d_wrist={pd:.0f} (no fwd trail, ball crossed court)")
                return
            print(f"  [hit-rejected] f={pi + frame_offset} t={(pi + frame_offset)/fps:.2f}s player={pp} "
                  f"no trajectory after hit ({len(fwd_trail)} pts)")
            return
        # 條件 2：軌跡需有實際位移（過濾靜態假偵測如場地標記）
        total_disp = sum(
            math.hypot(fwd_trail[k][0] - fwd_trail[k - 1][0],
                       fwd_trail[k][1] - fwd_trail[k - 1][1])
            for k in range(1, len(fwd_trail)))
        min_disp = img_h * 0.05
        if total_disp < min_disp:
            print(f"  [hit-rejected] f={pi + frame_offset} t={(pi + frame_offset)/fps:.2f}s player={pp} "
                  f"static detection after hit (disp={total_disp:.0f} < {min_disp:.0f})")
            return
        # 條件 3：軌跡需進入場地區域（過濾球僮回收球）
        in_court = any(
            (pp == "bottom" and fp[1] < img_h * 0.75) or
            (pp == "top" and fp[1] > img_h * 0.25)
            for fp in fwd_trail)
        if not in_court:
            # 備援：發球殘留軌跡在頂部，但球確實進入對方場地（快速發球）
            if _in_opp_court(pi, pp, serve_cross):
                contacts.append(pi)
                last_event_type = "contact"
                last_event = pi
                last_event_player = pp
                print(f"  [hit-fast] f={pi + frame_offset} t={(pi + frame_offset)/fps:.2f}s player={pp} "
                      f"d_wrist={pd:.0f} (fwd not in court, ball crossed court)")
                return
            print(f"  [hit-rejected] f={pi + frame_offset} t={(pi + frame_offset)/fps:.2f}s player={pp} "
                  f"trajectory not in court (out-of-play)")
            return
        contacts.append(pi)
        last_event_type = "contact"
        last_event = pi
        last_event_player = pp
        print(f"  [hit] f={pi + frame_offset} t={(pi + frame_offset)/fps:.2f}s player={pp} "
              f"d_wrist={pd:.0f} wrist_move={wm:.0f} thr={thr:.0f}")

    for i in range(W, n - W):
        if i in cut_set:
            _commit_pending()
            last_event = -100
            last_event_type = ""
            last_event_player = ""
            continue

        d = ball_wrist_dist[i]
        player_i = ball_wrist_player[i]

        # 球離開手腕範圍 → 確認上一個 pending
        if d is None or d > hit_radius:
            _commit_pending()
            continue

        # 球在手腕範圍內：
        # 如果有 pending 且是同一球員 → 更新為距離更小的幀（最後一次最近）
        if _pending is not None:
            pp_frame, pp_dist, pp_player = _pending
            if player_i == pp_player:
                # 同一球員、球還在手腕範圍 → 取距離更小的（或相同時取較晚的）
                if d <= pp_dist:
                    _pending = (i, d, player_i or "?")
                continue
            else:
                # 換球員了 → 先確認上一個
                _commit_pending()

        # 冷卻：同一球員用較長冷卻，防止同次揮拍重複觸發
        if last_event_type == "bounce":
            eff_cd = cd_bounce
        elif player_i == last_event_player:
            eff_cd = cd_same
        else:
            eff_cd = cd_frames
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

        _pending = (i, d, player_i or "?")

    _commit_pending()  # 結尾處理

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
        if any(abs(i - c) < cd_frames for c in contacts):
            continue
        # 不能太靠近其他 bounce
        if any(abs(i - b) < cd_bounce for b in bounces):
            continue
        # 遠離所有球員
        d = ball_wrist_dist[i]
        if d is not None and d < hit_radius * 2:
            continue

        bounces.append(i)
        print(f"  [bounce] f={i + frame_offset} t={(i + frame_offset)/fps:.2f}s "
              f"d_nearest={f'{d:.0f}' if d is not None else 'inf'}")

    print(f"[detect_events] {len(contacts)} hits, {len(bounces)} bounces")
    return contacts, bounces


# ─────────────────────────────────────────────────────────────────────────────
# 場地分區分類
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# 半場歸屬判定
# ─────────────────────────────────────────────────────────────────────────────

def assign_court_side(
    frame_idx: int,
    ball_positions: List[Optional[Tuple[float, float]]],
    player_top: List[Optional[Tuple[float, float]]],
    player_bottom: List[Optional[Tuple[float, float]]],
    H: Optional[np.ndarray],
    height: int,
) -> str:
    """判斷該幀球所在半場歸屬球員，回傳 'top' 或 'bottom'。"""
    bp = ball_positions[frame_idx] if frame_idx < len(ball_positions) else None
    if bp is not None and H is not None:
        w = project_to_world(bp, H)
        if w is not None:
            return "bottom" if w[1] < _NET_Y_M else "top"
    if bp is not None:
        return "bottom" if bp[1] > height * 0.5 else "top"
    tp = player_top[frame_idx] if frame_idx < len(player_top) else None
    bp2 = player_bottom[frame_idx] if frame_idx < len(player_bottom) else None
    if tp and bp2:
        return "top" if tp[1] < bp2[1] else "bottom"
    return "top" if tp else "bottom"


# ─────────────────────────────────────────────────────────────────────────────
# 發球偵測
# ─────────────────────────────────────────────────────────────────────────────

def find_serve_index(
    rally_contacts: List[int],
    ball_positions: List[Optional[Tuple[float, float]]],
    player_top: List[Optional[Tuple[float, float]]],
    player_bottom: List[Optional[Tuple[float, float]]],
    H: Optional[np.ndarray],
    height: int,
    fps: float,
) -> int:
    """找回合中發球的 contact index。

    邏輯：回合開頭連續同側的 contacts 都是發球前準備動作（拍球/拋球），
    發球是換邊前最後一個同側 contact（對方回擊前的那一拍）。
    """
    if len(rally_contacts) < 2:
        return 0
    first_side = assign_court_side(
        rally_contacts[0], ball_positions, player_top, player_bottom, H, height)
    for ci in range(1, len(rally_contacts)):
        side = assign_court_side(
            rally_contacts[ci], ball_positions, player_top, player_bottom, H, height)
        if side != first_side:
            return ci - 1
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# 勝利球落點
# ─────────────────────────────────────────────────────────────────────────────

def find_winner_landing(
    winner_player: str,
    rally_contacts: List[int],
    bounces_f: List[int],
    ball_positions: List[Optional[Tuple[float, float]]],
    next_rally_start: Optional[int],
    total_frames: int,
    H: np.ndarray,
    fps: float,
) -> Optional[Dict]:
    """找勝利球在對方半場的落點（世界座標），優先用 bounce，fallback 用球位置。"""
    last_fi = rally_contacts[-1]

    # 優先從 bounces_f 找最後一拍之後的彈跳
    for bf in bounces_f:
        if bf <= last_fi:
            continue
        if next_rally_start and bf >= next_rally_start:
            break
        bpos = ball_positions[bf] if bf < len(ball_positions) else None
        if bpos is None:
            continue
        bw = project_to_world(bpos, H)
        if bw is None:
            continue
        in_opp = (bw[1] > _NET_Y_M) if winner_player == "bottom" else (bw[1] < _NET_Y_M)
        if in_opp:
            return {"x": round(bw[0], 2), "y": round(bw[1], 2)}

    # fallback：最後一拍後在對方半場的第一個球位置
    end_fi = min(
        (next_rally_start - 1) if next_rally_start else total_frames,
        last_fi + int(fps * 3),
    )
    for fi in range(last_fi + 3, end_fi):
        bp = ball_positions[fi] if fi < len(ball_positions) else None
        if bp is None:
            continue
        bw = project_to_world(bp, H)
        if bw is None:
            continue
        in_opp = (bw[1] > _NET_Y_M) if winner_player == "bottom" else (bw[1] < _NET_Y_M)
        if in_opp:
            return {"x": round(bw[0], 2), "y": round(bw[1], 2)}

    return None


