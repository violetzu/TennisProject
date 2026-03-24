"""
分析邏輯判斷 (Analysis Logic)

負責：使用偵測 + 滑動窗口處理好的資料進行邏輯判斷。

包含：
  - smooth()                    滑動平均平滑
  - compute_frame_speeds_world() 逐幀球速（世界座標，km/h）
  - detect_events()             手腕距離法擊球 + vy 反轉觸地偵測
  - segment_rallies()           依時間間隔 + 切鏡切割回合
  - bounce_zone()               落地區域分類
  - player_court_zone()         球員站位區域分類
  - assign_court_side()         半場歸屬判定
  - find_serve_index()          發球偵測（越網判定）
  - determine_winners()         VLM 回合勝負判斷
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
WRIST_SEARCH_WINDOW = 5        # 局部最小值搜尋窗口（前後各 N 幀）
COOLDOWN_FRAMES = 8            # 擊球冷卻（不同球員交替）
SAME_PLAYER_COOLDOWN = 20      # 同一球員連擊冷卻（0.33s@60fps；防止同次揮拍重複觸發）
BOUNCE_COOLDOWN = 4            # bounce 後接 hit 允許更短冷卻
SWING_MIN_WRIST_RATIO = 0.05   # 手腕位移 / 球員縮放閾值（乘以球員 y 位置比例）
SWING_CHECK_WINDOW = 4         # 揮拍動作檢查窗口（前後各 N 幀）

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

    # ── 擊球：球-手腕距離的局部最小值（延遲確認）─────────────────────
    contacts: List[int] = []
    bounces: List[int] = []
    confidence: Dict[int, float] = {}
    last_event = -100
    last_event_type = ""
    last_event_player = ""

    # 延遲確認：球在手腕範圍內時持續更新候選，離開後才正式記錄
    _pending: Optional[Tuple[int, float, str]] = None  # (frame, dist, player)

    SW = SWING_CHECK_WINDOW

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
            print(f"  [hit-rejected] f={pi} t={pi/fps:.2f}s player={pp} "
                  f"wrist_move={wm:.0f} < {thr:.0f}")
            return
        conf = min(0.95, 0.5 + 0.5 * (1.0 - pd / hit_radius))
        contacts.append(pi)
        confidence[pi] = conf
        last_event_type = "contact"
        last_event = pi
        last_event_player = pp
        print(f"  [hit] f={pi} t={pi/fps:.2f}s player={pp} "
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
            eff_cd = BOUNCE_COOLDOWN
        elif player_i == last_event_player:
            eff_cd = SAME_PLAYER_COOLDOWN
        else:
            eff_cd = COOLDOWN_FRAMES
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
# VLM 回合勝負判斷
# ─────────────────────────────────────────────────────────────────────────────

def determine_winners(
    rally_groups: List[List[int]],
    thumb_dir,
    fps: float,
    vllm_cfg,
    progress_cb=None,
    progress_start: int = 97,
    progress_end: int = 99,
) -> Dict[int, str]:
    """逐回合呼叫 VLM 判斷勝負，回傳 {rally_idx: 'top'|'bottom'|'unknown'}。"""
    from .vlm_verify import verify_rally_winner

    results: Dict[int, str] = {}
    for ri, rc in enumerate(rally_groups):
        if not rc:
            continue
        next_f = rally_groups[ri + 1][0] if ri + 1 < len(rally_groups) else None
        results[ri] = verify_rally_winner(rc[-1], next_f, thumb_dir, fps, vllm_cfg)
        if progress_cb:
            span = progress_end - progress_start
            pct = progress_start + int((ri + 1) / max(len(rally_groups), 1) * span)
            progress_cb(pct, 100)
    return results


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


# ─────────────────────────────────────────────────────────────────────────────
# 分析主入口
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis(
    *,
    all_ball_positions: List[Optional[Tuple[float, float]]],
    all_wrist_top: List[Optional[Tuple[float, float]]],
    all_wrist_bottom: List[Optional[Tuple[float, float]]],
    all_player_top: List[Optional[Tuple[float, float]]],
    all_player_bottom: List[Optional[Tuple[float, float]]],
    width: int,
    height: int,
    fps: float,
    scene_cut_frames: List[int],
    last_valid_H,
    thumb_dir,
    vllm_cfg,
    progress_cb=None,
    precomputed_events: Optional[Tuple[List[int], List[int], Dict[int, float]]] = None,
) -> Dict:
    """事件偵測 + 球速 + 回合切割 + VLM 勝負，統一回傳分析結果。

    precomputed_events: 若提供 (contacts, bounces, confidence)，跳過 detect_events。
    """
    import shutil

    if precomputed_events is not None:
        contacts_f, bounces_f, event_confidence = precomputed_events
    else:
        contacts_f, bounces_f, event_confidence = detect_events(
            all_ball_positions, all_wrist_top, all_wrist_bottom,
            all_player_top, all_player_bottom,
            width, height, fps, scene_cut_frames,
        )

    if last_valid_H is not None:
        raw_speeds = compute_frame_speeds_world(all_ball_positions, fps, last_valid_H)
    else:
        raw_speeds = [None] * len(all_ball_positions)
    smooth_speeds = smooth(raw_speeds, 5)

    vlm_shot_types: Dict[int, str] = {f: "swing" for f in contacts_f}
    rally_groups = segment_rallies(contacts_f, fps, scene_cut_frames)

    if progress_cb:
        progress_cb(97, 100)

    try:
        vlm_winner_results = determine_winners(
            rally_groups, thumb_dir, fps, vllm_cfg, progress_cb)
    except Exception as exc:
        print(f"[VLM] winner判斷失敗: {exc}")
        vlm_winner_results = {}
    finally:
        shutil.rmtree(thumb_dir, ignore_errors=True)

    print(f"[analysis] {len(contacts_f)} contacts, "
          f"{len(bounces_f)} bounces, {len(rally_groups)} rallies")

    if progress_cb:
        progress_cb(99, 100)

    return {
        "contacts_f": contacts_f,
        "bounces_f": bounces_f,
        "event_confidence": event_confidence,
        "smooth_speeds": smooth_speeds,
        "vlm_shot_types": vlm_shot_types,
        "rally_groups": rally_groups,
        "vlm_winner_results": vlm_winner_results,
    }
