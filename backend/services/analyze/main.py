"""
綜合分析服務主流程 (Combine Analysis — Main)

核心規則：
  - 場地偵測（YOLO Pose, court.py）為前提條件：每幀獨立偵測，conf < 0.8 視為過場
  - 過場幀跳過所有分析，並重置球追蹤狀態（_reset_tracking）
  - 球速換算（analysis.py）只用 Homography 世界座標，無 pixel fallback
  - ActionRecognizer（CW_action_test.py）偵測揮拍後修正 Kalman 速度估計
  - 靜止假陽性（場地標線 / 帽子）透過 static_blacklist 機制過濾

分析 JSON 格式 → 見 backend/analyze.md
"""

from __future__ import annotations

import json
import math
import subprocess
import time
from collections import deque
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .CW_action_test import ActionRecognizer

from .ball import (
    RESET_AFTER_MISS, ROI_SIZE, STUCK_DIST_THRESHOLD, STUCK_FRAMES_LIMIT,
    STATIC_BLACKLIST_RADIUS, STATIC_BLACKLIST_TTL,
    WINDOW, extract_xyxy_conf, interpolate_window, is_valid_ball, new_kalman,
)
from .court import detect_court_yolo, draw_court, compute_net_y
from .player import build_action_candidates, detect_players, draw_skeleton
from .analysis import (
    MIN_BALL_SPEED_KMH,
    assign_player, bounce_zone, classify_shot_type,
    compute_frame_speeds_world,
    detect_events, find_winner_after, full_interpolate,
    player_court_zone, project_to_world,
    segment_rallies, smooth,
)

# ── 繪製開關 ──────────────────────────────────────────────────────────────────
DRAW_COURT_LINES = True    # 是否在影片上繪製場地線

# ── ActionRecognizer 觸發閾值 ─────────────────────────────────────────────────
SWING_SCORE_TH = 0.90



def analyze_combine(
    video_path: str,
    progress_cb: Optional[Callable[[int, int], None]],
    ball_model,
    pose_model,
    court_model,
    data_dir: str,
    job_id: str,
    *,
    width: Optional[int] = None,
    height: Optional[int] = None,
    fps: Optional[float] = None,
    total_frames: Optional[int] = None,
) -> Tuple[str, str]:
    """
    綜合分析主函式。

    執行流程：

    Phase 1  逐幀偵測（進度 0–97%）
      ├─ 場地偵測（YOLO Pose, conf≥0.8）→ H_img_to_world + kps
      │    └─ 偵測失敗（過場）→ 重置追蹤狀態，跳過 Pose / Ball，寫無標注幀
      ├─ Pose 偵測 → top / bottom 球員位置
      ├─ ActionRecognizer → 揮拍事件修正 Kalman 速度估計
      └─ Ball 偵測（Kalman + ROI + 靜止黑名單）→ 滑動窗口標注影片

    Phase 2  後處理（進度 97–99%）
      ├─ 全域球位置插值
      ├─ 事件偵測（contact / bounce）
      └─ 回合切割（時間間隔 + 切鏡邊界）

    Phase 3  統計彙整 → JSON（進度 99–100%）

    Returns:
        (json_path, annotated_video_path)
    """
    if ball_model is None or pose_model is None or court_model is None:
        raise ValueError("ball_model / pose_model / court_model 不可為 None")

    vpath = Path(video_path)
    if not vpath.exists():
        raise FileNotFoundError(f"影片不存在: {video_path}")

    # ── 輸出路徑 ──────────────────────────────────────────────────────────────
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    out_video_path = vpath.parent / f"{vpath.stem}_analyzed_{job_id}.mp4"
    out_json_path  = Path(data_dir) / f"analysis_{job_id}.json"

    # ── FFmpeg 管線 ───────────────────────────────────────────────────────────
    decode_cmd = [
        "/usr/bin/ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(vpath), "-an",
        "-vf", f"scale={width}:{height}",
        "-pix_fmt", "bgr24", "-f", "rawvideo", "-vsync", "0", "pipe:1",
    ]
    encode_cmd = [
        "/usr/bin/ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
        "-r", str(fps), "-i", "pipe:0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(out_video_path),
    ]

    dec = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    enc = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # ── 狀態初始化 ────────────────────────────────────────────────────────────

    H: Optional[np.ndarray] = None              # img→world Homography，None = 過場 / 未就緒
    court_corners: Optional[np.ndarray] = None  # 場地四角點像素座標 [TL, TR, BL, BR]
    court_kps: Optional[np.ndarray] = None      # YOLO 偵測到的全 14 個關鍵點像素座標
    net_y: Optional[float] = None               # 球網中心像素 y（由 H 逆投影）
    last_valid_H: Optional[np.ndarray] = None   # 最後一次成功的 H（Phase 2 球速計算用）
    n_processed: int = 0                        # 完整分析的幀數（不含過場跳過幀）

    _t: Dict[str, float] = {
        "court":  0.0,   # detect_court_yolo
        "pose":   0.0,   # pose model + detect_players
        "action": 0.0,   # ActionRecognizer
        "ball":   0.0,   # ball model + Kalman
        "write":  0.0,   # _write_buf_frame
    }

    tracking_mode = "global"
    last_center: Optional[Tuple[float, float]] = None
    miss_count = 0
    stuck_count = 0
    kalman_inited = False
    kalman = new_kalman()
    action = ActionRecognizer(fps=fps, img_w=width, img_h=height)
    # 靜止位置黑名單：List[(cx, cy, expire_frame)]
    static_blacklist: List[Tuple[float, float, int]] = []

    frame_buf: Deque[np.ndarray] = deque(maxlen=WINDOW)
    ball_box_buf: Deque[Optional[list]] = deque(maxlen=WINDOW)
    court_valid_buf: Deque[bool] = deque(maxlen=WINDOW)

    # 逐幀累計（長度 = total_frames_actual，過場幀填 None）
    all_ball_positions: List[Optional[Tuple[float, float]]] = []
    all_player_top: List[Optional[Tuple[float, float]]] = []
    all_player_bottom: List[Optional[Tuple[float, float]]] = []
    all_action_events: Dict[int, str] = {}   # frame_idx → 動作名稱（swing / overhead_hit）
    scene_cut_frames: List[int] = []         # 過場幀索引（H 由有效→None 的幀）

    idx = 0
    frame_bytes = width * height * 3

    def _reset_tracking() -> None:
        nonlocal tracking_mode, last_center, miss_count, stuck_count
        nonlocal kalman_inited, kalman, action
        tracking_mode = "global"
        last_center = None
        miss_count = 0
        stuck_count = 0
        kalman_inited = False
        kalman = new_kalman()
        action = ActionRecognizer(fps=fps, img_w=width, img_h=height)

    def _write_buf_frame(
        annotated_frame: np.ndarray,
        box: Optional[list],
        court_valid: bool,
    ) -> None:
        """推幀進滑動窗口，滿 WINDOW 後寫出最舊幀。
        court_valid=False 的幀即使插值有球框也不繪製。"""
        frame_buf.append(annotated_frame)
        ball_box_buf.append(box)
        court_valid_buf.append(court_valid)
        if len(frame_buf) == WINDOW:
            ib = interpolate_window(list(ball_box_buf))
            of = frame_buf[0].copy()
            if ib[0] and court_valid_buf[0]:
                x1_, y1_, x2_, y2_ = map(int, ib[0][:4])
                cv2.rectangle(of, (x1_, y1_), (x2_, y2_), (0, 255, 255), 2)
            if enc.stdin:
                enc.stdin.write(np.ascontiguousarray(of).tobytes())
            frame_buf.popleft()
            ball_box_buf.popleft()
            court_valid_buf.popleft()

    try:
        while True:
            raw = dec.stdout.read(frame_bytes) if dec.stdout else b""
            if not raw or len(raw) < frame_bytes:
                break

            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
            frame_draw = frame.copy()

            # ── 場地偵測（每幀獨立，信心 < 0.8 視為過場）───────────────────
            _t0 = time.perf_counter()
            result_try = detect_court_yolo(frame, court_model)
            if result_try is not None:
                H, court_corners, court_kps = result_try
                net_y = compute_net_y(H)
                last_valid_H = H
            else:
                if H is not None:
                    scene_cut_frames.append(idx)
                    _reset_tracking()
                H = None
                court_corners = None
                court_kps = None
                net_y = None
            _t["court"] += time.perf_counter() - _t0

            # ── 場地未就緒（過場）→ 跳過分析，寫出原始幀──────────────────
            if H is None:
                all_ball_positions.append(None)
                all_player_top.append(None)
                all_player_bottom.append(None)
                _write_buf_frame(frame_draw, None, court_valid=False)
                idx += 1
                if progress_cb and total_frames:
                    progress_cb(min(int(idx * 97 / total_frames), 96), 100)
                continue

            # ── 場地線繪製 ────────────────────────────────────────────────────
            if DRAW_COURT_LINES and court_corners is not None:
                draw_court(frame_draw, court_corners, H=H, kps=court_kps)

            # ── Pose 偵測 ──────────────────────────────────────────────────────
            _t0 = time.perf_counter()
            pose_results = pose_model.predict(
                source=frame, imgsz=1280, conf=0.01, verbose=False)
            pose_r = pose_results[0] if pose_results else None
            top_pos, bot_pos = detect_players(
                pose_r, width, height, court_corners, net_y)
            all_player_top.append(top_pos)
            all_player_bottom.append(bot_pos)
            draw_skeleton(frame_draw, pose_r, height, court_corners)
            _t["pose"] += time.perf_counter() - _t0

            # ── ActionRecognizer ───────────────────────────────────────────────
            _t0 = time.perf_counter()
            action_cands = build_action_candidates(pose_r, height, net_y)
            for ae in action.update_from_candidates(
                    action_cands, frame_idx=idx, ball_pos=last_center):
                if ae.name in ("swing", "overhead_hit") and float(ae.score) >= SWING_SCORE_TH:
                    all_action_events[idx] = ae.name
                    if kalman_inited:
                        kalman.statePost[2] *= -0.7
                        kalman.statePost[3] *= -0.7
                        kalman.errorCovPost *= 5.0
                    tracking_mode = "global"
                    break
            _t["action"] += time.perf_counter() - _t0

            # ── Ball 偵測（Kalman + ROI）──────────────────────────────────────
            _t0 = time.perf_counter()
            pred_c: Optional[Tuple[float, float]] = None
            if kalman_inited:
                pred = kalman.predict()
                pred_c = (float(pred[0]), float(pred[1]))

            detect_source = frame
            offset_x, offset_y = 0, 0
            current_imgsz = 1280

            if tracking_mode == "local" and pred_c:
                offset_x = int(max(0, min(pred_c[0] - ROI_SIZE // 2, width - ROI_SIZE)))
                offset_y = int(max(0, min(pred_c[1] - ROI_SIZE // 2, height - ROI_SIZE)))
                detect_source = frame[offset_y:offset_y + ROI_SIZE,
                                      offset_x:offset_x + ROI_SIZE]
                current_imgsz = ROI_SIZE

            ball_r_list = ball_model.predict(
                source=detect_source, imgsz=current_imgsz, conf=0.1, verbose=False)

            # 清除過期黑名單
            static_blacklist[:] = [(x, y, e) for x, y, e in static_blacklist if e > idx]

            chosen = None
            if ball_r_list:
                xyxy_roi, confs = extract_xyxy_conf(ball_r_list[0])
                cands = []
                for bx_box, conf in zip(xyxy_roi, confs):
                    gx1 = bx_box[0] + offset_x; gy1 = bx_box[1] + offset_y
                    gx2 = bx_box[2] + offset_x; gy2 = bx_box[3] + offset_y
                    g_box = [gx1, gy1, gx2, gy2]
                    if not is_valid_ball(g_box, width, height, frame):
                        continue
                    g_center = ((gx1 + gx2) / 2, (gy1 + gy2) / 2)
                    # 黑名單過濾：靜止假陽性位置
                    if any(math.hypot(g_center[0] - bx, g_center[1] - by) < STATIC_BLACKLIST_RADIUS
                           for bx, by, _ in static_blacklist):
                        continue
                    cands.append((g_box, conf, g_center))

                if cands:
                    chosen = (min(cands, key=lambda c: math.hypot(
                                  c[2][0] - pred_c[0], c[2][1] - pred_c[1]))
                              if tracking_mode == "local" and pred_c
                              else max(cands, key=lambda c: c[1]))

            chosen_box: Optional[list] = None
            if chosen:
                bx_g, _, (cbx, cby) = chosen
                if (last_center and
                        math.hypot(cbx - last_center[0], cby - last_center[1])
                        < STUCK_DIST_THRESHOLD):
                    stuck_count += 1
                else:
                    stuck_count = max(0, stuck_count - 1)
                    chosen_box = bx_g
                    if not kalman_inited:
                        kalman.statePost = np.array(
                            [[cbx], [cby], [0.0], [0.0]], dtype=np.float32)
                        kalman_inited = True
                    else:
                        kalman.correct(np.array(
                            [[np.float32(cbx)], [np.float32(cby)]], dtype=np.float32))
                    last_center = (cbx, cby)
                    miss_count = 0
                    tracking_mode = "local"

                if stuck_count >= STUCK_FRAMES_LIMIT:
                    if last_center is not None:
                        static_blacklist.append((last_center[0], last_center[1],
                                                 idx + STATIC_BLACKLIST_TTL))
                    tracking_mode, kalman_inited, last_center = "global", False, None
                    stuck_count = 0
            else:
                miss_count += 1
                if miss_count >= RESET_AFTER_MISS:
                    tracking_mode, kalman_inited, last_center = "global", False, None

            if chosen_box:
                cx = (chosen_box[0] + chosen_box[2]) / 2
                cy = (chosen_box[1] + chosen_box[3]) / 2
                all_ball_positions.append((cx, cy))
            else:
                all_ball_positions.append(None)
            _t["ball"] += time.perf_counter() - _t0

            _t0 = time.perf_counter()
            _write_buf_frame(frame_draw, chosen_box, court_valid=True)
            _t["write"] += time.perf_counter() - _t0
            n_processed += 1
            idx += 1
            if progress_cb and total_frames:
                progress_cb(min(int(idx * 97 / total_frames), 96), 100)

    finally:
        remaining_ib = interpolate_window(list(ball_box_buf))
        remaining_cv = list(court_valid_buf)
        for k, frm in enumerate(frame_buf):
            of = frm.copy()
            if remaining_ib[k] and remaining_cv[k]:
                x1_, y1_, x2_, y2_ = map(int, remaining_ib[k][:4])
                cv2.rectangle(of, (x1_, y1_), (x2_, y2_), (0, 255, 255), 2)
            if enc.stdin:
                enc.stdin.write(np.ascontiguousarray(of).tobytes())
        if dec.stdout:
            dec.stdout.close()
        dec.wait()
        if enc.stdin:
            enc.stdin.close()
        enc.wait()

    total_frames_actual = idx
    duration = total_frames_actual / fps if fps > 0 else 0.0

    # ── 耗時彙整 ──────────────────────────────────────────────────────────────
    n_all  = max(total_frames_actual, 1)
    n_proc = max(n_processed, 1)
    n_skip = total_frames_actual - n_processed
    total_measured = sum(_t.values())
    # court 對所有幀計時；其餘段落只在場地就緒時執行
    _n = {"court": n_all,
          "pose": n_proc, "action": n_proc, "ball": n_proc, "write": n_proc}
    skip_pct = n_skip / n_all * 100

    if progress_cb:
        progress_cb(97, 100)

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 2：後處理
    # ─────────────────────────────────────────────────────────────────────────
    _t2_start = time.perf_counter()

    contacts_f, bounces_f = detect_events(
        all_ball_positions, all_player_top, all_player_bottom,
        width, height, fps, scene_cut_frames,
    )
    _t2_events = time.perf_counter()

    if progress_cb:
        progress_cb(98, 100)

    rally_groups = segment_rallies(contacts_f, fps, scene_cut_frames)
    _t2_rallies = time.perf_counter()

    pos_interp = full_interpolate(all_ball_positions, max_gap=30)
    _t2_interp = time.perf_counter()

    # 球速：使用最後一次成功的 H 計算世界座標位移
    # （不同切鏡片段共用同一場地幾何，若全程無法偵測則填 None）
    if last_valid_H is not None:
        raw_speeds = compute_frame_speeds_world(pos_interp, fps, last_valid_H)
    else:
        raw_speeds = [None] * len(pos_interp)
    smooth_speeds = smooth(raw_speeds, 5)
    _t2_speed = time.perf_counter()

    _t2_total = _t2_speed - _t2_start

    def _get_speed_after(frame_idx: int, look_ahead: int = 5) -> Optional[float]:
        peaks = [smooth_speeds[j] for j in range(
            frame_idx, min(frame_idx + look_ahead, len(smooth_speeds)))
            if smooth_speeds[j] is not None and smooth_speeds[j] >= MIN_BALL_SPEED_KMH]
        return round(max(peaks), 1) if peaks else None

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 3：統計彙整 → JSON
    # ─────────────────────────────────────────────────────────────────────────
    _t3_start = time.perf_counter()

    def _speed_stats(speeds: List[float]) -> Dict:
        if not speeds:
            return {"avg_kmh": None, "max_kmh": None, "min_kmh": None, "count": 0}
        return {
            "avg_kmh": round(sum(speeds) / len(speeds), 1),
            "max_kmh": round(max(speeds), 1),
            "min_kmh": round(min(speeds), 1),
            "count": len(speeds),
        }

    player_stats: Dict = {
        "top":    {"shots": 0, "serves": 0, "winners": 0,
                   "shot_types": {"serve": 0, "overhead": 0, "swing": 0, "unknown": 0}},
        "bottom": {"shots": 0, "serves": 0, "winners": 0,
                   "shot_types": {"serve": 0, "overhead": 0, "swing": 0, "unknown": 0}},
    }
    all_speeds_kmh: List[float] = []
    serve_speeds_kmh: List[float] = []
    rally_speeds_kmh: List[float] = []
    depth_counts: Dict = {"net": 0, "service": 0, "baseline": 0}
    player_depth: Dict = {
        "top":    {"net": 0, "service": 0, "baseline": 0},
        "bottom": {"net": 0, "service": 0, "baseline": 0},
    }

    heatmap_contacts: List[Dict] = []
    heatmap_bounces: List[Dict] = []
    heatmap_top: List[Dict] = []
    heatmap_bot: List[Dict] = []

    rallies_out: List[Dict] = []

    for rally_idx, rally_contacts in enumerate(rally_groups):
        if not rally_contacts:
            continue

        next_start = (rally_groups[rally_idx + 1][0]
                      if rally_idx + 1 < len(rally_groups) else None)

        # 該回合範圍內的 bounce
        r_end_f = (next_start - 1) if next_start else total_frames_actual
        rally_bounces_f = [b for b in bounces_f
                           if rally_contacts[0] <= b <= r_end_f]

        # 發球者 = 回合第一拍的最近球員
        serve_ball = pos_interp[rally_contacts[0]] if rally_contacts[0] < len(pos_interp) else None
        server = (assign_player(rally_contacts[0], serve_ball,
                                all_player_top, all_player_bottom)
                  if serve_ball else "bottom")
        player_stats[server]["serves"] += 1

        # ── 逐拍建立 shots ────────────────────────────────────────────────
        shots_out: List[Dict] = []
        for seq, fi in enumerate(rally_contacts):
            ball_pos = pos_interp[fi] if fi < len(pos_interp) else None
            if ball_pos is None:
                continue

            player = assign_player(fi, ball_pos, all_player_top, all_player_bottom)
            is_serve = (seq == 0)
            shot_type = classify_shot_type(is_serve, all_action_events, fi)
            speed = _get_speed_after(fi)

            x_norm = round(ball_pos[0] / width, 4)
            y_norm = round(ball_pos[1] / height, 4)

            # 球員站位
            plist = all_player_top if player == "top" else all_player_bottom
            ppos = plist[fi] if fi < len(plist) else None
            player_pos = ({"x": round(ppos[0] / width, 4), "y": round(ppos[1] / height, 4)}
                          if ppos else None)

            # 世界座標
            ball_world = project_to_world(ball_pos, last_valid_H) if last_valid_H is not None else None
            player_world = (project_to_world(ppos, last_valid_H)
                            if ppos is not None and last_valid_H is not None else None)

            # 站位區域（優先世界座標，fallback 像素 y 區間）
            if player_world:
                p_zone = player_court_zone(player_world[1])
            else:
                dist_net = abs(y_norm - 0.5) / 0.5
                p_zone = "net" if dist_net < 0.17 else ("service" if dist_net < 0.54 else "baseline")

            # 統計累計
            player_stats[player]["shots"] += 1
            player_stats[player]["shot_types"][shot_type] += 1
            depth_counts[p_zone] += 1
            player_depth[player][p_zone] += 1
            if speed is not None:
                all_speeds_kmh.append(speed)
                (serve_speeds_kmh if is_serve else rally_speeds_kmh).append(speed)

            heatmap_contacts.append({"x": x_norm, "y": y_norm, "player": player})
            if ppos:
                bucket = heatmap_top if player == "top" else heatmap_bot
                bucket.append({"x": round(ppos[0] / width, 4),
                                "y": round(ppos[1] / height, 4)})

            shot: Dict = {
                "seq": seq + 1,
                "frame": fi,
                "time_sec": round(fi / fps, 3),
                "player": player,
                "is_serve": is_serve,
                "shot_type": shot_type,
                "speed_kmh": speed,
                "ball_pos": {"x": x_norm, "y": y_norm},
                "player_pos": player_pos,
                "player_zone": p_zone,
            }
            if ball_world:
                shot["ball_world"] = {"x": round(ball_world[0], 2),
                                      "y": round(ball_world[1], 2)}
            if player_world:
                shot["player_world"] = {"x": round(player_world[0], 2),
                                        "y": round(player_world[1], 2)}
            shots_out.append(shot)

        # ── 逐落點建立 bounces ────────────────────────────────────────────
        bounces_out: List[Dict] = []
        for bf in rally_bounces_f:
            bpos = pos_interp[bf] if bf < len(pos_interp) else None
            if bpos is None:
                continue
            x_n = round(bpos[0] / width, 4)
            y_n = round(bpos[1] / height, 4)
            b_world = project_to_world(bpos, last_valid_H) if last_valid_H is not None else None
            b_entry: Dict = {
                "frame": bf,
                "time_sec": round(bf / fps, 3),
                "pos": {"x": x_n, "y": y_n},
            }
            if b_world:
                b_entry["world"] = {"x": round(b_world[0], 2), "y": round(b_world[1], 2)}
                b_entry["zone"] = bounce_zone(b_world[0], b_world[1])
            bounces_out.append(b_entry)
            heatmap_bounces.append({"x": x_n, "y": y_n})

        # ── 回合結果 ──────────────────────────────────────────────────────
        winner_frame = find_winner_after(bounces_f, rally_contacts[-1], next_start, fps)
        winner_pos = None
        winner_player: Optional[str] = None
        if winner_frame is not None:
            wb = pos_interp[winner_frame] if winner_frame < len(pos_interp) else None
            if wb:
                winner_player = "bottom" if server == "top" else "top"
                player_stats[winner_player]["winners"] += 1
                winner_pos = {"x": round(wb[0] / width, 4), "y": round(wb[1] / height, 4)}

        cut_near = any(rally_contacts[0] <= sc <= rally_contacts[-1] + int(fps * 3)
                       for sc in scene_cut_frames)
        outcome_type = ("winner" if winner_frame is not None
                        else "scene_cut" if cut_near else "unknown")

        rallies_out.append({
            "id": rally_idx + 1,
            "start_frame": rally_contacts[0],
            "end_frame": rally_contacts[-1],
            "start_time_sec": round(rally_contacts[0] / fps, 2),
            "end_time_sec": round(rally_contacts[-1] / fps, 2),
            "duration_sec": round((rally_contacts[-1] - rally_contacts[0]) / fps, 2),
            "shot_count": len(shots_out),
            "server": server,
            "shots": shots_out,
            "bounces": bounces_out,
            "outcome": {
                "type": outcome_type,
                "winner_player": winner_player,
                "winner_pos": winner_pos,
            },
        })

    total_rallies = len(rallies_out)
    total_shots = sum(r["shot_count"] for r in rallies_out)
    avg_rally_length = round(total_shots / total_rallies, 1) if total_rallies > 0 else 0.0

    if progress_cb:
        progress_cb(99, 100)

    result = {
        "metadata": {
            "fps": round(fps, 3),
            "width": width,
            "height": height,
            "total_frames": total_frames_actual,
            "duration_sec": round(duration, 2),
            "court_detected": last_valid_H is not None,
            "scene_cuts": scene_cut_frames,
        },
        "summary": {
            "total_rallies": total_rallies,
            "total_shots": total_shots,
            "total_winners": sum(1 for r in rallies_out
                                 if r["outcome"]["type"] == "winner"),
            "avg_rally_length": avg_rally_length,
            "players": {
                "top":    player_stats["top"],
                "bottom": player_stats["bottom"],
            },
            "speed": {
                "all":    _speed_stats(all_speeds_kmh),
                "serves": _speed_stats(serve_speeds_kmh),
                "rally":  _speed_stats(rally_speeds_kmh),
            },
            "depth": {
                "total": depth_counts,
                "top":   player_depth["top"],
                "bottom": player_depth["bottom"],
            },
        },
        "rallies": rallies_out,
        "heatmap": {
            "contacts":       heatmap_contacts,
            "bounces":        heatmap_bounces,
            "top_player":     heatmap_top,
            "bottom_player":  heatmap_bot,
        },
    }

    _t3_agg = time.perf_counter()

    with out_json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    _t3_json = time.perf_counter()

    _t3_total = _t3_json - _t3_start
    _grand_total = total_measured + _t2_total + _t3_total

    def _pct(s: float) -> str:
        return f"{s / _grand_total * 100:>4.1f}%"

    print(f"\n[TIMING] {total_frames_actual} frames, {duration:.1f}s @ {fps:.1f}fps  |  "
          f"{n_processed} processed, {n_skip} skipped [{skip_pct:.1f}%]  |  "
          f"grand total {_grand_total:.3f}s")
    print(f"  {'Section':<18}  {'Total(s)':>8}  {'ms/frame':>9}  {'basis':>6}  {'%':>6}")
    print(f"  {'-'*18}  {'-'*8}  {'-'*9}  {'-'*6}  {'-'*6}")
    for name, sec in _t.items():
        basis = _n[name]
        label = "all" if basis == n_all else "proc"
        print(f"  {name:<18}  {sec:>8.3f}  {sec/basis*1000:>8.2f}  {label:>6}  {_pct(sec)}")
    print(f"  {'  detect_events':<18}  {_t2_events - _t2_start:>8.3f}  {'':>9}  {'':>6}  {_pct(_t2_events - _t2_start)}")
    print(f"  {'  segment_rallies':<18}  {_t2_rallies - _t2_events:>8.3f}  {'':>9}  {'':>6}  {_pct(_t2_rallies - _t2_events)}")
    print(f"  {'  full_interpolate':<18}  {_t2_interp - _t2_rallies:>8.3f}  {'':>9}  {'':>6}  {_pct(_t2_interp - _t2_rallies)}")
    print(f"  {'  speed_compute':<18}  {_t2_speed - _t2_interp:>8.3f}  {'':>9}  {'':>6}  {_pct(_t2_speed - _t2_interp)}")
    print(f"  {'  aggregation':<18}  {_t3_agg - _t3_start:>8.3f}  {'':>9}  {'':>6}  {_pct(_t3_agg - _t3_start)}")

    print(f"  {'-'*18}  {'-'*8}  {'-'*9}  {'-'*6}  {'-'*6}")
    print(f"  {'TOTAL':<18}  {_grand_total:>8.3f}  {_grand_total/n_all*1000:>8.2f}  {'all':>6}  100.0%")
    print(f"  → effective {1/(_grand_total/n_all):.1f} fps (all) / {1/(_grand_total/n_proc):.1f} fps (processed only)\n")

    if progress_cb:
        progress_cb(100, 100)

    return str(out_json_path), str(out_video_path)
