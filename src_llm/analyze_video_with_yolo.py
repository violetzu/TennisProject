import cv2
import os
from typing import Callable, Dict, Optional, Deque, Tuple, Any
from pathlib import Path
import subprocess
import numpy as np
from collections import deque

from .utils import get_video_meta


# ==========================================
#  小工具：從 Ultralytics 結果取出 boxes/conf
#  兼容 YOLO detect / pose 的 results 物件
# ==========================================
def _extract_xyxy_conf(result) -> Tuple[list, list]:
    """
    回傳: (xyxy_list, conf_list)
    每個都是 python list
    """
    if result is None or getattr(result, "boxes", None) is None or len(result.boxes) == 0:
        return [], []
    xyxy = result.boxes.xyxy
    conf = result.boxes.conf
    if xyxy is None or conf is None:
        return [], []
    return xyxy.cpu().tolist(), conf.cpu().tolist()


# ==========================================
#  優化函式 1：檢查球是否合理 (幾何 + 位置 + 顏色)
# ==========================================
def is_valid_ball(box, img_w, img_h, frame=None):
    x1, y1, x2, y2 = map(int, box[:4])
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    area = w * h

    # 1. 幾何過濾
    if area < 60 or area > 3000: return False
    ratio = w / (h + 1e-6)
    if ratio < 0.55 or ratio > 2.0: return False

    # 2. 位置過濾 (排除網子標籤 & 觀眾)
    # 網子高度約在 45%~55%，若物體在此且太靠邊(網柱)，排除
    if (img_h * 0.45 < cy < img_h * 0.55):
        if cx < img_w * 0.30 or cx > img_w * 0.70:
            return False
            
    if cy < img_h * 0.10: return False # 太高
    if cy > img_h * 0.95: return False # 太低

    # 3. 顏色過濾 (排除白線)
    if frame is not None:
        y1_c = max(0, y1); y2_c = min(img_h, y2)
        x1_c = max(0, x1); x2_c = min(img_w, x2)
        roi = frame[y1_c:y2_c, x1_c:x2_c]
        if roi.size > 0:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            # 檢查飽和度 (S)，網球飽和度高，白線極低
            avg_s = np.mean(hsv[:, :, 1])
            if avg_s < 40: return False

    return True


# ==========================================
#  優化函式 2：滑動視窗插值（只在 buffer 內補洞）
# ==========================================
def _interpolate_window_boxes(
    boxes_window: list,  # list[Optional[list[float]]], 長度 = WINDOW；元素為 [x1,y1,x2,y2] 或 None
    max_gap: int = 10,
) -> list:
    """
    在「同一個 window」內做線性插值：
    - 對每個 None 位置，找最近的 prev 和 next（都有 box 的）
    - 若 gap <= max_gap 就插值出 box
    回傳新的 list（同長度）
    """
    n = len(boxes_window)
    out = list(boxes_window)

    # 預先記錄所有有效 idx
    valid = [i for i, b in enumerate(out) if b is not None]
    if len(valid) < 2:
        return out

    for i in range(n):
        if out[i] is not None:
            continue

        # 找 prev
        prev_i = None
        for j in range(i - 1, -1, -1):
            if out[j] is not None:
                prev_i = j
                break

        # 找 next
        next_i = None
        for j in range(i + 1, n):
            if out[j] is not None:
                next_i = j
                break

        if prev_i is None or next_i is None:
            continue

        gap = next_i - prev_i
        if gap <= 0 or gap > max_gap:
            continue

        # 線性插值
        t = (i - prev_i) / gap
        b0 = np.array(out[prev_i], dtype=np.float32)
        b1 = np.array(out[next_i], dtype=np.float32)
        bi = (1 - t) * b0 + t * b1
        out[i] = bi.tolist()

    return out


# ==========================================
#  主函式：單趟解碼(FFmpeg) + 同幀跑球/姿態 + 滑窗插值 + 直接輸出
# ==========================================
def analyze_video_with_yolo(
    video_path: str,
    max_frames: Optional[int] = 128000,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    ball_model=None,
    pose_model=None,
) -> str:
    if ball_model is None or pose_model is None:
        raise ValueError("ball_model / pose_model 不可為 None")

    meta = get_video_meta(video_path)
    if not meta["fps"]: raise RuntimeError(f"無法開啟影片：{video_path}")
    fps, width, height = meta["fps"], meta["width"], meta["height"]

    src_path = Path(video_path)
    out_path = src_path.parent / (src_path.stem + "_yolo_final.mp4")

    # ---------------------------
    # Pose 繪圖設定
    # ---------------------------
    SKELETON_LINKS = [
        [5, 6], [5, 11], [6, 12], [11, 12], [5, 7], [7, 9], [6, 8], [8, 10],
        [11, 13], [13, 15], [12, 14], [14, 16]
    ]

    # ---------------------------
    # 球：範圍內挑最高 conf + 卡住重抓一次（沿用你原邏輯）
    # ---------------------------
    last_center = None
    last_seen_idx = None

    MAX_SPEED_PX_PER_S = 6000
    BASE_DIST = 100
    MAX_DIST_CLAMP = int(width * 0.20)

    STUCK_MOVE_PX = 6
    STUCK_FRAMES = 8
    stuck_count = 0
    force_global_next = False

    miss_count = 0
    RESET_AFTER_MISS = 10
    ####### ---------------------------
    # Kalman tracker：狀態 [x, y, vx, vy]（單位：pixel / frame）
    # ---------------------------
    kalman = cv2.KalmanFilter(4, 2)

    kalman.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=np.float32)

    kalman.transitionMatrix = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)

    # 這兩個噪聲可以調：越大=越不信任模型/越滑順
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 100.0  # (std≈10px)

    kalman.errorCovPost = np.eye(4, dtype=np.float32)
    kalman_inited = False

    # Kalman 的額外 gating（避免離譜誤判把 tracker 拉走）
    KALMAN_GATE_CLAMP = int(width * 0.15)  # 1080p/1920 建議先 0.12~0.18
    #######

    # ---------------------------
    # A 方法：滑動視窗 buffer
    # ---------------------------
    WINDOW = 12       # 建議 10~15
    MAX_GAP = 10      # window 內可補洞長度

    frame_buf: Deque[np.ndarray] = deque(maxlen=WINDOW)          # 已畫好「姿態」的 frame
    ball_box_buf: Deque[Optional[list]] = deque(maxlen=WINDOW)   # 每幀 ball box（raw）: [x1,y1,x2,y2] 或 None

    # ---------------------------
    # FFmpeg 解碼（raw bgr24）
    # ---------------------------
    # -vsync 0：避免補幀/丟幀導致幀數不一致
    # -an：不要音訊
    decode_cmd = [
        "/usr/bin/ffmpeg",
        "-hide_banner", "-loglevel", "error",
        "-i", str(video_path),
        "-an",
        "-vf", f"scale={width}:{height}",
        "-pix_fmt", "bgr24",
        "-f", "rawvideo",
        "-vsync", "0",
        "pipe:1",
    ]
    dec = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # FFmpeg 編碼輸出（raw bgr24 -> h264）
    encode_cmd = [
        "/usr/bin/ffmpeg", "-y",
        "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    enc = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    # 進度：單趟，所以 total_steps = max_frames（保守）
    total_steps = int(max_frames) if max_frames is not None else 0

    frame_size = width * height * 3
    idx = 0

    try:
        while True:
            if max_frames is not None and idx >= max_frames:
                break

            raw = dec.stdout.read(frame_size) if dec.stdout else b""
            if not raw or len(raw) < frame_size:
                break

            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
            frame_draw = frame.copy()

            # =========================================================
            # 1) 跑 pose（同幀）
            # =========================================================
            # Ultralytics: pose_model.predict(frame, ...)
            pose_results = pose_model.predict(
                source=frame, imgsz=1280, conf=0.10, verbose=False
            )
            pose_r = pose_results[0] if pose_results else None

            final_selection = []
            if pose_r is not None and getattr(pose_r, "keypoints", None) is not None and getattr(pose_r, "boxes", None) is not None:
                if pose_r.keypoints.data is not None and len(pose_r.boxes) > 0:
                    kps_list = pose_r.keypoints.data.cpu().tolist()
                    bx_list = pose_r.boxes.xyxy.cpu().tolist()

                    mid_y = height * 0.55
                    center_x = width / 2
                    upper_candidates = []
                    lower_candidates = []

                    for box, kp in zip(bx_list, kps_list):
                        x1, y1, x2, y2 = box[:4]
                        cx = (x1 + x2) / 2
                        feet_y = y2
                        area = (x2 - x1) * (y2 - y1)

                        if cx<width*0.05 or cx>width*0.95:
                            continue
                        if area < 100:
                            continue
                        if feet_y < height * 0.15:
                            continue

                        cand = {"area": area, "kps": kp}
                        if feet_y < mid_y:
                            upper_candidates.append(cand)
                        else:
                            lower_candidates.append(cand)

                    if upper_candidates:
                        final_selection.append(max(upper_candidates, key=lambda x: x["area"]))
                    if lower_candidates:
                        final_selection.append(max(lower_candidates, key=lambda x: x["area"]))

            # 畫骨架
            for cand in final_selection:
                kps = cand["kps"]
                for (x, y, conf) in kps:
                    if conf < 0.3:
                        continue
                    cv2.circle(frame_draw, (int(x), int(y)), 4, (0, 255, 0), -1)
                for i_link, j_link in SKELETON_LINKS:
                    if i_link >= len(kps) or j_link >= len(kps):
                        continue
                    x1, y1, c1 = kps[i_link]
                    x2, y2, c2 = kps[j_link]
                    if c1 < 0.3 or c2 < 0.3:
                        continue
                    cv2.line(frame_draw, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # =========================================================
            # 2) 跑 ball（同幀）+ 你的 gating/卡住重抓邏輯
            # =========================================================
            ball_results = ball_model.predict(
                source=frame, imgsz=1280, conf=0.35, iou=0.5, verbose=False
            )
            ball_r = ball_results[0] if ball_results else None
            xyxy, confs = _extract_xyxy_conf(ball_r)

            # --- Kalman predict（有 init 才 predict）---
            pred_center = None
            if kalman_inited:
                pred = kalman.predict()
                pred_center = (float(pred[0]), float(pred[1]))

            chosen_box = None

            if xyxy:
                # 1) 過濾候選
                candidates = []
                for box, conf in zip(xyxy, confs):
                    if not is_valid_ball(box, width, height, frame=frame):
                        continue

                    cx = (box[0] + box[2]) / 2.0
                    cy = (box[1] + box[3]) / 2.0

                    # ✅ 額外：排除左下角比分板（US Open 轉播很常誤判）
                    if cy > height * 0.78 and cx < width * 0.32:
                        continue

                    candidates.append((box, float(conf), (cx, cy)))

                # 2) 全局最高 or 範圍內最高
                use_global = (not kalman_inited) or (pred_center is None) or force_global_next

                chosen = None
                if candidates:
                    if use_global:
                        # 全局先挑 conf 最高
                        cand = max(candidates, key=lambda t: t[1])

                        # 但如果 Kalman 已 init，仍要離預測不要太遠，避免離譜誤判
                        if kalman_inited and pred_center is not None:
                            _, _, (cx, cy) = cand
                            dist_to_pred = float(np.hypot(cx - pred_center[0], cy - pred_center[1]))
                            if dist_to_pred <= KALMAN_GATE_CLAMP:
                                chosen = cand
                            else:
                                chosen = None
                        else:
                            chosen = cand
                    else:
                        # 範圍內挑 conf 最高（以 Kalman 預測點為中心）
                        gap = idx - last_seen_idx if last_seen_idx is not None else 1
                        gap = max(int(gap), 1)

                        max_dist = BASE_DIST + (MAX_SPEED_PX_PER_S / fps) * gap
                        max_dist = min(max_dist, MAX_DIST_CLAMP)

                        # 再加一層 Kalman clamp（兩者取 min）
                        max_dist = min(max_dist, KALMAN_GATE_CLAMP)

                        gated = []
                        for box, conf, (cx, cy) in candidates:
                            dist = float(np.hypot(cx - pred_center[0], cy - pred_center[1]))
                            if dist <= max_dist:
                                gated.append((box, conf, (cx, cy)))

                        if gated:
                            chosen = max(gated, key=lambda t: t[1])

                if chosen is not None:
                    box, conf, (cx, cy) = chosen
                    chosen_box = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]

                    # --- Kalman correct：只在「接受的測量」時更新 ---
                    meas = np.array([[np.float32(cx)], [np.float32(cy)]], dtype=np.float32)
                    if not kalman_inited:
                        kalman.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
                        kalman.statePre = kalman.statePost.copy()
                        kalman_inited = True
                    else:
                        kalman.correct(meas)

                    # 原本卡住判定（可留著）
                    if last_center is not None:
                        move = float(np.hypot(cx - last_center[0], cy - last_center[1]))
                        if move < STUCK_MOVE_PX:
                            stuck_count += 1
                        else:
                            stuck_count = 0
                    else:
                        stuck_count = 0

                    if stuck_count >= STUCK_FRAMES:
                        force_global_next = True
                        stuck_count = 0
                    else:
                        force_global_next = False

                    last_center = (cx, cy)
                    last_seen_idx = idx
                    miss_count = 0
                else:
                    miss_count += 1
            else:
                miss_count += 1

            # 太久沒球：重置（連 Kalman 一起 reset）
            if miss_count >= RESET_AFTER_MISS:
                last_center = None
                last_seen_idx = None
                force_global_next = False
                stuck_count = 0
                miss_count = 0
                kalman_inited = False


            # =========================================================
            # 3) 丟進 buffer（frame 已畫姿態，球先不畫）
            # =========================================================
            frame_buf.append(frame_draw)
            ball_box_buf.append(chosen_box)

            # =========================================================
            # 4) 若 buffer 滿了：在 window 內插值 -> 把最舊那幀定稿輸出
            # =========================================================
            if len(frame_buf) == WINDOW:
                # window 插值（只補洞）
                interp_boxes = _interpolate_window_boxes(list(ball_box_buf), max_gap=MAX_GAP)

                # 最舊那幀
                out_frame = frame_buf[0].copy()
                out_box = interp_boxes[0]

                if out_box is not None:
                    x1, y1, x2, y2 = map(int, out_box[:4])
                    # 插值畫橘色 + 小點
                    is_interp = (ball_box_buf[0] is None)
                    color = (0, 165, 255) if is_interp else (0, 255, 255)
                    cv2.rectangle(out_frame, (x1, y1), (x2, y2), color, 2)
                    if is_interp:
                        cv2.circle(out_frame, ((x1 + x2) // 2, (y1 + y2) // 2), 3, color, -1)

                if enc.stdin:
                    enc.stdin.write(np.ascontiguousarray(out_frame).tobytes())

                # pop 最舊（因為已輸出）
                frame_buf.popleft()
                ball_box_buf.popleft()

            idx += 1
            if progress_callback and total_steps:
                progress_callback(min(idx, total_steps), total_steps)

        # =========================================================
        # 影片結尾：flush buffer（尾端無法用「未來幀」插值，只能用 window 內現有資料補一次）
        # =========================================================
        while len(frame_buf) > 0:
            interp_boxes = _interpolate_window_boxes(list(ball_box_buf), max_gap=MAX_GAP)
            out_frame = frame_buf[0].copy()
            out_box = interp_boxes[0]

            if out_box is not None:
                x1, y1, x2, y2 = map(int, out_box[:4])
                is_interp = (ball_box_buf[0] is None)
                color = (0, 165, 255) if is_interp else (0, 255, 255)
                cv2.rectangle(out_frame, (x1, y1), (x2, y2), color, 2)
                if is_interp:
                    cv2.circle(out_frame, ((x1 + x2) // 2, (y1 + y2) // 2), 3, color, -1)

            if enc.stdin:
                enc.stdin.write(np.ascontiguousarray(out_frame).tobytes())

            frame_buf.popleft()
            ball_box_buf.popleft()

    finally:
        # 關閉解碼
        try:
            if dec.stdout:
                dec.stdout.close()
        except Exception:
            pass
        try:
            dec.wait(timeout=5)
        except Exception:
            pass

        # 關閉編碼
        try:
            if enc.stdin:
                enc.stdin.close()
        except Exception:
            pass
        try:
            enc.wait(timeout=30)
        except Exception:
            pass

        # 若 ffmpeg 有錯，丟出更清楚訊息
        if dec.returncode not in (0, None):
            err = b""
            try:
                err = dec.stderr.read() if dec.stderr else b""
            except Exception:
                pass
            raise RuntimeError(f"FFmpeg 解碼失敗 (code={dec.returncode}).\n{err.decode('utf-8', errors='ignore')}")

        if enc.returncode not in (0, None):
            err = b""
            try:
                err = enc.stderr.read() if enc.stderr else b""
            except Exception:
                pass
            raise RuntimeError(f"FFmpeg 編碼失敗 (code={enc.returncode}).\n{err.decode('utf-8', errors='ignore')}")

    return str(out_path)
