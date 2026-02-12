import cv2
import os
import time
from typing import Callable, Dict, Optional, Deque, Tuple, Any
from pathlib import Path
import subprocess
import numpy as np
from collections import deque
from .CW_action_test import ActionRecognizer

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
#  優化函式 1：檢查球是否合理 (抗模糊 + 抗誤判版)
# ==========================================
def is_valid_ball(box, img_w, img_h, frame=None):
    x1, y1, x2, y2 = map(int, box[:4])
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    area = w * h

    # 1. 幾何過濾 ( Turn 6 修正：放寬以容忍動態模糊)
    # 遠處的球面積很小 (20px)，高速球會變得很長 (Ratio > 2.0)
    if area < 10 or area > 3000: return False
    
    ratio = w / (h + 1e-6)
    if ratio < 0.15 or ratio > 6.0: return False # 容忍長條拖影

    # 2. 位置過濾 (排除網子標籤 & 觀眾)
    # 網子高度約在 45%~55%，若物體在此且太靠邊(網柱)，排除
    if (img_h * 0.44 < cy < img_h * 0.56): 
        if cx < img_w * 0.30 or cx > img_w * 0.70:
            return False
            
    if cy < img_h * 0.05: return False # 太高
    if cy > img_h * 0.98: return False # 太低

    # 3. 顏色過濾 ( Turn 6 修正：降低飽和度門檻)
    if frame is not None:
        y1_c = max(0, y1); y2_c = min(img_h, y2)
        x1_c = max(0, x1); x2_c = min(img_w, x2)
        roi = frame[y1_c:y2_c, x1_c:x2_c]
        if roi.size > 0:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            # 檢查飽和度 (S)
            avg_s = np.mean(hsv[:, :, 1])
            # 高速移動時黃色球會跟藍色地板混色，S值會掉到 20 左右，只過濾純白線 (<15)
            if avg_s < 15: return False 

    return True


# ==========================================
#  優化函式 2：滑動視窗插值
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
#  主函式
# ==========================================
def _interpolate_window_boxes(boxes_window: list, max_gap: int = 10) -> list:
    n = len(boxes_window)
    out = list(boxes_window)
    valid = [i for i, b in enumerate(out) if b is not None]
    if len(valid) < 2: return out

    for i in range(n):
        if out[i] is not None: continue
        prev_i = next((j for j in range(i - 1, -1, -1) if out[j] is not None), None)
        next_i = next((j for j in range(i + 1, n) if out[j] is not None), None)
        if prev_i is None or next_i is None: continue
        gap = next_i - prev_i
        if gap > max_gap: continue
        t = (i - prev_i) / gap
        b0 = np.array(out[prev_i], dtype=np.float32)
        b1 = np.array(out[next_i], dtype=np.float32)
        bi = (1 - t) * b0 + t * b1
        out[i] = bi.tolist()
    return out

# ==========================================
#  主函式
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
    out_path = src_path.parent / (src_path.stem + "_yolo_roi.mp4")
    
    action = ActionRecognizer(fps=fps, img_w=width, img_h=height)

    SKELETON_LINKS = [
        [5, 6], [5, 11], [6, 12], [11, 12], [5, 7], [7, 9], [6, 8], [8, 10],
        [11, 13], [13, 15], [12, 14], [14, 16]
    ]

    ROI_SIZE = 256
    tracking_mode = "global"  
    last_center, last_seen_idx, miss_count = None, None, 0
    stuck_count, kalman_inited = 0, False
    STUCK_FRAMES_LIMIT, STUCK_DIST_THRESHOLD, RESET_AFTER_MISS = 6, 3.0, 15
    
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.01
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 10.0 
    kalman.errorCovPost = np.eye(4, dtype=np.float32)

    WINDOW, MAX_GAP = 12, 10
    frame_buf: Deque[np.ndarray] = deque(maxlen=WINDOW)
    ball_box_buf: Deque[Optional[list]] = deque(maxlen=WINDOW)

    decode_cmd = ["/usr/bin/ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video_path), "-an", "-vf", f"scale={width}:{height}", "-pix_fmt", "bgr24", "-f", "rawvideo", "-vsync", "0", "pipe:1"]
    dec = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    encode_cmd = ["/usr/bin/ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_path)]
    enc = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    idx = 0
    total_steps = int(max_frames) if max_frames is not None else 0
    last_action_txt, last_action_until = "", -1

    try:
        while True:
            if max_frames is not None and idx >= max_frames: break
            raw = dec.stdout.read(width * height * 3) if dec.stdout else b""
            if not raw or len(raw) < width * height * 3: break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
            frame_draw = frame.copy()

            # 1. Pose Estimation
            pose_results = pose_model.predict(source=frame, imgsz=1280, conf=0.15, verbose=False)
            pose_r = pose_results[0] if pose_results else None
            final_selection = []
            if pose_r and getattr(pose_r, "keypoints", None) is not None:
                kps_list, bx_list = pose_r.keypoints.data.cpu().tolist(), pose_r.boxes.xyxy.cpu().tolist()
                for box, kp in zip(bx_list, kps_list):
                    x1, y1, x2, y2 = box[:4]
                    feet_y = y2
                    area = (x2 - x1) * (y2 - y1)
                    if area < 150 or feet_y < height * 0.1: continue
                    if sum(1 for (_, _, c) in kp if c >= 0.3) < 8: continue
                    if feet_y > height * 0.55:
                        final_selection.append({"area": area, "kps": kp})
                if final_selection: 
                    final_selection = [max(final_selection, key=lambda x: x["area"])]

            # 繪製骨架
            for cand in final_selection:
                kps = cand["kps"]
                for (x, y, conf) in kps:
                    if conf >= 0.3: cv2.circle(frame_draw, (int(x), int(y)), 4, (0, 255, 0), -1)
                for i, j in SKELETON_LINKS:
                    if kps[i][2] >= 0.3 and kps[j][2] >= 0.3:
                        cv2.line(frame_draw, (int(kps[i][0]), int(kps[i][1])), (int(kps[j][0]), int(kps[j][1])), (0, 255, 0), 2)

            # 2. Action Recognition
            events = action.update_from_candidates(final_selection, frame_idx=idx, ball_pos=last_center)
            if events:
                e = events[-1]
                if e.name == "swing" and float(e.score) >= 0.90:
                    last_action_txt = f"{e.name} ({e.score:.2f})"
                    last_action_until = idx + int(0.5 * fps)
        
                    # --- 核心邏輯：擊球後應對 ---
                    if kalman_inited:
                        # 1. 速度反向：將 Kalman 狀態中的 vx, vy 反轉並稍微減速
                        kalman.statePost[2] *= -0.7 
                        kalman.statePost[3] *= -0.7
                        # 2. 降低信心：讓 Kalman 在擊球後更願意聽從 YOLO 的新偵測結果
                        kalman.errorCovPost *= 5.0 
            
                    # 3. 模式切換：擊球後 5 幀內強制切換到全域，因為小 ROI 很容易跟丟反向的球
                    tracking_mode = "global"

            # 3. Ball Detection (ROI 邏輯)
            pred_c = None
            if kalman_inited:
                pred = kalman.predict()
                pred_c = (float(pred[0]), float(pred[1]))

            detect_source = frame
            offset_x, offset_y = 0, 0
            current_imgsz = 1280

            if tracking_mode == "local" and pred_c:
                offset_x = int(max(0, min(pred_c[0] - ROI_SIZE // 2, width - ROI_SIZE)))
                offset_y = int(max(0, min(pred_c[1] - ROI_SIZE // 2, height - ROI_SIZE)))
                detect_source = frame[offset_y:offset_y+ROI_SIZE, offset_x:offset_x+ROI_SIZE]
                current_imgsz = ROI_SIZE
            
            ball_r_list = ball_model.predict(source=detect_source, imgsz=current_imgsz, conf=0.1, verbose=False)
            
            chosen = None
            if ball_r_list:
                xyxy_roi, confs = _extract_xyxy_conf(ball_r_list[0])
                cands = []
                for bx, conf in zip(xyxy_roi, confs):
                    gx1, gy1, gx2, gy2 = bx[0]+offset_x, bx[1]+offset_y, bx[2]+offset_x, bx[3]+offset_y
                    g_box = [gx1, gy1, gx2, gy2]
                    g_center = ((gx1+gx2)/2, (gy1+gy2)/2)
                    if is_valid_ball(g_box, width, height, frame):
                        cands.append((g_box, conf, g_center))
                
                if cands:
                    if tracking_mode == "local":
                        chosen = min(cands, key=lambda c: np.hypot(c[2][0]-pred_c[0], c[2][1]-pred_c[1]))
                    else:
                        chosen = max(cands, key=lambda c: c[1])

            # 4. 更新狀態與 Kalman
            chosen_box = None
            if chosen:
                bx, _, (cbx, cby) = chosen
                if last_center and np.hypot(cbx-last_center[0], cby-last_center[1]) < STUCK_DIST_THRESHOLD:
                    stuck_count += 1
                else:
                    stuck_count = max(0, stuck_count - 1)
                    chosen_box = bx
                    if not kalman_inited:
                        kalman.statePost = np.array([[cbx], [cby], [0], [0]], dtype=np.float32)
                        kalman_inited = True
                    else:
                        kalman.correct(np.array([[np.float32(cbx)], [np.float32(cby)]], dtype=np.float32))
                    last_center, last_seen_idx, miss_count = (cbx, cby), idx, 0
                    tracking_mode = "local"
                
                if stuck_count >= STUCK_FRAMES_LIMIT:
                    tracking_mode, kalman_inited, last_center = "global", False, None
            else:
                miss_count += 1
                

                
                 
                if miss_count >= RESET_AFTER_MISS:
                    tracking_mode, kalman_inited, last_center = "global", False, None

            # 5. Buffer & FFmpeg 寫出
            frame_buf.append(frame_draw)
            ball_box_buf.append(chosen_box)
            if len(frame_buf) == WINDOW:
                ib = _interpolate_window_boxes(list(ball_box_buf), max_gap=MAX_GAP)
                of = frame_buf[0].copy()
                if ib[0]:
                    x1, y1, x2, y2 = map(int, ib[0][:4])
                    cv2.rectangle(of, (x1, y1), (x2, y2), (0, 255, 255), 2)
                
                if enc.stdin: enc.stdin.write(np.ascontiguousarray(of).tobytes())
                frame_buf.popleft(); ball_box_buf.popleft()

            idx += 1
            if progress_callback and total_steps:
                progress_callback(min(idx, total_steps), total_steps)

    finally:
        if dec.stdout: dec.stdout.close()
        dec.wait()
        if enc.stdin: enc.stdin.close()
        enc.wait()
        
    return str(out_path)