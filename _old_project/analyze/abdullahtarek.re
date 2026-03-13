import cv2
import os
from typing import Callable, Dict, Optional
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
from .utils import get_video_meta

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
#  優化函式 2：軌跡插值
# ==========================================
def interpolate_ball_trajectory(ball_dict: Dict[int, list], max_gap=10) -> Dict[int, list]:
    if not ball_dict: return {}
    frames = sorted(ball_dict.keys())
    full_idx = np.arange(frames[0], frames[-1] + 1)
    
    data = []
    for f in full_idx:
        if f in ball_dict and len(ball_dict[f]) > 0:
            data.append(ball_dict[f][0][:4])
        else:
            data.append([np.nan]*4)
            
    df = pd.DataFrame(data, index=full_idx, columns=['x1', 'y1', 'x2', 'y2'])
    df = df.interpolate(method='linear', limit=max_gap, limit_direction='both')
    
    new_ball_dict = {}
    for idx, row in df.iterrows():
        if not np.isnan(row['x1']):
            new_ball_dict[int(idx)] = [[row['x1'], row['y1'], row['x2'], row['y2'], -1.0]]
    return new_ball_dict

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
    meta = get_video_meta(video_path)
    if not meta["fps"]: raise RuntimeError(f"無法開啟影片：{video_path}")
    fps, width, height = meta["fps"], meta["width"], meta["height"]

    src_path = Path(video_path)
    out_path = src_path.parent / (src_path.stem + "_yolo_final.mp4")

    # =========================================================
    #  第一趟：球偵測 (已修復 idx 重複計數問題)
    # =========================================================
    ball_boxes_by_frame: Dict[int, list] = {}
    idx = 0
    total_steps = max_frames * 2 

    for r in ball_model.track(
        source=video_path, stream=True, tracker="bytetrack.yaml", persist=True,
        imgsz=1280, conf=0.15, iou=0.5, verbose=False
    ):
        if idx >= max_frames: break

        current_frame = r.orig_img 

        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().tolist()
            confs = r.boxes.conf.cpu().tolist()
            best_box = None
            max_conf = -1.0

            for box, conf in zip(xyxy, confs):
                if is_valid_ball(box, width, height, frame=current_frame):
                    if conf > max_conf:
                        max_conf = conf
                        best_box = box
            
            if best_box is not None:
                ball_boxes_by_frame[idx] = [best_box]

        # [重要修正] 這裡只加一次！骨架不會再跑掉了
        idx += 1
        if progress_callback: progress_callback(min(idx, max_frames), total_steps)

    print("正在優化球的軌跡...")
    ball_boxes_by_frame = interpolate_ball_trajectory(ball_boxes_by_frame)

    # =========================================================
    #  第二趟：人物 (透視漏斗過濾法 - 解決球僮問題)
    # =========================================================
    SKELETON_LINKS = [
        [5, 6], [5, 11], [6, 12], [11, 12], [5, 7], [7, 9], [6, 8], [8, 10],
        [11, 13], [13, 15], [12, 14], [14, 16]
    ]

    ffmpeg_cmd = [
        "/usr/bin/ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_path)
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    idx = 0
    try:
        for r in pose_model(source=video_path, stream=True, imgsz=1280, conf=0.15, verbose=False):
            if idx >= max_frames: break
            frame = r.orig_img.copy()
            
            # --- 1. 繪製球 ---
            for box in ball_boxes_by_frame.get(idx, []):
                x1, y1, x2, y2 = map(int, box[:4])
                is_interp = (len(box) > 4 and box[4] == -1.0)
                color = (0, 165, 255) if is_interp else (0, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                if is_interp: cv2.circle(frame, ((x1+x2)//2, (y1+y2)//2), 3, color, -1)

            # --- 2. 繪製人物 (透視漏斗篩選) ---
            final_selection = []
            
            # [除錯用] 繪製漏斗邊界 (紅線) - 跑一次確認沒問題後可註解掉
            # cv2.line(frame, (int(width*(0.5-0.08)), int(height*0.15)), (int(width*(0.5-0.5)), int(height)), (0,0,255), 2)
            # cv2.line(frame, (int(width*(0.5+0.08)), int(height*0.15)), (int(width*(0.5+0.5)), int(height)), (0,0,255), 2)

            if r.keypoints and r.boxes and r.keypoints.data is not None:
                kps_list = r.keypoints.data.cpu().tolist()
                bx_list = r.boxes.xyxy.cpu().tolist()
                
                mid_y = height * 0.55
                center_x = width / 2
                upper_candidates = []
                lower_candidates = []

                for i, (box, kp) in enumerate(zip(bx_list, kps_list)):
                    x1, y1, x2, y2 = box[:4]
                    cx = (x1 + x2) / 2
                    feet_y = y2 
                    area = (x2 - x1) * (y2 - y1)

                    # --- [關鍵] 透視漏斗過濾 ---
                    dist_from_center = abs(cx - center_x) / width
                    y_ratio = feet_y / height
                    
                    # 遠端(top)只有 8% 寬度，近端(bottom)有 50% 寬度
                    max_allowed_dist = 0.08 + (0.42 * y_ratio)
                    
                    # 踢掉超出漏斗的人 (球僮/線審)
                    if dist_from_center > max_allowed_dist:
                        continue

                    if area < 400: continue
                    if feet_y < height * 0.15: continue

                    cand = {"area": area, "kps": kp}
                    if feet_y < mid_y:
                        upper_candidates.append(cand)
                    else:
                        lower_candidates.append(cand)

                # 選面積最大
                if upper_candidates:
                    final_selection.append(max(upper_candidates, key=lambda x: x["area"]))
                if lower_candidates:
                    final_selection.append(max(lower_candidates, key=lambda x: x["area"]))

            # --- 3. 畫骨架 ---
            for cand in final_selection:
                kps = cand["kps"]
                for (x, y, conf) in kps:
                    if conf < 0.3: continue
                    cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
                for i_link, j_link in SKELETON_LINKS:
                    if i_link >= len(kps) or j_link >= len(kps): continue
                    x1, y1, c1 = kps[i_link]
                    x2, y2, c2 = kps[j_link]
                    if c1 < 0.3 or c2 < 0.3: continue
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            if proc.stdin:
                proc.stdin.write(np.ascontiguousarray(frame).tobytes())

            idx += 1
            if progress_callback: progress_callback(min(idx, max_frames) + max_frames, total_steps)

    finally:
        if proc.stdin: proc.stdin.close()
        proc.wait()

    return str(out_path)