import cv2
from typing import Callable, Dict, Optional
from pathlib import Path
import subprocess
import numpy as np

from .utils import get_video_meta

def analyze_video_with_yolo(
    video_path: str,
    max_frames: int = 300,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    ball_model=None,
    pose_model=None,
) -> str:
    """
    YOLO 分析 + 繪圖（同步函式，會在背景執行緒裡跑）

    - 第一趟：偵測 / 追蹤球的位置
    - 第二趟：偵測球員姿態，選出兩位球員畫骨架
    - 每幀畫好後以 raw BGR pipe 給 ffmpeg 壓 mp4
    - progress_callback(done, total) 用來回報進度
    """
    # --- 影片基本資訊 ---
    meta = get_video_meta(video_path)
    if not meta["fps"]:
        raise RuntimeError(f"無法開啟影片：{video_path}")

    fps = meta["fps"]
    width, height = meta["width"], meta["height"]

    src_path = Path(video_path)
    out_name = src_path.stem + "_yolo.mp4"
    out_path = src_path.parent / out_name

    # =========================================================
    #  第一趟：球偵測 + 追蹤 → 存每幀的有效球框
    # =========================================================
    ball_boxes_by_frame: Dict[int, list] = {}
    ball_hist: Dict[int, list] = {}
    idx = 0

    total_steps = max_frames * 2  # 第一趟 + 第二趟

    for r in ball_model.track(
        source=video_path,
        stream=True,
        tracker="bytetrack.yaml",
        persist=True,
        imgsz=1280,
        conf=0.20,
        iou=0.5,
        verbose=False,
    ):
        if idx >= max_frames:
            break

        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().tolist()
            ids = r.boxes.id.cpu().tolist() if r.boxes.id is not None else [None] * len(xyxy)

            for box, tid in zip(xyxy, ids):
                if tid is None:
                    continue
                tid = int(tid)

                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2

                # Anti-Static
                if tid not in ball_hist:
                    ball_hist[tid] = []
                ball_hist[tid].append((cx, cy))
                if len(ball_hist[tid]) > 30:
                    ball_hist[tid].pop(0)

                is_static = False
                if len(ball_hist[tid]) > 10:
                    pts = ball_hist[tid][-15:]
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    if (max(xs) - min(xs) < 20) and (max(ys) - min(ys) < 20):
                        is_static = True

                if not is_static:
                    ball_boxes_by_frame.setdefault(idx, []).append(box)

        idx += 1

        if progress_callback:
            progress_callback(min(idx, max_frames), total_steps)

    # =========================================================
    #  第二趟：人物 / 姿態 + 球員篩選 → 繪圖 + pipe raw 給 ffmpeg
    # =========================================================
    SKELETON_LINKS = [
        [5, 6], [5, 11], [6, 12], [11, 12],
        [5, 7], [7, 9], [6, 8], [8, 10],
        [11, 13], [13, 15], [12, 14], [14, 16],
    ]

    ffmpeg_cmd = [
        "/usr/bin/ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]

    proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    idx = 0

    try:
        for r in pose_model(
            source=video_path,
            stream=True,
            imgsz=1280,
            conf=0.15,
            verbose=False,
        ):
            if idx >= max_frames:
                break

            frame = r.orig_img.copy()
            img_h, img_w = r.orig_shape
            center_x = img_w / 2

            # 這幀沒人，只畫球框
            if not (r.keypoints and r.boxes and r.keypoints.data is not None):
                for box in ball_boxes_by_frame.get(idx, []):
                    x1, y1, x2, y2 = map(int, box[:4])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

                if proc.stdin:
                    out_frame = np.ascontiguousarray(frame)
                    proc.stdin.write(out_frame.tobytes())

                idx += 1
                if progress_callback:
                    step = max_frames + min(idx, max_frames)
                    progress_callback(step, total_steps)
                continue

            kps_list = r.keypoints.data.cpu().tolist()
            bx_list = r.boxes.xyxy.cpu().tolist()

            near_candidates = []
            far_candidates = []

            # ===== 球員選擇演算法 =====
            for i, (b, kp) in enumerate(zip(bx_list, kps_list)):
                cx = (b[0] + b[2]) / 2
                cy = (b[1] + b[3]) / 2
                area = (b[2] - b[0]) * (b[3] - b[1])

                # 太上面直接略過
                if b[3] < img_h * 0.1:
                    continue

                if cy > img_h * 0.5:
                    # 近場（畫面下半），排除超邊緣雜訊
                    if img_w * 0.05 < cx < img_w * 0.95:
                        near_candidates.append({"area": area, "kps": kp, "pid": i})
                else:
                    # 遠場：裁判 / 球童過濾
                    if cx > img_w * 0.80:
                        continue
                    if cx > img_w * 0.70 and b[3] > img_h * 0.40:
                        continue
                    if cx < img_w * 0.10:
                        continue

                    norm_y = b[3] / (img_h * 0.5)
                    ideal_y = 0.35
                    is_too_far = norm_y < 0.20

                    y_score = 1.0 - abs(norm_y - ideal_y) * 3.0
                    dist_ratio = abs(cx - center_x) / (img_w / 2)
                    score = area * y_score * (1 - dist_ratio * 0.2)

                    if is_too_far:
                        score *= 0.1

                    if score > 0:
                        far_candidates.append({
                            "area": area,
                            "score": score,
                            "kps": kp,
                            "pid": i
                        })

            final_selection = []
            if near_candidates:
                final_selection.append(max(near_candidates, key=lambda x: x["area"]))
            if far_candidates:
                final_selection.append(max(far_candidates, key=lambda x: x["score"]))

            # ===== 繪圖 =====

            # 球框（黃）
            for box in ball_boxes_by_frame.get(idx, []):
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

            # 球員骨架（綠）
            for cand in final_selection:
                kps = cand["kps"]

                for (x, y, conf) in kps:
                    if conf < 0.3:
                        continue
                    cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

                for i_link, j_link in SKELETON_LINKS:
                    if i_link >= len(kps) or j_link >= len(kps):
                        continue
                    x1, y1, c1 = kps[i_link]
                    x2, y2, c2 = kps[j_link]
                    if c1 < 0.3 or c2 < 0.3:
                        continue
                    cv2.line(
                        frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 255, 0),
                        2,
                    )

            # 寫給 ffmpeg
            if proc.stdin:
                out_frame = np.ascontiguousarray(frame)
                proc.stdin.write(out_frame.tobytes())

            idx += 1
            if progress_callback:
                step = max_frames + min(idx, max_frames)
                progress_callback(step, total_steps)

    finally:
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        except Exception:
            pass

        returncode = proc.wait()
        err = b""
        if proc.stderr:
            try:
                err = proc.stderr.read()
            except Exception:
                pass

        if returncode != 0:
            if err:
                print(err.decode(errors="ignore"))
            raise RuntimeError("ffmpeg 轉 H.264 失敗（pipe rawvideo）")

    return str(out_path)
