# tools_ball.py
import os
import io
import uuid
from typing import Dict, Any
import cv2
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

APP_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_MODEL = YOLO(os.path.join(APP_DIR, "../model/ball/yolov8_ball_09250900_best.pt"))  # 你的 ball.pt 放這裡

def _grid9(cx: float, cy: float, W: int, H: int) -> str:
    cols = ["左", "中", "右"]
    rows = ["上", "中", "下"]
    col = 0 if cx < W/3 else (1 if cx < 2*W/3 else 2)
    row = 0 if cy < H/3 else (1 if cy < 2*H/3 else 2)
    return cols[col] + rows[row]

def _draw_annotated(frame_bgr, bbox, center, label_text: str, out_dir: str) -> str:
    """畫框＋中心點＋文字，存成 PNG，回傳檔名"""
    os.makedirs(out_dir, exist_ok=True)

    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    x1, y1, x2, y2 = bbox

    # 框框
    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=4)
    # 中心點
    r = 5
    draw.ellipse(
        [center[0]-r, center[1]-r, center[0]+r, center[1]+r],
        outline=(0, 255, 0), width=4
    )

    # 文字
    try:
        font = ImageFont.truetype("NotoSansCJK-Regular.ttc", 24)
    except Exception:
        font = None
    draw.text(
        (x1, max(0, y1-28)),
        label_text,
        fill=(255, 255, 255),
        font=font,
        stroke_width=2,
        stroke_fill=(0, 0, 0),
    )

    name = f"{uuid.uuid4().hex}.png"
    path = os.path.join(out_dir, name)
    pil.save(path, format="PNG")
    return name

def ball_position_on_frame(
    video_path: str,
    frame_id: int,
    out_dir: str,
    base_url: str = "/image"
) -> Dict[str, Any]:
    """
    核心工具：
    - 給影片路徑 + 幀數
    - 用 ball.pt 找球
    - 回傳結構化資訊（found/grid9/center/bbox/image_url/...）
    """
    if not os.path.exists(video_path):
        return {"ok": False, "error": f"video not found: {video_path}"}

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_id))
    ok, frame = cap.read()
    cap.release()

    if not ok:
        return {"ok": False, "error": f"cannot read frame {frame_id}"}

    result = YOLO_MODEL(frame, conf=0.25)[0]
    W, H = result.orig_shape[1], result.orig_shape[0]

    if len(result.boxes) == 0:
        # 沒偵測到球
        return {
            "ok": True,
            "frame_id": frame_id,
            "found": False,
            "message": f"第 {frame_id} 幀沒有偵測到球",
            "objects": [],
        }

    # 取置信度最高那顆球
    best = max(result.boxes, key=lambda b: float(b.conf))
    x1, y1, x2, y2 = [float(v) for v in best.xyxy[0]]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    grid = _grid9(cx, cy, W, H)

    bbox = [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
    center = [round(cx, 1), round(cy, 1)]
    conf = round(float(best.conf), 3)

    label = f"ball | {grid} | conf {conf:.2f}"
    img_name = _draw_annotated(frame, [x1, y1, x2, y2], [cx, cy], label, out_dir)
    image_url = f"{base_url}/{img_name}"

    return {
        "ok": True,
        "frame_id": frame_id,
        "found": True,
        "grid9": grid,
        "center": center,
        "bbox": bbox,
        "conf": conf,
        "image_url": image_url,
        "message": f"第 {frame_id} 幀的球在畫面 {grid}，中心點 {center}，bbox={bbox}",
    }
