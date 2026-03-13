"""
court.py 場地偵測測試腳本（YOLO 版）

用法：
    python test_court.py [image1] [image2] ...
    未指定時自動測試 image1.png ~ image5.png

輸出：result_<stem>.png 存在當前目錄
  - 綠圈 + 索引號：全 14 個 YOLO keypoint
  - 紅線：draw_court() 繪製的場地線
"""

import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from ultralytics import YOLO
from services.combine.court import detect_court_yolo, draw_court, WORLD_CORNERS

COURT_MODEL_PATH = str(Path(__file__).parent.parent / "models" / "court" / "best.pt")
_model = None

def get_model():
    global _model
    if _model is None:
        _model = YOLO(COURT_MODEL_PATH)
    return _model


def run_one(img_path: str) -> bool:
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"  [ERROR] 無法讀取：{img_path}")
        return False

    h, w = frame.shape[:2]
    result = detect_court_yolo(frame, get_model())
    stem = Path(img_path).stem

    if result is None:
        print(f"  [FAIL]  {stem}  {w}x{h}")
        cv2.imwrite(f"result_{stem}.png", frame)
        return False

    H, img_corners, kps = result
    tl, tr, bl, br = img_corners

    slope_far  = (float(tr[1]) - float(tl[1])) / max(abs(float(tr[0]) - float(tl[0])), 1)
    slope_near = (float(br[1]) - float(bl[1])) / max(abs(float(br[0]) - float(bl[0])), 1)
    slope_diff = slope_far - slope_near
    far_w  = float(tr[0]) - float(tl[0])
    near_w = float(br[0]) - float(bl[0])

    test_world = cv2.perspectiveTransform(
        img_corners.reshape(1, -1, 2), H).reshape(-1, 2)
    max_err = float(np.max(np.linalg.norm(WORLD_CORNERS - test_world, axis=1)))

    print(f"  [OK]    {stem}  {w}x{h}  H-err={max_err:.3f}m")
    print(f"    far   ({tl[0]:6.0f},{tl[1]:5.0f})→({tr[0]:6.0f},{tr[1]:5.0f})"
          f"  w={far_w:6.0f}px  slope={slope_far:+.4f}")
    print(f"    near  ({bl[0]:6.0f},{bl[1]:5.0f})→({br[0]:6.0f},{br[1]:5.0f})"
          f"  w={near_w:6.0f}px  slope={slope_near:+.4f}")
    print(f"    slope_diff={slope_diff:+.4f}  ratio={near_w/far_w:.2f}x")

    # 印出所有 14 個 keypoint 座標（方便確認索引語意）
    print("    kps (index: x, y):")
    for i, (x, y) in enumerate(kps):
        print(f"      [{i:2d}] ({x:7.1f}, {y:6.1f})")

    out = frame.copy()
    draw_court(out, img_corners, H=H, kps=kps)

    # 繪製所有 keypoint（索引號 + 圓圈）
    for i, (x, y) in enumerate(kps):
        if x == 0 and y == 0:
            continue  # 未標注點略過
        cx, cy = int(x), int(y)
        cv2.circle(out, (cx, cy), 6, (0, 255, 0), -1)
        cv2.putText(out, str(i), (cx + 7, cy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    out_path = f"result_{stem}.png"
    cv2.imwrite(out_path, out)
    print(f"    → {out_path}")
    return True


def main():
    targets = (sys.argv[1:] if len(sys.argv) > 1
               else [f"image{i}.png" for i in range(1, 6)])
    print(f"Testing {len(targets)} image(s)...\n")
    results = [run_one(t) for t in targets]
    ok = sum(results)
    print(f"\n[SUMMARY] {ok}/{len(targets)} passed")


if __name__ == "__main__":
    main()
