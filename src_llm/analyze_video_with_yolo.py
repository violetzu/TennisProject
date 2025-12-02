# =========================================================
#  核心邏輯：高解析度球偵測 + 靜止過濾 + 裁判/球員篩選演算法
# =========================================================
from typing import Optional, Dict, List
# 如果你的檔案結構是扁平的，請用 from get_yolo_models ...
# 如果是在 src_llm 資料夾下，請保留 from .get_yolo_models ...

from .get_yolo_models import get_yolo_models


def analyze_video_with_yolo(video_path: str, max_frames: int, session_data: Dict) -> Dict[str, List[Dict]]:
    """
    Args:
        video_path: 影片路徑
        max_frames: 最大分析幀數
        session_data: 從 main.py 傳過來的 Session 字典物件 (直接修改它來更新進度)
    """
    
    # 確保 session_data 有初始化 (雖然 main.py 應該已經做了，但防呆一下)
    if "progress" not in session_data:
        session_data["progress"] = 0

    ball_model, pose_model = get_yolo_models()
    ball_tracks = []
    poses = []
    ball_hist = {}  # 用於記錄球的軌跡歷史，做靜止判定用

    # --- 階段 1: 球偵測 (進度 0% - 50%) ---
    idx = 0
    # 使用 ByteTrack 追蹤器
    for r in ball_model.track(source=video_path, stream=True, tracker="bytetrack.yaml", persist=True, imgsz=1280, conf=0.20, iou=0.5, verbose=False):
        if idx >= max_frames: break
        
        # 【修正】直接修改傳入的字典，不需要去查全域變數 SESSION_STORE
        session_data["progress"] = int((idx / max_frames) * 50)

        if r.boxes and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().tolist()
            ids = r.boxes.id.cpu().tolist() if r.boxes.id is not None else [None]*len(xyxy)
            for box, tid in zip(xyxy, ids):
                if tid is None: continue
                tid = int(tid)
                cx = (box[0]+box[2])/2
                cy = (box[1]+box[3])/2
                
                # --- 演算法：靜止過濾 (Anti-Static Filter) ---
                if tid not in ball_hist: ball_hist[tid] = []
                ball_hist[tid].append((cx, cy))
                if len(ball_hist[tid]) > 30: ball_hist[tid].pop(0)
                
                is_static = False
                if len(ball_hist[tid]) > 10:
                    pts = ball_hist[tid][-15:]
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    if (max(xs)-min(xs) < 20) and (max(ys)-min(ys) < 20):
                        is_static = True
                
                if not is_static:
                    ball_tracks.append({
                        "frame": idx, "track_id": tid,
                        "x": cx, "y": cy, "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]
                    })
        idx += 1

    # --- 階段 2: 人物/姿態辨識 (進度 50% - 100%) ---
    idx = 0
    for r in pose_model(source=video_path, stream=True, imgsz=1280, conf=0.25, verbose=False):
        if idx >= max_frames: break
        
        # 【修正】直接修改傳入的字典
        session_data["progress"] = 50 + int((idx / max_frames) * 50)

        img_h, img_w = r.orig_shape
        center_x = img_w / 2
        
        if r.keypoints and r.boxes and r.keypoints.data is not None:
            kps = r.keypoints.data.cpu().tolist()
            bx = r.boxes.xyxy.cpu().tolist()
            
            near_candidates = [] 
            far_candidates = []  

            for i, (b, kp) in enumerate(zip(bx, kps)):
                cx = (b[0]+b[2])/2
                cy = (b[1]+b[3])/2
                area = (b[2]-b[0]) * (b[3]-b[1])
                
                if b[3] < img_h * 0.1: continue

                # --- 演算法：場地分區與權重計算 ---
                if cy > img_h * 0.5:
                    if cx > img_w * 0.05 and cx < img_w * 0.95:
                        near_candidates.append({"area": area, "kps": kp, "pid": i})
                else:
                    dist_ratio = abs(cx - center_x) / (img_w / 2)
                    score = area * (1 - dist_ratio)
                    if cx > img_w * 0.10 and cx < img_w * 0.90:
                        far_candidates.append({"area": area, "score": score, "kps": kp, "pid": i})

            final_selection = []
            if near_candidates:
                final_selection.append(max(near_candidates, key=lambda x: x["area"]))
            
            if far_candidates:
                final_selection.append(max(far_candidates, key=lambda x: x["score"]))

            for cand in final_selection:
                poses.append({
                    "frame": idx,
                    "person_id": cand["pid"],
                    "keypoints": cand["kps"]
                })

        idx += 1

    return {"ball_tracks": ball_tracks, "poses": poses}