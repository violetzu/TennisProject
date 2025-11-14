#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
旋轉視角3D姿態分析系統
Pipeline: YOLO人物檢測 → MediaPipe BlazePose 3D → 旋轉3D視角

特色：
- 只追蹤最大的人物
- 使用站立方向的三維座標系統
- 旋轉鏡頭視角
- XYZ座標軸顯示
- 高品質3D渲染

使用方法：
python rotating_3d_pose_pipeline.py \
  --input input.mp4 \
  --output output_rotating.mp4 \
  --yolo-model yolov8n.pt
"""

import os
import sys
import math
import argparse
from pathlib import Path
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np

# YOLOv8
try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None
    print(f"Warning: Ultralytics not available: {e}")

# MediaPipe Pose (3D)
try:
    import mediapipe as mp
except Exception as e:
    mp = None
    print(f"Warning: MediaPipe not available: {e}")

# MediaPipe BlazePose的33個關鍵點連接
BLAZEPOSE_CONNECTIONS = [
    # 軀幹核心
    (11, 12), (23, 24), (11, 23), (12, 24),
    # 左臂
    (11, 13), (13, 15),
    # 右臂  
    (12, 14), (14, 16),
    # 左腿
    (23, 25), (25, 27),
    # 右腿
    (24, 26), (26, 28),
    # 腳部
    (27, 29), (28, 30),
    (29, 31), (30, 32),
]

def pick_largest_person(boxes: List[np.ndarray]) -> Optional[int]:
    """選擇面積最大的人物"""
    if boxes is None or len(boxes) == 0:
        return None
    
    # 計算每個邊界框的面積
    areas = []
    for box in boxes:
        area = (box[2] - box[0]) * (box[3] - box[1])
        areas.append(area)
    
    # 回傳最大面積的索引
    idx = int(np.argmax(areas))
    return idx

def crop_with_margin(img: np.ndarray, 
                    xyxy: Tuple[float, float, float, float], 
                    margin: float = 0.15) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """裁剪影像並加上邊距"""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(float, xyxy)
    bw, bh = x2 - x1, y2 - y1
    cx, cy = x1 + bw / 2, y1 + bh / 2
    side = max(bw, bh) * (1 + margin * 2)
    x1n = int(max(0, cx - side / 2))
    y1n = int(max(0, cy - side / 2))
    x2n = int(min(w, cx + side / 2))
    y2n = int(min(h, cy + side / 2))
    crop = img[y1n:y2n, x1n:x2n].copy()
    return crop, (x1n, y1n, x2n, y2n)

def convert_to_standing_coordinates(pose_3d: np.ndarray, scale_factor: float = 1000.0) -> np.ndarray:
    """將MediaPipe座標轉換為站立方向的座標系統"""
    # MediaPipe座標系統: x右, y上, z向前（相對於人體）
    # 目標座標系統: x右, y前, z上（站立方向）
    
    pose_converted = pose_3d.copy()
    
    # 重新映射座標軸
    # 原始: x=左右, y=上下, z=前後
    # 目標: x=左右, y=前後, z=上下
    temp_x = pose_converted[:, 0] * scale_factor  # 保持左右方向
    temp_y = -pose_converted[:, 2] * scale_factor # z變成y（前後），翻轉符號
    temp_z = -pose_converted[:, 1] * scale_factor # y變成z（上下），翻轉符號讓上為正
    
    pose_converted[:, 0] = temp_x
    pose_converted[:, 1] = temp_y  
    pose_converted[:, 2] = temp_z
    
    # 將人物置中到地面
    if len(pose_converted) > 0:
        # 使用腳踝位置作為地面參考
        left_ankle = 27 if len(pose_converted) > 27 else -1
        right_ankle = 28 if len(pose_converted) > 28 else -1
        
        if left_ankle != -1 and right_ankle != -1:
            ground_level = min(pose_converted[left_ankle, 2], pose_converted[right_ankle, 2])
            pose_converted[:, 2] -= ground_level  # 讓腳踝接觸地面
        
        # 水平置中
        center_x = np.mean(pose_converted[:, 0])
        center_y = np.mean(pose_converted[:, 1])
        pose_converted[:, 0] -= center_x
        pose_converted[:, 1] -= center_y
    
    return pose_converted

def draw_2d_skeleton(frame: np.ndarray, 
                    landmarks: List[Tuple[float, float, float]], 
                    visibility_thr: float = 0.5) -> None:
    """在原影片上繪製2D骨架"""
    if not landmarks:
        return
    
    # 繪製連線
    color_skeleton = (0, 255, 0)  # 綠色
    color_points = (0, 180, 255)  # 橙色
    
    for i, j in BLAZEPOSE_CONNECTIONS:
        if 0 <= i < len(landmarks) and 0 <= j < len(landmarks):
            xi, yi, vi = landmarks[i]
            xj, yj, vj = landmarks[j]
            if vi >= visibility_thr and vj >= visibility_thr:
                cv2.line(frame, (int(xi), int(yi)), (int(xj), int(yj)), color_skeleton, 2)
    
    # 繪製關鍵點
    for (x, y, v) in landmarks:
        if v >= visibility_thr:
            cv2.circle(frame, (int(x), int(y)), 3, color_points, -1)

def draw_coordinate_axes(canvas: np.ndarray, view_w: int, view_h: int, 
                        rotation_angle: float, axis_length: float = 100):
    """繪製XYZ座標軸"""
    # 座標軸原點（左下角）
    origin_x = int(view_w * 0.15)
    origin_y = int(view_h * 0.85)
    
    # 旋轉角度
    angle_rad = math.radians(rotation_angle)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    # 仰視角度
    elevation = math.radians(30)
    cos_elev = math.cos(elevation)
    sin_elev = math.sin(elevation)
    
    # 定義座標軸方向（在3D空間中）
    axes = {
        'X': (axis_length, 0, 0),      # 紅色 - 右方
        'Y': (0, axis_length, 0),      # 綠色 - 前方
        'Z': (0, 0, axis_length)       # 藍色 - 上方
    }
    
    colors = {
        'X': (0, 0, 255),    # 紅色
        'Y': (0, 255, 0),    # 綠色
        'Z': (255, 0, 0)     # 藍色
    }
    
    for axis_name, (x, y, z) in axes.items():
        # 應用旋轉
        rot_x = x * cos_a - y * sin_a
        rot_y = x * sin_a + y * cos_a
        rot_z = z
        
        # 應用仰視角度投影
        y_proj = rot_y * cos_elev - rot_z * sin_elev
        z_proj = rot_y * sin_elev + rot_z * cos_elev
        
        # 轉換到螢幕座標
        end_x = int(origin_x + rot_x)
        end_y = int(origin_y - z_proj)  # 螢幕Y軸向下
        
        # 繪製軸線
        cv2.arrowedLine(canvas, (origin_x, origin_y), (end_x, end_y), 
                       colors[axis_name], 3, tipLength=0.1)
        
        # 標記軸名稱
        label_x = int(end_x + 10)
        label_y = int(end_y + 5)
        cv2.putText(canvas, axis_name, (label_x, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[axis_name], 2)

def render_rotating_3d_pose(view_w: int, view_h: int,
                           keypoints3d: np.ndarray,
                           frame_idx: int) -> np.ndarray:
    """渲染旋轉視角的3D姿態"""
    
    canvas = np.zeros((view_h, view_w, 3), dtype=np.uint8)
    
    if keypoints3d is None or keypoints3d.shape[0] == 0:
        cv2.putText(canvas, 'No pose detected', (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
        return canvas
    
    # 轉換為站立座標系統
    pose_standing = convert_to_standing_coordinates(keypoints3d)
    
    # 旋轉視角（模仿你的範例）
    rotation_angle = (frame_idx * 0.5) % 360  # 每幀旋轉2度
    angle_rad = math.radians(rotation_angle)
    
    # 繞Z軸旋轉（上下軸）
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    rotated_pose = pose_standing.copy()
    for i in range(len(rotated_pose)):
        x, y, z = rotated_pose[i]
        rotated_pose[i, 0] = x * cos_a - y * sin_a
        rotated_pose[i, 1] = x * sin_a + y * cos_a
        rotated_pose[i, 2] = z
    
    # 3D到2D投影（等距投影，類似matplotlib的3D效果）
    # 視角: 仰視30度
    elevation = math.radians(30)
    cos_elev = math.cos(elevation)
    sin_elev = math.sin(elevation)
    
    # 投影計算
    proj_x = []
    proj_y = []
    depths = []
    
    for i in range(len(rotated_pose)):
        x, y, z = rotated_pose[i]
        
        # 應用仰視角度
        y_rot = y * cos_elev - z * sin_elev
        z_rot = y * sin_elev + z * cos_elev
        
        # 簡單的正交投影
        screen_x = x
        screen_y = -z_rot  # 翻轉Y軸以符合螢幕座標
        
        proj_x.append(screen_x)
        proj_y.append(screen_y) 
        depths.append(y_rot)  # 用於深度排序
    
    # 計算適當的縮放和位移，確保骨架完全可見
    if len(proj_x) > 0:
        min_x, max_x = min(proj_x), max(proj_x)
        min_y, max_y = min(proj_y), max(proj_y)
        
        # 計算範圍並加上邊距
        range_x = max_x - min_x
        range_y = max_y - min_y
        
        if range_x == 0:
            range_x = 1
        if range_y == 0:
            range_y = 1
            
        # 縮放因子，確保骨架適合畫面（保留20%邊距）
        scale_x = (view_w * 0.6) / range_x
        scale_y = (view_h * 0.6) / range_y
        scale = min(scale_x, scale_y)
        
        # 計算中心位置
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        # 將座標轉換到螢幕空間
        screen_x = [((x - center_x) * scale) + (view_w / 2) for x in proj_x]
        screen_y = [((y - center_y) * scale) + (view_h / 2) for y in proj_y]
        
        # 正規化深度用於視覺效果
        if max(depths) != min(depths):
            norm_depths = [(d - min(depths)) / (max(depths) - min(depths)) for d in depths]
        else:
            norm_depths = [0.5] * len(depths)
    
        # 繪製 XYZ 座標軸
        draw_coordinate_axes(canvas, view_w, view_h, rotation_angle, scale * 100)
        
        # 繪製骨架連線
        skeleton_color = (255, 255, 255)  # 白色骨架
        for i, j in BLAZEPOSE_CONNECTIONS:
            if i < len(screen_x) and j < len(screen_x):
                # 根據深度調整線條粗細
                depth_factor = (norm_depths[i] + norm_depths[j]) / 2
                thickness = max(2, int(4 * (1 - depth_factor * 0.3)))
                
                p1 = (int(screen_x[i]), int(screen_y[i]))
                p2 = (int(screen_x[j]), int(screen_y[j]))
                cv2.line(canvas, p1, p2, skeleton_color, thickness)
        
        # 繪製關鍵點
        for i in range(len(screen_x)):
            depth_factor = norm_depths[i]
            radius = max(3, int(6 * (1 - depth_factor * 0.2)))
            
            # 根據深度調整顏色亮度
            brightness = 0.7 + 0.3 * (1 - depth_factor)
            point_color = tuple(int(255 * brightness) for _ in range(3))
            
            cv2.circle(canvas, (int(screen_x[i]), int(screen_y[i])), 
                      radius, point_color, -1)
    
    # 繪製資訊
    cv2.putText(canvas, f'3D Pose (Angle: {rotation_angle}°)', (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(canvas, 'Standing coordinate system', (10, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    return canvas

def run_rotating_pipeline(
    path_input: str,
    path_output: str, 
    yolo_model_path: str = "yolov8n.pt",
    confidence_threshold: float = 0.5,
    enable_smoothing: bool = True,
    output_fps: Optional[float] = None,
    resize_width: Optional[int] = None,
):
    """運行旋轉視角3D姿態管線"""
    
    if YOLO is None or mp is None:
        raise RuntimeError("需要安裝 ultralytics 和 mediapipe 套件")
    
    cap = cv2.VideoCapture(path_input)
    if not cap.isOpened():
        raise FileNotFoundError(f"無法開啟輸入影片: {path_input}")
    
    # 獲取影片資訊
    in_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = output_fps or in_fps
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 計算輸出尺寸
    if resize_width:
        scale = resize_width / in_w
        draw_w = resize_width
        draw_h = int(in_h * scale)
    else:
        draw_w, draw_h = in_w, in_h
    
    # 左右佈局
    out_w = draw_w * 2
    out_h = draw_h
    
    # 初始化模型
    yolo = YOLO(yolo_model_path)
    mp_pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=confidence_threshold,
        min_tracking_confidence=confidence_threshold,
        smooth_landmarks=enable_smoothing,
    )
    
    # 視頻寫入器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path_output, fourcc, fps, (out_w, out_h))
    
    frame_idx = 0
    pose_history = deque(maxlen=5)  # 用於平滑
    
    try:
        print(f"開始處理影片... 輸出尺寸: {out_w}x{out_h}")
        
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            
            frame_idx += 1
            
            # 調整大小
            if resize_width:
                frame_draw = cv2.resize(frame_bgr, (draw_w, draw_h))
            else:
                frame_draw = frame_bgr.copy()
            
            # YOLO 人物檢測
            results = yolo.predict(source=frame_draw, imgsz=640, verbose=False)[0]
            boxes = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []
            scores = results.boxes.conf.cpu().numpy() if results.boxes is not None else []
            classes = results.boxes.cls.cpu().numpy().astype(int) if results.boxes is not None else []
            
            # 篩選並選擇最大的人物
            person_detections = []
            for box, score, cls in zip(boxes, scores, classes):
                if cls == 0 and score >= confidence_threshold:
                    person_detections.append(box)
            
            person_idx = pick_largest_person(person_detections)
            
            pose_3d = None
            pose_2d = None
            
            if person_idx is not None:
                bbox = person_detections[person_idx]
                crop, (x1c, y1c, x2c, y2c) = crop_with_margin(frame_draw, bbox, margin=0.15)
                
                if crop.size > 0:
                    # MediaPipe 處理
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    mp_results = mp_pose.process(crop_rgb)
                    
                    if mp_results.pose_landmarks and mp_results.pose_world_landmarks:
                        # 2D 關鍵點
                        lmks2d = []
                        for lm in mp_results.pose_landmarks.landmark:
                            x = x1c + lm.x * (x2c - x1c)
                            y = y1c + lm.y * (y2c - y1c)
                            v = getattr(lm, 'visibility', 1.0)
                            lmks2d.append((x, y, v))
                        
                        pose_2d = lmks2d
                        
                        # 3D 關鍵點
                        pts3d = np.array([[lm.x, lm.y, lm.z] for lm in mp_results.pose_world_landmarks.landmark])
                        
                        # 平滑處理
                        if enable_smoothing:
                            pose_history.append(pts3d)
                            if len(pose_history) > 1:
                                # 簡單的移動平均
                                pose_3d = np.mean(pose_history, axis=0)
                            else:
                                pose_3d = pts3d
                        else:
                            pose_3d = pts3d
                
                # 繪製邊界框
                x1, y1, x2, y2 = bbox.astype(int)
                cv2.rectangle(frame_draw, (x1, y1), (x2, y2), (255, 120, 0), 2)
                cv2.putText(frame_draw, 'Person', (x1, y1 - 6),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 120, 0), 2)
            
            # 繪製2D骨架
            if pose_2d:
                draw_2d_skeleton(frame_draw, pose_2d)
            else:
                cv2.putText(frame_draw, 'No person detected', (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            # 渲染3D視圖
            pose_3d_view = render_rotating_3d_pose(
                view_w=draw_w,
                view_h=draw_h,
                keypoints3d=pose_3d,
                frame_idx=frame_idx
            )
            
            # 左右組合
            combined_frame = np.concatenate([frame_draw, pose_3d_view], axis=1)
            
            # 添加分隔線
            cv2.line(combined_frame, (draw_w, 0), (draw_w, draw_h), (100, 100, 100), 2)
            
            # 添加資訊
            cv2.putText(combined_frame, f"Frame {frame_idx}", (10, out_h - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 寫入影片
            out.write(combined_frame)
            
            # 進度顯示
            if frame_idx % 30 == 0:
                print(f"已處理 {frame_idx} 幀...")
                
    finally:
        out.release()
        cap.release()
        mp_pose.close()
    
    print(f"處理完成！輸出儲存至: {path_output}")

def parse_args():
    parser = argparse.ArgumentParser(description='旋轉視角3D姿態分析系統')
    
    # 必要參數
    parser.add_argument('--input', required=True, help='輸入影片路徑')
    parser.add_argument('--output', required=True, help='輸出影片路徑')
    
    # 模型參數
    parser.add_argument('--yolo-model', default='yolov8n.pt',
                       help='YOLO模型路徑 (預設: yolov8n.pt)')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='人物檢測信心度閾值 (預設: 0.5)')
    
    # 處理參數
    parser.add_argument('--no-smoothing', action='store_true',
                       help='停用姿態平滑化')
    
    # 輸出參數  
    parser.add_argument('--fps', type=float, default=None,
                       help='輸出影片FPS (預設: 與輸入相同)')
    parser.add_argument('--width', type=int, default=None,
                       help='調整輸出寬度 (預設: 與輸入相同)')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    run_rotating_pipeline(
        path_input=args.input,
        path_output=args.output,
        yolo_model_path=args.yolo_model,
        confidence_threshold=args.confidence,
        enable_smoothing=not args.no_smoothing,
        output_fps=args.fps,
        resize_width=args.width,
    )