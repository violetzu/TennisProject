# -*- coding: utf-8 -*-
"""主程式：負責協同各模組完成網球賽事標註流程。"""

import os

# TensorFlow / TFLite：2=過濾 INFO 和 WARNING，3=只顯示致命錯誤
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Abseil/glog：2=WARNING 以上才顯示，3=ERROR 以上
os.environ["GLOG_minloglevel"] = "3"
os.environ["ABSL_LOGGING_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
from tqdm import tqdm

from src import (
    CourtDetectorNet,
    CourtReference,
    BounceDetector,
    BallDetector,
    PersonDetector,
    Pose3DVisualizer,
)
from src.utils import scene_detect
import argparse
import torch


def read_video(path_video):
    """讀取影片並依序擷取所有影格。"""
    cap = cv2.VideoCapture(path_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break    
    cap.release()
    return frames, fps

def write_video(imgs_res, fps, path_output_video):
    """以指定的 FPS 將處理後影格輸出成影片，依副檔名自動選擇編碼器。"""
    height, width = imgs_res[0].shape[:2]
    # 副檔名 → fourcc 對應表
    codec_map = {
        '.mp4': 'mp4v',
        '.m4v': 'mp4v',
        '.avi': 'DIVX'
    }
    ext = os.path.splitext(path_output_video)[1].lower()
    fourcc_str = codec_map.get(ext, 'XVID')  # 預設用 XVID
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)

    out = cv2.VideoWriter(path_output_video, fourcc, fps, (width, height))
    for frame in imgs_res:
        out.write(frame)
    out.release()   

def get_court_img():
    """生成網球場示意圖，提供小地圖繪製使用。"""
    court_reference = CourtReference()
    court = court_reference.build_court_reference()
    court = cv2.dilate(court, np.ones((10, 10), dtype=np.uint8))
    court_img = (np.stack((court, court, court), axis=2)*255).astype(np.uint8)
    return court_img

def main(frames, scenes, bounces, ball_track, homography_matrices, kps_court,
         persons_top, persons_bottom, pose_visualizer=None, draw_trace=False, trace=7):
    """
    :params
        frames: list of original images
        scenes: list of beginning and ending of video fragment
        bounces: list of image numbers where ball touches the ground
        ball_track: list of (x,y) ball coordinates
        homography_matrices: list of homography matrices
        kps_court: list of 14 key points of tennis court
        draw_trace: whether to draw ball trace
        trace: the length of ball trace
    :return
        imgs_res: list of resulting images

    主要迴圈會依場景逐格完成：
        * 將網球與選手資訊繪製在原始畫面上
        * 透過單應矩陣將資料投影到小地圖
        * 對有追蹤結果的片段增加球軌跡與落點標記
    """
    imgs_res = []
    width_minimap = 166
    height_minimap = 350
    is_track = [x is not None for x in homography_matrices]
    pose_enabled = pose_visualizer is not None

    progress = tqdm(total=len(frames), desc='Rendering output', unit='frame')

    for num_scene in range(len(scenes)):
        start_idx, end_idx = scenes[num_scene]
        sum_track = sum(is_track[start_idx:end_idx])
        len_track = end_idx - start_idx
        eps = 1e-15
        scene_rate = sum_track / (len_track + eps)
        use_projection = scene_rate > 0.5
        court_img = get_court_img() if use_projection else None

        for i in range(start_idx, end_idx):
            original_frame = frames[i]
            img_res = original_frame.copy()
            inv_mat = homography_matrices[i] if use_projection else None
            players_top = persons_top[i] if i < len(persons_top) else []
            players_bottom = persons_bottom[i] if i < len(persons_bottom) else []
            target_player = players_bottom[0] if players_bottom else None
            pose_view = None
            if pose_enabled:
                pose_view = pose_visualizer.render(target_player, i, img_res.shape)
            players_for_minimap = players_top + players_bottom

            if use_projection:
                if ball_track[i][0]:
                    if draw_trace:
                        for j in range(trace):
                            if i - j >= 0 and ball_track[i - j][0]:
                                draw_x = int(ball_track[i - j][0])
                                draw_y = int(ball_track[i - j][1])
                                cv2.circle(img_res, (draw_x, draw_y), radius=3, color=(0, 255, 0), thickness=2)
                    else:
                        cv2.circle(
                            img_res,
                            (int(ball_track[i][0]), int(ball_track[i][1])),
                            radius=5,
                            color=(0, 255, 0),
                            thickness=2,
                        )
                        cv2.putText(
                            img_res,
                            'ball',
                            (int(ball_track[i][0]) + 8, int(ball_track[i][1]) + 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2,
                        )

                if kps_court[i] is not None:
                    for kp in kps_court[i]:
                        cv2.circle(
                            img_res,
                            (int(kp[0, 0]), int(kp[0, 1])),
                            radius=0,
                            color=(0, 0, 255),
                            thickness=10,
                        )

                height, width, _ = img_res.shape

                if i in bounces and inv_mat is not None:
                    ball_point = np.array(ball_track[i], dtype=np.float32).reshape(1, 1, 2)
                    ball_point = cv2.perspectiveTransform(ball_point, inv_mat)
                    court_img = cv2.circle(
                        court_img,
                        (int(ball_point[0, 0, 0]), int(ball_point[0, 0, 1])),
                        radius=0,
                        color=(0, 255, 255),
                        thickness=50,
                    )

                minimap = court_img.copy()

                for player in players_for_minimap:
                    person_point = np.array(player.foot, dtype=np.float32).reshape(1, 1, 2)
                    person_point = cv2.perspectiveTransform(person_point, inv_mat)
                    cv2.circle(
                        minimap,
                        (int(person_point[0, 0, 0]), int(person_point[0, 0, 1])),
                        radius=0,
                            color=(255, 0, 0),
                            thickness=80,
                        )

                minimap = cv2.resize(minimap, (width_minimap, height_minimap))
                img_res[30:(30 + height_minimap), (width - 30 - width_minimap):(width - 30), :] = minimap
            else:
                height, width, _ = img_res.shape

            if pose_enabled:
                Pose3DVisualizer.draw_annotations(
                    img_res,
                    players_top,
                    players_bottom,
                    pose_visualizer.last_bbox,
                )
                if pose_view is not None:
                    if pose_view.shape[0] != img_res.shape[0] or pose_view.shape[1] != img_res.shape[1]:
                        pose_view = cv2.resize(pose_view, (img_res.shape[1], img_res.shape[0]))
                    combined_frame = np.concatenate([img_res, pose_view], axis=1)
                    cv2.line(combined_frame, (img_res.shape[1], 0), (img_res.shape[1], img_res.shape[0]), (100, 100, 100), 2)
                    cv2.putText(
                        combined_frame,
                        f'Frame {i}',
                        (10, height - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )
                    imgs_res.append(combined_frame)
                    progress.update(1)
                    continue
            else:
                for player in players_for_minimap:
                    Pose3DVisualizer.draw_2d_skeleton(img_res, player.landmarks)

            imgs_res.append(img_res)
            progress.update(1)

    progress.close()
    return imgs_res
 



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_ball_track_model', type=str, required=True,
                        help='path to pretrained model for ball detection')
    parser.add_argument('--path_court_model', type=str, required=True,
                        help='path to pretrained model for court detection')
    parser.add_argument('--path_bounce_model', type=str, required=True,
                        help='path to pretrained model for bounce detection')
    parser.add_argument('--path_person_model', type=str, required=True,
                        help='YOLO model path for player detection')
    parser.add_argument('--path_input_video', type=str, required=True, help='path to input video')
    parser.add_argument('--path_output_video', type=str, required=True, help='path to output video')
    parser.add_argument('--person_confidence', type=float, default=0.5,
                        help='confidence threshold for player detection')
    parser.add_argument('--person_resize_width', type=int, default=None,
                        help='optional resize width before detection')
    parser.add_argument('--pose_no_smoothing', action='store_true',
                        help='disable temporal smoothing for the 3D pose view')
    parser.add_argument('--pose_smoothing_window', type=int, default=5,
                        help='number of frames used for 3D pose smoothing')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    frames, fps = read_video(args.path_input_video)
    scenes = scene_detect(args.path_input_video)

    print('ball detection')
    ball_detector = BallDetector(args.path_ball_track_model, device)
    ball_track = ball_detector.infer_model(frames)

    print('court detection')
    court_detector = CourtDetectorNet(args.path_court_model, device)
    homography_matrices, kps_court = court_detector.infer_model(frames)

    print('person detection')
    person_detector = PersonDetector(
        path_model=args.path_person_model,
        device=device,
        confidence_threshold=args.person_confidence,
        resize_width=args.person_resize_width,
    )
    persons_top, persons_bottom = person_detector.track_players(frames, homography_matrices)

    # Bounce 偵測：運用球座標序列判斷球是否觸地
    bounce_detector = BounceDetector(args.path_bounce_model)
    x_ball = [x[0] for x in ball_track]
    y_ball = [x[1] for x in ball_track]
    bounces = bounce_detector.predict(x_ball, y_ball)

    pose_visualizer = Pose3DVisualizer(
        enable_smoothing=not args.pose_no_smoothing,
        smoothing_window=args.pose_smoothing_window,
    )

    imgs_res = main(
        frames,
        scenes,
        bounces,
        ball_track,
        homography_matrices,
        kps_court,
        persons_top,
        persons_bottom,
        pose_visualizer=pose_visualizer,
        draw_trace=True,
    )

    write_video(imgs_res, fps, args.path_output_video)

    person_detector.close()
