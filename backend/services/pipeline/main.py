# main.py
import argparse
import json
import os
import sys
from config import BASE_DIR

if os.path.join(BASE_DIR, "..") not in sys.path:
	sys.path.insert(0, os.path.join(BASE_DIR, ".."))

from services.pipeline.utils.video_utils import read_video, save_video, get_video_fps
from services.pipeline.trackers.player_tracker import PlayerTracker
from services.pipeline.trackers.ball_tracker import BallTracker
from services.pipeline.detectors.detect_players import serialize_player_frames
from services.pipeline.detectors.detect_ball import save_video_info_json
from services.pipeline.court_line_detector import CourtLineDetector
from services.pipeline.court_homography import (
	compute_homography,
	project_player_positions,
	project_ball_positions,
	build_world_frames,
	save_world_coordinate_json,
	render_minicourt_video,
)
from analysis.ball_speed import update_ball_speed
# from event.alternating_detector import detect_events_shot_based, ShotDetectorConfig
from event.alternating_detector import detect_alternating_events, AlternatingDetectorConfig
DATA_DIR = os.path.join(BASE_DIR, "data")
VIDEO_DIR = os.path.join(BASE_DIR, "videos")

COURT_MODEL_PTAH = os.path.join(BASE_DIR, "models", "keypoints_model.pth")
BALL_MODEL_PTAH = os.path.join(BASE_DIR,  "models", "best_ball_detect_model.pt")
BALL_DECTIONS_MODEL_PTAH = os.path.join(BASE_DIR, "models", "ball_detections.pkl")



def run_pipeline(input_path: str, output_name: str | None = None):
	"""執行完整管線，回傳產物路徑字典。"""

	input_path = os.path.abspath(input_path)
	if not os.path.exists(input_path):
		raise FileNotFoundError(f"影片不存在: {input_path}")

	basename = output_name or os.path.splitext(os.path.basename(input_path))[0]

	# 準備路徑
	output_video_dir = os.path.join(VIDEO_DIR, "output_videos")
	minicourt_dir = os.path.join(VIDEO_DIR, "output_minicourt")
	os.makedirs(output_video_dir, exist_ok=True)
	os.makedirs(minicourt_dir, exist_ok=True)
	os.makedirs(DATA_DIR, exist_ok=True)

	output_video_path = os.path.join(output_video_dir, f"{basename}.mp4")
	minicourt_video_path = os.path.join(minicourt_dir, f"minicourt_{basename}.mp4")
	video_json_path = os.path.join(DATA_DIR, f"video_info_{basename}.json")
	world_json_path = os.path.join(DATA_DIR, f"world_info_{basename}.json")

	print(f"讀取影片中: {input_path}")
	video_frames = read_video(input_path)
	fps = get_video_fps(input_path)
	print(f"總影格數: {len(video_frames)} / FPS = {fps}")

	# 1. Player Tracking
	print("執行 Player Tracking...")
	player_tracker = PlayerTracker(model_path="yolo11s.pt", conf=0.25)
	player_detections = player_tracker.detect_and_track_frames(video_frames)

	# 2. Court Line Detection
	print("執行 Court Line Detection...")
	court_line_detector = CourtLineDetector(COURT_MODEL_PTAH)
	court_keypoints = court_line_detector.predict(video_frames[0])

	# 選出左右兩個玩家
	player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)
	player_frames_payload = serialize_player_frames(player_detections)

	# 3. Ball Detection
	print("執行 Ball Detection（使用專屬網球模型）...")
	ball_tracker = BallTracker(model_path=BALL_MODEL_PTAH, conf=0.15)

	raw_ball_detections, detection_mask = ball_tracker.detect_frames(
		video_frames,
		read_from_stub=False,
		stub_path=BALL_DECTIONS_MODEL_PTAH,
	)
	smooth_ball_positions = ball_tracker.interpolate_ball_positions(raw_ball_detections)

	save_video_info_json(
		output_path=video_json_path,
		video_name=os.path.basename(input_path),
		fps=fps,
		player_frames=player_frames_payload,
		detected_balls=raw_ball_detections,
		interpolated_balls=smooth_ball_positions,
		detection_mask=detection_mask,
	)
	print(f"球員/球位置 JSON 已輸出至: {video_json_path}")

	# 4. 同質變換 + 世界座標
	print("計算球場同質變換與世界座標...")
	H_img_to_world, _ = compute_homography(court_keypoints)
	player_world_positions = project_player_positions(H_img_to_world, player_detections)
	ball_world_positions = project_ball_positions(H_img_to_world, smooth_ball_positions)

	world_frames = build_world_frames(
		fps=fps,
		player_frames_image=player_detections,
		player_frames_world=player_world_positions,
		player_serialized=player_frames_payload,
		ball_frames_image=smooth_ball_positions,
		ball_frames_world=ball_world_positions,
		detection_mask=detection_mask,
	)

	save_world_coordinate_json(
		output_path=world_json_path,
		video_name=os.path.basename(input_path),
		fps=fps,
		frames=world_frames,
		homography_matrix=H_img_to_world,
	)
	print(f"世界座標 JSON 已輸出至: {world_json_path}")

	print("計算球速並寫回 JSON...")
	update_ball_speed(
		world_json_path=world_json_path,
		video_json_path=video_json_path,
		smoothing_window=5,
	)

	print("偵測擊球與彈跳事件 (shot-based detection)...")
	cfg = AlternatingDetectorConfig()
	cfg.fps = fps
	contact_events, bounce_events = detect_alternating_events(
		video_json_path=video_json_path,
		world_json_path=world_json_path,
		config=cfg,
		# verbose=True,
	)
	print(f"擊球事件數量: {len(contact_events)}")
	print(f"彈跳事件數量: {len(bounce_events)}")

	actual_minicourt_path = render_minicourt_video(world_frames, minicourt_video_path, fps)
	print(f"Mini court 影片輸出至: {actual_minicourt_path}")

	# 5. 合成畫面
	print("繪製輸出影片內容...")
	output_frames = player_tracker.draw_bboxes(video_frames, player_detections)
	output_frames = ball_tracker.draw_bboxes(output_frames, smooth_ball_positions, detection_mask)
	output_frames = court_line_detector.draw_keypoints_on_video(output_frames, court_keypoints)

	# 6. 輸出影片
	print(f"輸出影片中: {output_video_path}")
	actual_output_path = save_video(output_frames, output_video_path, fps=fps)
	print(f"輸出影片已儲存至: {actual_output_path}")

	return {
		"world_json": world_json_path,
		"video_json": video_json_path,
		"output_video": actual_output_path,
		"minicourt_video": actual_minicourt_path,
	}


# def parse_args():
# 	parser = argparse.ArgumentParser(description="Run tennis analysis pipeline")
# 	parser.add_argument("--input", required=False, default=None, help="輸入影片絕對路徑，預設使用 backend/videos/input_videos/input_video1.mp4")
# 	parser.add_argument("--output-name", required=False, default=None, help="輸出檔名基底（不含副檔名），預設取輸入影片檔名")
# 	parser.add_argument("--json-out", action="store_true", help="以 PIPELINE_JSON:{{...}} 形式輸出產物路徑，便於 IPC 解析")
# 	return parser.parse_args()


# def main():
# 	args = parse_args()
# 	outputs = run_pipeline(input_path=args.input, output_name=args.output_name)
# 	if args.json_out:
# 		print("PIPELINE_JSON:" + json.dumps(outputs, ensure_ascii=False))
# 	else:
# 		print("Pipeline 完成，產物路徑：")
# 		for k, v in outputs.items():
# 			print(f"  {k}: {v}")


# if __name__ == "__main__":
# 	main()
