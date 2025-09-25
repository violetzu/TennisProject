#!/bin/bash

# ====== 資料夾變數 ======
BALL_DIR="./model/ball"
COURT_DIR="./model/court"
BOUNCE_DIR="./model/bounce"
INPUT_DIR="./input"
OUTPUT_DIR="./output"

mkdir -p "$BALL_DIR" "$COURT_DIR" "$BOUNCE_DIR" "$INPUT_DIR" "$OUTPUT_DIR"

# ====== 模型清單 ======
BALL_MODEL="yolov8_ball_09250900_best.pt"
COURT_MODEL="court.pt"
BOUNCE_MODEL="ctb_regr_bounce.cbm"

BALL_URL="https://drive.google.com/uc?export=download&id=10DHUZL3RQuNMA4gAxfPzOhqVL-JLCJLf"
COURT_URL="https://drive.google.com/uc?export=download&id=1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG"
BOUNCE_URL="https://drive.google.com/uc?export=download&id=1Eo5HDnAQE8y_FbOftKZ8pjiojwuy2BmJ"

# ====== 檢查模型檔案 ======
if [ ! -f "$BALL_DIR/$BALL_MODEL" ]; then
  echo "下載 Ball 模型..."
  wget -q --show-progress -O "$BALL_DIR/$BALL_MODEL" "$BALL_URL"
fi

if [ ! -f "$COURT_DIR/$COURT_MODEL" ]; then
  echo "下載 Court 模型..."
  wget -q --show-progress -O "$COURT_DIR/$COURT_MODEL" "$COURT_URL"
fi

if [ ! -f "$BOUNCE_DIR/$BOUNCE_MODEL" ]; then
  echo "下載 Bounce 模型..."
  wget -q --show-progress -O "$BOUNCE_DIR/$BOUNCE_MODEL" "$BOUNCE_URL"
fi

# ====== 影片清單 ======
declare -A VIDEO_URLS=(
  ["最好辨識的.mp4"]="https://drive.google.com/uc?export=download&id=1ttWh0nV9lqFnOBOOA92f3X_uMuRACLZ5"
  ["抓不到落點的.mp4"]="https://drive.google.com/uc?export=download&id=1Hb6mlEmQhOkPuhPKYrrwRuCMEwYkNLJi"
  ["巨人評審.mp4"]="https://drive.google.com/uc?export=download&id=1ESsqFBvpI3X3HQJRggmtCjTnYrkwKlEI"
  ["哪來那麼多球僮.mp4"]="https://drive.google.com/uc?export=download&id=1MmoShAfNSuhsFhm6JohbaWi2Dr9WJk2I"
  ["會動的場地.mp4"]="https://drive.google.com/uc?export=download&id=1tqz4EVIVq08MocZzFuj3UHxfuai-b9ZG"
)

# ====== 檢查 & 下載影片 ======
for video in "${!VIDEO_URLS[@]}"; do
  if [ ! -f "$INPUT_DIR/$video" ]; then
    echo "下載 $video ..."
    wget -q --show-progress -O "$INPUT_DIR/$video" "${VIDEO_URLS[$video]}"
  fi
done

# ====== 執行處理 ======
for video in "${!VIDEO_URLS[@]}"; do
  base=$(basename "$video" .mp4)
  echo "開始處理 $base ..."
  python main.py \
    --path_ball_track_model "$BALL_DIR/$BALL_MODEL" \
    --path_court_model "$COURT_DIR/$COURT_MODEL" \
    --path_bounce_model "$BOUNCE_DIR/$BOUNCE_MODEL" \
    --path_input_video "$INPUT_DIR/$video" \
    --path_output_video "$OUTPUT_DIR/${base}_out.mp4" #可直接改avi或mp4
done