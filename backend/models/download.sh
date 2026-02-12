#!/usr/bin/env bash
set -e
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
BALL_DIR="$BASE_DIR/ball"
# BALL_MODEL="best.pt"
# BALL_URL="https://drive.google.com/uc?export=download&id=1Ca7riJgmfSxZRxafuUprcscp7bF75ARn"

POSE_DIR="$BASE_DIR/person"
POSE_MODEL="yolo11n-pose.pt"
POSE_URL="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt"

mkdir -p "$BALL_DIR" "$POSE_DIR"

# # -------- Ball model --------
# if [ ! -f "$BALL_DIR/$BALL_MODEL" ]; then
#   echo "下載 Ball 模型..."
#   gdown "$BALL_URL" -O "$BALL_DIR/$BALL_MODEL"
#   # wget -O "$BALL_DIR/$BALL_MODEL" "$BALL_URL"
# else
#   echo "Ball model 已存在"
# fi

# -------- Pose model --------
if [ ! -f "$POSE_DIR/$POSE_MODEL" ]; then
  echo "下載 Pose 模型..."
  wget -O "$POSE_DIR/$POSE_MODEL" "$POSE_URL"
else
  echo "Pose model 已存在"
fi

echo "✅ All models ready."
