#!/usr/bin/env bash
set -e
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

POSE_MODEL="yolo26s-pose.pt"
POSE_URL="https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-pose.pt"

BALL_MODEL="ball_best.pt"
BALL_ID="1n87EmSlq3GWAVVEyVcq2Ze5BqHGHTxhb"

COURT_MODEL="court_best.pt"
COURT_ID="1USHocv-lM1hmb_U0vD6QieLzqzJhHwGl"

# -------- Pose model --------
if [ ! -f "$BASE_DIR/$POSE_MODEL" ]; then
  if [ -z "$POSE_URL" ]; then
    echo "⚠️  Pose model URL not set, skipping $POSE_MODEL"
  else
    echo "下載 Pose 模型..."
    wget -O "$BASE_DIR/$POSE_MODEL" "$POSE_URL"
  fi
else
  echo "Pose model 已存在"
fi

# -------- Ball model --------
if [ ! -f "$BASE_DIR/$BALL_MODEL" ]; then
  echo "下載 Ball 模型..."
  gdown "$BALL_ID" -O "$BASE_DIR/$BALL_MODEL"
else
  echo "Ball model 已存在"
fi

# -------- Court model --------
if [ ! -f "$BASE_DIR/$COURT_MODEL" ]; then
  echo "下載 Court 模型..."
  gdown "$COURT_ID" -O "$BASE_DIR/$COURT_MODEL"
else
  echo "Court model 已存在"
fi

echo "✅ All models ready."
