#!/usr/bin/env bash
# TrackNet 資料集訓練 YOLOBall — A100 40GB
# 資料: tracknet_yolo_dataset (19,106 張, 1280×720, 16px bbox)
# 用法: cd backend/models && bash ball_train.sh

EPOCHS=300
PATIENCE=30
IMGSZ=1280
MODEL=yolo26s
BATCH=48
CONF=0.1

NAME=${MODEL}-epochs${EPOCHS}-imgsz${IMGSZ}-CONF${CONF}-Recall@-Accuracy@

apt-get update && apt-get install -y tmux
tmux new -s train

yolo train \
project=ball \
name="$NAME" \
model=${MODEL}.pt \
data="tracknet_yolo_dataset/data.yaml" \
epochs=$EPOCHS \
imgsz=$IMGSZ \
batch=$BATCH \
rect=True \
mixup=0.1 \
mosaic=1.0 \
patience=$PATIENCE \
conf=$CONF \
workers=12 \
amp=True