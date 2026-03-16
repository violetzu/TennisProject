EPOCHS=500
PATIENCE=50
IMGSZ=1280
MODEL=yolo26s
CONF=0.1

NAME=${MODEL}-epochs${EPOCHS}-imgsz${IMGSZ}-conf${CONF}-mAP50@

yolo train \
project=ball \
name="$NAME" \
model=${MODEL}.pt \
data="ball_dataset/data.yaml" \
epochs=$EPOCHS \
imgsz=$IMGSZ \
rect=True \
mixup=0.1 \
patience=$PATIENCE \
conf=$CONF \