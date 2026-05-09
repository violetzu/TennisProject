EPOCHS=100
PATIENCE=20
IMGSZ=320
MODEL=yolo26n-pose

NAME=${MODEL}-epochs${EPOCHS}-imgsz${IMGSZ}-mAP50@

yolo pose train \
project=court \
name="$NAME" \
model=${MODEL}.pt \
data="court_dataset/data.yaml" \
epochs=$EPOCHS \
imgsz=$IMGSZ \
rect=True \
mixup=0.1 \
patience=$PATIENCE \
degrees=5.0 \
scale=0.3 \
shear=2.0 \
perspective=0.0001 \
hsv_h=0.015 \
hsv_s=0.4 \
hsv_v=0.2 \
kobj=2.0 \