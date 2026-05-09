yolo val \
project=val \
model="runs/detect/ball/tracknet-yolo26s-epochs300-imgsz1280-batch48-mAP50@3/weights/best.pt" \
data=ball_dataset/data.yaml \
imgsz=1280 \
conf=0.1