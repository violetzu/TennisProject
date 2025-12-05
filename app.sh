BALL_DIR="./model/ball"
BALL_MODEL="yolov8_ball_09250900_best.pt"
BALL_URL="https://drive.google.com/uc?export=download&id=10DHUZL3RQuNMA4gAxfPzOhqVL-JLCJLf"

if [ ! -f "$BALL_DIR/$BALL_MODEL" ]; then
  echo "下載 Ball 模型..."
  wget -q --show-progress -O "$BALL_DIR/$BALL_MODEL" "$BALL_URL"
fi