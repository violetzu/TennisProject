#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
IMAGE_NAME="${COURTTEST_IMAGE:-courttest-runner}"

usage() {
  cat <<'EOF'
Usage:
  backend/test/courttest/run.sh build-image
  backend/test/courttest/run.sh eval-all    [-- extra-cli-args]
  backend/test/courttest/run.sh eval-yolo   [-- extra-cli-args]
  backend/test/courttest/run.sh eval-hough
EOF
}

docker_run() {
  docker run --rm --gpus all \
    -v "$ROOT_DIR/backend":/workspace/backend \
    "$IMAGE_NAME" "$@"
}

CMD="${1:-}"
shift || true

case "$CMD" in
  build-image)
    docker build -f "$ROOT_DIR/backend/test/courttest/Dockerfile" \
      -t "$IMAGE_NAME" "$ROOT_DIR"
    ;;
  eval-all)
    docker_run python3 -m backend.test.courttest eval-all "$@"
    ;;
  eval-yolo)
    docker_run python3 -m backend.test.courttest eval-yolo "$@"
    ;;
  eval-hough)
    docker_run python3 -m backend.test.courttest eval-hough "$@"
    ;;
  *)
    usage; exit 1
    ;;
esac
