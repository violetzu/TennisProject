#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
IMAGE_NAME="${BALLTEST_IMAGE:-balltest-runner}"
DOCKER_GPU_ARGS="${BALLTEST_DOCKER_GPU_ARGS:---gpus all}"
DOCKER_RUN_EXTRA="${BALLTEST_DOCKER_RUN_EXTRA:---ipc=host}"

DEFAULT_VARIANTS=(yolo26m yolov8s yolo11s)
VARIANTS=()
EXTRA_ARGS=()

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --variants)
      shift
      while [[ "$#" -gt 0 && "$1" != "--" ]]; do
        VARIANTS+=("$1")
        shift
      done
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    *)
      VARIANTS+=("$1")
      shift
      ;;
  esac
done

if [[ "${#VARIANTS[@]}" -eq 0 ]]; then
  VARIANTS=("${DEFAULT_VARIANTS[@]}")
fi

log() {
  echo "[$(date -Iseconds)] $*"
}

wait_for_train_yolo_idle() {
  while docker ps --no-trunc --format '{{.Command}}' | grep -F 'python3 -m backend.test.balltest train-yolo' >/dev/null; do
    log "another train-yolo job is still running; sleep 60s"
    sleep 60
  done
}

run_variant() {
  local variant="$1"
  log "starting variant=${variant}"
  docker run ${DOCKER_GPU_ARGS} ${DOCKER_RUN_EXTRA} --rm \
    -v "${ROOT_DIR}:/workspace" \
    -w /workspace \
    "${IMAGE_NAME}" \
    python3 -m backend.test.balltest train-yolo --variant "${variant}" "${EXTRA_ARGS[@]}"
  log "finished variant=${variant}"
}

log "series variants: ${VARIANTS[*]}"
if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
  log "shared extra args: ${EXTRA_ARGS[*]}"
fi

for variant in "${VARIANTS[@]}"; do
  wait_for_train_yolo_idle
  run_variant "${variant}"
done

log "train-yolo-series completed"
