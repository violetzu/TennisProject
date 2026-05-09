#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
IMAGE_NAME="${BALLTEST_IMAGE:-balltest-runner}"
DOCKER_GPU_ARGS="${BALLTEST_DOCKER_GPU_ARGS:---gpus all}"
DOCKER_RUN_EXTRA="${BALLTEST_DOCKER_RUN_EXTRA:---ipc=host}"

DEFAULT_STEPS=(
  "train-tracknet --batch-size 8"
  "train-tracknet-v3 --batch-size 24"
  "train-tracknet-v2 --batch-size 24"
  "train-tracknet-v4 --batch-size 24"
  "train-tracknet-v5 --batch-size 20"
)
STEPS=()

canonical_step() {
  case "$1" in
    train-tracknet|v1)
      printf '%s\n' "train-tracknet --batch-size 8"
      ;;
    train-tracknet-v2|v2)
      printf '%s\n' "train-tracknet-v2 --batch-size 24"
      ;;
    train-tracknet-v3|v3)
      printf '%s\n' "train-tracknet-v3 --batch-size 24"
      ;;
    train-tracknet-v4|v4)
      printf '%s\n' "train-tracknet-v4 --batch-size 24"
      ;;
    train-tracknet-v5|v5)
      printf '%s\n' "train-tracknet-v5 --batch-size 20"
      ;;
    *)
      echo "unknown tracknet series step: $1" >&2
      exit 1
      ;;
  esac
}

while [[ "$#" -gt 0 ]]; do
  STEPS+=("$(canonical_step "$1")")
  shift
done

if [[ "${#STEPS[@]}" -eq 0 ]]; then
  STEPS=("${DEFAULT_STEPS[@]}")
fi

log() {
  echo "[$(date -Iseconds)] $*"
}

wait_for_training_idle() {
  while docker ps --no-trunc --format '{{.Command}}' | grep -F "python3 -m backend.test.balltest train-" >/dev/null; do
    log "another training job is still running; sleep 60s"
    sleep 60
  done
}

run_tracknet() {
  local command="$1"
  shift
  log "starting ${command} $*"
  docker run ${DOCKER_GPU_ARGS} ${DOCKER_RUN_EXTRA} --rm \
    -v "${ROOT_DIR}:/workspace" \
    -w /workspace \
    "${IMAGE_NAME}" \
    python3 -m backend.test.balltest "${command}" "$@"
  log "finished ${command}"
}

log "series order: ${STEPS[*]}"

for step in "${STEPS[@]}"; do
  wait_for_training_idle
  # shellcheck disable=SC2206
  args=($step)
  run_tracknet "${args[@]}"
done

log "train-tracknet-series completed"
