#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BALLTEST_DIR="$ROOT_DIR/backend/test/balltest"
TMUX_ROOT="$BALLTEST_DIR/artifacts/tmux"
LOG_DIR="$TMUX_ROOT/logs"
CMD_DIR="$TMUX_ROOT/cmd"
STATE_DIR="$TMUX_ROOT/state"

IMAGE_NAME="${BALLTEST_IMAGE:-balltest-runner}"
SESSION_PREFIX="${BALLTEST_SESSION_PREFIX:-balltest}"
DOCKER_GPU_ARGS="${BALLTEST_DOCKER_GPU_ARGS:---gpus all}"
DOCKER_RUN_EXTRA="${BALLTEST_DOCKER_RUN_EXTRA:---ipc=host}"

mkdir -p "$LOG_DIR" "$CMD_DIR" "$STATE_DIR"

usage() {
  cat <<'EOF'
Usage:
  backend/test/balltest/run.sh build-image
  backend/test/balltest/run.sh prepare [-- extra-cli-args]
  backend/test/balltest/run.sh train-yolo [-- extra-cli-args]
  backend/test/balltest/run.sh train-yolo-series [series-args]
  backend/test/balltest/run.sh train-tracknet-series
  backend/test/balltest/run.sh cache-yolo [-- extra-cli-args]
  backend/test/balltest/run.sh train-tracknet [-- extra-cli-args]
  backend/test/balltest/run.sh train-tracknet-v2 [-- extra-cli-args]
  backend/test/balltest/run.sh train-tracknet-v3 [-- extra-cli-args]
  backend/test/balltest/run.sh train-tracknet-v4 [-- extra-cli-args]
  backend/test/balltest/run.sh train-tracknet-v5 [-- extra-cli-args]
  backend/test/balltest/run.sh eval-methods [-- extra-cli-args]
  backend/test/balltest/run.sh eval-trackers [-- extra-cli-args]
  backend/test/balltest/run.sh eval-ablation [-- extra-cli-args]
  backend/test/balltest/run.sh report
  backend/test/balltest/run.sh list
  backend/test/balltest/run.sh attach <alias|session>
  backend/test/balltest/run.sh logs <alias|session>
  backend/test/balltest/run.sh kill <alias|session>

Examples:
  backend/test/balltest/run.sh build-image
  backend/test/balltest/run.sh prepare -- --force
  backend/test/balltest/run.sh train-yolo -- --resume
  backend/test/balltest/run.sh train-yolo -- --variant yolo26m
  backend/test/balltest/run.sh train-yolo-series
  backend/test/balltest/run.sh train-yolo-series --variants yolov8s yolo11s -- --epochs 160 --patience 30
  backend/test/balltest/run.sh train-tracknet-series
  backend/test/balltest/run.sh train-tracknet -- --resume
  backend/test/balltest/run.sh train-tracknet-v2
  backend/test/balltest/run.sh train-tracknet-v3
  backend/test/balltest/run.sh train-tracknet-v4 -- --resume
  backend/test/balltest/run.sh train-tracknet-v5 -- --resume
  backend/test/balltest/run.sh eval-methods -- --methods yolo26s_raw yolo26m_raw yolov8s_raw yolo11s_raw
  backend/test/balltest/run.sh eval-trackers -- --methods yolo_balltracker
  backend/test/balltest/run.sh eval-trackers -- --methods yolo_balltracker yolo_sort yolo_bytetrack
  backend/test/balltest/run.sh eval-ablation
EOF
}

require_bin() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required command: $1" >&2
    exit 1
  fi
}

shell_join() {
  local out=""
  local arg
  for arg in "$@"; do
    out+=" $(printf '%q' "$arg")"
  done
  printf '%s' "${out# }"
}

latest_file() {
  printf '%s/%s.latest\n' "$STATE_DIR" "$1"
}

meta_file() {
  printf '%s/%s.meta\n' "$STATE_DIR" "$1"
}

write_state() {
  local alias="$1"
  local session="$2"
  local log_path="$3"
  local cmd_path="$4"
  printf '%s\n' "$session" > "$(latest_file "$alias")"
  cat > "$(meta_file "$alias")" <<EOF
session=$session
log=$log_path
cmd=$cmd_path
EOF
}

read_meta_value() {
  local target="$1"
  local key="$2"
  local meta
  meta="$(meta_file "$target")"
  if [[ ! -f "$meta" ]]; then
    return 1
  fi
  awk -F= -v k="$key" '$1==k {print substr($0, index($0, "=")+1)}' "$meta"
}

resolve_session() {
  local target="$1"
  if tmux has-session -t "$target" 2>/dev/null; then
    printf '%s\n' "$target"
    return 0
  fi
  if [[ -f "$(latest_file "$target")" ]]; then
    local latest
    latest="$(cat "$(latest_file "$target")")"
    if tmux has-session -t "$latest" 2>/dev/null; then
      printf '%s\n' "$latest"
      return 0
    fi
    echo "latest session for alias '$target' is not running anymore; use logs $target" >&2
    exit 1
  fi
  echo "cannot resolve tmux target: $target" >&2
  exit 1
}

resolve_log() {
  local target="$1"
  if [[ -f "$(meta_file "$target")" ]]; then
    read_meta_value "$target" "log"
    return 0
  fi
  local session
  session="$(resolve_session "$target")"
  local alias
  for alias_meta in "$STATE_DIR"/*.meta; do
    [[ -e "$alias_meta" ]] || continue
    if grep -q "^session=$session$" "$alias_meta"; then
      awk -F= '$1=="log" {print substr($0, index($0, "=")+1)}' "$alias_meta"
      return 0
    fi
  done
  echo "cannot resolve log for: $target" >&2
  exit 1
}

start_tmux_job() {
  local alias="$1"
  local command_str="$2"
  local ts session log_path cmd_path
  ts="$(date +%Y%m%d-%H%M%S)"
  session="${SESSION_PREFIX}.${alias}.${ts}"
  log_path="$LOG_DIR/${session}.log"
  cmd_path="$CMD_DIR/${session}.sh"

  cat > "$cmd_path" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd $(printf '%q' "$ROOT_DIR")
{
  echo "[start] \$(date -Iseconds)"
  echo "[cwd] $ROOT_DIR"
  echo "[alias] $alias"
  echo "[session] $session"
  echo "[command] $command_str"
  eval "$command_str"
} 2>&1 | tee -a $(printf '%q' "$log_path")
status=\${PIPESTATUS[0]}
echo "[end] \$(date -Iseconds) exit=\$status" | tee -a $(printf '%q' "$log_path")
exit "\$status"
EOF
  chmod +x "$cmd_path"

  tmux new-session -d -s "$session" "bash $(printf '%q' "$cmd_path")"
  write_state "$alias" "$session" "$log_path" "$cmd_path"
  echo "started tmux session: $session"
  echo "log: $log_path"
  echo "attach: backend/test/balltest/run.sh attach $alias"
}

docker_cli_cmd() {
  local cli_args=("$@")
  printf 'docker run %s %s --rm -v %q:/workspace -w /workspace %q python3 -m backend.test.balltest %s' \
    "$DOCKER_GPU_ARGS" \
    "$DOCKER_RUN_EXTRA" \
    "$ROOT_DIR" \
    "$IMAGE_NAME" \
    "$(shell_join "${cli_args[@]}")"
}

build_image_cmd() {
  printf 'docker build --progress=plain -f %q -t %q %q' \
    "$BALLTEST_DIR/Dockerfile" \
    "$IMAGE_NAME" \
    "$ROOT_DIR"
}

parse_extra_args() {
  EXTRA_ARGS=()
  if [[ "${1:-}" == "--" ]]; then
    shift
  fi
  if [[ "$#" -gt 0 ]]; then
    EXTRA_ARGS=("$@")
  fi
}

list_jobs() {
  echo "[running tmux sessions]"
  tmux list-sessions 2>/dev/null | grep "^${SESSION_PREFIX}\." || true
  echo
  echo "[latest aliases]"
  for meta in "$STATE_DIR"/*.meta; do
    [[ -e "$meta" ]] || continue
    local alias
    alias="$(basename "$meta" .meta)"
    echo "$alias -> $(read_meta_value "$alias" "session")"
  done
}

main() {
  require_bin tmux
  require_bin docker

  local action="${1:-}"
  if [[ -z "$action" ]]; then
    usage
    exit 1
  fi
  shift || true

  case "$action" in
    build-image)
      start_tmux_job "build-image" "$(build_image_cmd)"
      ;;
    train-yolo-series)
      start_tmux_job "train-yolo-series" "$(printf '%q' "$BALLTEST_DIR/run_yolo_series.sh") $(shell_join "$@")"
      ;;
    train-tracknet-series)
      start_tmux_job "train-tracknet-series" "$(printf '%q' "$BALLTEST_DIR/run_tracknet_series.sh") $(shell_join "$@")"
      ;;
    prepare|train-yolo|cache-yolo|train-tracknet|train-tracknet-v2|train-tracknet-v3|train-tracknet-v4|train-tracknet-v5|eval-methods|eval-trackers|eval-ablation)
      parse_extra_args "$@"
      start_tmux_job "$action" "$(docker_cli_cmd "$action" "${EXTRA_ARGS[@]}")"
      ;;
    report)
      start_tmux_job "report" "$(docker_cli_cmd report)"
      ;;
    list)
      list_jobs
      ;;
    attach)
      local target="${1:-}"
      if [[ -z "$target" ]]; then
        echo "attach requires <alias|session>" >&2
        exit 1
      fi
      tmux attach -t "$(resolve_session "$target")"
      ;;
    logs)
      local target="${1:-}"
      if [[ -z "$target" ]]; then
        echo "logs requires <alias|session>" >&2
        exit 1
      fi
      tail -n 200 -f "$(resolve_log "$target")"
      ;;
    kill)
      local target="${1:-}"
      if [[ -z "$target" ]]; then
        echo "kill requires <alias|session>" >&2
        exit 1
      fi
      tmux kill-session -t "$(resolve_session "$target")"
      ;;
    help|-h|--help)
      usage
      ;;
    *)
      echo "unknown action: $action" >&2
      usage
      exit 1
      ;;
  esac
}

main "$@"
