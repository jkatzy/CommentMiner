#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/home/jovyan/work}
REPO_ID=${REPO_ID:-Jkatzy/code-comments}
WATCH_PID=${WATCH_PID:-}
WATCH_LOG=${WATCH_LOG:-}
MAX_RESTARTS=${MAX_RESTARTS:-3}
SLEEP_SECONDS=${SLEEP_SECONDS:-60}
UPLOAD=${UPLOAD:-0}
DOWNLOAD_LEGACY=${DOWNLOAD_LEGACY:-1}
SCAN_STACK_V2=${SCAN_STACK_V2:-1}
SCAN_LEGACY=${SCAN_LEGACY:-1}
FINALIZE=${FINALIZE:-1}
SCANCODE_BACKEND=${SCANCODE_BACKEND:-api}
SCANCODE_PROCESSES=${SCANCODE_PROCESSES:-1}
WORKERS=${WORKERS:-80}
BATCH_SIZE=${BATCH_SIZE:-5000}
PROGRESS_EVERY=${PROGRESS_EVERY:-10}
STACK_V2_OUTPUT=${STACK_V2_OUTPUT:-var/comment-dataset-the-stack-v2-dedup-license-scan}
LEGACY_OUTPUT=${LEGACY_OUTPUT:-var/code-comments-source-license-scan}
FINAL_OUTPUT=${FINAL_OUTPUT:-var/code-comments-license-scan}

cd "$ROOT_DIR"
mkdir -p var/logs

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

runner_completed() {
  [[ -n "$WATCH_LOG" && -f "$WATCH_LOG" ]] && grep -qE '^[0-9TZ:-]+ Done$' "$WATCH_LOG"
}

run_upload_verifier_if_ready() {
  if [[ "$UPLOAD" == "1" && -d "$FINAL_OUTPUT" ]]; then
    log "Running standalone upload verifier"
    REPO_ID="$REPO_ID" FINAL_OUTPUT="$FINAL_OUTPUT" \
      scripts/verify-code-comments-license-upload.sh
  fi
}

restart_count=0
current_pid="$WATCH_PID"
current_log="$WATCH_LOG"

while true; do
  if [[ -n "$current_pid" ]] && ps -p "$current_pid" >/dev/null 2>&1; then
    sleep "$SLEEP_SECONDS"
    continue
  fi

  WATCH_LOG="$current_log"
  if runner_completed; then
    log "Watched runner completed"
    run_upload_verifier_if_ready
    exit 0
  fi

  if (( restart_count >= MAX_RESTARTS )); then
    log "Maximum restarts reached; leaving scan for manual inspection"
    exit 1
  fi

  restart_count=$((restart_count + 1))
  current_log="var/logs/code-comments-license-scan-runner-$(date -u +%Y%m%dT%H%M%SZ)-watchdog-${restart_count}.log"
  log "Watched runner is not active and not complete; restarting attempt=${restart_count} log=${current_log}"
  rm -rf \
    "${STACK_V2_OUTPUT}/.tmp/"* \
    "${LEGACY_OUTPUT}/.tmp/"* 2>/dev/null || true
  setsid env \
    ROOT_DIR="$ROOT_DIR" \
    REPO_ID="$REPO_ID" \
    UPLOAD="$UPLOAD" \
    DOWNLOAD_LEGACY="$DOWNLOAD_LEGACY" \
    SCAN_STACK_V2="$SCAN_STACK_V2" \
    SCAN_LEGACY="$SCAN_LEGACY" \
    FINALIZE="$FINALIZE" \
    SCANCODE_BACKEND="$SCANCODE_BACKEND" \
    SCANCODE_PROCESSES="$SCANCODE_PROCESSES" \
    WORKERS="$WORKERS" \
    BATCH_SIZE="$BATCH_SIZE" \
    PROGRESS_EVERY="$PROGRESS_EVERY" \
    STACK_V2_OUTPUT="$STACK_V2_OUTPUT" \
    LEGACY_OUTPUT="$LEGACY_OUTPUT" \
    FINAL_OUTPUT="$FINAL_OUTPUT" \
    "$ROOT_DIR/scripts/run-code-comments-license-scan.sh" \
    > "$current_log" 2>&1 < /dev/null &
  current_pid=$!
  log "Restarted runner pid=${current_pid}"
done
