#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG_PATH="${COMMENTMINER_STACK_CONFIG:-config/the-stack.sample.json}"
DATASET_NAME="${COMMENTMINER_STACK_DATASET:-the-stack}"
OUTPUT_ROOT="${COMMENTMINER_OUTPUT_ROOT:-$REPO_ROOT/var/server-output/the-stack}"
STATE_ROOT="${COMMENTMINER_STATE_ROOT:-$REPO_ROOT/var/server-state/the-stack}"
LOG_LEVEL="${COMMENTMINER_LOG_LEVEL:-INFO}"
PROGRESS_EVERY="${COMMENTMINER_PROGRESS_EVERY:-1000}"
MAX_COMMENT_START_ROW="${COMMENTMINER_MAX_COMMENT_START_ROW:-3}"
TOKEN_ENV="${COMMENTMINER_TOKEN_ENV:-}"
NO_TQDM="${COMMENTMINER_NO_TQDM:-0}"
MAX_AUTO_WORKERS="${COMMENTMINER_MAX_AUTO_WORKERS:-8}"
WORKERS_OVERRIDE="${COMMENTMINER_WORKERS:-${WORKERS:-}}"
AUTO_WORKERS=4
WORKER_SOURCE="auto"

if command -v nproc >/dev/null 2>&1; then
  AUTO_WORKERS="$(nproc)"
fi

if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
  AUTO_WORKERS="${SLURM_CPUS_PER_TASK}"
fi

if ! [[ "$AUTO_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid auto-detected worker count: '$AUTO_WORKERS'" >&2
  exit 1
fi

if ! [[ "$MAX_AUTO_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid COMMENTMINER_MAX_AUTO_WORKERS value: '$MAX_AUTO_WORKERS'" >&2
  exit 1
fi

if (( AUTO_WORKERS > MAX_AUTO_WORKERS )); then
  AUTO_WORKERS="$MAX_AUTO_WORKERS"
fi

if [[ -n "$WORKERS_OVERRIDE" ]]; then
  WORKERS="$WORKERS_OVERRIDE"
  WORKER_SOURCE="override"
else
  WORKERS="$AUTO_WORKERS"
fi

if ! [[ "$WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid worker count: '$WORKERS'" >&2
  exit 1
fi

declare -a TEMP_CONFIGS=()
declare -a FAILED_LANGUAGES=()
declare -a LANGUAGES=()

cleanup() {
  if ((${#TEMP_CONFIGS[@]} == 0)); then
    return
  fi
  rm -f "${TEMP_CONFIGS[@]}"
}

trap cleanup EXIT

discover_languages() {
  local cmd=(
    uv run commentminer
    --log-level WARNING
    list-languages
    "$CONFIG_PATH"
    "$DATASET_NAME"
  )
  if [[ -n "$TOKEN_ENV" ]]; then
    cmd+=(--token-env "$TOKEN_ENV")
  fi
  "${cmd[@]}" | awk '/^- / {print substr($0, 3)}'
}

write_language_config() {
  local language="$1"
  local temp_config="$2"
  local language_output_root="$OUTPUT_ROOT/$language"
  local language_state_root="$STATE_ROOT/$language"

  uv run python - "$CONFIG_PATH" "$temp_config" "$language_output_root" "$language_state_root" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

source_config = Path(sys.argv[1])
temp_config = Path(sys.argv[2])
language_output_root = Path(sys.argv[3]).resolve()
language_state_root = Path(sys.argv[4]).resolve()

config = json.loads(source_config.read_text(encoding="utf-8"))
storage = config["storage"]
storage["working_directory"] = str((language_state_root / "work").resolve())
storage["output_directory"] = str(language_output_root)
storage["checkpoint_directory"] = str((language_state_root / "checkpoints").resolve())
storage["download_directory"] = str((language_state_root / "downloads").resolve())
storage["huggingface_cache_directory"] = str((language_state_root / "hf-cache").resolve())

temp_config.write_text(json.dumps(config, indent=2), encoding="utf-8")
PY
}

run_language() {
  local language="$1"
  local temp_config
  temp_config="$(mktemp "${TMPDIR:-/tmp}/commentminer-${language}-XXXXXX.json")"
  TEMP_CONFIGS+=("$temp_config")
  write_language_config "$language" "$temp_config"

  local cmd=(
    uv run commentminer
    --log-level "$LOG_LEVEL"
    mine-dataset
    "$temp_config"
    "$DATASET_NAME"
    --language "$language"
    --workers "$WORKERS"
    --progress-every "$PROGRESS_EVERY"
    --max-comment-start-row "$MAX_COMMENT_START_ROW"
  )
  if [[ -n "$TOKEN_ENV" ]]; then
    cmd+=(--token-env "$TOKEN_ENV")
  fi
  if [[ "$NO_TQDM" == "1" ]]; then
    cmd+=(--no-tqdm)
  fi

  echo "=== Starting language: $language ==="
  echo "Output root: $OUTPUT_ROOT/$language/the-stack"
  echo "State root:  $STATE_ROOT/$language"
  echo "Workers:     $WORKERS ($WORKER_SOURCE)"

  if "${cmd[@]}"; then
    echo "=== Completed language: $language ==="
    return 0
  fi

  echo "=== Failed language: $language ===" >&2
  return 1
}

main() {
  mkdir -p "$OUTPUT_ROOT" "$STATE_ROOT"

  if (($# > 0)); then
    LANGUAGES=("$@")
  else
    mapfile -t LANGUAGES < <(discover_languages)
  fi

  if ((${#LANGUAGES[@]} == 0)); then
    echo "No languages found for dataset '$DATASET_NAME' using config '$CONFIG_PATH'." >&2
    return 1
  fi

  for language in "${LANGUAGES[@]}"; do
    if ! run_language "$language"; then
      FAILED_LANGUAGES+=("$language")
    fi
  done

  if ((${#FAILED_LANGUAGES[@]} > 0)); then
    echo "Languages with failed runs:" >&2
    printf ' - %s\n' "${FAILED_LANGUAGES[@]}" >&2
    return 1
  fi

  echo "Finished all languages."
  echo "Per-language outputs are stored under: $OUTPUT_ROOT/<language>/the-stack/<run_id>/"
  return 0
}

main "$@"
