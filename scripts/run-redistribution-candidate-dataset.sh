#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_ROOT"

INPUT_SOURCE=${INPUT_SOURCE:-${INPUT:-Jkatzy/code-comments}}
OUTPUT=${OUTPUT:-var/redistribution-candidate-comments-java-100k}
DATASET=${DATASET:-the-stack-v2-dedup}
SOURCE_LANGUAGE=${SOURCE_LANGUAGE:-Java}
SOURCE_FILES_LIMIT=${SOURCE_FILES_LIMIT:-100000}
ALL_LANGUAGES=${ALL_LANGUAGES:-0}
COMMENT_ROWS_LIMIT=${COMMENT_ROWS_LIMIT:-}
SCANCODE_SCORE_BELOW=${SCANCODE_SCORE_BELOW:-}
SCAN_WORKERS=${SCAN_WORKERS:-1}
FUZZY_THRESHOLD=${FUZZY_THRESHOLD:-0.82}
INCLUDE_GOVERNMENT_SEEDS=${INCLUDE_GOVERNMENT_SEEDS:-0}
INCLUDE_PROVENANCE_SEEDS=${INCLUDE_PROVENANCE_SEEDS:-0}
INCLUDE_FUNDING_SEEDS=${INCLUDE_FUNDING_SEEDS:-0}
INCLUDE_EXPORT_CONTROL_SEEDS=${INCLUDE_EXPORT_CONTROL_SEEDS:-0}
INCLUDE_UNPUBLISHED_WORK_SEEDS=${INCLUDE_UNPUBLISHED_WORK_SEEDS:-0}
SCAN_ONLY=${SCAN_ONLY:-0}
BATCH_SIZE=${BATCH_SIZE:-8192}
JUDGE_BATCH_SIZE=${JUDGE_BATCH_SIZE:-64}
JUDGE_WORKERS=${JUDGE_WORKERS:-4}
JUDGE_MAX_BATCH_CHARS=${JUDGE_MAX_BATCH_CHARS:-160000}
JUDGE_MAX_COMMENT_CHARS=${JUDGE_MAX_COMMENT_CHARS:-12000}
JUDGE_MAX_ATTEMPTS=${JUDGE_MAX_ATTEMPTS:-3}
JUDGE_TIMEOUT_SECONDS=${JUDGE_TIMEOUT_SECONDS:-900}
JUDGE_CACHE=${JUDGE_CACHE:-${OUTPUT}-judge-cache.sqlite}
CODEX_COMMAND=${CODEX_COMMAND:-codex}
CODEX_MODEL=${CODEX_MODEL:-gpt-5.6-luna}
CODEX_REASONING_EFFORT=${CODEX_REASONING_EFFORT:-max}
REVISION=${REVISION:-0d4c83fac76705d2e2388186b628543a4916dab8}
TOKEN_ENV=${TOKEN_ENV:-}
HF_CACHE_DIRECTORY=${HF_CACHE_DIRECTORY:-}
OVERWRITE=${OVERWRITE:-0}

case "$INCLUDE_GOVERNMENT_SEEDS" in
  0|1) ;;
  *) echo "INCLUDE_GOVERNMENT_SEEDS must be 0 or 1" >&2; exit 2 ;;
esac
case "$ALL_LANGUAGES" in
  0|1) ;;
  *) echo "ALL_LANGUAGES must be 0 or 1" >&2; exit 2 ;;
esac
if [[ "$ALL_LANGUAGES" == 1 ]]; then
  if [[ -z "$COMMENT_ROWS_LIMIT" ]]; then
    echo "COMMENT_ROWS_LIMIT is required when ALL_LANGUAGES=1" >&2
    exit 2
  fi
  if [[ -z "$SCANCODE_SCORE_BELOW" ]]; then
    echo "SCANCODE_SCORE_BELOW is required when ALL_LANGUAGES=1" >&2
    exit 2
  fi
fi
case "$INCLUDE_PROVENANCE_SEEDS" in
  0|1) ;;
  *) echo "INCLUDE_PROVENANCE_SEEDS must be 0 or 1" >&2; exit 2 ;;
esac
case "$INCLUDE_FUNDING_SEEDS" in
  0|1) ;;
  *) echo "INCLUDE_FUNDING_SEEDS must be 0 or 1" >&2; exit 2 ;;
esac
case "$INCLUDE_EXPORT_CONTROL_SEEDS" in
  0|1) ;;
  *) echo "INCLUDE_EXPORT_CONTROL_SEEDS must be 0 or 1" >&2; exit 2 ;;
esac
case "$INCLUDE_UNPUBLISHED_WORK_SEEDS" in
  0|1) ;;
  *) echo "INCLUDE_UNPUBLISHED_WORK_SEEDS must be 0 or 1" >&2; exit 2 ;;
esac
case "$SCAN_ONLY" in
  0|1) ;;
  *) echo "SCAN_ONLY must be 0 or 1" >&2; exit 2 ;;
esac
case "$OVERWRITE" in
  0|1) ;;
  *) echo "OVERWRITE must be 0 or 1" >&2; exit 2 ;;
esac

command=(
  uv run commentminer build-redistribution-candidate-dataset
  "$OUTPUT"
  --input-source "$INPUT_SOURCE"
  --dataset "$DATASET"
  --language "$SOURCE_LANGUAGE"
  --source-files-limit "$SOURCE_FILES_LIMIT"
  --scan-workers "$SCAN_WORKERS"
  --fuzzy-threshold "$FUZZY_THRESHOLD"
  --batch-size "$BATCH_SIZE"
  --judge-batch-size "$JUDGE_BATCH_SIZE"
  --judge-workers "$JUDGE_WORKERS"
  --judge-max-batch-chars "$JUDGE_MAX_BATCH_CHARS"
  --judge-max-comment-chars "$JUDGE_MAX_COMMENT_CHARS"
  --judge-max-attempts "$JUDGE_MAX_ATTEMPTS"
  --judge-timeout-seconds "$JUDGE_TIMEOUT_SECONDS"
  --judge-cache "$JUDGE_CACHE"
  --codex-command "$CODEX_COMMAND"
  --codex-model "$CODEX_MODEL"
  --codex-reasoning-effort "$CODEX_REASONING_EFFORT"
  --revision "$REVISION"
)

if [[ "$ALL_LANGUAGES" == 1 ]]; then
  command+=(
    --all-languages
    --comment-rows-limit "$COMMENT_ROWS_LIMIT"
    --scancode-score-below "$SCANCODE_SCORE_BELOW"
  )
fi

if [[ "$INCLUDE_GOVERNMENT_SEEDS" == 1 ]]; then
  command+=(--include-government-seeds)
fi
if [[ "$INCLUDE_PROVENANCE_SEEDS" == 1 ]]; then
  command+=(--include-provenance-seeds)
fi
if [[ "$INCLUDE_FUNDING_SEEDS" == 1 ]]; then
  command+=(--include-funding-seeds)
fi
if [[ "$INCLUDE_EXPORT_CONTROL_SEEDS" == 1 ]]; then
  command+=(--include-export-control-seeds)
fi
if [[ "$INCLUDE_UNPUBLISHED_WORK_SEEDS" == 1 ]]; then
  command+=(--include-unpublished-work-seeds)
fi
if [[ "$SCAN_ONLY" == 1 ]]; then
  command+=(--scan-only)
fi
if [[ "$OVERWRITE" == 1 ]]; then
  command+=(--overwrite)
fi
if [[ -n "$TOKEN_ENV" ]]; then
  command+=(--token-env "$TOKEN_ENV")
fi
if [[ -n "$HF_CACHE_DIRECTORY" ]]; then
  command+=(--hf-cache-directory "$HF_CACHE_DIRECTORY")
fi

"${command[@]}" "$@"

if [[ "$SCAN_ONLY" == 0 ]]; then
  uv run commentminer verify-redistribution-candidate-dataset "$OUTPUT"
fi
