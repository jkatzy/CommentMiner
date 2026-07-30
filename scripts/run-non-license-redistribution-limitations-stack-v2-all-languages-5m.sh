#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export INPUT_SOURCE=${INPUT_SOURCE:-var/code-comments-license-scan}
export OUTPUT=${OUTPUT:-var/non-license-redistribution-limitations-stack-v2-all-languages-5m}
export JUDGE_CACHE=${JUDGE_CACHE:-var/non-license-redistribution-limitations-stack-v2-all-languages-5m-judge-cache.sqlite}
export ALL_LANGUAGES=1
export COMMENT_ROWS_LIMIT=${COMMENT_ROWS_LIMIT:-5000000}
export SCANCODE_SCORE_BELOW=${SCANCODE_SCORE_BELOW:-0.9}
export SCAN_WORKERS=${SCAN_WORKERS:-32}
export BATCH_SIZE=${BATCH_SIZE:-65536}
export INCLUDE_GOVERNMENT_SEEDS=1
export INCLUDE_PROVENANCE_SEEDS=1
export INCLUDE_FUNDING_SEEDS=1
export INCLUDE_EXPORT_CONTROL_SEEDS=1
export INCLUDE_UNPUBLISHED_WORK_SEEDS=1
export JUDGE_BATCH_SIZE=${JUDGE_BATCH_SIZE:-64}
export JUDGE_WORKERS=${JUDGE_WORKERS:-8}
export CODEX_MODEL=${CODEX_MODEL:-gpt-5.6-luna}
export CODEX_REASONING_EFFORT=${CODEX_REASONING_EFFORT:-low}

"$SCRIPT_DIR/run-redistribution-candidate-dataset.sh" \
  --judgment-profile non_license_limitations \
  "$@"
