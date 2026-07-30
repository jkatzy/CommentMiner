#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export INPUT_SOURCE=${INPUT_SOURCE:-var/code-comments-license-scan}
export OUTPUT=${OUTPUT:-var/non-license-redistribution-limitations-java-100k}
export JUDGE_CACHE=${JUDGE_CACHE:-var/non-license-redistribution-limitations-java-100k-judge-cache.sqlite}
export JUDGE_BATCH_SIZE=${JUDGE_BATCH_SIZE:-4}
export JUDGE_WORKERS=${JUDGE_WORKERS:-4}
export CODEX_MODEL=${CODEX_MODEL:-gpt-5.6-luna}
export CODEX_REASONING_EFFORT=${CODEX_REASONING_EFFORT:-max}

"$SCRIPT_DIR/run-redistribution-candidate-dataset.sh" \
  --judgment-profile non_license_limitations \
  "$@"
