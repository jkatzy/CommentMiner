#!/usr/bin/env bash
set -euo pipefail

parallel="${1:-${PARALLEL:-4}}"
config="${CONFIG:-config/pipeline.example.json}"
dataset="${DATASET:-the-stack-v2}"
log_root="${LOG_ROOT:-var/logs/the-stack-v2-parallel-$(date -u +%Y%m%dT%H%M%SZ)}"
prefetch_files="${PREFETCH_FILES:-2}"
download_workers="${DOWNLOAD_WORKERS:-2}"
extraction_workers="${EXTRACTION_WORKERS:-4}"
extraction_buffer="${EXTRACTION_BUFFER:-16}"
progress_every="${PROGRESS_EVERY:-10000}"
total_content_download_workers="${TOTAL_CONTENT_DOWNLOAD_WORKERS:-128}"
content_download_workers="${CONTENT_DOWNLOAD_WORKERS:-}"
content_prefetch_records="${CONTENT_PREFETCH_RECORDS:-}"

if ! [[ "$parallel" =~ ^[1-9][0-9]*$ ]]; then
  echo "parallel must be a positive integer, got: $parallel" >&2
  exit 2
fi
if ! [[ "$total_content_download_workers" =~ ^[1-9][0-9]*$ ]]; then
  echo "TOTAL_CONTENT_DOWNLOAD_WORKERS must be a positive integer, got: $total_content_download_workers" >&2
  exit 2
fi
if [[ -z "$content_download_workers" ]]; then
  content_download_workers=$(( (total_content_download_workers + parallel - 1) / parallel ))
fi
if ! [[ "$content_download_workers" =~ ^[1-9][0-9]*$ ]]; then
  echo "CONTENT_DOWNLOAD_WORKERS must be a positive integer, got: $content_download_workers" >&2
  exit 2
fi
if [[ -z "$content_prefetch_records" ]]; then
  content_prefetch_records=$(( content_download_workers * 4 ))
fi
if ! [[ "$content_prefetch_records" =~ ^[1-9][0-9]*$ ]]; then
  echo "CONTENT_PREFETCH_RECORDS must be a positive integer, got: $content_prefetch_records" >&2
  exit 2
fi
if (( content_prefetch_records < content_download_workers )); then
  echo "CONTENT_PREFETCH_RECORDS must be >= CONTENT_DOWNLOAD_WORKERS" >&2
  exit 2
fi

mkdir -p "$log_root"

echo "Dataset: $dataset"
echo "Config: $config"
echo "Parallel workers: $parallel"
echo "Content download workers per language: $content_download_workers"
echo "Approximate total content download workers: $(( parallel * content_download_workers ))"
echo "Log root: $log_root"
echo "Summary: $log_root/summary.log"

uv run commentminer list-languages "$config" "$dataset" |
  sed -n "s/^- //p" |
  xargs -r -P "$parallel" -I {} bash -c '
    set -u

    language="$1"
    log_root="$2"
    config="$3"
    dataset="$4"
    prefetch_files="$5"
    download_workers="$6"
    extraction_workers="$7"
    extraction_buffer="$8"
    progress_every="$9"
    content_download_workers="${10}"
    content_prefetch_records="${11}"

    mkdir -p "$log_root"
    safe=$(printf "%s" "$language" | sed "s/[^A-Za-z0-9._+-]/-/g")
    log="$log_root/${safe}.log"

    printf "%s START %s\n" "$(date -u +%FT%TZ)" "$language" >> "$log_root/summary.log"

    if uv run commentminer --log-level INFO mine-dataset "$config" "$dataset" \
      --language "$language" \
      --prefetch-files "$prefetch_files" \
      --download-workers "$download_workers" \
      --content-download-workers "$content_download_workers" \
      --content-prefetch-records "$content_prefetch_records" \
      --extraction-workers "$extraction_workers" \
      --extraction-buffer "$extraction_buffer" \
      --progress-every "$progress_every" \
      --no-tqdm > "$log" 2>&1; then
      status=0
    else
      status=$?
    fi

    printf "%s DONE(%s) %s\n" "$(date -u +%FT%TZ)" "$status" "$language" >> "$log_root/summary.log"
    exit "$status"
  ' _ {} "$log_root" "$config" "$dataset" "$prefetch_files" "$download_workers" "$extraction_workers" "$extraction_buffer" "$progress_every" "$content_download_workers" "$content_prefetch_records"
