#!/usr/bin/env bash
set -euo pipefail

config="${CONFIG:-config/pipeline.example.json}"
dataset="${DATASET:-the-stack-v2}"
package_size="${PACKAGE_SIZE:-10000}"
metadata_download_workers="${METADATA_DOWNLOAD_WORKERS:-4}"
package_workers="${PACKAGE_WORKERS:-64}"
package_worker_backend="${PACKAGE_WORKER_BACKEND:-process}"
content_download_workers="${CONTENT_DOWNLOAD_WORKERS:-2048}"
content_prefetch_records="${CONTENT_PREFETCH_RECORDS:-}"
extraction_workers="${EXTRACTION_WORKERS:-4}"
extraction_buffer="${EXTRACTION_BUFFER:-}"
max_files_per_language="${MAX_FILES_PER_LANGUAGE:-}"
max_languages="${MAX_LANGUAGES:-}"
max_packages="${MAX_PACKAGES:-}"
languages="${LANGUAGES:-}"
token_env="${TOKEN_ENV:-}"
progress_every="${PROGRESS_EVERY:-10000}"
log_level="${LOG_LEVEL:-INFO}"

args=(
  uv run commentminer --log-level "$log_level" mine-stack-v2-packages
  "$config"
  "$dataset"
  --package-size "$package_size"
  --metadata-download-workers "$metadata_download_workers"
  --package-workers "$package_workers"
  --package-worker-backend "$package_worker_backend"
  --content-download-workers "$content_download_workers"
  --extraction-workers "$extraction_workers"
  --progress-every "$progress_every"
  --no-tqdm
)

if [[ -n "$content_prefetch_records" ]]; then
  args+=(--content-prefetch-records "$content_prefetch_records")
fi
if [[ -n "$extraction_buffer" ]]; then
  args+=(--extraction-buffer "$extraction_buffer")
fi
if [[ -n "$max_files_per_language" ]]; then
  args+=(--max-files-per-language "$max_files_per_language")
fi
if [[ -n "$max_languages" ]]; then
  args+=(--max-languages "$max_languages")
fi
if [[ -n "$max_packages" ]]; then
  args+=(--max-packages "$max_packages")
fi
if [[ -n "$languages" ]]; then
  args+=(--languages "$languages")
fi
if [[ -n "$token_env" ]]; then
  args+=(--token-env "$token_env")
fi
if [[ "${SKIP_ERRORS:-}" == "1" ]]; then
  args+=(--skip-errors)
fi
if [[ "${RERUN_COMPLETED_PACKAGES:-}" == "1" ]]; then
  args+=(--rerun-completed-packages)
fi
if [[ "${CACHE_SOURCE_FILES:-}" == "1" ]]; then
  args+=(--cache-source-files)
fi

printf "Config: %s\n" "$config"
printf "Dataset: %s\n" "$dataset"
printf "Package size: %s ids\n" "$package_size"
printf "Metadata download workers: %s\n" "$metadata_download_workers"
printf "Package workers: %s\n" "$package_workers"
printf "Package worker backend: %s\n" "$package_worker_backend"
printf "Total S3 content download workers: %s\n" "$content_download_workers"

exec "${args[@]}"
