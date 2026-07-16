#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/home/jovyan/work}
REPO_ID=${REPO_ID:-Jkatzy/code-comments}
STACK_V2_INPUT=${STACK_V2_INPUT:-var/comment-dataset-the-stack-v2-dedup}
STACK_V2_OUTPUT=${STACK_V2_OUTPUT:-var/comment-dataset-the-stack-v2-dedup-license-scan}
STACK_V2_DATASET=${STACK_V2_DATASET:-the-stack-v2-dedup}
LEGACY_INPUT=${LEGACY_INPUT:-var/code-comments-source}
LEGACY_OUTPUT=${LEGACY_OUTPUT:-var/code-comments-source-license-scan}
FINAL_OUTPUT=${FINAL_OUTPUT:-var/code-comments-license-scan}
DETECTION_CACHE=${DETECTION_CACHE:-var/code-comments-license-score-cache.sqlite}
DATASETS_LEGACY=${DATASETS_LEGACY:-the-heap,the-stack}
BATCH_SIZE=${BATCH_SIZE:-5000}
WORKERS=${WORKERS:-80}
SCANCODE_BACKEND=${SCANCODE_BACKEND:-api}
SCANCODE_PROCESSES=${SCANCODE_PROCESSES:-1}
PROGRESS_EVERY=${PROGRESS_EVERY:-10}
DOWNLOAD_WORKERS=${DOWNLOAD_WORKERS:-8}
UPLOAD=${UPLOAD:-0}
DOWNLOAD_LEGACY=${DOWNLOAD_LEGACY:-1}
SCAN_STACK_V2=${SCAN_STACK_V2:-1}
SCAN_LEGACY=${SCAN_LEGACY:-1}
FINALIZE=${FINALIZE:-1}

cd "$ROOT_DIR"
mkdir -p var/logs

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

wait_for_existing_process() {
  label=$1
  pattern=$2
  while pgrep -f "$pattern" >/dev/null; do
    log "Waiting for existing ${label} process to finish"
    sleep 300
  done
}

download_legacy_configs() {
  wait_for_existing_process "legacy download" "snapshot_download.*${LEGACY_INPUT}"
  log "Downloading legacy configs from ${REPO_ID} into ${LEGACY_INPUT}"
  REPO_ID="$REPO_ID" LEGACY_INPUT="$LEGACY_INPUT" DATASETS_LEGACY="$DATASETS_LEGACY" \
    DOWNLOAD_WORKERS="$DOWNLOAD_WORKERS" uv run python - <<'PY'
import os
from huggingface_hub import snapshot_download

datasets = [item for item in os.environ["DATASETS_LEGACY"].split(",") if item]
snapshot_download(
    repo_id=os.environ["REPO_ID"],
    repo_type="dataset",
    local_dir=os.environ["LEGACY_INPUT"],
    allow_patterns=["README.md", *(f"{dataset}/**" for dataset in datasets)],
    max_workers=int(os.environ["DOWNLOAD_WORKERS"]),
)
PY
}

verify_legacy_download() {
  log "Verifying legacy download coverage"
  REPO_ID="$REPO_ID" LEGACY_INPUT="$LEGACY_INPUT" DATASETS_LEGACY="$DATASETS_LEGACY" \
    uv run python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import HfApi

repo_id = os.environ["REPO_ID"]
root = Path(os.environ["LEGACY_INPUT"])
datasets = [item for item in os.environ["DATASETS_LEGACY"].split(",") if item]
remote_files = HfApi().list_repo_files(repo_id=repo_id, repo_type="dataset")
errors = []
for dataset in datasets:
    remote_count = sum(
        1 for path in remote_files
        if path.startswith(f"{dataset}/") and path.endswith(".parquet")
    )
    local_count = sum(1 for _ in (root / dataset).glob("*/*.parquet"))
    print(f"{dataset}: local={local_count} remote={remote_count}")
    if local_count != remote_count:
        errors.append(f"{dataset}: local={local_count} remote={remote_count}")
if errors:
    raise SystemExit("Incomplete legacy download: " + "; ".join(errors))
PY
}

scan_stack_v2() {
  wait_for_existing_process "Stack v2 scan" "commentminer scan-hf-comment-licenses ${STACK_V2_INPUT}.*${STACK_V2_OUTPUT}"
  log "Scanning Stack v2 config"
  uv run commentminer scan-hf-comment-licenses \
    "$STACK_V2_INPUT" \
    --output-directory "$STACK_V2_OUTPUT" \
    --detection-cache "$DETECTION_CACHE" \
    --datasets "$STACK_V2_DATASET" \
    --scanner-backend "$SCANCODE_BACKEND" \
    --batch-size "$BATCH_SIZE" \
    --workers "$WORKERS" \
    --scancode-processes "$SCANCODE_PROCESSES" \
    --progress-every "$PROGRESS_EVERY"
}

scan_legacy_configs() {
  wait_for_existing_process "legacy scan" "commentminer scan-hf-comment-licenses ${LEGACY_INPUT}.*${LEGACY_OUTPUT}"
  log "Scanning legacy configs"
  uv run commentminer scan-hf-comment-licenses \
    "$LEGACY_INPUT" \
    --output-directory "$LEGACY_OUTPUT" \
    --detection-cache "$DETECTION_CACHE" \
    --datasets "$DATASETS_LEGACY" \
    --scanner-backend "$SCANCODE_BACKEND" \
    --batch-size "$BATCH_SIZE" \
    --workers "$WORKERS" \
    --scancode-processes "$SCANCODE_PROCESSES" \
    --progress-every "$PROGRESS_EVERY"
}

stage_final_dataset() {
  log "Staging combined license-scanned dataset at ${FINAL_OUTPUT}"
  tmp_output="${FINAL_OUTPUT}.tmp"
  rm -rf "$tmp_output"
  mkdir -p "$tmp_output/license-scan-manifests"

  if [[ -f "${LEGACY_INPUT}/README.md" ]]; then
    cp "${LEGACY_INPUT}/README.md" "${tmp_output}/README.md"
  elif [[ -f "${STACK_V2_INPUT}/README.md" ]]; then
    cp "${STACK_V2_INPUT}/README.md" "${tmp_output}/README.md"
  fi
  if [[ -f "${tmp_output}/README.md" ]]; then
    TMP_OUTPUT="$tmp_output" uv run python - <<'PY'
import os
from pathlib import Path

readme = Path(os.environ["TMP_OUTPUT"]) / "README.md"
text = readme.read_text(encoding="utf-8")
old = "Each row contains `dataset`, `record_id`, `opening_comment`, `language`, `path`, `repo`, `extracted_at`, and `metadata`."
new = "Each row contains `dataset`, `record_id`, `opening_comment`, `language`, `path`, `repo`, `extracted_at`, `metadata`, `comment_license_detection`, and `comment_license_score`."
if old in text:
    text = text.replace(old, new)
note = (
    "\\n## License Detection\\n\\n"
    "`comment_license_detection` is a JSON string produced by ScanCode Toolkit. "
    "It records whether the opening comment matched a known license notice, the detected license expressions, filtered license matches, ScanCode scan errors, and the best raw ScanCode license score. "
    "`comment_license_score` is a numeric column with the best raw ScanCode score across license matches and clues for every row, or 0 when no score was found. "
    "The `contains_license_notice` flag is counted with minimum ScanCode score 95 and minimum match coverage 95.\\n"
)
if "## License Detection" not in text:
    text = text.rstrip() + note
readme.write_text(text + "\\n", encoding="utf-8")
PY
  fi

  IFS=',' read -r -a legacy_datasets <<< "$DATASETS_LEGACY"
  for dataset in "${legacy_datasets[@]}"; do
    cp -al "${LEGACY_OUTPUT}/${dataset}" "${tmp_output}/${dataset}"
  done
  cp -al "${STACK_V2_OUTPUT}/${STACK_V2_DATASET}" "${tmp_output}/${STACK_V2_DATASET}"

  cp "${LEGACY_OUTPUT}/manifest.json" "${tmp_output}/license-scan-manifests/legacy.json"
  cp "${STACK_V2_OUTPUT}/manifest.json" "${tmp_output}/license-scan-manifests/${STACK_V2_DATASET}.json"

  rm -rf "$FINAL_OUTPUT"
  mv "$tmp_output" "$FINAL_OUTPUT"
}

verify_final_dataset() {
  log "Verifying staged final dataset coverage"
  LEGACY_INPUT="$LEGACY_INPUT" STACK_V2_INPUT="$STACK_V2_INPUT" \
    FINAL_OUTPUT="$FINAL_OUTPUT" DATASETS_LEGACY="$DATASETS_LEGACY" \
    STACK_V2_DATASET="$STACK_V2_DATASET" uv run python - <<'PY'
import json
import math
import os
from pathlib import Path
import pyarrow.parquet as pq

legacy_root = Path(os.environ["LEGACY_INPUT"])
stack_v2_root = Path(os.environ["STACK_V2_INPUT"])
stack_v2_dataset = os.environ["STACK_V2_DATASET"]
legacy_datasets = [item for item in os.environ["DATASETS_LEGACY"].split(",") if item]
source_roots = {dataset: legacy_root for dataset in legacy_datasets}
source_roots[stack_v2_dataset] = stack_v2_root
final_root = Path(os.environ["FINAL_OUTPUT"])
errors = []
positive_scores = 0
threshold_hits = 0


def column_null_count(path, column_name):
    schema = pq.read_schema(path)
    if column_name not in schema.names:
        errors.append(f"missing {column_name} in {path}")
        return 0
    column_index = schema.names.index(column_name)
    metadata = pq.read_metadata(path)
    null_count = 0
    for row_group_index in range(metadata.num_row_groups):
        column = metadata.row_group(row_group_index).column(column_index)
        statistics = column.statistics
        if statistics is None or statistics.null_count is None:
            values = pq.read_table(path, columns=[column_name], use_threads=False).column(column_name)
            return sum(1 for value in values.to_pylist() if value is None)
        null_count += statistics.null_count
    return null_count


for dataset, source_root in source_roots.items():
    source_by_path = {
        path.relative_to(source_root / dataset): path
        for path in sorted((source_root / dataset).glob("*/*.parquet"))
    }
    final_by_path = {
        path.relative_to(final_root / dataset): path
        for path in sorted((final_root / dataset).glob("*/*.parquet"))
    }
    print(f"{dataset}: final={len(final_by_path)} source={len(source_by_path)}")
    if set(final_by_path) != set(source_by_path):
        missing = sorted(str(path) for path in set(source_by_path) - set(final_by_path))
        extra = sorted(str(path) for path in set(final_by_path) - set(source_by_path))
        errors.append(f"{dataset}: shard paths differ missing={missing[:5]} extra={extra[:5]}")
    source_rows = 0
    final_rows = 0
    for relative_path, source_path in source_by_path.items():
        source_row_count = pq.read_metadata(source_path).num_rows
        source_rows += source_row_count
        final_path = final_by_path.get(relative_path)
        if final_path is None:
            continue
        final_row_count = pq.read_metadata(final_path).num_rows
        final_rows += final_row_count
        if final_row_count != source_row_count:
            errors.append(
                f"{dataset}/{relative_path}: final_rows={final_row_count} source_rows={source_row_count}"
            )
        schema = pq.read_schema(final_path)
        if "comment_license_detection" not in schema.names:
            errors.append(f"{dataset}: missing comment_license_detection in {final_path}")
        if "comment_license_score" not in schema.names:
            errors.append(f"{dataset}: missing comment_license_score in {final_path}")
        else:
            null_scores = column_null_count(final_path, "comment_license_score")
            if null_scores:
                errors.append(f"{dataset}: {null_scores} null comment_license_score values in {final_path}")
        if not {"comment_license_detection", "comment_license_score"} <= set(schema.names):
            continue
        for batch in pq.ParquetFile(final_path).iter_batches(
            batch_size=100_000,
            columns=["comment_license_detection", "comment_license_score"],
        ):
            for detection_json, score_value in zip(
                batch.column(0).to_pylist(),
                batch.column(1).to_pylist(),
                strict=True,
            ):
                try:
                    score = float(score_value)
                except (TypeError, ValueError):
                    errors.append(f"{dataset}/{relative_path}: invalid score {score_value!r}")
                    continue
                if not math.isfinite(score) or not 0.0 <= score <= 100.0:
                    errors.append(f"{dataset}/{relative_path}: score outside 0-100: {score!r}")
                positive_scores += score > 0.0
                try:
                    detection = json.loads(detection_json)
                except (TypeError, json.JSONDecodeError):
                    errors.append(f"{dataset}/{relative_path}: invalid detection JSON")
                    continue
                if not isinstance(detection, dict):
                    errors.append(f"{dataset}/{relative_path}: detection is not an object")
                    continue
                try:
                    best_score = float(detection.get("best_license_score", 0.0))
                except (TypeError, ValueError):
                    errors.append(f"{dataset}/{relative_path}: invalid best_license_score")
                    continue
                if not math.isclose(score, best_score, rel_tol=0.0, abs_tol=1e-9):
                    errors.append(
                        f"{dataset}/{relative_path}: score={score} best_license_score={best_score}"
                    )
                threshold_hits += bool(detection.get("contains_license_notice"))
                engine_errors = [
                    str(item)
                    for item in detection.get("scan_errors", []) or []
                    if not str(item).startswith("Opening comment truncated from ")
                ]
                if engine_errors:
                    errors.append(f"{dataset}/{relative_path}: ScanCode errors={engine_errors[:3]}")
    print(f"{dataset}: final_rows={final_rows} source_rows={source_rows}")
    if final_rows != source_rows:
        errors.append(f"{dataset}: final_rows={final_rows} source_rows={source_rows}")
if positive_scores == 0:
    errors.append("No positive ScanCode scores found across the complete dataset")
if threshold_hits == 0:
    errors.append("No threshold-qualified license detections found across the complete dataset")
print(f"positive_scores={positive_scores} threshold_hits={threshold_hits}")
if errors:
    raise SystemExit("Final dataset verification failed: " + "; ".join(errors))
PY
}

upload_final_dataset() {
  if [[ "$UPLOAD" != "1" ]]; then
    log "UPLOAD=0, skipping upload"
    return
  fi
  uv run python - <<'PY'
from huggingface_hub import HfApi

try:
    whoami = HfApi().whoami()
except Exception as exc:
    raise SystemExit(f"Hugging Face authentication unavailable: {exc}") from exc
print("Authenticated as", whoami.get("name") or whoami)
PY
  log "Uploading ${FINAL_OUTPUT} to ${REPO_ID}"
  uv run python - <<PY
from huggingface_hub import HfApi

HfApi().upload_large_folder(
    repo_id="${REPO_ID}",
    repo_type="dataset",
    folder_path="${FINAL_OUTPUT}",
    num_workers=8,
    print_report_every=60,
)
PY
}

verify_uploaded_dataset() {
  if [[ "$UPLOAD" != "1" ]]; then
    return
  fi
  log "Verifying uploaded dataset file coverage and score schema"
  REPO_ID="$REPO_ID" FINAL_OUTPUT="$FINAL_OUTPUT" \
    DATASETS="${DATASETS_LEGACY},${STACK_V2_DATASET}" \
    scripts/verify-code-comments-license-upload.sh
}

if [[ "$DOWNLOAD_LEGACY" == "1" ]]; then
  download_legacy_configs
  verify_legacy_download
else
  log "DOWNLOAD_LEGACY=0, skipping legacy download and download verification"
fi
if [[ "$SCAN_STACK_V2" == "1" ]]; then
  scan_stack_v2
else
  log "SCAN_STACK_V2=0, skipping Stack v2 scan"
fi
if [[ "$SCAN_LEGACY" == "1" ]]; then
  scan_legacy_configs
else
  log "SCAN_LEGACY=0, skipping legacy scan"
fi
if [[ "$FINALIZE" == "1" ]]; then
  stage_final_dataset
  verify_final_dataset
  upload_final_dataset
  verify_uploaded_dataset
else
  log "FINALIZE=0, skipping final staging, verification, and upload"
fi
log "Done"
