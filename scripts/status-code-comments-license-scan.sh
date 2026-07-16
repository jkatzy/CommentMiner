#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/home/jovyan/work}
STACK_V2_INPUT=${STACK_V2_INPUT:-var/comment-dataset-the-stack-v2-dedup}
STACK_V2_OUTPUT=${STACK_V2_OUTPUT:-var/comment-dataset-the-stack-v2-dedup-license-scan}
STACK_V2_DATASET=${STACK_V2_DATASET:-the-stack-v2-dedup}
LEGACY_INPUT=${LEGACY_INPUT:-var/code-comments-source}
LEGACY_OUTPUT=${LEGACY_OUTPUT:-var/code-comments-source-license-scan}
FINAL_OUTPUT=${FINAL_OUTPUT:-var/code-comments-license-scan}
DETECTION_CACHE=${DETECTION_CACHE:-var/code-comments-license-score-cache.sqlite}
DATASETS_LEGACY=${DATASETS_LEGACY:-redpajama-github,the-heap,the-stack}
cd "$ROOT_DIR"

echo "== Processes =="
pgrep -af 'run-code-comments-license-scan' || true
printf 'commentminer_scan_processes: '
(pgrep -f 'commentminer scan-hf-comment-licenses' || true) | wc -l
printf 'commentminer_cache_prewarm_processes: '
(pgrep -f 'commentminer prewarm-hf-license-cache' || true) | wc -l
printf 'scancode_processes: '
(pgrep -f '/scancode ' || true) | wc -l

echo
echo "== Checkpoints =="
STACK_V2_INPUT="$STACK_V2_INPUT" STACK_V2_OUTPUT="$STACK_V2_OUTPUT" \
  STACK_V2_DATASET="$STACK_V2_DATASET" LEGACY_INPUT="$LEGACY_INPUT" \
  LEGACY_OUTPUT="$LEGACY_OUTPUT" DATASETS_LEGACY="$DATASETS_LEGACY" \
  uv run python - <<'PY'
import json
import os
from collections import Counter
from pathlib import Path

checkpoints = [
    (
        os.environ["STACK_V2_DATASET"],
        Path(os.environ["STACK_V2_OUTPUT"]) / "license-scan-checkpoint.json",
        Path(os.environ["STACK_V2_INPUT"]),
        [os.environ["STACK_V2_DATASET"]],
    ),
    (
        "legacy",
        Path(os.environ["LEGACY_OUTPUT"]) / "license-scan-checkpoint.json",
        Path(os.environ["LEGACY_INPUT"]),
        [item for item in os.environ["DATASETS_LEGACY"].split(",") if item],
    ),
]
for name, path, input_root, datasets in checkpoints:
    print(name)
    expected_shards = sum(
        1
        for dataset in datasets
        for _ in (input_root / dataset).glob("*/*.parquet")
    )
    if not path.exists():
        print("  checkpoint: missing")
        print(f"  expected_shards: {expected_shards}")
        continue
    data = json.loads(path.read_text())
    stats = data.get("shard_stats", {})
    completed_shards = data.get("completed_shards", [])
    completed = len(completed_shards)
    records = sum(item.get("records_scanned", 0) for item in stats.values())
    detections = sum(item.get("records_with_detected_license", 0) for item in stats.values())
    print(f"  completed_shards: {completed}/{expected_shards}")
    by_dataset = Counter(str(item).split("/", 1)[0] for item in completed_shards)
    for dataset in datasets:
        print(f"  {dataset}: {by_dataset.get(dataset, 0)}")
    print(f"  records_scanned: {records}")
    print(f"  records_with_detected_license: {detections}")
    print(f"  updated_at: {data.get('updated_at')}")
PY

echo
echo "== Local Source Counts =="
find "$LEGACY_INPUT" -mindepth 2 -maxdepth 3 -type f -name 'part-*.parquet' -printf '%P\n' 2>/dev/null \
  | awk -F/ '{count[$1]++} END {for (k in count) print k, count[k]}' \
  | LC_ALL=C sort || true

echo
echo "== Detection Cache =="
cache_path="$DETECTION_CACHE"
if [[ ! -f "$cache_path" ]]; then
  echo "cache: missing"
else
  printf 'cache_path: %s\n' "$cache_path"
  du -h "$cache_path" | awk '{print "cache_size: " $1}'
  if [[ -f "${cache_path}-wal" ]]; then
    du -h "${cache_path}-wal" | awk '{print "cache_wal_size: " $1}'
  fi
  if [[ "${CACHE_COUNT:-0}" == "1" ]]; then
    if ! DETECTION_CACHE="$DETECTION_CACHE" timeout "${CACHE_COUNT_TIMEOUT:-30}" uv run python - <<'PY'; then
import os
from pathlib import Path
import sqlite3

cache = Path(os.environ["DETECTION_CACHE"])
with sqlite3.connect(cache) as connection:
    count = connection.execute("SELECT COUNT(*) FROM license_detection_cache").fetchone()[0]
print(f"cached_detections: {count}")
PY
      echo "cached_detections: count timed out after ${CACHE_COUNT_TIMEOUT:-30}s"
    fi
  else
    echo "cached_detections: skipped; set CACHE_COUNT=1 to run the expensive count"
  fi
fi

echo
echo "== Latest Output Shards =="
STACK_V2_OUTPUT="$STACK_V2_OUTPUT" LEGACY_OUTPUT="$LEGACY_OUTPUT" uv run python - <<'PY'
import os
from pathlib import Path
import pyarrow.parquet as pq

roots = [
    Path(os.environ["STACK_V2_OUTPUT"]),
    Path(os.environ["LEGACY_OUTPUT"]),
]
for root in roots:
    print(root)
    if not root.exists():
        print("  missing")
        continue
    paths = sorted(root.glob("*/*/part-*.parquet"), key=lambda path: path.stat().st_mtime, reverse=True)[:5]
    if not paths:
        print("  no parquet outputs yet")
        continue
    for path in paths:
        schema = pq.read_schema(path)
        columns = set(schema.names)
        has_detection = "comment_license_detection" in columns
        has_score = "comment_license_score" in columns
        null_scores = "n/a"
        if has_score:
            metadata = pq.read_metadata(path)
            column_index = schema.names.index("comment_license_score")
            null_count = 0
            for row_group_index in range(metadata.num_row_groups):
                statistics = metadata.row_group(row_group_index).column(column_index).statistics
                if statistics is None or statistics.null_count is None:
                    values = pq.read_table(
                        path,
                        columns=["comment_license_score"],
                        use_threads=False,
                    ).column("comment_license_score")
                    null_count = sum(1 for value in values.to_pylist() if value is None)
                    break
                null_count += statistics.null_count
            null_scores = str(null_count)
        print(
            f"  {path.relative_to(root)} rows={pq.read_metadata(path).num_rows} "
            f"has_detection={has_detection} has_score={has_score} null_scores={null_scores}"
        )
PY

echo
echo "== Sizes =="
du -sh \
  "$STACK_V2_INPUT" \
  "$STACK_V2_OUTPUT" \
  "$LEGACY_INPUT" \
  "$LEGACY_OUTPUT" \
  "$FINAL_OUTPUT" \
  "$DETECTION_CACHE" 2>/dev/null || true
df -h /home/jovyan/work | tail -1
