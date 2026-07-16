#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/home/jovyan/work}
REPO_ID=${REPO_ID:-Jkatzy/code-comments}
FINAL_OUTPUT=${FINAL_OUTPUT:-var/code-comments-license-scan}
DATASETS=${DATASETS:-redpajama-github,the-heap,the-stack,the-stack-v2-dedup}

cd "$ROOT_DIR"

REPO_ID="$REPO_ID" FINAL_OUTPUT="$FINAL_OUTPUT" DATASETS="$DATASETS" uv run python - <<'PY'
import json
import math
import os
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq

repo_id = os.environ["REPO_ID"]
final_root = Path(os.environ["FINAL_OUTPUT"])
datasets = [item for item in os.environ["DATASETS"].split(",") if item]
if not final_root.exists():
    raise SystemExit(f"Final output directory does not exist: {final_root}")

local_files = sorted(
    path.relative_to(final_root).as_posix()
    for path in final_root.glob("*/*/part-*.parquet")
)
if not local_files:
    raise SystemExit(f"No local parquet files found under {final_root}")

remote_files = set(HfApi().list_repo_files(repo_id=repo_id, repo_type="dataset"))
missing = [path for path in local_files if path not in remote_files]
extra_remote_parquet = sorted(
    path for path in remote_files
    if path.endswith(".parquet") and path not in set(local_files)
)
print(
    f"remote_parquet_files={sum(1 for path in remote_files if path.endswith('.parquet'))} "
    f"local_parquet_files={len(local_files)}"
)
if missing:
    raise SystemExit("Uploaded dataset is missing files: " + "; ".join(missing[:20]))
if extra_remote_parquet:
    raise SystemExit(
        "Uploaded dataset has stale extra Parquet files: "
        + "; ".join(extra_remote_parquet[:20])
    )

for dataset in datasets:
    sample = next((path for path in local_files if path.startswith(f"{dataset}/")), None)
    if sample is None:
        raise SystemExit(f"No local sample found for {dataset}")
    downloaded = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=sample,
    )
    schema = pq.read_schema(downloaded)
    missing_columns = [
        column for column in ["comment_license_detection", "comment_license_score"]
        if column not in schema.names
    ]
    if missing_columns:
        raise SystemExit(f"Remote sample {sample} missing columns: {missing_columns}")

    metadata = pq.read_metadata(downloaded)
    column_index = schema.names.index("comment_license_score")
    null_scores = 0
    for row_group_index in range(metadata.num_row_groups):
        statistics = metadata.row_group(row_group_index).column(column_index).statistics
        if statistics is None or statistics.null_count is None:
            values = pq.read_table(
                downloaded,
                columns=["comment_license_score"],
                use_threads=False,
            ).column("comment_license_score")
            null_scores = sum(1 for value in values.to_pylist() if value is None)
            break
        null_scores += statistics.null_count
    if null_scores:
        raise SystemExit(f"Remote sample {sample} has {null_scores} null comment_license_score values")
    sample_table = pq.read_table(
        downloaded,
        columns=["comment_license_detection", "comment_license_score"],
        use_threads=False,
    )
    for detection_json, score_value in zip(
        sample_table.column("comment_license_detection").to_pylist(),
        sample_table.column("comment_license_score").to_pylist(),
        strict=True,
    ):
        score = float(score_value)
        if not math.isfinite(score) or not 0.0 <= score <= 100.0:
            raise SystemExit(f"Remote sample {sample} has invalid score {score!r}")
        detection = json.loads(detection_json)
        best_score = float(detection.get("best_license_score", 0.0))
        if not math.isclose(score, best_score, rel_tol=0.0, abs_tol=1e-9):
            raise SystemExit(
                f"Remote sample {sample} score={score} best_license_score={best_score}"
            )
    print(f"{dataset}: verified remote sample {sample} rows={sample_table.num_rows}")
PY
