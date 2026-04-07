from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
import logging
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

from .pipeline import JsonlShardWriter


_LOGGER = logging.getLogger(__name__)
_DEFAULT_MAX_RECORDS_PER_SHARD = 100_000
_DEFAULT_MAX_BYTES_PER_SHARD = 128 * 1024 * 1024


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


@dataclass(slots=True)
class AggregateCommentRunStats:
    output_directory: Path
    dataset_name: str
    source_field: str
    input_directories: list[Path]
    source_datasets: list[str] = field(default_factory=list)
    records_aggregated: int = 0
    shards_written: int = 0


@dataclass(slots=True)
class _SourceRun:
    input_directory: Path
    dataset_name: str
    manifest: dict[str, Any] | None
    shards: list[Path]


def aggregate_comment_runs(
    input_directories: Sequence[Path],
    *,
    output_root: Path | None = None,
    dataset_name: str = "combined-comments",
    source_field: str = "source_dataset",
    max_records_per_shard: int = _DEFAULT_MAX_RECORDS_PER_SHARD,
    max_bytes_per_shard: int = _DEFAULT_MAX_BYTES_PER_SHARD,
    progress_every: int = 1000,
) -> AggregateCommentRunStats:
    if not input_directories:
        raise ValueError("At least one input run directory is required")
    if not dataset_name.strip():
        raise ValueError("dataset_name must not be empty")
    if not source_field.strip():
        raise ValueError("source_field must not be empty")
    if max_records_per_shard < 1:
        raise ValueError(f"max_records_per_shard must be >= 1, got {max_records_per_shard}")
    if max_bytes_per_shard < 1:
        raise ValueError(f"max_bytes_per_shard must be >= 1, got {max_bytes_per_shard}")
    if progress_every < 1:
        raise ValueError(f"progress_every must be >= 1, got {progress_every}")

    source_runs = [_load_source_run(path) for path in input_directories]
    output_root = (output_root or _default_output_root(source_runs)).resolve()
    _validate_output_root(output_root, source_runs)

    writer = JsonlShardWriter(
        output_root,
        dataset_name,
        max_records_per_shard=max_records_per_shard,
        max_bytes_per_shard=max_bytes_per_shard,
    )
    stats = AggregateCommentRunStats(
        output_directory=writer.dataset_directory,
        dataset_name=dataset_name,
        source_field=source_field,
        input_directories=[item.input_directory for item in source_runs],
        source_datasets=sorted({item.dataset_name for item in source_runs}),
    )
    source_records: list[dict[str, Any]] = []

    _LOGGER.info(
        "Starting comment aggregation dataset_name=%s source_runs=%s output_directory=%s",
        dataset_name,
        len(source_runs),
        writer.dataset_directory,
    )

    try:
        for source_run in source_runs:
            records_from_run = 0
            _LOGGER.info(
                "Aggregating run input_directory=%s source_dataset=%s shards=%s",
                source_run.input_directory,
                source_run.dataset_name,
                len(source_run.shards),
            )
            for payload in _iter_run_payloads(source_run.shards):
                payload[source_field] = source_run.dataset_name
                payload["dataset"] = dataset_name
                writer.write_json_line(json.dumps(payload, ensure_ascii=False) + "\n")
                stats.records_aggregated += 1
                records_from_run += 1
                if stats.records_aggregated % progress_every == 0:
                    _LOGGER.info(
                        "Aggregation progress dataset_name=%s records_aggregated=%s current_source_dataset=%s",
                        dataset_name,
                        stats.records_aggregated,
                        source_run.dataset_name,
                    )

            source_records.append(
                {
                    "input_directory": str(source_run.input_directory),
                    "source_dataset": source_run.dataset_name,
                    "records_aggregated": records_from_run,
                    "shards": [path.name for path in source_run.shards],
                    "source_manifest": source_run.manifest,
                }
            )
    finally:
        writer.close()

    stats.shards_written = len(writer.shard_paths)
    _write_aggregate_manifest(writer, stats, source_records)
    _LOGGER.info(
        "Finished comment aggregation dataset_name=%s records_aggregated=%s shards_written=%s",
        dataset_name,
        stats.records_aggregated,
        stats.shards_written,
    )
    return stats


def _load_source_run(path: Path) -> _SourceRun:
    input_directory = path.resolve()
    if not input_directory.exists() or not input_directory.is_dir():
        raise ValueError(f"Input run directory does not exist: {input_directory}")

    shards = sorted(input_directory.glob("part-*.jsonl"))
    if not shards:
        raise ValueError(f"No input shard files found in: {input_directory}")

    manifest_path = input_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    dataset_name = _resolve_source_dataset(input_directory, manifest, shards)
    return _SourceRun(
        input_directory=input_directory,
        dataset_name=dataset_name,
        manifest=manifest,
        shards=shards,
    )


def _resolve_source_dataset(
    input_directory: Path,
    manifest: dict[str, Any] | None,
    shards: Sequence[Path],
) -> str:
    if manifest and manifest.get("dataset"):
        return str(manifest["dataset"])
    for shard in shards:
        with shard.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                dataset_name = payload.get("dataset")
                if dataset_name:
                    return str(dataset_name)
                break
    return input_directory.parent.name


def _default_output_root(source_runs: Sequence[_SourceRun]) -> Path:
    base_candidates = [str(item.input_directory.parent.parent) for item in source_runs]
    return Path(os.path.commonpath(base_candidates)).resolve()


def _validate_output_root(output_root: Path, source_runs: Sequence[_SourceRun]) -> None:
    for source_run in source_runs:
        if output_root == source_run.input_directory or output_root.is_relative_to(source_run.input_directory):
            raise ValueError(
                f"Output root {output_root} must not be inside an input run directory: {source_run.input_directory}"
            )


def _iter_run_payloads(shards: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for shard in shards:
        with shard.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                if "opening_comment" not in payload:
                    raise ValueError(f"Shard {shard} does not look like an extracted comment run")
                yield payload


def _write_aggregate_manifest(
    writer: JsonlShardWriter,
    stats: AggregateCommentRunStats,
    source_records: list[dict[str, Any]],
) -> None:
    manifest = {
        "dataset": stats.dataset_name,
        "run_id": writer.run_id,
        "created_at": _utc_now(),
        "source_field": stats.source_field,
        "records_aggregated": stats.records_aggregated,
        "shards_written": stats.shards_written,
        "source_datasets": stats.source_datasets,
        "input_directories": [str(path) for path in stats.input_directories],
        "shards": [path.name for path in writer.shard_paths],
        "source_runs": source_records,
    }
    (writer.dataset_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
