from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterable, Sequence

from .pipeline import JsonlShardWriter


_LOGGER = logging.getLogger(__name__)
_DEFAULT_MAX_RECORDS_PER_SHARD = 100_000
_DEFAULT_MAX_BYTES_PER_SHARD = 128 * 1024 * 1024
_DEFAULT_HASH_WORKERS = max(1, min(8, os.cpu_count() or 1))
_DEFAULT_SORT_PARALLELISM = max(1, min(8, os.cpu_count() or 1))


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


@dataclass(slots=True)
class DeduplicateCommentRunStats:
    input_directory: Path
    output_directory: Path
    input_dataset_name: str
    dataset_name: str
    source_field: str
    records_seen: int = 0
    unique_comments: int = 0
    duplicate_occurrences: int = 0
    shards_written: int = 0


@dataclass(slots=True)
class _HashTask:
    ordinal: int
    payload: dict[str, Any]


@dataclass(slots=True)
class _HashResult:
    digest: str
    ordinal: int
    payload_json: str


def deduplicate_comment_run(
    input_directory: Path,
    *,
    output_root: Path | None = None,
    dataset_name: str | None = None,
    source_field: str = "source_dataset",
    hash_workers: int = _DEFAULT_HASH_WORKERS,
    hash_batch_size: int = 1000,
    sort_parallelism: int = _DEFAULT_SORT_PARALLELISM,
    sort_command: str = "sort",
    max_records_per_shard: int = _DEFAULT_MAX_RECORDS_PER_SHARD,
    max_bytes_per_shard: int = _DEFAULT_MAX_BYTES_PER_SHARD,
    progress_every: int = 1000,
) -> DeduplicateCommentRunStats:
    if not source_field.strip():
        raise ValueError("source_field must not be empty")
    if hash_workers < 1:
        raise ValueError(f"hash_workers must be >= 1, got {hash_workers}")
    if hash_batch_size < 1:
        raise ValueError(f"hash_batch_size must be >= 1, got {hash_batch_size}")
    if sort_parallelism < 1:
        raise ValueError(f"sort_parallelism must be >= 1, got {sort_parallelism}")
    if max_records_per_shard < 1:
        raise ValueError(f"max_records_per_shard must be >= 1, got {max_records_per_shard}")
    if max_bytes_per_shard < 1:
        raise ValueError(f"max_bytes_per_shard must be >= 1, got {max_bytes_per_shard}")
    if progress_every < 1:
        raise ValueError(f"progress_every must be >= 1, got {progress_every}")

    input_directory = input_directory.resolve()
    if not input_directory.exists() or not input_directory.is_dir():
        raise ValueError(f"Input run directory does not exist: {input_directory}")

    input_shards = sorted(input_directory.glob("part-*.jsonl"))
    if not input_shards:
        raise ValueError(f"No input shard files found in: {input_directory}")

    manifest_path = input_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    input_dataset_name = _resolve_input_dataset_name(input_directory, manifest, input_shards)
    resolved_dataset_name = dataset_name or f"{input_dataset_name}-deduplicated"
    if not resolved_dataset_name.strip():
        raise ValueError("dataset_name must not be empty")

    output_root = (output_root or input_directory.parent.parent).resolve()
    if output_root == input_directory or output_root.is_relative_to(input_directory):
        raise ValueError(f"Output root {output_root} must not be inside the input run directory")

    writer = JsonlShardWriter(
        output_root,
        resolved_dataset_name,
        max_records_per_shard=max_records_per_shard,
        max_bytes_per_shard=max_bytes_per_shard,
    )
    stats = DeduplicateCommentRunStats(
        input_directory=input_directory,
        output_directory=writer.dataset_directory,
        input_dataset_name=input_dataset_name,
        dataset_name=resolved_dataset_name,
        source_field=source_field,
    )
    temp_root = writer.dataset_directory / ".tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    unsorted_hashes_path = temp_root / "comment-hashes.tsv"
    sorted_hashes_path = temp_root / "comment-hashes.sorted.tsv"

    _LOGGER.info(
        "Starting comment deduplication input_directory=%s dataset_name=%s output_directory=%s hash_workers=%s sort_parallelism=%s",
        input_directory,
        resolved_dataset_name,
        writer.dataset_directory,
        hash_workers,
        sort_parallelism,
    )

    try:
        _write_unsorted_hashes(
            input_shards,
            unsorted_hashes_path,
            stats=stats,
            hash_workers=hash_workers,
            hash_batch_size=hash_batch_size,
            progress_every=progress_every,
        )
        _sort_hashed_comments(
            unsorted_hashes_path,
            sorted_hashes_path,
            sort_command=sort_command,
            sort_parallelism=sort_parallelism,
            temp_root=temp_root,
        )
        _write_deduplicated_output(
            sorted_hashes_path,
            writer,
            stats=stats,
            source_field=source_field,
            progress_every=progress_every,
        )
    except Exception:
        writer.close()
        if writer.dataset_directory.exists():
            shutil.rmtree(writer.dataset_directory)
        dataset_root = writer.dataset_directory.parent
        if dataset_root.exists() and not any(dataset_root.iterdir()):
            dataset_root.rmdir()
        raise
    finally:
        writer.close()
        if temp_root.exists():
            shutil.rmtree(temp_root)

    stats.shards_written = len(writer.shard_paths)
    _write_dedup_manifest(
        writer,
        stats,
        input_shards=input_shards,
        input_manifest=manifest,
        hash_workers=hash_workers,
        hash_batch_size=hash_batch_size,
        sort_parallelism=sort_parallelism,
        sort_command=sort_command,
    )
    _LOGGER.info(
        "Finished comment deduplication dataset_name=%s records_seen=%s unique_comments=%s duplicate_occurrences=%s shards_written=%s",
        resolved_dataset_name,
        stats.records_seen,
        stats.unique_comments,
        stats.duplicate_occurrences,
        stats.shards_written,
    )
    return stats


def _resolve_input_dataset_name(
    input_directory: Path,
    manifest: dict[str, Any] | None,
    input_shards: Sequence[Path],
) -> str:
    if manifest and manifest.get("dataset"):
        return str(manifest["dataset"])
    for shard in input_shards:
        for payload in _iter_run_payloads([shard]):
            dataset_name = payload.get("dataset")
            if dataset_name:
                return str(dataset_name)
            break
    return input_directory.parent.name


def _write_unsorted_hashes(
    input_shards: Sequence[Path],
    output_path: Path,
    *,
    stats: DeduplicateCommentRunStats,
    hash_workers: int,
    hash_batch_size: int,
    progress_every: int,
) -> None:
    tasks: list[_HashTask] = []
    ordinal = 0

    with output_path.open("w", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=hash_workers) as executor:
        for payload in _iter_run_payloads(input_shards):
            tasks.append(_HashTask(ordinal=ordinal, payload=payload))
            ordinal += 1
            if len(tasks) >= hash_batch_size:
                _flush_hash_batch(tasks, handle, executor, stats=stats, progress_every=progress_every)
                tasks = []
        if tasks:
            _flush_hash_batch(tasks, handle, executor, stats=stats, progress_every=progress_every)


def _flush_hash_batch(
    tasks: list[_HashTask],
    handle,
    executor: ThreadPoolExecutor,
    *,
    stats: DeduplicateCommentRunStats,
    progress_every: int,
) -> None:
    for result in executor.map(_hash_comment_payload, tasks):
        handle.write(f"{result.digest}\t{result.ordinal:020d}\t{result.payload_json}\n")
        stats.records_seen += 1
        if stats.records_seen % progress_every == 0:
            _LOGGER.info(
                "Deduplication hashing progress dataset_name=%s records_seen=%s",
                stats.dataset_name,
                stats.records_seen,
            )


def _hash_comment_payload(task: _HashTask) -> _HashResult:
    payload_json = json.dumps(task.payload, ensure_ascii=False)
    normalized_comment = "".join(character for character in str(task.payload.get("opening_comment", "")) if character.isalnum())
    digest = hashlib.sha256(normalized_comment.encode("utf-8")).hexdigest()
    return _HashResult(digest=digest, ordinal=task.ordinal, payload_json=payload_json)


def _sort_hashed_comments(
    input_path: Path,
    output_path: Path,
    *,
    sort_command: str,
    sort_parallelism: int,
    temp_root: Path,
) -> None:
    command = [
        sort_command,
        "-t",
        "\t",
        "-k1,1",
        "-k2,2n",
        f"--parallel={sort_parallelism}",
        f"--temporary-directory={temp_root}",
        "-o",
        str(output_path),
        str(input_path),
    ]
    env = os.environ.copy()
    env.setdefault("LC_ALL", "C")
    subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )


def _write_deduplicated_output(
    sorted_hashes_path: Path,
    writer: JsonlShardWriter,
    *,
    stats: DeduplicateCommentRunStats,
    source_field: str,
    progress_every: int,
) -> None:
    current_hash: str | None = None
    representative: dict[str, Any] | None = None
    occurrences: list[dict[str, Any]] = []
    source_datasets: set[str] = set()

    with sorted_hashes_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            digest, _ordinal, payload_json = line.rstrip("\n").split("\t", 2)
            payload = json.loads(payload_json)

            if current_hash is None:
                current_hash = digest
            elif digest != current_hash:
                _write_dedup_group(
                    writer,
                    stats=stats,
                    digest=current_hash,
                    representative=representative,
                    occurrences=occurrences,
                    source_datasets=source_datasets,
                )
                if stats.unique_comments % progress_every == 0:
                    _LOGGER.info(
                        "Deduplication grouping progress dataset_name=%s unique_comments=%s duplicate_occurrences=%s",
                        stats.dataset_name,
                        stats.unique_comments,
                        stats.duplicate_occurrences,
                    )
                current_hash = digest
                representative = None
                occurrences = []
                source_datasets = set()

            if representative is None:
                representative = payload
            occurrences.append(_build_occurrence_metadata(payload))
            source_value = payload.get(source_field) or payload.get("dataset")
            if source_value is not None:
                source_datasets.add(str(source_value))

    if current_hash is not None:
        _write_dedup_group(
            writer,
            stats=stats,
            digest=current_hash,
            representative=representative,
            occurrences=occurrences,
            source_datasets=source_datasets,
        )


def _write_dedup_group(
    writer: JsonlShardWriter,
    *,
    stats: DeduplicateCommentRunStats,
    digest: str,
    representative: dict[str, Any] | None,
    occurrences: list[dict[str, Any]],
    source_datasets: set[str],
) -> None:
    if representative is None:
        return
    payload = {
        "dataset": stats.dataset_name,
        "record_id": digest,
        "opening_comment": representative.get("opening_comment", ""),
        "normalized_comment_hash": digest,
        "occurrence_count": len(occurrences),
        "source_datasets": sorted(source_datasets),
        "occurrences": occurrences,
        "deduplicated_at": _utc_now(),
    }
    writer.write_json_line(json.dumps(payload, ensure_ascii=False) + "\n")
    stats.unique_comments += 1
    stats.duplicate_occurrences += max(0, len(occurrences) - 1)


def _build_occurrence_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "opening_comment"}


def _iter_run_payloads(shards: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for shard in shards:
        with shard.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                if "opening_comment" not in payload:
                    raise ValueError(f"Shard {shard} does not look like an extracted or aggregated comment run")
                yield payload


def _write_dedup_manifest(
    writer: JsonlShardWriter,
    stats: DeduplicateCommentRunStats,
    *,
    input_shards: Sequence[Path],
    input_manifest: dict[str, Any] | None,
    hash_workers: int,
    hash_batch_size: int,
    sort_parallelism: int,
    sort_command: str,
) -> None:
    manifest = {
        "dataset": stats.dataset_name,
        "run_id": writer.run_id,
        "created_at": _utc_now(),
        "input_directory": str(stats.input_directory),
        "input_dataset": stats.input_dataset_name,
        "source_field": stats.source_field,
        "records_seen": stats.records_seen,
        "unique_comments": stats.unique_comments,
        "duplicate_occurrences": stats.duplicate_occurrences,
        "shards_written": stats.shards_written,
        "hash_algorithm": "sha256",
        "normalization": "remove all whitespace and non-alphanumeric characters",
        "hash_workers": hash_workers,
        "hash_batch_size": hash_batch_size,
        "sort_command": sort_command,
        "sort_parallelism": sort_parallelism,
        "input_shards": [path.name for path in input_shards],
        "output_shards": [path.name for path in writer.shard_paths],
        "input_manifest": input_manifest,
    }
    (writer.dataset_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
