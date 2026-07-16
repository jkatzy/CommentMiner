from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from .models import _json_safe


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _path_segment(value: str | None) -> str:
    if value is None:
        return "unknown"
    cleaned = value.strip()
    if not cleaned:
        return "unknown"
    return cleaned.replace("/", "%2F")


def _split_name(value: str | None) -> str:
    if value is None:
        return "unknown"
    cleaned = value.strip()
    if not cleaned:
        return "unknown"
    cleaned = cleaned.replace("#", "_sharp")
    cleaned = cleaned.replace("++", "_plus_plus")
    cleaned = cleaned.replace("+", "_plus")
    split = re.sub(r"[^A-Za-z0-9]+", "_", cleaned).strip("_")
    return split or "unknown"


@dataclass(slots=True)
class ExportGroupStats:
    dataset: str
    language: str
    records: int = 0
    shards: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ExportStats:
    output_directory: Path
    records_written: int = 0
    records_skipped_duplicate: int = 0
    groups: dict[tuple[str, str], ExportGroupStats] = field(default_factory=dict)


class _GroupShardWriter:
    def __init__(
        self,
        root: Path,
        dataset: str,
        language: str,
        *,
        output_format: str,
        max_records_per_shard: int,
        max_bytes_per_shard: int,
        shard_prefix: str = "part",
    ) -> None:
        self.root = root
        self.dataset = dataset
        self.language = language
        self.directory = root / _path_segment(dataset) / _path_segment(language)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.output_format = output_format
        self.max_records_per_shard = max_records_per_shard
        self.max_bytes_per_shard = max_bytes_per_shard
        self.shard_prefix = shard_prefix
        self._handle = None
        self._buffered_records: list[dict[str, Any]] = []
        self._current_bytes = 0
        self._current_records = 0
        self._next_shard_index = self._discover_next_shard_index()
        self.extension = "parquet" if output_format == "parquet" else "jsonl"
        self.shard_paths: list[Path] = sorted(self.directory.glob(f"part-*.{self.extension}"))

    def _discover_next_shard_index(self) -> int:
        next_index = 0
        for path in list(self.directory.glob(f"{self.shard_prefix}-*.jsonl")) + list(
            self.directory.glob(f"{self.shard_prefix}-*.parquet")
        ):
            try:
                next_index = max(next_index, int(path.stem.split("-")[-1]) + 1)
            except ValueError:
                continue
        return next_index

    def write(self, record: dict[str, Any]) -> Path:
        safe_record = _normalize_export_record(record)
        encoded = (json.dumps(safe_record, ensure_ascii=False) + "\n").encode("utf-8")
        if self._handle is None:
            self._open_next_shard()
        elif (
            self._current_records >= self.max_records_per_shard
            or self._current_bytes + len(encoded) > self.max_bytes_per_shard
        ):
            self._open_next_shard()

        if self.output_format == "jsonl":
            assert self._handle is not None
            self._handle.write(encoded.decode("utf-8"))
        else:
            self._buffered_records.append(safe_record)
        self._current_bytes += len(encoded)
        self._current_records += 1
        return self.shard_paths[-1]

    def close(self) -> None:
        if self.output_format == "parquet" and self._buffered_records:
            assert self.shard_paths
            _write_parquet_records(self.shard_paths[-1], self._buffered_records)
            self._buffered_records = []
        if self.output_format == "jsonl" and self._handle is not None:
            self._handle.close()
        self._handle = None

    def _open_next_shard(self) -> None:
        self.close()
        path = self.directory / f"{self.shard_prefix}-{self._next_shard_index:05d}.{self.extension}"
        self._next_shard_index += 1
        if self.output_format == "jsonl":
            self._handle = path.open("a", encoding="utf-8")
        else:
            self._handle = object()
        self._current_bytes = 0
        self._current_records = 0
        if path not in self.shard_paths:
            self.shard_paths.append(path)


class _WriterCache:
    def __init__(
        self,
        root: Path,
        *,
        output_format: str,
        max_open_writers: int,
        max_records_per_shard: int,
        max_bytes_per_shard: int,
        shard_prefix: str = "part",
    ) -> None:
        self.root = root
        self.output_format = output_format
        self.max_open_writers = max_open_writers
        self.max_records_per_shard = max_records_per_shard
        self.max_bytes_per_shard = max_bytes_per_shard
        self.shard_prefix = shard_prefix
        self._writers: OrderedDict[tuple[str, str], _GroupShardWriter] = OrderedDict()

    def writer_for(self, dataset: str, language: str) -> _GroupShardWriter:
        key = (dataset, language)
        writer = self._writers.get(key)
        if writer is not None:
            self._writers.move_to_end(key)
            return writer

        while len(self._writers) >= self.max_open_writers:
            _, stale = self._writers.popitem(last=False)
            stale.close()

        writer = _GroupShardWriter(
            self.root,
            dataset,
            language,
            output_format=self.output_format,
            max_records_per_shard=self.max_records_per_shard,
            max_bytes_per_shard=self.max_bytes_per_shard,
            shard_prefix=self.shard_prefix,
        )
        self._writers[key] = writer
        return writer

    def close(self) -> None:
        for writer in self._writers.values():
            writer.close()
        self._writers.clear()


def _iter_jsonl_records(paths: list[Path]) -> Iterator[dict[str, Any]]:
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def iter_mined_comment_record_groups(output_directory: Path) -> Iterator[tuple[Path, Iterator[dict[str, Any]]]]:
    for group_directory in _mined_comment_group_directories(output_directory):
        paths = sorted(group_directory.glob("*/part-*.jsonl"))
        yield group_directory, _iter_jsonl_records(paths)


def _mined_comment_group_directories(output_directory: Path) -> list[Path]:
    groups: list[Path] = []
    for group_directory in sorted(path for path in output_directory.iterdir() if path.is_dir()):
        paths = sorted(group_directory.glob("*/part-*.jsonl"))
        if paths:
            groups.append(group_directory)
    return groups


def iter_mined_comment_records(output_directory: Path) -> Iterator[dict[str, Any]]:
    for _, records in iter_mined_comment_record_groups(output_directory):
        yield from records


def export_huggingface_dataset(
    input_directory: Path,
    output_directory: Path,
    *,
    output_format: str = "parquet",
    max_records_per_shard: int,
    max_bytes_per_shard: int,
    dedupe_record_ids: bool = False,
    dedupe_scope: str = "global",
    dataset_card_layout: str = "dataset-language-configs",
    max_open_writers: int = 64,
    workers: int = 1,
) -> ExportStats:
    if output_format not in {"parquet", "jsonl"}:
        raise ValueError(f"Unsupported export format: {output_format}")
    if dedupe_scope not in {"global", "input-group"}:
        raise ValueError(f"Unsupported dedupe scope: {dedupe_scope}")
    if dataset_card_layout not in {"dataset-language-configs", "language-splits"}:
        raise ValueError(f"Unsupported dataset card layout: {dataset_card_layout}")
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    if workers > 1 and dedupe_record_ids and dedupe_scope == "global":
        raise ValueError("parallel export does not support global record-id dedupe")
    output_directory.mkdir(parents=True, exist_ok=True)
    group_directories = _mined_comment_group_directories(input_directory)
    if workers > 1 and group_directories:
        stats = _export_huggingface_dataset_parallel(
            group_directories,
            output_directory,
            output_format=output_format,
            max_records_per_shard=max_records_per_shard,
            max_bytes_per_shard=max_bytes_per_shard,
            dedupe_record_ids=dedupe_record_ids,
            dedupe_scope=dedupe_scope,
            max_open_writers=max_open_writers,
            workers=workers,
        )
    else:
        stats = _export_mined_comment_groups(
            group_directories,
            output_directory,
            output_format=output_format,
            max_records_per_shard=max_records_per_shard,
            max_bytes_per_shard=max_bytes_per_shard,
            dedupe_record_ids=dedupe_record_ids,
            dedupe_scope=dedupe_scope,
            max_open_writers=max_open_writers,
        )

    _write_export_manifest(
        stats,
        output_format=output_format,
        dedupe_scope=dedupe_scope if dedupe_record_ids else None,
        dataset_card_layout=dataset_card_layout,
        workers=workers,
    )
    _write_dataset_card(
        stats,
        output_format=output_format,
        layout=dataset_card_layout,
    )
    return stats


def _export_mined_comment_groups(
    group_directories: Iterable[Path],
    output_directory: Path,
    *,
    output_format: str,
    max_records_per_shard: int,
    max_bytes_per_shard: int,
    dedupe_record_ids: bool,
    dedupe_scope: str,
    max_open_writers: int,
    shard_prefix: str = "part",
) -> ExportStats:
    writers = _WriterCache(
        output_directory,
        output_format=output_format,
        max_open_writers=max_open_writers,
        max_records_per_shard=max_records_per_shard,
        max_bytes_per_shard=max_bytes_per_shard,
        shard_prefix=shard_prefix,
    )
    stats = ExportStats(output_directory=output_directory)
    global_seen_record_ids: set[tuple[str, str]] = set()

    try:
        for group_directory in group_directories:
            records = _iter_jsonl_records(sorted(group_directory.glob("*/part-*.jsonl")))
            group_seen_record_ids: set[tuple[str, str]] = set()
            seen_record_ids = (
                global_seen_record_ids
                if dedupe_scope == "global"
                else group_seen_record_ids
            )
            for record in records:
                dataset = str(record.get("dataset") or "unknown")
                language = str(record.get("language") or "unknown")
                record_id = str(record.get("record_id") or "")
                dedupe_key = (dataset, record_id)
                if dedupe_record_ids and dedupe_key in seen_record_ids:
                    stats.records_skipped_duplicate += 1
                    continue
                if dedupe_record_ids:
                    seen_record_ids.add(dedupe_key)

                record = _normalize_export_record(record)
                writer = writers.writer_for(dataset, language)
                shard_path = writer.write(record)

                group_key = (dataset, language)
                group = stats.groups.get(group_key)
                if group is None:
                    group = ExportGroupStats(dataset=dataset, language=language)
                    stats.groups[group_key] = group
                group.records += 1
                relative_shard = str(shard_path.relative_to(output_directory))
                if relative_shard not in group.shards:
                    group.shards.append(relative_shard)
                stats.records_written += 1
    finally:
        writers.close()
    return stats


def _export_huggingface_dataset_parallel(
    group_directories: list[Path],
    output_directory: Path,
    *,
    output_format: str,
    max_records_per_shard: int,
    max_bytes_per_shard: int,
    dedupe_record_ids: bool,
    dedupe_scope: str,
    max_open_writers: int,
    workers: int,
) -> ExportStats:
    worker_count = min(workers, len(group_directories))
    chunks = _contiguous_chunks(group_directories, worker_count)
    stats = ExportStats(output_directory=output_directory)
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                _export_mined_comment_groups,
                chunk,
                output_directory,
                output_format=output_format,
                max_records_per_shard=max_records_per_shard,
                max_bytes_per_shard=max_bytes_per_shard,
                dedupe_record_ids=dedupe_record_ids,
                dedupe_scope=dedupe_scope,
                max_open_writers=max_open_writers,
                shard_prefix=f"part-{worker_index:05d}",
            )
            for worker_index, chunk in enumerate(chunks)
            if chunk
        ]
        for future in as_completed(futures):
            _merge_export_stats(stats, future.result())
    return stats


def _contiguous_chunks(items: list[Path], workers: int) -> list[list[Path]]:
    chunk_size = max(1, math.ceil(len(items) / workers))
    return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]


def _merge_export_stats(target: ExportStats, source: ExportStats) -> None:
    target.records_written += source.records_written
    target.records_skipped_duplicate += source.records_skipped_duplicate
    for key, source_group in source.groups.items():
        group = target.groups.get(key)
        if group is None:
            group = ExportGroupStats(
                dataset=source_group.dataset,
                language=source_group.language,
            )
            target.groups[key] = group
        group.records += source_group.records
        seen_shards = set(group.shards)
        for shard in source_group.shards:
            if shard not in seen_shards:
                group.shards.append(shard)
                seen_shards.add(shard)


def _write_export_manifest(
    stats: ExportStats,
    *,
    output_format: str,
    dedupe_scope: str | None,
    dataset_card_layout: str,
    workers: int,
) -> None:
    payload = {
        "created_at": _utc_now(),
        "records_written": stats.records_written,
        "records_skipped_duplicate": stats.records_skipped_duplicate,
        "dedupe_scope": dedupe_scope,
        "format": output_format,
        "layout": f"<dataset>/<language>/part-*.{output_format}",
        "dataset_card_layout": dataset_card_layout,
        "workers": workers,
        "groups": [
            {
                "dataset": group.dataset,
                "language": group.language,
                "records": group.records,
                "shards": sorted(group.shards),
            }
            for group in sorted(stats.groups.values(), key=lambda item: (item.dataset, item.language))
        ],
    }
    (stats.output_directory / "manifest.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def _write_dataset_card(
    stats: ExportStats,
    *,
    output_format: str,
    layout: str,
) -> None:
    lines = [
        "---",
        "configs:",
    ]
    groups = sorted(stats.groups.values(), key=lambda item: (item.dataset, item.language))
    if layout == "dataset-language-configs":
        for group in groups:
            config_name = f"{group.dataset}__{group.language}".replace("/", "-")
            lines.extend(
                [
                    f"- config_name: {json.dumps(config_name)}",
                    "  data_files:",
                    "  - split: train",
                    f"    path: {json.dumps(sorted(group.shards))}",
                ]
            )
    elif layout == "language-splits":
        groups_by_dataset: OrderedDict[str, list[ExportGroupStats]] = OrderedDict()
        for group in groups:
            groups_by_dataset.setdefault(group.dataset, []).append(group)
        for dataset, dataset_groups in groups_by_dataset.items():
            used_splits: set[str] = set()
            lines.extend(
                [
                    f"- config_name: {json.dumps(dataset)}",
                    "  data_files:",
                ]
            )
            for group in dataset_groups:
                split_name = _split_name(group.language)
                if split_name in used_splits:
                    suffix = 2
                    while f"{split_name}_{suffix}" in used_splits:
                        suffix += 1
                    split_name = f"{split_name}_{suffix}"
                used_splits.add(split_name)
                shard_pattern = (
                    f"{_path_segment(group.dataset)}/"
                    f"{_path_segment(group.language)}/"
                    f"part-*.{output_format}"
                )
                lines.extend(
                    [
                        f"  - split: {json.dumps(split_name)}",
                        f"    path: {json.dumps(shard_pattern)}",
                    ]
                )
    else:
        raise ValueError(f"Unsupported dataset card layout: {layout}")
    lines.extend(
        [
            "---",
            "",
            "# Comment Dataset",
            "",
            "Opening comments extracted from code datasets with CommentMiner and ML4SE-toolkit.",
            "",
            f"Files are grouped as `<dataset>/<language>/part-*.{output_format}`.",
            "",
            (
                "The Hugging Face dataset card declares one config per source dataset "
                "and one split-safe language name per language."
                if layout == "language-splits"
                else (
                    "The Hugging Face dataset card declares one config per "
                    "source dataset and language."
                )
            ),
            "",
            "Each row contains `dataset`, `record_id`, `opening_comment`, `language`, `path`, `repo`, `extracted_at`, and `metadata`.",
            "",
            "For Parquet exports, `metadata` is stored as a JSON string so every source dataset shares one stable schema.",
            "",
        ]
    )
    (stats.output_directory / "README.md").write_text("\n".join(lines), encoding="utf-8")


_EXPORT_SCHEMA = pa.schema(
    [
        ("dataset", pa.string()),
        ("record_id", pa.string()),
        ("opening_comment", pa.string()),
        ("language", pa.string()),
        ("path", pa.string()),
        ("repo", pa.string()),
        ("extracted_at", pa.string()),
        ("metadata", pa.string()),
    ]
)


def _normalize_export_record(record: dict[str, Any]) -> dict[str, Any]:
    safe_record = _json_safe(record)
    metadata = safe_record.get("metadata")
    if not isinstance(metadata, str):
        metadata = json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True)
    return {
        "dataset": _string_or_none(safe_record.get("dataset")),
        "record_id": _string_or_none(safe_record.get("record_id")),
        "opening_comment": _string_or_none(safe_record.get("opening_comment")),
        "language": _string_or_none(safe_record.get("language")),
        "path": _string_or_none(safe_record.get("path")),
        "repo": _string_or_none(safe_record.get("repo")),
        "extracted_at": _string_or_none(safe_record.get("extracted_at")),
        "metadata": metadata,
    }


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _write_parquet_records(path: Path, records: list[dict[str, Any]]) -> None:
    table = pa.Table.from_pylist(records, schema=_EXPORT_SCHEMA)
    pq.write_table(table, path)
