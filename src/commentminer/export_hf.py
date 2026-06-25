from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any, Iterator

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
    ) -> None:
        self.root = root
        self.dataset = dataset
        self.language = language
        self.directory = root / _path_segment(dataset) / _path_segment(language)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.output_format = output_format
        self.max_records_per_shard = max_records_per_shard
        self.max_bytes_per_shard = max_bytes_per_shard
        self._handle = None
        self._buffered_records: list[dict[str, Any]] = []
        self._current_bytes = 0
        self._current_records = 0
        self._next_shard_index = self._discover_next_shard_index()
        self.extension = "parquet" if output_format == "parquet" else "jsonl"
        self.shard_paths: list[Path] = sorted(self.directory.glob(f"part-*.{self.extension}"))

    def _discover_next_shard_index(self) -> int:
        next_index = 0
        for path in list(self.directory.glob("part-*.jsonl")) + list(self.directory.glob("part-*.parquet")):
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
        path = self.directory / f"part-{self._next_shard_index:05d}.{self.extension}"
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
    ) -> None:
        self.root = root
        self.output_format = output_format
        self.max_open_writers = max_open_writers
        self.max_records_per_shard = max_records_per_shard
        self.max_bytes_per_shard = max_bytes_per_shard
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
        )
        self._writers[key] = writer
        return writer

    def close(self) -> None:
        for writer in self._writers.values():
            writer.close()
        self._writers.clear()


def iter_mined_comment_records(output_directory: Path) -> Iterator[dict[str, Any]]:
    for path in sorted(output_directory.glob("*/*/part-*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def export_huggingface_dataset(
    input_directory: Path,
    output_directory: Path,
    *,
    output_format: str = "parquet",
    max_records_per_shard: int,
    max_bytes_per_shard: int,
    dedupe_record_ids: bool = False,
    max_open_writers: int = 64,
) -> ExportStats:
    if output_format not in {"parquet", "jsonl"}:
        raise ValueError(f"Unsupported export format: {output_format}")
    output_directory.mkdir(parents=True, exist_ok=True)
    writers = _WriterCache(
        output_directory,
        output_format=output_format,
        max_open_writers=max_open_writers,
        max_records_per_shard=max_records_per_shard,
        max_bytes_per_shard=max_bytes_per_shard,
    )
    stats = ExportStats(output_directory=output_directory)
    seen_record_ids: set[tuple[str, str]] = set()

    try:
        for record in iter_mined_comment_records(input_directory):
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

    _write_export_manifest(stats, output_format=output_format)
    _write_dataset_card(stats, output_format=output_format)
    return stats


def _write_export_manifest(stats: ExportStats, *, output_format: str) -> None:
    payload = {
        "created_at": _utc_now(),
        "records_written": stats.records_written,
        "records_skipped_duplicate": stats.records_skipped_duplicate,
        "format": output_format,
        "layout": f"<dataset>/<language>/part-*.{output_format}",
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


def _write_dataset_card(stats: ExportStats, *, output_format: str) -> None:
    lines = [
        "---",
        "configs:",
    ]
    for group in sorted(stats.groups.values(), key=lambda item: (item.dataset, item.language)):
        config_name = f"{group.dataset}__{group.language}".replace("/", "-")
        lines.extend(
            [
                f"- config_name: {json.dumps(config_name)}",
                "  data_files:",
                "  - split: train",
                f"    path: {json.dumps(sorted(group.shards))}",
            ]
        )
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
