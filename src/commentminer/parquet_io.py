from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from .models import _json_safe


COMMENT_SCHEMA = pa.schema(
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


def normalize_comment_record(record: dict[str, Any]) -> dict[str, str | None]:
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


def iter_comment_records(
    paths: Iterable[Path],
    *,
    batch_size: int = 65_536,
) -> Iterator[dict[str, Any]]:
    for path in paths:
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            yield from batch.to_pylist()


def write_comment_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records, schema=COMMENT_SCHEMA)
    temporary_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        pq.write_table(table, temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def estimated_record_bytes(record: dict[str, Any]) -> int:
    """Estimate logical record bytes without serializing the record to JSON.

    Normalized comment records contain only strings and nulls. The fixed allowance
    covers JSON field names and syntax, while the 12.5% margin covers common string
    escaping. This remains intentionally conservative because it is only used to
    rotate output shards, not to report their actual Parquet size.
    """
    value_bytes = sum(
        len(value.encode("utf-8")) if isinstance(value, str) else 4
        for value in record.values()
    )
    return 192 + value_bytes + value_bytes // 8


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
