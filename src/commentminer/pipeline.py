from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Iterator

from .config import PipelineConfig
from .models import CommentExtractor, CommentRecord, DatasetSource, ExtractedComment, InputRecord


_SLUG_PATTERN = re.compile(r"[^a-zA-Z0-9]+")
_COMMENT_ID_SEPARATOR = "::comment::"
_LOGGER = logging.getLogger(__name__)


def _slugify(value: str) -> str:
    slug = _SLUG_PATTERN.sub("-", value).strip("-").lower()
    return slug or "dataset"


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _normalize_comment(comment: str | None) -> str | None:
    if comment is None:
        return None
    normalized = comment.replace("\r\n", "\n").strip()
    return normalized or None


def _json_default(value: object) -> object:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, set):
        return list(value)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            pass
    return str(value)


def _comment_record_payload(record: CommentRecord) -> dict[str, object]:
    return {
        "dataset": record.dataset,
        "record_id": record.record_id,
        "opening_comment": record.opening_comment,
        "language": record.language,
        "path": record.path,
        "repo": record.repo,
        "extracted_at": record.extracted_at,
        "metadata": record.metadata,
    }


@dataclass(slots=True)
class DatasetCheckpoint:
    dataset: str
    last_record_id: str | None = None
    records_seen: int = 0
    comments_written: int = 0
    updated_at: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "DatasetCheckpoint":
        return cls(
            dataset=str(data["dataset"]),
            last_record_id=str(data["last_record_id"]) if data.get("last_record_id") is not None else None,
            records_seen=int(data.get("records_seen", 0)),
            comments_written=int(data.get("comments_written", 0)),
            updated_at=str(data["updated_at"]) if data.get("updated_at") is not None else None,
        )


@dataclass(slots=True)
class PipelineRunStats:
    dataset: str
    run_id: str
    records_seen: int = 0
    comments_written: int = 0
    skipped_without_comment: int = 0
    shards_written: int = 0


@dataclass(frozen=True, slots=True)
class _ExtractionResult:
    record: InputRecord
    comments: list[ExtractedComment]


class CheckpointStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, dataset_name: str) -> Path:
        return self.root / f"{_slugify(dataset_name)}.json"

    def load(self, dataset_name: str) -> DatasetCheckpoint:
        path = self.path_for(dataset_name)
        if not path.exists():
            return DatasetCheckpoint(dataset=dataset_name)
        raw = json.loads(path.read_text(encoding="utf-8"))
        return DatasetCheckpoint.from_dict(raw)

    def save(self, checkpoint: DatasetCheckpoint) -> Path:
        checkpoint.updated_at = _utc_now()
        path = self.path_for(checkpoint.dataset)
        temp_path = path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(checkpoint.to_dict(), indent=2), encoding="utf-8")
        temp_path.replace(path)
        return path


class JsonlShardWriter:
    def __init__(
        self,
        root: Path,
        dataset_name: str,
        *,
        max_records_per_shard: int,
        max_bytes_per_shard: int,
        run_id: str | None = None,
    ) -> None:
        self.run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.dataset_name = dataset_name
        self.dataset_directory = root / _slugify(dataset_name) / self.run_id
        self.dataset_directory.mkdir(parents=True, exist_ok=True)
        self.max_records_per_shard = max_records_per_shard
        self.max_bytes_per_shard = max_bytes_per_shard
        self._handle = None
        self._current_bytes = 0
        self._current_records = 0
        self._next_shard_index = 0
        self.shard_paths: list[Path] = []

    def _open_next_shard(self) -> None:
        self.close()
        shard_path = self.dataset_directory / f"part-{self._next_shard_index:05d}.jsonl"
        self._next_shard_index += 1
        self._handle = shard_path.open("a", encoding="utf-8")
        self._current_bytes = 0
        self._current_records = 0
        self.shard_paths.append(shard_path)

    def write(self, record: CommentRecord) -> None:
        line = (
            json.dumps(
                _comment_record_payload(record),
                ensure_ascii=False,
                separators=(",", ":"),
                default=_json_default,
            )
            + "\n"
        )
        self.write_json_line(line)

    def write_json_line(self, line: str) -> None:
        encoded = line.encode("utf-8")

        if self._handle is None:
            self._open_next_shard()
        elif (
            self._current_records >= self.max_records_per_shard
            or self._current_bytes + len(encoded) > self.max_bytes_per_shard
        ):
            self._open_next_shard()

        assert self._handle is not None
        self._handle.write(line)
        self._current_records += 1
        self._current_bytes += len(encoded)

    def flush(self) -> None:
        if self._handle is not None:
            self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def _normalize_extracted_comments(value: object) -> list[ExtractedComment]:
    if value is None:
        return []
    if isinstance(value, ExtractedComment):
        comment = _normalize_comment(value.text)
        if comment is None:
            return []
        return [
            ExtractedComment(
                text=comment,
                start_line=value.start_line,
                index=value.index,
            )
        ]
    if isinstance(value, str):
        comment = _normalize_comment(value)
        return [ExtractedComment(text=comment)] if comment is not None else []
    try:
        iterator = iter(value)  # type: ignore[arg-type]
    except TypeError:
        comment = _normalize_comment(str(value))
        return [ExtractedComment(text=comment)] if comment is not None else []

    comments: list[ExtractedComment] = []
    for index, item in enumerate(iterator):
        if isinstance(item, ExtractedComment):
            comment = _normalize_comment(item.text)
            if comment is None:
                continue
            comments.append(
                ExtractedComment(
                    text=comment,
                    start_line=item.start_line,
                    index=item.index if item.index is not None else index,
                )
            )
        else:
            comment = _normalize_comment(str(item))
            if comment is not None:
                comments.append(ExtractedComment(text=comment, index=index))
    return comments


def _extract_comments(extractor: CommentExtractor, record: InputRecord) -> list[ExtractedComment]:
    extract_many = getattr(extractor, "extract_opening_comments", None)
    if callable(extract_many):
        return _normalize_extracted_comments(extract_many(record))
    return _normalize_extracted_comments(extractor.extract_opening_comment(record))


def _extract_record_comments(
    extractor: CommentExtractor,
    record: InputRecord,
) -> _ExtractionResult:
    return _ExtractionResult(record=record, comments=_extract_comments(extractor, record))


def _require_positive_int(name: str, value: int) -> int:
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


def _iter_extraction_results(
    records: Iterator[InputRecord],
    extractor: CommentExtractor,
    *,
    max_workers: int,
    max_pending: int,
    max_records: int | None,
) -> Iterator[_ExtractionResult]:
    if max_workers == 1:
        records_seen = 0
        while True:
            if max_records is not None and records_seen >= max_records:
                return
            try:
                record = next(records)
            except StopIteration:
                return
            records_seen += 1
            yield _extract_record_comments(extractor, record)
        return

    executor = ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="commentminer-extract",
    )
    pending: deque[Future[_ExtractionResult]] = deque()
    submitted_records = 0
    source_exhausted = False

    try:
        while pending or not source_exhausted:
            while (
                not source_exhausted
                and len(pending) < max_pending
                and (max_records is None or submitted_records < max_records)
            ):
                try:
                    record = next(records)
                except StopIteration:
                    source_exhausted = True
                    break
                pending.append(executor.submit(_extract_record_comments, extractor, record))
                submitted_records += 1

            if not pending:
                return
            yield pending.popleft().result()
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


def _build_comment_record(
    record: InputRecord,
    extracted_comment: ExtractedComment,
    *,
    multiple_comments: bool,
) -> CommentRecord:
    metadata = dict(record.metadata)
    if extracted_comment.index is not None:
        metadata["comment_index"] = extracted_comment.index
    if extracted_comment.start_line is not None:
        metadata["comment_start_line"] = extracted_comment.start_line
    record_id = record.record_id
    if multiple_comments:
        record_id = f"{record.record_id}{_COMMENT_ID_SEPARATOR}{extracted_comment.index or 0}"

    return CommentRecord(
        dataset=record.dataset,
        record_id=record_id,
        opening_comment=extracted_comment.text,
        language=record.language,
        path=record.path,
        repo=record.repo,
        extracted_at=_utc_now(),
        metadata=metadata,
    )


def _write_run_manifest(
    writer: JsonlShardWriter,
    stats: PipelineRunStats,
    checkpoint: DatasetCheckpoint,
    checkpoint_path: Path,
) -> None:
    manifest = {
        "dataset": stats.dataset,
        "run_id": stats.run_id,
        "created_at": _utc_now(),
        "records_seen": stats.records_seen,
        "comments_written": stats.comments_written,
        "skipped_without_comment": stats.skipped_without_comment,
        "checkpoint": checkpoint.to_dict(),
        "checkpoint_path": str(checkpoint_path),
        "shards": [path.name for path in writer.shard_paths],
    }
    manifest_path = writer.dataset_directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run_dataset(
    source: DatasetSource,
    extractor: CommentExtractor,
    config: PipelineConfig,
    *,
    max_records: int | None = None,
    progress_every: int = 1000,
    extraction_workers: int = 1,
    extraction_buffer: int | None = None,
) -> PipelineRunStats:
    config.ensure_directories()
    extraction_workers = _require_positive_int("extraction_workers", extraction_workers)
    if extraction_buffer is None:
        extraction_buffer = extraction_workers * 4
    extraction_buffer = _require_positive_int("extraction_buffer", extraction_buffer)
    if extraction_buffer < extraction_workers:
        raise ValueError(
            "extraction_buffer must be greater than or equal to extraction_workers"
        )

    checkpoint_store = CheckpointStore(config.storage.checkpoint_directory)
    checkpoint = checkpoint_store.load(source.name)
    writer = JsonlShardWriter(
        config.storage.output_directory,
        source.name,
        max_records_per_shard=config.storage.max_records_per_shard,
        max_bytes_per_shard=config.storage.max_bytes_per_shard,
    )
    stats = PipelineRunStats(dataset=source.name, run_id=writer.run_id)
    iterator = iter(source.iter_records(start_after=checkpoint.last_record_id))
    _LOGGER.info(
        "Starting mining run dataset=%s run_id=%s resume_from=%s max_records=%s extraction_workers=%s extraction_buffer=%s",
        source.name,
        writer.run_id,
        checkpoint.last_record_id,
        max_records,
        extraction_workers,
        extraction_buffer,
    )

    try:
        results = _iter_extraction_results(
            iterator,
            extractor,
            max_workers=extraction_workers,
            max_pending=extraction_buffer,
            max_records=max_records,
        )
        for result in results:
            record = result.record
            stats.records_seen += 1
            checkpoint.records_seen += 1
            checkpoint.last_record_id = record.record_id

            comments = result.comments
            if not comments:
                stats.skipped_without_comment += 1
            else:
                multiple_comments = len(comments) > 1
                for comment in comments:
                    writer.write(
                        _build_comment_record(
                            record,
                            comment,
                            multiple_comments=multiple_comments,
                        )
                    )
                    stats.comments_written += 1
                    checkpoint.comments_written += 1

            if stats.records_seen % config.checkpoint_interval_records == 0:
                writer.flush()
                checkpoint_store.save(checkpoint)
            if progress_every > 0 and stats.records_seen % progress_every == 0:
                _LOGGER.info(
                    "Mining progress dataset=%s run_id=%s records_seen=%s comments_written=%s skipped_without_comment=%s last_record_id=%s",
                    source.name,
                    writer.run_id,
                    stats.records_seen,
                    stats.comments_written,
                    stats.skipped_without_comment,
                    checkpoint.last_record_id,
                )
    finally:
        writer.flush()
        checkpoint_path = checkpoint_store.save(checkpoint)
        close = getattr(iterator, "close", None)
        if callable(close):
            close()
        writer.close()

    stats.shards_written = len(writer.shard_paths)
    _write_run_manifest(writer, stats, checkpoint, checkpoint_path)
    _LOGGER.info(
        "Finished mining run dataset=%s run_id=%s records_seen=%s comments_written=%s skipped_without_comment=%s shards_written=%s",
        source.name,
        writer.run_id,
        stats.records_seen,
        stats.comments_written,
        stats.skipped_without_comment,
        stats.shards_written,
    )
    return stats
