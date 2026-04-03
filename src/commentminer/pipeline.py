from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from tqdm.auto import tqdm

from .config import PipelineConfig
from .models import CommentExtractor, CommentRecord, DatasetSource, InputRecord
from .sources import TheStackParquetSource


_SLUG_PATTERN = re.compile(r"[^a-zA-Z0-9]+")
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
    failed_shards: int = 0


@dataclass(slots=True)
class CompletedShardResult:
    remote_path: str
    temp_output_path: Path | None
    records_seen: int
    comments_written: int
    skipped_without_comment: int
    last_record_id: str | None


@dataclass(slots=True)
class FailedShard:
    shard: str
    error: str


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
        line = json.dumps(record.to_dict(), ensure_ascii=False) + "\n"
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
        self._handle.flush()
        self._current_records += 1
        self._current_bytes += len(encoded)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def _build_comment_record(record: InputRecord, opening_comment: str) -> CommentRecord:
    return CommentRecord(
        dataset=record.dataset,
        record_id=record.record_id,
        opening_comment=opening_comment,
        language=record.language,
        path=record.path,
        repo=record.repo,
        extracted_at=_utc_now(),
        metadata=dict(record.metadata),
    )


def _write_run_manifest(
    writer: JsonlShardWriter,
    stats: PipelineRunStats,
    checkpoint: DatasetCheckpoint,
    checkpoint_path: Path,
    *,
    failed_shards: list[FailedShard] | None = None,
) -> None:
    manifest = {
        "dataset": stats.dataset,
        "run_id": stats.run_id,
        "created_at": _utc_now(),
        "records_seen": stats.records_seen,
        "comments_written": stats.comments_written,
        "skipped_without_comment": stats.skipped_without_comment,
        "failed_shards": [asdict(item) for item in failed_shards or []],
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
) -> PipelineRunStats:
    config.ensure_directories()
    checkpoint_store = CheckpointStore(config.storage.checkpoint_directory)
    checkpoint = checkpoint_store.load(source.name)
    writer = JsonlShardWriter(
        config.storage.output_directory,
        source.name,
        max_records_per_shard=config.storage.max_records_per_shard,
        max_bytes_per_shard=config.storage.max_bytes_per_shard,
    )
    stats = PipelineRunStats(dataset=source.name, run_id=writer.run_id)
    iterator = source.iter_records(start_after=checkpoint.last_record_id)
    _LOGGER.info(
        "Starting mining run dataset=%s run_id=%s resume_from=%s max_records=%s",
        source.name,
        writer.run_id,
        checkpoint.last_record_id,
        max_records,
    )

    try:
        for record in iterator:
            stats.records_seen += 1
            checkpoint.records_seen += 1
            checkpoint.last_record_id = record.record_id

            comment = _normalize_comment(extractor.extract_opening_comment(record))
            if comment is None:
                stats.skipped_without_comment += 1
            else:
                writer.write(_build_comment_record(record, comment))
                stats.comments_written += 1
                checkpoint.comments_written += 1

            if stats.records_seen % config.checkpoint_interval_records == 0:
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
            if max_records is not None and stats.records_seen >= max_records:
                break
    finally:
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


def _temp_output_path(temp_root: Path, remote_path: str) -> Path:
    digest = hashlib.sha1(remote_path.encode("utf-8")).hexdigest()[:12]
    return temp_root / f"{_slugify(remote_path)}-{digest}.jsonl"


def _write_json_line(handle, record: CommentRecord) -> None:
    handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def _merge_temp_output(writer: JsonlShardWriter, temp_output_path: Path | None) -> None:
    if temp_output_path is None or not temp_output_path.exists():
        return
    with temp_output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            writer.write_json_line(line)
    temp_output_path.unlink()


def _process_stack_shard(
    source: TheStackParquetSource,
    remote,
    extractor_factory: Callable[[], CommentExtractor],
    temp_root: Path,
    *,
    show_progress: bool,
) -> CompletedShardResult:
    extractor = extractor_factory()
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_output_path = _temp_output_path(temp_root, remote.path)
    handle = None
    records_seen = 0
    comments_written = 0
    skipped_without_comment = 0
    last_record_id: str | None = None
    try:
        for record in source.iter_shard_records(remote, show_progress=show_progress):
            records_seen += 1
            last_record_id = record.record_id
            comment = _normalize_comment(extractor.extract_opening_comment(record))
            if comment is None:
                skipped_without_comment += 1
                continue
            if handle is None:
                handle = temp_output_path.open("w", encoding="utf-8")
            _write_json_line(handle, _build_comment_record(record, comment))
            comments_written += 1
        return CompletedShardResult(
            remote_path=remote.path,
            temp_output_path=temp_output_path if comments_written > 0 else None,
            records_seen=records_seen,
            comments_written=comments_written,
            skipped_without_comment=skipped_without_comment,
            last_record_id=last_record_id,
        )
    except Exception:
        if handle is not None:
            handle.close()
        if temp_output_path.exists():
            temp_output_path.unlink()
        raise
    finally:
        if handle is not None and not handle.closed:
            handle.close()
        if comments_written == 0 and temp_output_path.exists():
            temp_output_path.unlink()


def run_sharded_dataset(
    source: TheStackParquetSource,
    extractor_factory: Callable[[], CommentExtractor],
    config: PipelineConfig,
    *,
    max_workers: int = 1,
    progress_every: int = 1000,
) -> PipelineRunStats:
    if max_workers < 1:
        raise ValueError(f"max_workers must be >= 1, got {max_workers}")

    config.ensure_directories()
    checkpoint_store = CheckpointStore(config.storage.checkpoint_directory)
    checkpoint = checkpoint_store.load(source.name)
    writer = JsonlShardWriter(
        config.storage.output_directory,
        source.name,
        max_records_per_shard=config.storage.max_records_per_shard,
        max_bytes_per_shard=config.storage.max_bytes_per_shard,
    )
    temp_root = writer.dataset_directory / ".tmp"
    pending_shards = source.pending_shards()
    stats = PipelineRunStats(dataset=source.name, run_id=writer.run_id)
    failed_shards: list[FailedShard] = []
    progress_target = progress_every if progress_every > 0 else None
    shard_progress = tqdm(
        total=len(pending_shards),
        desc=f"{source.name}:shards",
        unit="shard",
        dynamic_ncols=True,
        leave=False,
        disable=not source.show_progress or max_workers <= 1,
    )
    _LOGGER.info(
        "Starting sharded mining run dataset=%s run_id=%s pending_shards=%s workers=%s",
        source.name,
        writer.run_id,
        len(pending_shards),
        max_workers,
    )

    def _update_success(remote, result: CompletedShardResult) -> None:
        nonlocal progress_target
        _merge_temp_output(writer, result.temp_output_path)
        source.mark_shard_completed(remote)
        stats.records_seen += result.records_seen
        stats.comments_written += result.comments_written
        stats.skipped_without_comment += result.skipped_without_comment
        checkpoint.records_seen += result.records_seen
        checkpoint.comments_written += result.comments_written
        checkpoint.last_record_id = result.last_record_id
        checkpoint_store.save(checkpoint)
        if progress_target is not None and stats.records_seen >= progress_target:
            _LOGGER.info(
                "Mining progress dataset=%s run_id=%s records_seen=%s comments_written=%s skipped_without_comment=%s last_record_id=%s",
                source.name,
                writer.run_id,
                stats.records_seen,
                stats.comments_written,
                stats.skipped_without_comment,
                checkpoint.last_record_id,
            )
            while stats.records_seen >= progress_target:
                progress_target += progress_every

    futures: dict[Future[CompletedShardResult], object] = {}
    shard_iter = iter(pending_shards)

    def _submit_next(executor: ThreadPoolExecutor) -> bool:
        try:
            remote = next(shard_iter)
        except StopIteration:
            return False
        future = executor.submit(
            _process_stack_shard,
            source,
            remote,
            extractor_factory,
            temp_root,
            show_progress=source.show_progress and max_workers == 1,
        )
        futures[future] = remote
        return True

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for _ in range(min(max_workers, len(pending_shards))):
                _submit_next(executor)
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    remote = futures.pop(future)
                    shard_progress.update(1)
                    try:
                        result = future.result()
                    except Exception as exc:
                        source.note_shard_failure(remote, exc)
                        failed_shards.append(FailedShard(shard=remote.path, error=str(exc)))
                    else:
                        _LOGGER.info(
                            "Finished shard dataset=%s remote_path=%s records_seen=%s comments_written=%s skipped_without_comment=%s",
                            source.name,
                            result.remote_path,
                            result.records_seen,
                            result.comments_written,
                            result.skipped_without_comment,
                        )
                        _update_success(remote, result)
                    _submit_next(executor)
    finally:
        shard_progress.close()
        checkpoint_path = checkpoint_store.save(checkpoint)
        writer.close()
        if temp_root.exists():
            for path in temp_root.iterdir():
                path.unlink()
            temp_root.rmdir()

    stats.shards_written = len(writer.shard_paths)
    stats.failed_shards = len(failed_shards)
    _write_run_manifest(
        writer,
        stats,
        checkpoint,
        checkpoint_path,
        failed_shards=failed_shards,
    )
    _LOGGER.info(
        "Finished sharded mining run dataset=%s run_id=%s records_seen=%s comments_written=%s skipped_without_comment=%s shards_written=%s failed_shards=%s",
        source.name,
        writer.run_id,
        stats.records_seen,
        stats.comments_written,
        stats.skipped_without_comment,
        stats.shards_written,
        stats.failed_shards,
    )
    return stats
