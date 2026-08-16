from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
import hashlib
import json
import logging
import math
from multiprocessing import get_context
import os
from pathlib import Path
from queue import Empty
import random
import re
import shutil
import time
import traceback
from typing import Any, Callable, Iterator, Sequence
from urllib.parse import unquote

import httpx
from huggingface_hub import HfFileSystem
from huggingface_hub.errors import HfHubHTTPError
import pyarrow.parquet as pq

from .config import DatasetSpec, PipelineConfig
from .extractors import ML4SEOpeningCommentExtractor
from .models import InputRecord
from .pipeline import PipelineRunStats, run_dataset
from .sources import ShardRowCursor


_LOGGER = logging.getLogger(__name__)
_LISTING_CACHE_VERSION = 1
_STACK_V3_SOURCE_LINE_LIMIT = 250
_UNICODE_SURROGATE_PATTERN = re.compile("[\ud800-\udfff]")


class StackV3MemorySafeguardError(RuntimeError):
    """Raised when continuing a Stack v3 attempt would risk exhausting memory."""


class StackV3ShardDeferred(RuntimeError):
    """Raised to checkpoint and requeue a shard when host memory is low."""


@dataclass(frozen=True, slots=True)
class StackV3BucketShard:
    index: int
    path: str
    language: str
    size: int

    @property
    def digest(self) -> str:
        return hashlib.sha1(self.path.encode("utf-8")).hexdigest()[:12]

    def source_name(self, dataset_name: str) -> str:
        return f"{dataset_name}__stack-v3-shard-{self.index:08d}-{self.digest}"


@dataclass(frozen=True, slots=True)
class StackV3BucketPlan:
    dataset: str
    bucket_id: str
    languages: tuple[str, ...]
    shards: tuple[StackV3BucketShard, ...]

    @property
    def bytes_planned(self) -> int:
        return sum(shard.size for shard in self.shards)


@dataclass(slots=True)
class StackV3BucketMiningSummary:
    dataset: str
    bucket_id: str
    languages_planned: int
    shards_planned: int
    shards_skipped: int
    shards_completed: int
    bytes_planned: int
    records_seen: int = 0
    comments_written: int = 0
    failed_shards: list[str] = field(default_factory=list)
    run_stats: list[PipelineRunStats] = field(default_factory=list)


@dataclass(slots=True)
class _StackV3WorkerOutcome:
    task_id: int
    status: str
    stats: PipelineRunStats | None = None
    error: str | None = None
    traceback_text: str | None = None


class StackV3BucketShardSource:
    def __init__(
        self,
        dataset: DatasetSpec,
        shard: StackV3BucketShard,
        local_path: Path,
        *,
        max_repo_ids: int = 32,
        min_available_memory_gb: int = 0,
    ) -> None:
        self.dataset = dataset
        self.shard = shard
        self.local_path = local_path
        self.max_repo_ids = max_repo_ids
        self.min_available_memory_gb = min_available_memory_gb
        self.name = shard.source_name(dataset.name)

    def iter_records(self, start_after: str | None = None) -> Iterator[InputRecord]:
        start_row = 0
        if start_after:
            cursor = ShardRowCursor.parse(start_after)
            if cursor.remote_path == self.shard.path:
                start_row = cursor.row_index + 1

        parquet_file = pq.ParquetFile(self.local_path)
        available = set(parquet_file.schema.names)
        columns = [
            column
            for column in ("content_id", "content", "size_bytes", "dedup_cluster", "repo_ids")
            if column in available
        ]
        row_index = 0
        batches = iter(
            parquet_file.iter_batches(
                batch_size=self.dataset.batch_size,
                columns=columns,
            )
        )
        while True:
            _raise_if_memory_low(
                self.min_available_memory_gb,
                shard_path=self.shard.path,
            )
            try:
                batch = next(batches)
            except StopIteration:
                break
            for row in batch.to_pylist():
                if row_index < start_row:
                    row_index += 1
                    continue
                content_id = str(row.get("content_id") or f"row-{row_index}")
                content, record_language, content_truncated = _stack_v3_content(
                    self.shard.language,
                    row.get("content"),
                )
                raw_repo_ids = row.get("repo_ids") or []
                repo_ids = [int(value) for value in raw_repo_ids[: self.max_repo_ids]]
                metadata: dict[str, Any] = {
                    "content_id": content_id,
                    "size_bytes": row.get("size_bytes"),
                    "dedup_cluster": row.get("dedup_cluster"),
                    "repo_ids": repo_ids,
                    "repo_count": len(raw_repo_ids),
                    "repo_ids_truncated": len(raw_repo_ids) > self.max_repo_ids,
                    "remote_path": self.shard.path,
                    "row_index": row_index,
                    "selected_language": self.shard.language,
                    "detect_language_from_content": True,
                }
                if content_truncated:
                    metadata["content_truncated"] = True
                    metadata["content_line_limit"] = _STACK_V3_SOURCE_LINE_LIMIT
                if record_language != self.shard.language:
                    metadata["container_language"] = self.shard.language
                    metadata["selected_language"] = record_language
                yield InputRecord(
                    dataset=self.dataset.name,
                    record_id=ShardRowCursor(self.shard.path, row_index).to_record_id(),
                    content=content,
                    language=record_language,
                    repo=str(repo_ids[0]) if repo_ids else None,
                    metadata=metadata,
                )
                row_index += 1


def _stack_v3_content(language: str, raw_content: Any) -> tuple[str, str, bool]:
    if isinstance(raw_content, bytes):
        content = raw_content.decode("utf-8", errors="replace")
    else:
        content = str(raw_content or "")
    if language.casefold() != "jupyter notebook":
        content, truncated = _truncate_to_line_limit(
            content,
            _STACK_V3_SOURCE_LINE_LIMIT,
        )
        if language.casefold() == "unity3d asset":
            content = _UNICODE_SURROGATE_PATTERN.sub("\ufffd", content)
        return content, language, truncated

    try:
        notebook = json.loads(content)
    except (TypeError, ValueError, json.JSONDecodeError):
        return content, language, False
    if not isinstance(notebook, dict):
        return content, language, False

    code_cells: list[str] = []
    for cell in notebook.get("cells") or []:
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        source = cell.get("source") or ""
        if isinstance(source, list):
            code_cells.append("".join(str(line) for line in source))
        else:
            code_cells.append(str(source))
    if not code_cells:
        return "", language, False

    metadata = notebook.get("metadata") or {}
    detected = None
    if isinstance(metadata, dict):
        language_info = metadata.get("language_info") or {}
        kernelspec = metadata.get("kernelspec") or {}
        if isinstance(language_info, dict):
            detected = language_info.get("name")
        if not detected and isinstance(kernelspec, dict):
            detected = kernelspec.get("language")
    return "\n".join(code_cells), str(detected or language), False


def _truncate_to_line_limit(text: str, max_lines: int) -> tuple[str, bool]:
    if max_lines < 1 or not text:
        return text, False
    search_from = 0
    for _ in range(max_lines):
        newline = text.find("\n", search_from)
        if newline == -1:
            return text, False
        search_from = newline + 1
    if search_from >= len(text):
        return text, False
    return text[:search_from], True


def plan_stack_v3_bucket_shards(
    config: PipelineConfig,
    dataset: DatasetSpec,
    *,
    token: str | bool | None = None,
    languages: Sequence[str] | None = None,
    exclude_languages: Sequence[str] | None = None,
    max_languages: int | None = None,
    max_shards: int | None = None,
    listing_workers: int = 64,
    refresh_listing: bool = False,
    filesystem: HfFileSystem | None = None,
) -> StackV3BucketPlan:
    if dataset.source != "huggingface_bucket":
        raise ValueError(f"Dataset '{dataset.name}' is not a Hugging Face Storage Bucket")
    listing_workers = _positive_int("listing_workers", listing_workers)
    config.ensure_directories()
    bucket_id = dataset.resolve_repo_id()
    cache_path = _listing_cache_path(config, dataset)
    cached = None if refresh_listing else _load_listing_cache(cache_path, bucket_id)
    if cached is None:
        fs = filesystem or HfFileSystem(token=token)
        full_inventory = _list_bucket_shards(
            fs,
            bucket_id,
            languages=None,
            max_languages=None,
            listing_workers=listing_workers,
        )
        full_inventory.sort(key=lambda shard: (shard.language, shard.path))
        shards = [
            StackV3BucketShard(
                index=index,
                path=shard.path,
                language=shard.language,
                size=shard.size,
            )
            for index, shard in enumerate(full_inventory)
        ]
        _save_listing_cache(cache_path, bucket_id, shards)
    else:
        shards = cached

    selected = set(languages or ())
    if selected:
        shards = [shard for shard in shards if shard.language in selected]
    excluded = set(exclude_languages or ())
    if excluded:
        shards = [shard for shard in shards if shard.language not in excluded]
    if max_languages is not None:
        allowed = set(sorted({shard.language for shard in shards})[:max_languages])
        shards = [shard for shard in shards if shard.language in allowed]
    shards.sort(key=lambda shard: (shard.language, shard.path))
    if max_shards is not None:
        shards = shards[: _positive_int("max_shards", max_shards)]
    return StackV3BucketPlan(
        dataset=dataset.name,
        bucket_id=bucket_id,
        languages=tuple(sorted({shard.language for shard in shards})),
        shards=tuple(shards),
    )


def mine_stack_v3_bucket_shards(
    config: PipelineConfig,
    dataset: DatasetSpec,
    *,
    token: str | bool | None = None,
    languages: Sequence[str] | None = None,
    exclude_languages: Sequence[str] | None = None,
    max_languages: int | None = None,
    max_shards: int | None = None,
    listing_workers: int = 64,
    shard_workers: int = 128,
    max_extraction_workers: int = 80,
    max_comment_start_row: int = 10,
    progress_every: int = 10_000,
    skip_completed_shards: bool = True,
    skip_errors: bool = False,
    refresh_listing: bool = False,
    shuffle_shards: bool = False,
    shuffle_seed: int = 0,
    min_free_gb: int = 20,
    min_available_memory_gb: int = 16,
    worker_memory_margin: float = 1.25,
    memory_recovery_seconds: float = 30.0,
    shard_launch_interval_seconds: float = 2.0,
    transient_retry_initial_seconds: float = 10.0,
    transient_retry_max_seconds: float = 300.0,
    extractor_factory: Callable[[], ML4SEOpeningCommentExtractor] | None = None,
) -> StackV3BucketMiningSummary:
    shard_workers = _positive_int("shard_workers", shard_workers)
    max_extraction_workers = min(
        shard_workers,
        _positive_int("max_extraction_workers", max_extraction_workers),
    )
    if min_free_gb < 0:
        raise ValueError("min_free_gb must be at least 0")
    if min_available_memory_gb < 0:
        raise ValueError("min_available_memory_gb must be at least 0")
    if worker_memory_margin < 1:
        raise ValueError("worker_memory_margin must be at least 1")
    if memory_recovery_seconds < 0:
        raise ValueError("memory_recovery_seconds must be at least 0")
    if shard_launch_interval_seconds < 0:
        raise ValueError("shard_launch_interval_seconds must be at least 0")
    if transient_retry_initial_seconds < 0:
        raise ValueError("transient_retry_initial_seconds must be at least 0")
    if transient_retry_max_seconds < transient_retry_initial_seconds:
        raise ValueError(
            "transient_retry_max_seconds must be at least transient_retry_initial_seconds"
        )
    plan = plan_stack_v3_bucket_shards(
        config,
        dataset,
        token=token,
        languages=languages,
        exclude_languages=exclude_languages,
        max_languages=max_languages,
        max_shards=max_shards,
        listing_workers=listing_workers,
        refresh_listing=refresh_listing,
    )
    extractor_factory = extractor_factory or (
        lambda: ML4SEOpeningCommentExtractor(max_start_row=max_comment_start_row)
    )
    extractor = extractor_factory()
    unsupported = [
        language
        for language in plan.languages
        if not extractor.supports_language_value(language)
    ]
    if unsupported:
        _LOGGER.warning(
            "Stack v3 has %s partitions without a direct ML4SE parser; content-based fallback remains enabled: %s",
            len(unsupported),
            ", ".join(unsupported),
        )

    pending: list[StackV3BucketShard] = []
    skipped = 0
    for shard in plan.shards:
        if skip_completed_shards and _completion_path(config, dataset, shard).exists():
            skipped += 1
        else:
            pending.append(shard)
    summary = StackV3BucketMiningSummary(
        dataset=dataset.name,
        bucket_id=plan.bucket_id,
        languages_planned=len(plan.languages),
        shards_planned=len(plan.shards),
        shards_skipped=skipped,
        shards_completed=0,
        bytes_planned=plan.bytes_planned,
    )
    if not pending:
        return summary

    if shuffle_shards:
        random.Random(shuffle_seed).shuffle(pending)
        _LOGGER.info(
            "Shuffled %s pending Stack v3 shards across languages with seed=%s",
            len(pending),
            shuffle_seed,
        )

    queue = deque(pending)
    _LOGGER.info(
        "Starting Stack v3 with %s shard workers, at most %s simultaneous extractors, "
        "and %s GiB minimum available-memory headroom",
        shard_workers,
        max_extraction_workers,
        min_available_memory_gb,
    )
    context = get_context("fork")
    extraction_slots = context.BoundedSemaphore(max_extraction_workers)
    result_queue = context.Queue()
    active: dict[int, tuple[Any, StackV3BucketShard]] = {}
    next_task_id = 0
    memory_paused = False
    memory_recovered_since: float | None = None
    next_launch_at = 0.0
    next_memory_gate_log = 0.0
    worker_memory_estimate_bytes = 0
    transient_retry_counts: dict[str, int] = {}
    transient_retry_not_before: dict[str, float] = {}
    active_registry_path = _active_worker_registry_path(config, dataset)
    _write_active_worker_registry(active_registry_path, active)

    def refresh_worker_memory_estimate() -> int:
        nonlocal worker_memory_estimate_bytes
        current = [
            rss
            for process, _ in active.values()
            if process.pid is not None
            for rss in [_process_rss_bytes(process.pid)]
            if rss is not None
        ]
        if current:
            worker_memory_estimate_bytes = max(current)
        return worker_memory_estimate_bytes

    def launch_memory_ready(*, log_block: bool = True) -> bool:
        nonlocal next_memory_gate_log
        if min_available_memory_gb <= 0:
            return True
        available = _available_memory_bytes()
        if available is None:
            return True
        worker_estimate = refresh_worker_memory_estimate()
        required = _required_launch_memory_bytes(
            min_available_memory_gb,
            worker_estimate,
            worker_memory_margin,
        )
        if available >= required:
            return True
        now = time.monotonic()
        if log_block and now >= next_memory_gate_log:
            _LOGGER.warning(
                "Stack v3 memory launch gate is closed: %.1f GiB available, %.1f GiB "
                "required (%.1f GiB floor + %.2fx largest-worker %.1f GiB); "
                "queued=%s active=%s",
                available / 1024**3,
                required / 1024**3,
                min_available_memory_gb,
                worker_memory_margin,
                worker_estimate / 1024**3,
                len(queue),
                len(active),
            )
            next_memory_gate_log = now + 60.0
        return False

    def refresh_memory_pause() -> None:
        nonlocal memory_paused, memory_recovered_since, next_launch_at
        if not memory_paused:
            return
        now = time.monotonic()
        if not launch_memory_ready():
            memory_recovered_since = None
            return
        if memory_recovered_since is None:
            memory_recovered_since = now
            _LOGGER.warning(
                "Stack v3 memory is above the guarded launch requirement; waiting %.0f "
                "seconds for sustained recovery before paced relaunch",
                memory_recovery_seconds,
            )
            return
        if now - memory_recovered_since < memory_recovery_seconds:
            return
        memory_paused = False
        memory_recovered_since = None
        next_launch_at = now
        _LOGGER.warning(
            "Stack v3 memory recovery was sustained for %.0f seconds; resuming "
            "paced shard launches every %.1f seconds",
            memory_recovery_seconds,
            shard_launch_interval_seconds,
        )

    def fill() -> int:
        nonlocal next_task_id, next_launch_at
        submitted = 0
        if memory_paused:
            return submitted
        if min_available_memory_gb > 0 and time.monotonic() < next_launch_at:
            return submitted
        while queue and len(active) < shard_workers:
            if not launch_memory_ready():
                return submitted
            if min_free_gb and (
                shutil.disk_usage(config.storage.output_directory).free
                < min_free_gb * 1024**3
            ):
                raise RuntimeError(
                    "Stopping Stack v3 before scheduling another shard: "
                    f"free disk is below {min_free_gb} GiB"
                )
            shard = _pop_ready_stack_v3_shard(
                queue,
                transient_retry_not_before,
                now=time.monotonic(),
            )
            if shard is None:
                return submitted
            task_id = next_task_id
            next_task_id += 1
            process = context.Process(
                target=_stack_v3_worker_entry,
                args=(
                    result_queue,
                    task_id,
                    config,
                    dataset,
                    shard,
                ),
                kwargs={
                    "token": token,
                    "max_comment_start_row": max_comment_start_row,
                    "progress_every": progress_every,
                    "extraction_slots": extraction_slots,
                    "min_available_memory_gb": min_available_memory_gb,
                },
                name=f"stack-v3-{shard.index:08d}",
            )
            try:
                process.start()
            except Exception:
                queue.appendleft(shard)
                raise
            active[task_id] = process, shard
            _write_active_worker_registry(active_registry_path, active)
            submitted += 1
            if min_available_memory_gb > 0:
                # Start guarded workers one at a time.  This gives the new process
                # time to download/open its shard and expose a meaningful RSS before
                # deciding whether the following worker will fit.
                next_launch_at = time.monotonic() + shard_launch_interval_seconds
                return submitted
        return submitted

    def stop_active() -> None:
        for process, _ in active.values():
            if process.is_alive():
                process.terminate()
        for process, _ in active.values():
            process.join(timeout=5)

    try:
        fill()
        while active or queue:
            refresh_worker_memory_estimate()
            refresh_memory_pause()
            if not active:
                fill()
                if not active and queue:
                    time.sleep(0.5)
                continue
            try:
                outcome = result_queue.get(timeout=0.5)
            except Empty:
                dead_without_result = [
                    (task_id, process, shard)
                    for task_id, (process, shard) in active.items()
                    if process.exitcode is not None
                ]
                if dead_without_result:
                    task_id, process, shard = dead_without_result[0]
                    process.join(timeout=1)
                    active.pop(task_id, None)
                    raise StackV3MemorySafeguardError(
                        "A Stack v3 worker exited without reporting a result; treating this as "
                        f"possible memory exhaustion path={shard.path} exitcode={process.exitcode}"
                    )
                fill()
                continue

            process_and_shard = active.pop(outcome.task_id, None)
            if process_and_shard is None:
                continue
            _write_active_worker_registry(active_registry_path, active)
            process, shard = process_and_shard
            # The outcome is the worker's final action.  Reap it completely before
            # scheduling a replacement so all Arrow/Python allocations are gone.
            process.join()
            if outcome.status == "deferred":
                queue.append(shard)
                memory_paused = min_available_memory_gb > 0
                memory_recovered_since = None
                _LOGGER.warning(
                    "Requeued Stack v3 shard after memory-pressure checkpoint path=%s "
                    "queued=%s active=%s; replacement launches are paused reason=%s",
                    shard.path,
                    len(queue),
                    len(active),
                    outcome.error,
                )
            elif outcome.status == "memory_error":
                raise StackV3MemorySafeguardError(outcome.error or "Stack v3 memory failure")
            elif outcome.status == "transient_error":
                retry_count = transient_retry_counts.get(shard.digest, 0) + 1
                transient_retry_counts[shard.digest] = retry_count
                delay = min(
                    transient_retry_max_seconds,
                    transient_retry_initial_seconds * (2 ** min(retry_count - 1, 30)),
                )
                transient_retry_not_before[shard.digest] = time.monotonic() + delay
                queue.append(shard)
                _LOGGER.warning(
                    "Requeued Stack v3 shard after transient download failure path=%s "
                    "retry=%s delay_seconds=%.1f queued=%s active=%s error=%s",
                    shard.path,
                    retry_count,
                    delay,
                    len(queue),
                    len(active),
                    outcome.error,
                )
            elif outcome.status == "error":
                if not skip_errors:
                    raise RuntimeError(outcome.error or f"Stack v3 shard failed: {shard.path}")
                summary.failed_shards.append(shard.path)
                _LOGGER.error(
                    "Stack v3 shard failed path=%s error=%s\n%s",
                    shard.path,
                    outcome.error,
                    outcome.traceback_text or "",
                )
            else:
                transient_retry_counts.pop(shard.digest, None)
                transient_retry_not_before.pop(shard.digest, None)
                assert outcome.stats is not None
                stats = outcome.stats
                summary.run_stats.append(stats)
                summary.shards_completed += 1
                summary.records_seen += stats.records_seen
                summary.comments_written += stats.comments_written
            if not memory_paused:
                fill()
    except BaseException:
        stop_active()
        raise
    finally:
        _clear_active_worker_registry(active_registry_path, parent_pid=os.getpid())
        result_queue.close()
        result_queue.join_thread()
    return summary


def _stack_v3_worker_entry(
    result_queue: Any,
    task_id: int,
    config: PipelineConfig,
    dataset: DatasetSpec,
    shard: StackV3BucketShard,
    *,
    token: str | bool | None,
    max_comment_start_row: int,
    progress_every: int,
    extraction_slots: Any | None,
    min_available_memory_gb: int,
) -> None:
    try:
        stats = _mine_stack_v3_shard_process(
            config,
            dataset,
            shard,
            token=token,
            max_comment_start_row=max_comment_start_row,
            progress_every=progress_every,
            extraction_slots=extraction_slots,
            min_available_memory_gb=min_available_memory_gb,
        )
    except StackV3ShardDeferred as exc:
        outcome = _StackV3WorkerOutcome(task_id=task_id, status="deferred", error=str(exc))
    except StackV3MemorySafeguardError as exc:
        outcome = _StackV3WorkerOutcome(
            task_id=task_id,
            status="memory_error",
            error=str(exc),
            traceback_text=traceback.format_exc(),
        )
    except BaseException as exc:
        status = "transient_error" if _is_transient_stack_v3_error(exc) else "error"
        outcome = _StackV3WorkerOutcome(
            task_id=task_id,
            status=status,
            error=f"{type(exc).__name__}: {exc}",
            traceback_text=traceback.format_exc(),
        )
    else:
        outcome = _StackV3WorkerOutcome(task_id=task_id, status="success", stats=stats)
    result_queue.put(outcome)


def _mine_stack_v3_shard_process(
    config: PipelineConfig,
    dataset: DatasetSpec,
    shard: StackV3BucketShard,
    *,
    token: str | bool | None,
    max_comment_start_row: int,
    progress_every: int,
    extraction_slots: Any | None = None,
    min_available_memory_gb: int = 0,
) -> PipelineRunStats:
    scratch = config.storage.download_directory / dataset.name / "bucket-shards"
    scratch.mkdir(parents=True, exist_ok=True)
    local_path = scratch / f"{shard.digest}.parquet"
    partial_path = local_path.with_suffix(".partial")
    fs = HfFileSystem(token=token)
    try:
        _raise_if_memory_low(min_available_memory_gb, shard_path=shard.path)
        if not local_path.exists() or local_path.stat().st_size != shard.size:
            partial_path.unlink(missing_ok=True)
            _LOGGER.info(
                "Downloading Stack v3 shard language=%s path=%s size=%s",
                shard.language,
                shard.path,
                shard.size,
            )
            fs.get(shard.path, str(partial_path))
            partial_path.replace(local_path)
        with _extraction_memory_guard(
            extraction_slots,
            min_available_memory_gb=min_available_memory_gb,
            shard_path=shard.path,
        ):
            source = StackV3BucketShardSource(
                dataset,
                shard,
                local_path,
                min_available_memory_gb=min_available_memory_gb,
            )
            output_config = replace(
                config,
                storage=replace(
                    config.storage,
                    output_directory=config.storage.output_directory / dataset.name,
                ),
            )
            try:
                stats = run_dataset(
                    source,
                    ML4SEOpeningCommentExtractor(max_start_row=max_comment_start_row),
                    output_config,
                    progress_every=progress_every,
                    extraction_workers=1,
                )
            except MemoryError as exc:
                raise StackV3MemorySafeguardError(
                    f"Memory allocation failed while extracting Stack v3 shard {shard.path}"
                ) from exc
        completion = _completion_path(config, dataset, shard)
        completion.parent.mkdir(parents=True, exist_ok=True)
        temp = completion.with_suffix(".tmp")
        temp.write_text(
            json.dumps(
                {
                    "path": shard.path,
                    "language": shard.language,
                    "size": shard.size,
                    "records_seen": stats.records_seen,
                    "comments_written": stats.comments_written,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        temp.replace(completion)
        return stats
    finally:
        partial_path.unlink(missing_ok=True)
        local_path.unlink(missing_ok=True)


@contextmanager
def _extraction_memory_guard(
    extraction_slots: Any | None,
    *,
    min_available_memory_gb: int,
    shard_path: str,
) -> Iterator[None]:
    if extraction_slots is not None:
        while not extraction_slots.acquire(timeout=0.5):
            _raise_if_memory_low(
                min_available_memory_gb,
                shard_path=shard_path,
            )
    try:
        _raise_if_memory_low(
            min_available_memory_gb,
            shard_path=shard_path,
        )
        yield
    finally:
        if extraction_slots is not None:
            extraction_slots.release()


def _wait_for_memory_headroom(
    min_available_memory_gb: int,
    *,
    shard_path: str,
    poll_seconds: float = 5.0,
) -> None:
    if min_available_memory_gb <= 0:
        return
    required = min_available_memory_gb * 1024**3
    next_log = 0.0
    while True:
        available = _available_memory_bytes()
        if available is None or available >= required:
            return
        now = time.monotonic()
        if now >= next_log:
            _LOGGER.warning(
                "Stack v3 scheduler waiting for memory shard=%s: %.1f GiB available, %.1f GiB required",
                shard_path,
                available / 1024**3,
                required / 1024**3,
            )
            next_log = now + 60.0
        time.sleep(poll_seconds)


def _memory_has_headroom(min_available_memory_gb: int) -> bool:
    if min_available_memory_gb <= 0:
        return True
    available = _available_memory_bytes()
    return available is None or available >= min_available_memory_gb * 1024**3


def _required_launch_memory_bytes(
    min_available_memory_gb: int,
    largest_worker_rss_bytes: int,
    worker_memory_margin: float = 1.25,
) -> int:
    floor = min_available_memory_gb * 1024**3
    worker_reserve = math.ceil(max(0, largest_worker_rss_bytes) * worker_memory_margin)
    return floor + worker_reserve


def _process_rss_bytes(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _pop_ready_stack_v3_shard(
    queue: deque[StackV3BucketShard],
    retry_not_before: dict[str, float],
    *,
    now: float,
) -> StackV3BucketShard | None:
    for _ in range(len(queue)):
        shard = queue.popleft()
        if retry_not_before.get(shard.digest, 0.0) <= now:
            retry_not_before.pop(shard.digest, None)
            return shard
        queue.append(shard)
    return None


def _is_transient_stack_v3_error(exc: BaseException) -> bool:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, HfHubHTTPError):
            status = current.response.status_code
            if status in {408, 425, 429} or status >= 500:
                return True
        if isinstance(current, (httpx.TimeoutException, httpx.TransportError)):
            return True
        if isinstance(current, (TimeoutError, ConnectionError)):
            return True
        current = current.__cause__ or current.__context__
    return False


def _active_worker_registry_path(config: PipelineConfig, dataset: DatasetSpec) -> Path:
    return (
        config.storage.working_directory
        / "stack-v3-bucket"
        / dataset.name
        / "active-workers.json"
    )


def _write_active_worker_registry(
    path: Path,
    active: dict[int, tuple[Any, StackV3BucketShard]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "parent_pid": os.getpid(),
        "updated_at": time.time(),
        "workers": [
            {
                "task_id": task_id,
                "pid": process.pid,
                "digest": shard.digest,
                "path": shard.path,
                "language": shard.language,
            }
            for task_id, (process, shard) in sorted(active.items())
            if process.pid is not None
        ],
    }
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    temporary.replace(path)


def _clear_active_worker_registry(path: Path, *, parent_pid: int) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("parent_pid") or 0) != parent_pid:
            return
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return
    path.unlink(missing_ok=True)


def _raise_if_memory_low(min_available_memory_gb: int, *, shard_path: str) -> None:
    if _memory_has_headroom(min_available_memory_gb):
        return
    available = _available_memory_bytes()
    available_gb = available / 1024**3 if available is not None else 0.0
    raise StackV3ShardDeferred(
        f"Deferring Stack v3 shard {shard_path}: {available_gb:.1f} GiB available, "
        f"{min_available_memory_gb} GiB required"
    )


def _available_memory_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _list_bucket_shards(
    fs: HfFileSystem,
    bucket_id: str,
    *,
    languages: Sequence[str] | None,
    max_languages: int | None,
    listing_workers: int,
) -> list[StackV3BucketShard]:
    root = f"buckets/{bucket_id}/contents"
    directories = [
        item
        for item in fs.ls(root, detail=True)
        if item.get("type") == "directory" and "/language=" in str(item.get("name"))
    ]
    selected = set(languages or ())
    if selected:
        directories = [
            item
            for item in directories
            if unquote(str(item["name"]).rsplit("language=", 1)[1]) in selected
        ]
    directories.sort(key=lambda item: unquote(str(item["name"])))
    if max_languages is not None:
        directories = directories[: _positive_int("max_languages", max_languages)]

    def list_one(item: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
        language = unquote(str(item["name"]).rsplit("language=", 1)[1])
        files = [
            child
            for child in fs.ls(str(item["name"]), detail=True)
            if child.get("type") == "file" and str(child.get("name", "")).endswith(".parquet")
        ]
        return language, files

    shards: list[StackV3BucketShard] = []
    with ThreadPoolExecutor(max_workers=listing_workers) as executor:
        for language, files in executor.map(list_one, directories):
            for item in files:
                shards.append(
                    StackV3BucketShard(
                        index=0,
                        path=str(item["name"]),
                        language=language,
                        size=int(item.get("size") or 0),
                    )
                )
    return shards


def _listing_cache_path(config: PipelineConfig, dataset: DatasetSpec) -> Path:
    return config.storage.working_directory / "stack-v3-bucket" / f"{dataset.name}-listing.json"


def _load_listing_cache(path: Path, bucket_id: str) -> list[StackV3BucketShard] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != _LISTING_CACHE_VERSION or payload.get("bucket_id") != bucket_id:
            return None
        return [StackV3BucketShard(**item) for item in payload.get("shards", [])]
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _save_listing_cache(
    path: Path,
    bucket_id: str,
    shards: Sequence[StackV3BucketShard],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(
        json.dumps(
            {
                "version": _LISTING_CACHE_VERSION,
                "bucket_id": bucket_id,
                "shards": [
                    {
                        "index": shard.index,
                        "path": shard.path,
                        "language": shard.language,
                        "size": shard.size,
                    }
                    for shard in shards
                ],
            }
        ),
        encoding="utf-8",
    )
    temp.replace(path)


def _completion_path(
    config: PipelineConfig,
    dataset: DatasetSpec,
    shard: StackV3BucketShard,
) -> Path:
    return (
        config.storage.working_directory
        / "stack-v3-bucket"
        / dataset.name
        / "completed"
        / f"{shard.digest}.json"
    )


def _positive_int(name: str, value: int) -> int:
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value
