from __future__ import annotations

from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from functools import partial
import gc
import hashlib
import logging
from pathlib import Path
from typing import Callable, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from .config import DatasetSpec, PipelineConfig
from .downloader import HuggingFaceDownloader, RemoteFile
from .extractors import ML4SEOpeningCommentExtractor
from .models import CommentExtractor, InputRecord
from .pipeline import CheckpointStore, PipelineRunStats, run_dataset
from .sources import (
    AiohttpSoftwareHeritageContentFetcher,
    ShardRowCursor,
    StackV2ContentFetcher,
    StackV2SWHContentSource,
    _raise_fd_limit_for_stack_v2_content_workers,
    _stack_v2_content_fetcher_from_dataset,
)


_LOGGER = logging.getLogger(__name__)
_PROCESSED_CHECKPOINT_NAMESPACE = "processed-shards"


@dataclass(frozen=True, slots=True)
class StackV2IdSegment:
    language: str | None
    remote: RemoteFile
    start_row: int
    end_row: int

    @property
    def id_count(self) -> int:
        return max(0, self.end_row - self.start_row)


@dataclass(frozen=True, slots=True)
class StackV2IdPackage:
    index: int
    segments: tuple[StackV2IdSegment, ...]

    @property
    def id_count(self) -> int:
        return sum(segment.id_count for segment in self.segments)

    @property
    def languages(self) -> tuple[str | None, ...]:
        seen: set[str | None] = set()
        languages: list[str | None] = []
        for segment in self.segments:
            if segment.language in seen:
                continue
            seen.add(segment.language)
            languages.append(segment.language)
        return tuple(languages)

    def source_name(self, dataset_name: str) -> str:
        fingerprint = hashlib.sha1()
        for segment in self.segments:
            fingerprint.update(segment.remote.path.encode("utf-8"))
            fingerprint.update(b":")
            fingerprint.update(str(segment.start_row).encode("ascii"))
            fingerprint.update(b"-")
            fingerprint.update(str(segment.end_row).encode("ascii"))
            fingerprint.update(b";")
        digest = fingerprint.hexdigest()[:12]
        return f"{dataset_name}__stack-v2-idpkg-{self.index:08d}-{digest}"


@dataclass(frozen=True, slots=True)
class StackV2IdPackagePlan:
    dataset: str
    repo_id: str
    revision: str
    package_size: int
    packages: tuple[StackV2IdPackage, ...]
    remote_files: tuple[RemoteFile, ...]

    @property
    def id_count(self) -> int:
        return sum(package.id_count for package in self.packages)


@dataclass(slots=True)
class StackV2PackageMiningSummary:
    dataset: str
    repo_id: str
    revision: str
    package_size: int
    packages_planned: int
    packages_skipped: int
    packages_completed: int
    ids_planned: int
    records_seen: int = 0
    comments_written: int = 0
    failed_packages: list[str] = field(default_factory=list)
    run_stats: list[PipelineRunStats] = field(default_factory=list)


class StackV2SWHContentPackageSource(StackV2SWHContentSource):
    """Stack v2 source that processes one pre-planned package of blob-id rows."""

    def __init__(
        self,
        config: PipelineConfig,
        dataset: DatasetSpec,
        package: StackV2IdPackage,
        *,
        source_name: str | None = None,
        show_progress: bool = True,
        token: str | bool | None = None,
        downloader: HuggingFaceDownloader | None = None,
        cache_source_files: bool | None = None,
        content_fetcher: StackV2ContentFetcher | None = None,
        content_executor: ThreadPoolExecutor | None = None,
        content_download_workers: int | None = None,
        content_prefetch_records: int | None = None,
        content_language_filter: Callable[[str], bool] | None = None,
        skip_missing_content: bool | None = None,
    ) -> None:
        super().__init__(
            config,
            dataset,
            language=None,
            show_progress=show_progress,
            token=token,
            downloader=downloader,
            max_files=None,
            prefetch_files=1,
            download_workers=1,
            cache_source_files=cache_source_files,
            content_fetcher=content_fetcher,
            content_executor=content_executor,
            content_download_workers=content_download_workers,
            content_prefetch_records=content_prefetch_records,
            content_language_filter=content_language_filter,
            skip_missing_content=skip_missing_content,
        )
        self.package = package
        self.name = source_name or package.source_name(dataset.name)
        self._active_stop_row: int | None = None
        self._active_segment: StackV2IdSegment | None = None

    def iter_records(self, start_after: str | None = None):
        self.config.ensure_directories()
        resume_cursor = ShardRowCursor.parse(start_after) if start_after else None
        resume_matched = resume_cursor is None
        _LOGGER.info(
            "Preparing Stack v2 id package dataset=%s source=%s package=%s ids=%s segments=%s resume_from=%s",
            self.dataset.name,
            self.name,
            self.package.index,
            self.package.id_count,
            len(self.package.segments),
            start_after,
        )

        for segment in self.package.segments:
            segment_start = segment.start_row
            if not resume_matched:
                if resume_cursor is None or resume_cursor.remote_path != segment.remote.path:
                    continue
                resume_matched = True
                segment_start = max(segment.start_row, resume_cursor.row_index + 1)

            if segment_start >= segment.end_row:
                continue

            local_path = self._local_or_downloaded_metadata_path(segment.remote)
            self._active_stop_row = segment.end_row
            self._active_segment = segment
            segment_resume = (
                ShardRowCursor(segment.remote.path, segment_start - 1)
                if segment_start > 0
                else None
            )
            try:
                yield from super()._iter_file_records(
                    segment.remote,
                    local_path,
                    segment_resume,
                )
            finally:
                self._active_stop_row = None
                self._active_segment = None
                resume_cursor = None

    def _local_or_downloaded_metadata_path(self, remote: RemoteFile) -> Path:
        local_path = self.downloader._download_root(  # noqa: SLF001
            self.config,
            self.dataset,
            None,
        ) / remote.path
        if local_path.exists():
            return local_path
        return self.downloader.download_remote_file(
            self.config,
            self.dataset,
            remote.path,
            language=None,
            token=self.token,
            use_cache=self.cache_source_files,
        )

    def _stop_row_for_remote(self, remote: RemoteFile) -> int | None:
        return self._active_stop_row

    def _row_to_input_record_with_content(
        self,
        remote_path: str,
        row_index: int,
        row: dict[str, object],
        content: str,
        *,
        content_fetch_status: str,
    ) -> InputRecord:
        record = super()._row_to_input_record_with_content(
            remote_path,
            row_index,
            row,
            content,
            content_fetch_status=content_fetch_status,
        )
        metadata = dict(record.metadata)
        metadata["stack_v2_id_package_index"] = self.package.index
        if self._active_segment is not None:
            metadata["stack_v2_id_package_start_row"] = self._active_segment.start_row
            metadata["stack_v2_id_package_end_row"] = self._active_segment.end_row
            if self._active_segment.language is not None:
                metadata["selected_language"] = self._active_segment.language
        return InputRecord(
            dataset=record.dataset,
            record_id=record.record_id,
            content=record.content,
            language=record.language,
            path=record.path,
            repo=record.repo,
            metadata=metadata,
        )


def plan_stack_v2_id_packages(
    config: PipelineConfig,
    dataset: DatasetSpec,
    *,
    downloader: HuggingFaceDownloader | None = None,
    token: str | bool | None = None,
    languages: Sequence[str | None] | None = None,
    max_languages: int | None = None,
    max_files_per_language: int | None = None,
    package_size: int = 10_000,
    metadata_download_workers: int = 4,
    cache_source_files: bool = False,
) -> StackV2IdPackagePlan:
    if dataset.source != "huggingface_hub":
        raise ValueError(f"Dataset '{dataset.name}' is not backed by Hugging Face")
    if dataset.extra.get("content_backend") != "softwareheritage_s3":
        raise ValueError(
            f"Dataset '{dataset.name}' is not configured as a Stack v2 SWH content source"
        )
    package_size = _require_positive_int("package_size", package_size)
    metadata_download_workers = _require_positive_int(
        "metadata_download_workers",
        metadata_download_workers,
    )
    config.ensure_directories()
    downloader = downloader or HuggingFaceDownloader()
    resolved_languages = _resolve_languages(
        config,
        dataset,
        downloader=downloader,
        token=token,
        languages=languages,
        max_languages=max_languages,
    )

    remote_refs: list[tuple[str | None, RemoteFile]] = []
    for language in resolved_languages:
        plan = downloader.plan_download(
            config,
            dataset,
            language=language,
            token=token,
            max_files=max_files_per_language,
            checkpoint_namespace=_PROCESSED_CHECKPOINT_NAMESPACE,
        )
        remote_refs.extend((language, remote) for remote in plan.pending_files)

    remote_refs.sort(key=lambda item: ((item[0] or ""), item[1].path))
    _LOGGER.info(
        "Planning Stack v2 id packages dataset=%s languages=%s metadata_files=%s package_size=%s metadata_download_workers=%s",
        dataset.name,
        len(resolved_languages),
        len(remote_refs),
        package_size,
        metadata_download_workers,
    )
    row_counts = _download_metadata_row_counts(
        config,
        dataset,
        downloader=downloader,
        token=token,
        remote_refs=remote_refs,
        metadata_download_workers=metadata_download_workers,
        cache_source_files=cache_source_files,
    )
    packages = _build_id_packages(row_counts, package_size=package_size)
    remote_files = tuple(remote for _, remote, row_count in row_counts if row_count > 0)
    return StackV2IdPackagePlan(
        dataset=dataset.name,
        repo_id=dataset.resolve_repo_id(),
        revision=dataset.revision,
        package_size=package_size,
        packages=tuple(packages),
        remote_files=remote_files,
    )


def mine_stack_v2_id_packages(
    config: PipelineConfig,
    dataset: DatasetSpec,
    *,
    downloader: HuggingFaceDownloader | None = None,
    token: str | bool | None = None,
    languages: Sequence[str | None] | None = None,
    max_languages: int | None = None,
    max_files_per_language: int | None = None,
    max_packages: int | None = None,
    package_size: int = 10_000,
    metadata_download_workers: int = 4,
    package_workers: int = 4,
    package_worker_backend: str | None = None,
    package_worker_max_tasks_per_child: int | None = None,
    content_download_workers: int = 2048,
    content_prefetch_records: int | None = None,
    extraction_workers: int = 1,
    extraction_buffer: int | None = None,
    max_comment_start_row: int = 10,
    cache_source_files: bool = False,
    show_progress: bool = True,
    progress_every: int = 1000,
    skip_completed_packages: bool = True,
    skip_errors: bool = False,
    content_fetcher: StackV2ContentFetcher | None = None,
    extractor_factory: Callable[[], CommentExtractor] | None = None,
) -> StackV2PackageMiningSummary:
    package_workers = _require_positive_int("package_workers", package_workers)
    package_worker_backend = _package_worker_backend_option(
        package_worker_backend,
        dataset.extra.get("package_worker_backend"),
        dataset.extra.get("stack_v2_package_worker_backend"),
        default="thread",
    )
    package_worker_max_tasks_per_child = _package_worker_max_tasks_per_child_option(
        package_worker_max_tasks_per_child,
        dataset.extra.get("package_worker_max_tasks_per_child"),
        dataset.extra.get("stack_v2_package_worker_max_tasks_per_child"),
        default=1 if package_worker_backend == "process" else None,
    )
    content_download_workers = _require_positive_int(
        "content_download_workers",
        content_download_workers,
    )
    if content_prefetch_records is None:
        if package_worker_backend == "process":
            content_prefetch_records = content_download_workers
        else:
            configured_prefetch = dataset.extra.get("content_prefetch_records")
            if configured_prefetch is None:
                configured_prefetch = dataset.extra.get("swh_content_prefetch_records")
            content_prefetch_records = (
                int(configured_prefetch)
                if configured_prefetch is not None
                else content_download_workers
            )
    content_prefetch_records = _require_positive_int(
        "content_prefetch_records",
        content_prefetch_records,
    )
    if content_prefetch_records < content_download_workers:
        raise ValueError("content_prefetch_records must be >= content_download_workers")
    extraction_workers = _require_positive_int("extraction_workers", extraction_workers)
    if extraction_buffer is not None:
        extraction_buffer = _require_positive_int("extraction_buffer", extraction_buffer)
        if extraction_buffer < extraction_workers:
            raise ValueError("extraction_buffer must be >= extraction_workers")

    downloader = downloader or HuggingFaceDownloader()
    extractor_factory = extractor_factory or (
        partial(_default_opening_comment_extractor, max_comment_start_row)
    )
    language_filter = _content_language_filter(extractor_factory())
    plan = plan_stack_v2_id_packages(
        config,
        dataset,
        downloader=downloader,
        token=token,
        languages=languages,
        max_languages=max_languages,
        max_files_per_language=max_files_per_language,
        package_size=package_size,
        metadata_download_workers=metadata_download_workers,
        cache_source_files=cache_source_files,
    )
    packages = list(plan.packages)
    if max_packages is not None:
        max_packages = _require_positive_int("max_packages", max_packages)
        packages = packages[:max_packages]

    summary = StackV2PackageMiningSummary(
        dataset=plan.dataset,
        repo_id=plan.repo_id,
        revision=plan.revision,
        package_size=plan.package_size,
        packages_planned=len(packages),
        packages_skipped=0,
        packages_completed=0,
        ids_planned=sum(package.id_count for package in packages),
    )
    if skip_completed_packages:
        pending_packages = []
        for package in packages:
            if _package_is_complete(config, dataset, package):
                summary.packages_skipped += 1
            else:
                pending_packages.append(package)
        packages = pending_packages
    if not packages:
        return summary

    _raise_fd_limit_for_stack_v2_content_workers(content_download_workers)
    if package_worker_backend == "process":
        if content_fetcher is not None:
            raise ValueError(
                "package_worker_backend='process' cannot use an injected content_fetcher"
            )
        package_content_workers = max(
            1,
            (content_download_workers + package_workers - 1) // package_workers,
        )
        shared_fetcher: StackV2ContentFetcher | None = None
        shared_content_executor = None
        process_language_filter = None
    else:
        shared_fetcher = content_fetcher or _stack_v2_content_fetcher_from_dataset(
            dataset,
            min_pool_connections=content_download_workers,
        )
        if isinstance(shared_fetcher, AiohttpSoftwareHeritageContentFetcher):
            package_content_workers = content_download_workers
            shared_content_executor = None
            shared_fetcher.warm(concurrency=content_download_workers)
        else:
            package_content_workers = content_download_workers
            shared_content_executor = ThreadPoolExecutor(
                max_workers=content_download_workers,
                thread_name_prefix="commentminer-stack-v2-shared-content",
            )
        process_language_filter = language_filter
    try:
        run_stats, failed_packages = _run_packages(
            config,
            dataset,
            packages,
            downloader=downloader,
            token=token,
            package_workers=package_workers,
            package_worker_backend=package_worker_backend,
            package_worker_max_tasks_per_child=package_worker_max_tasks_per_child,
            content_download_workers=package_content_workers,
            content_prefetch_records=content_prefetch_records,
            content_fetcher=shared_fetcher,
            content_executor=shared_content_executor,
            content_language_filter=process_language_filter,
            extractor_factory=extractor_factory,
            extraction_workers=extraction_workers,
            extraction_buffer=extraction_buffer,
            cache_source_files=cache_source_files,
            show_progress=show_progress,
            progress_every=progress_every,
            skip_errors=skip_errors,
        )
    finally:
        if shared_content_executor is not None:
            shared_content_executor.shutdown(wait=True, cancel_futures=True)
        if content_fetcher is None and isinstance(
            shared_fetcher,
            AiohttpSoftwareHeritageContentFetcher,
        ):
            shared_fetcher.close()

    summary.run_stats.extend(run_stats)
    summary.failed_packages.extend(failed_packages)
    summary.packages_completed = len(run_stats)
    summary.records_seen = sum(stats.records_seen for stats in run_stats)
    summary.comments_written = sum(stats.comments_written for stats in run_stats)
    return summary


def _resolve_languages(
    config: PipelineConfig,
    dataset: DatasetSpec,
    *,
    downloader: HuggingFaceDownloader,
    token: str | bool | None,
    languages: Sequence[str | None] | None,
    max_languages: int | None,
) -> list[str | None]:
    if languages is not None:
        resolved = list(languages)
    elif dataset.supports_language_selection():
        configured = dataset.available_languages()
        resolved = configured or downloader.list_languages(
            dataset,
            token=token,
            cache_directory=downloader.remote_file_cache_directory(config),
        )
    else:
        resolved = [None]

    if max_languages is not None:
        max_languages = _require_positive_int("max_languages", max_languages)
        resolved = resolved[:max_languages]
    return resolved or [None]


def _download_metadata_row_counts(
    config: PipelineConfig,
    dataset: DatasetSpec,
    *,
    downloader: HuggingFaceDownloader,
    token: str | bool | None,
    remote_refs: Sequence[tuple[str | None, RemoteFile]],
    metadata_download_workers: int,
    cache_source_files: bool,
) -> list[tuple[str | None, RemoteFile, int]]:
    def load_row_count(ref: tuple[str | None, RemoteFile]) -> tuple[str | None, RemoteFile, int]:
        language, remote = ref
        local_path = downloader.download_remote_file(
            config,
            dataset,
            remote.path,
            language=None,
            token=token,
            use_cache=cache_source_files,
        )
        parquet_file = pq.ParquetFile(local_path)
        if parquet_file.metadata is None:
            raise ValueError(f"Stack v2 parquet metadata is missing row count: {remote.path}")
        return language, remote, parquet_file.metadata.num_rows

    if metadata_download_workers == 1:
        return [load_row_count(ref) for ref in remote_refs]

    with ThreadPoolExecutor(
        max_workers=metadata_download_workers,
        thread_name_prefix="commentminer-stack-v2-metadata",
    ) as executor:
        futures = [executor.submit(load_row_count, ref) for ref in remote_refs]
        return [future.result() for future in futures]


def _build_id_packages(
    row_counts: Sequence[tuple[str | None, RemoteFile, int]],
    *,
    package_size: int,
) -> list[StackV2IdPackage]:
    packages: list[StackV2IdPackage] = []
    current_segments: list[StackV2IdSegment] = []
    current_count = 0

    def flush() -> None:
        nonlocal current_segments, current_count
        if not current_segments:
            return
        packages.append(
            StackV2IdPackage(
                index=len(packages),
                segments=tuple(current_segments),
            )
        )
        current_segments = []
        current_count = 0

    for language, remote, row_count in row_counts:
        start_row = 0
        while start_row < row_count:
            available = package_size - current_count
            take = min(available, row_count - start_row)
            end_row = start_row + take
            current_segments.append(
                StackV2IdSegment(
                    language=language,
                    remote=remote,
                    start_row=start_row,
                    end_row=end_row,
                )
            )
            current_count += take
            start_row = end_row
            if current_count >= package_size:
                flush()
    flush()
    return packages


def _package_is_complete(
    config: PipelineConfig,
    dataset: DatasetSpec,
    package: StackV2IdPackage,
) -> bool:
    checkpoint = CheckpointStore(config.storage.checkpoint_directory).load(
        package.source_name(dataset.name)
    )
    return checkpoint.records_seen >= package.id_count and package.id_count > 0


def _run_packages(
    config: PipelineConfig,
    dataset: DatasetSpec,
    packages: Sequence[StackV2IdPackage],
    *,
    downloader: HuggingFaceDownloader,
    token: str | bool | None,
    package_workers: int,
    package_worker_backend: str,
    package_worker_max_tasks_per_child: int | None,
    content_download_workers: int,
    content_prefetch_records: int,
    content_fetcher: StackV2ContentFetcher | None,
    content_executor: ThreadPoolExecutor | None,
    content_language_filter: Callable[[str], bool] | None,
    extractor_factory: Callable[[], CommentExtractor],
    extraction_workers: int,
    extraction_buffer: int | None,
    cache_source_files: bool,
    show_progress: bool,
    progress_every: int,
    skip_errors: bool,
) -> tuple[list[PipelineRunStats], list[str]]:
    if package_workers == 1 and package_worker_backend != "process":
        stats: list[PipelineRunStats] = []
        failed_packages: list[str] = []
        for package in packages:
            try:
                assert content_fetcher is not None
                stats.append(
                    _run_package(
                        config,
                        dataset,
                        package,
                        downloader=downloader,
                        token=token,
                        content_download_workers=content_download_workers,
                        content_prefetch_records=content_prefetch_records,
                        content_fetcher=content_fetcher,
                        content_executor=content_executor,
                        content_language_filter=content_language_filter,
                        extractor=extractor_factory(),
                        extraction_workers=extraction_workers,
                        extraction_buffer=extraction_buffer,
                        cache_source_files=cache_source_files,
                        show_progress=show_progress,
                        progress_every=progress_every,
                    )
                )
            except Exception:
                if not skip_errors:
                    raise
                failed_packages.append(package.source_name(dataset.name))
                _LOGGER.exception("Stack v2 package failed package=%s", package.index)
        return stats, failed_packages

    package_queue = deque(packages)
    run_stats: list[PipelineRunStats] = []
    failed_packages: list[str] = []
    pending: dict[Future[PipelineRunStats], StackV2IdPackage] = {}
    process_executor_kwargs = (
        {"max_tasks_per_child": package_worker_max_tasks_per_child}
        if package_worker_backend == "process"
        and package_worker_max_tasks_per_child is not None
        else {}
    )
    executor_context = (
        ProcessPoolExecutor(max_workers=package_workers, **process_executor_kwargs)
        if package_worker_backend == "process"
        else ThreadPoolExecutor(
            max_workers=package_workers,
            thread_name_prefix="commentminer-stack-v2-package",
        )
    )
    with executor_context as executor:
        def fill() -> None:
            while len(pending) < package_workers and package_queue:
                package = package_queue.popleft()
                if package_worker_backend == "process":
                    future = executor.submit(
                        _run_package_process,
                        config,
                        dataset,
                        package,
                        token=token,
                        content_download_workers=content_download_workers,
                        content_prefetch_records=content_prefetch_records,
                        extractor_factory=extractor_factory,
                        extraction_workers=extraction_workers,
                        extraction_buffer=extraction_buffer,
                        cache_source_files=cache_source_files,
                        show_progress=show_progress,
                        progress_every=progress_every,
                    )
                else:
                    assert content_fetcher is not None
                    future = executor.submit(
                        _run_package,
                        config,
                        dataset,
                        package,
                        downloader=downloader,
                        token=token,
                        content_download_workers=content_download_workers,
                        content_prefetch_records=content_prefetch_records,
                        content_fetcher=content_fetcher,
                        content_executor=content_executor,
                        content_language_filter=content_language_filter,
                        extractor=extractor_factory(),
                        extraction_workers=extraction_workers,
                        extraction_buffer=extraction_buffer,
                        cache_source_files=cache_source_files,
                        show_progress=show_progress,
                        progress_every=progress_every,
                    )
                pending[future] = package

        fill()
        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                package = pending.pop(future)
                try:
                    run_stats.append(future.result())
                except Exception:
                    if not skip_errors:
                        raise
                    failed_packages.append(package.source_name(dataset.name))
                    _LOGGER.exception("Stack v2 package failed package=%s", package.index)
                fill()
    return run_stats, failed_packages


def _run_package_process(
    config: PipelineConfig,
    dataset: DatasetSpec,
    package: StackV2IdPackage,
    *,
    token: str | bool | None,
    content_download_workers: int,
    content_prefetch_records: int,
    extractor_factory: Callable[[], CommentExtractor],
    extraction_workers: int,
    extraction_buffer: int | None,
    cache_source_files: bool,
    show_progress: bool,
    progress_every: int,
) -> PipelineRunStats:
    _raise_fd_limit_for_stack_v2_content_workers(content_download_workers)
    downloader = HuggingFaceDownloader()
    content_fetcher = _stack_v2_content_fetcher_from_dataset(
        dataset,
        min_pool_connections=content_download_workers,
    )
    if isinstance(content_fetcher, AiohttpSoftwareHeritageContentFetcher):
        content_fetcher.warm(concurrency=content_download_workers)
    extractor = extractor_factory()
    try:
        return _run_package(
            config,
            dataset,
            package,
            downloader=downloader,
            token=token,
            content_download_workers=content_download_workers,
            content_prefetch_records=content_prefetch_records,
            content_fetcher=content_fetcher,
            content_executor=None,
            content_language_filter=_content_language_filter(extractor),
            extractor=extractor,
            extraction_workers=extraction_workers,
            extraction_buffer=extraction_buffer,
            cache_source_files=cache_source_files,
            show_progress=show_progress,
            progress_every=progress_every,
        )
    finally:
        if isinstance(content_fetcher, AiohttpSoftwareHeritageContentFetcher):
            content_fetcher.close()
        _release_package_worker_memory()


def _run_package(
    config: PipelineConfig,
    dataset: DatasetSpec,
    package: StackV2IdPackage,
    *,
    downloader: HuggingFaceDownloader,
    token: str | bool | None,
    content_download_workers: int,
    content_prefetch_records: int,
    content_fetcher: StackV2ContentFetcher,
    content_executor: ThreadPoolExecutor | None,
    content_language_filter: Callable[[str], bool] | None,
    extractor: CommentExtractor,
    extraction_workers: int,
    extraction_buffer: int | None,
    cache_source_files: bool,
    show_progress: bool,
    progress_every: int,
) -> PipelineRunStats:
    source_name = package.source_name(dataset.name)
    _LOGGER.info(
        "Starting Stack v2 id package dataset=%s source=%s package=%s ids=%s languages=%s",
        dataset.name,
        source_name,
        package.index,
        package.id_count,
        ",".join(language or "all" for language in package.languages),
    )
    source = StackV2SWHContentPackageSource(
        config,
        dataset,
        package,
        source_name=source_name,
        show_progress=show_progress,
        token=token,
        downloader=downloader,
        cache_source_files=cache_source_files,
        content_fetcher=content_fetcher,
        content_executor=content_executor,
        content_download_workers=content_download_workers,
        content_prefetch_records=content_prefetch_records,
        content_language_filter=content_language_filter,
    )
    stats = run_dataset(
        source,
        extractor,
        config,
        progress_every=progress_every,
        extraction_workers=extraction_workers,
        extraction_buffer=extraction_buffer,
    )
    _LOGGER.info(
        "Completed Stack v2 id package dataset=%s source=%s package=%s records=%s comments=%s",
        dataset.name,
        source_name,
        package.index,
        stats.records_seen,
        stats.comments_written,
    )
    return stats


def _content_language_filter(extractor: CommentExtractor) -> Callable[[str], bool] | None:
    supports_language = getattr(extractor, "supports_language_value", None)
    if callable(supports_language):
        return supports_language
    return None


def _default_opening_comment_extractor(max_comment_start_row: int) -> CommentExtractor:
    return ML4SEOpeningCommentExtractor(max_start_row=max_comment_start_row)


def _release_package_worker_memory() -> None:
    gc.collect()
    try:
        pa.default_memory_pool().release_unused()
    except Exception:  # pragma: no cover - depends on optional Arrow allocator support.
        _LOGGER.debug("Unable to release unused PyArrow memory", exc_info=True)


def _package_worker_backend_option(*values: str | None, default: str) -> str:
    for value in values:
        if value is None:
            continue
        result = str(value).strip().lower()
        if not result:
            continue
        if result not in {"thread", "process"}:
            raise ValueError(
                f"package_worker_backend must be 'thread' or 'process', got {value!r}"
            )
        return result
    return default


def _package_worker_max_tasks_per_child_option(
    *values: object,
    default: int | None,
) -> int | None:
    for value in values:
        if value is None:
            continue
        result = int(value)
        if result < 0:
            raise ValueError(
                "package_worker_max_tasks_per_child must be non-negative"
            )
        return result or None
    return default


def _require_positive_int(name: str, value: int) -> int:
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be >= 1, got {result}")
    return result
