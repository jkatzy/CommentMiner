from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import logging
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
from tempfile import TemporaryDirectory
from typing import Any, Callable, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


_LOGGER = logging.getLogger(__name__)
_STACK_V2_MIN_LICENSE_SCORE = 95.0
_STACK_V2_MIN_MATCH_COVERAGE = 95.0
_SQLITE_MAX_VARIABLES = 900
_SCANNER_BACKENDS = {"api", "cli"}
_LICENSE_DETECTION_CACHE_VERSION = 3
_LICENSE_SCAN_MIN_SCORE = 0.0
_HF_LICENSE_SCAN_COLUMNS = {"comment_license_detection", "comment_license_score"}
_MAX_SCANCODE_API_QUERY_CHARS = 10_000
_HISTOGRAM_PARQUET_BATCH_SIZE = 1_000_000


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _normalized_scanner_backend(scanner_backend: str) -> str:
    normalized = scanner_backend.strip().lower()
    if normalized not in _SCANNER_BACKENDS:
        allowed = ", ".join(sorted(_SCANNER_BACKENDS))
        raise ValueError(f"scanner_backend must be one of {allowed}, got {scanner_backend!r}")
    return normalized


def _scancode_identity(*, scanner_backend: str, scancode_command: str) -> str:
    if scanner_backend == "api":
        try:
            package_version = version("scancode-toolkit")
        except PackageNotFoundError:
            package_version = "not-installed"
        return f"api:scancode-toolkit:{package_version}"

    executable = shutil.which(scancode_command)
    if executable is None:
        return f"cli:{scancode_command}:not-found"
    executable_path = Path(executable).resolve()
    try:
        stat = executable_path.stat()
    except OSError:
        return f"cli:{executable_path}:unreadable"
    return f"cli:{executable_path}:{stat.st_size}:{stat.st_mtime_ns}"


def _license_scan_configuration(
    *,
    scanner_backend: str,
    scancode_command: str,
    min_license_score: float,
    min_match_coverage: float,
) -> dict[str, Any]:
    return {
        "cache_version": _LICENSE_DETECTION_CACHE_VERSION,
        "scanner_backend": scanner_backend,
        "scancode_identity": _scancode_identity(
            scanner_backend=scanner_backend,
            scancode_command=scancode_command,
        ),
        "scan_min_license_score": _LICENSE_SCAN_MIN_SCORE,
        "min_license_score": float(min_license_score),
        "min_match_coverage": float(min_match_coverage),
        "max_scancode_api_query_chars": (
            _MAX_SCANCODE_API_QUERY_CHARS if scanner_backend == "api" else None
        ),
    }


def _source_shard_fingerprint(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
    }


def _checkpoint_source_matches(path: Path, shard_stats: dict[str, int] | None) -> bool:
    if shard_stats is None:
        return False
    try:
        fingerprint = _source_shard_fingerprint(path)
    except OSError:
        return False
    return all(shard_stats.get(key) == value for key, value in fingerprint.items())


@dataclass(slots=True)
class LicenseScanCheckpoint:
    source_directory: str
    scan_configuration: dict[str, Any] = field(default_factory=dict)
    completed_shards: list[str] = field(default_factory=list)
    shard_stats: dict[str, dict[str, int]] = field(default_factory=dict)
    updated_at: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["completed_shards"] = sorted(self.completed_shards)
        payload["shard_stats"] = {
            shard: self.shard_stats[shard] for shard in sorted(self.shard_stats)
        }
        return payload

    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        source_directory: Path,
        scan_configuration: dict[str, Any],
    ) -> "LicenseScanCheckpoint":
        if not path.exists():
            return cls(
                source_directory=str(source_directory),
                scan_configuration=scan_configuration,
            )
        raw = json.loads(path.read_text(encoding="utf-8"))
        checkpoint = cls(
            source_directory=str(raw.get("source_directory", source_directory)),
            scan_configuration=dict(raw.get("scan_configuration") or {}),
            completed_shards=[str(item) for item in raw.get("completed_shards", [])],
            shard_stats={
                str(shard): {
                    "records_scanned": int(stats.get("records_scanned", 0)),
                    "records_with_detected_license": int(
                        stats.get("records_with_detected_license", 0)
                    ),
                    "batches_run": int(stats.get("batches_run", 0)),
                    "source_size": int(stats.get("source_size", -1)),
                    "source_mtime_ns": int(stats.get("source_mtime_ns", -1)),
                }
                for shard, stats in (raw.get("shard_stats") or {}).items()
                if isinstance(stats, dict)
            },
            updated_at=str(raw["updated_at"]) if raw.get("updated_at") is not None else None,
        )
        if (
            checkpoint.source_directory != str(source_directory)
            or checkpoint.scan_configuration != scan_configuration
        ):
            return cls(
                source_directory=str(source_directory),
                scan_configuration=scan_configuration,
            )
        return checkpoint

    def save(self, path: Path) -> Path:
        self.updated_at = _utc_now()
        temp_path = path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        temp_path.replace(path)
        return path


@dataclass(slots=True)
class LicenseScanStats:
    input_directory: Path
    output_directory: Path
    records_scanned: int = 0
    records_with_detected_license: int = 0
    records_without_detected_license: int = 0
    shards_processed: int = 0
    shards_skipped: int = 0
    batches_run: int = 0


@dataclass(slots=True)
class LicenseCachePrewarmStats:
    input_directory: Path
    detection_cache_path: Path
    records_seen: int = 0
    unique_comments_seen: int = 0
    cached_comments: int = 0
    comments_scanned: int = 0
    unique_comments_with_detected_license: int = 0
    shards_processed: int = 0
    batches_run: int = 0


@dataclass(slots=True)
class LicenseScoreHistogram:
    input_directory: Path
    shard_format: str
    bin_edges: list[tuple[float, float]]
    bin_counts: list[int]
    shards_read: int = 0
    shards_without_score_column: int = 0
    records_seen: int = 0
    scores_seen: int = 0
    scores_binned: int = 0
    missing_scores: int = 0
    invalid_scores: int = 0
    scores_outside_range: int = 0
    min_score: float | None = None
    max_score: float | None = None
    score_sum: float = 0.0

    @property
    def mean_score(self) -> float | None:
        if self.scores_seen == 0:
            return None
        return self.score_sum / self.scores_seen


@dataclass(slots=True)
class _ScannedBatch:
    headers: list[dict[str, Any]]
    records_scanned: int
    detections: int


@dataclass(slots=True)
class _ScannedParquetShard:
    relative_path: str
    output_path: str
    records_scanned: int
    detections: int
    batches_run: int
    headers: list[dict[str, Any]]
    source_size: int
    source_mtime_ns: int


@dataclass(slots=True)
class _PrewarmedParquetShard:
    relative_path: str
    records_seen: int
    unique_comments_seen: int
    cached_comments: int
    comments_scanned: int
    detections: int
    batches_run: int
    headers: list[dict[str, Any]]


def scan_huggingface_comment_licenses(
    input_directory: Path,
    *,
    output_directory: Path | None = None,
    scancode_command: str = "scancode",
    scancode_processes: int = 1,
    scanner_backend: str = "api",
    detection_cache_path: Path | None = None,
    batch_size: int = 500,
    min_license_score: float = _STACK_V2_MIN_LICENSE_SCORE,
    min_match_coverage: float = _STACK_V2_MIN_MATCH_COVERAGE,
    workers: int = 1,
    progress_every: int = 10,
    dataset_names: Iterable[str] | None = None,
    languages: Iterable[str] | None = None,
    max_shards: int | None = None,
    runner: Callable[..., dict[str, Any]] | None = None,
) -> LicenseScanStats:
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if scancode_processes < -1:
        raise ValueError(f"scancode_processes must be >= -1, got {scancode_processes}")
    scanner_backend = _normalized_scanner_backend(scanner_backend)
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    if progress_every < 1:
        raise ValueError(f"progress_every must be >= 1, got {progress_every}")
    if max_shards is not None and max_shards < 1:
        raise ValueError(f"max_shards must be >= 1, got {max_shards}")
    if runner is not None and workers > 1:
        raise ValueError("parallel Hugging Face license scans do not support an injected runner")
    scan_configuration = _license_scan_configuration(
        scanner_backend=scanner_backend,
        scancode_command=scancode_command,
        min_license_score=min_license_score,
        min_match_coverage=min_match_coverage,
    )

    input_directory = input_directory.resolve()
    if not input_directory.exists() or not input_directory.is_dir():
        raise ValueError(f"Input Hugging Face dataset directory does not exist: {input_directory}")

    dataset_filter = set(dataset_names or [])
    language_filter = set(languages or [])
    input_shards = _hf_parquet_shards(
        input_directory,
        dataset_filter=dataset_filter,
        language_filter=language_filter,
    )
    if not input_shards:
        raise ValueError(f"No Hugging Face Parquet shards found in: {input_directory}")
    if scanner_backend == "api" and runner is None:
        _warm_scancode_api(min_license_score=min_license_score)

    output_directory = (
        output_directory or input_directory.parent / f"{input_directory.name}-license-scan"
    ).resolve()
    if output_directory == input_directory:
        raise ValueError("Output directory must differ from the input dataset directory")
    output_directory.mkdir(parents=True, exist_ok=True)
    temp_root = output_directory / ".tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    detection_cache_path = (
        detection_cache_path or output_directory / "license-detection-cache.sqlite"
    ).resolve()
    _ensure_license_detection_cache(detection_cache_path)

    checkpoint_path = output_directory / "license-scan-checkpoint.json"
    checkpoint = LicenseScanCheckpoint.from_path(
        checkpoint_path,
        source_directory=input_directory,
        scan_configuration=scan_configuration,
    )
    completed_shards = set(checkpoint.completed_shards)
    if _prune_stale_hf_checkpoint_entries(
        checkpoint,
        completed_shards=completed_shards,
        input_directory=input_directory,
        output_directory=output_directory,
    ):
        completed_shards = set(checkpoint.completed_shards)
        checkpoint.save(checkpoint_path)
    if _hydrate_hf_checkpoint_stats(
        checkpoint,
        completed_shards=completed_shards,
        input_directory=input_directory,
        output_directory=output_directory,
    ):
        checkpoint.save(checkpoint_path)
    pending_shards = [
        path
        for path in input_shards
        if not _hf_output_shard_is_current(
            output_directory / path.relative_to(input_directory),
            path.relative_to(input_directory).as_posix(),
            completed_shards=completed_shards,
        )
    ]
    if max_shards is not None:
        pending_shards = pending_shards[:max_shards]
    stats = LicenseScanStats(input_directory=input_directory, output_directory=output_directory)
    for relative_path in checkpoint.completed_shards:
        shard_stats = checkpoint.shard_stats.get(relative_path)
        if not shard_stats:
            continue
        stats.records_scanned += shard_stats["records_scanned"]
        stats.records_with_detected_license += shard_stats["records_with_detected_license"]
        stats.records_without_detected_license += (
            shard_stats["records_scanned"] - shard_stats["records_with_detected_license"]
        )
        stats.batches_run += shard_stats["batches_run"]
    stats.shards_skipped = len(input_shards) - len(pending_shards)
    scancode_headers: list[dict[str, Any]] = []

    _LOGGER.info(
        "Starting Hugging Face comment license scan input_directory=%s output_directory=%s shards=%s pending=%s workers=%s batch_size=%s scanner_backend=%s scancode_command=%s scancode_processes=%s",
        input_directory,
        output_directory,
        len(input_shards),
        len(pending_shards),
        workers,
        batch_size,
        scanner_backend,
        scancode_command,
        scancode_processes,
    )

    try:
        if workers == 1:
            for input_shard in pending_shards:
                result = _scan_hf_parquet_shard(
                    input_shard,
                    input_directory,
                    output_directory,
                    temp_root,
                    detection_cache_path,
                    scancode_command=scancode_command,
                    scancode_processes=scancode_processes,
                    scanner_backend=scanner_backend,
                    batch_size=batch_size,
                    min_license_score=min_license_score,
                    min_match_coverage=min_match_coverage,
                    runner=runner,
                )
                _apply_scanned_parquet_shard(
                    stats,
                    checkpoint,
                    checkpoint_path,
                    completed_shards,
                    result,
                    progress_every=progress_every,
                )
                if result.headers and not scancode_headers:
                    scancode_headers = result.headers
        else:
            current_workers = min(workers, len(pending_shards) or 1)
            retry_count = 0
            remaining_shards = list(pending_shards)
            while remaining_shards:
                if scanner_backend == "api":
                    _warm_scancode_api(min_license_score=min_license_score)
                executor_kwargs: dict[str, Any] = {
                    "max_workers": min(current_workers, len(remaining_shards) or 1)
                }
                if scanner_backend == "api":
                    fork_context = _fork_multiprocessing_context()
                    if fork_context is not None:
                        executor_kwargs["mp_context"] = fork_context
                try:
                    with ProcessPoolExecutor(**executor_kwargs) as executor:
                        futures = {
                            executor.submit(
                                _scan_hf_parquet_shard,
                                input_shard,
                                input_directory,
                                output_directory,
                                temp_root,
                                detection_cache_path,
                                scancode_command=scancode_command,
                                scancode_processes=scancode_processes,
                                scanner_backend=scanner_backend,
                                batch_size=batch_size,
                                min_license_score=min_license_score,
                                min_match_coverage=min_match_coverage,
                                runner=None,
                            ): input_shard
                            for input_shard in remaining_shards
                        }
                        for future in as_completed(futures):
                            result = future.result()
                            _apply_scanned_parquet_shard(
                                stats,
                                checkpoint,
                                checkpoint_path,
                                completed_shards,
                                result,
                                progress_every=progress_every,
                            )
                            if result.headers and not scancode_headers:
                                scancode_headers = result.headers
                except BrokenProcessPool:
                    retry_count += 1
                    if retry_count > 3:
                        raise
                    previous_workers = current_workers
                    current_workers = max(1, int(current_workers * 0.75))
                    remaining_shards = _pending_hf_shards_after_pool_failure(
                        remaining_shards,
                        input_directory=input_directory,
                        output_directory=output_directory,
                        completed_shards=completed_shards,
                    )
                    _LOGGER.warning(
                        "Hugging Face license scan worker pool terminated abruptly; retrying remaining shards retry=%s previous_workers=%s next_workers=%s pending=%s",
                        retry_count,
                        previous_workers,
                        current_workers,
                        len(remaining_shards),
                    )
                    continue
                break
    finally:
        if temp_root.exists() and not any(temp_root.iterdir()):
            shutil.rmtree(temp_root)

    _write_hf_license_scan_manifest(
        input_directory=input_directory,
        output_directory=output_directory,
        checkpoint_path=checkpoint_path,
        stats=stats,
        input_shards=input_shards,
        batch_size=batch_size,
        min_license_score=min_license_score,
        min_match_coverage=min_match_coverage,
        scancode_command=scancode_command,
        scancode_processes=scancode_processes,
        scanner_backend=scanner_backend,
        scan_configuration=scan_configuration,
        detection_cache_path=detection_cache_path,
        scancode_headers=scancode_headers,
        workers=workers,
        dataset_filter=sorted(dataset_filter),
        language_filter=sorted(language_filter),
    )
    _LOGGER.info(
        "Finished Hugging Face comment license scan records_scanned=%s records_with_detected_license=%s shards_processed=%s shards_skipped=%s",
        stats.records_scanned,
        stats.records_with_detected_license,
        stats.shards_processed,
        stats.shards_skipped,
    )
    return stats


def prewarm_huggingface_license_detection_cache(
    input_directory: Path,
    *,
    detection_cache_path: Path,
    scancode_command: str = "scancode",
    scancode_processes: int = 1,
    scanner_backend: str = "api",
    batch_size: int = 500,
    min_license_score: float = _STACK_V2_MIN_LICENSE_SCORE,
    min_match_coverage: float = _STACK_V2_MIN_MATCH_COVERAGE,
    workers: int = 1,
    progress_every: int = 10,
    dataset_names: Iterable[str] | None = None,
    languages: Iterable[str] | None = None,
    max_shards: int | None = None,
    runner: Callable[..., dict[str, Any]] | None = None,
) -> LicenseCachePrewarmStats:
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if scancode_processes < -1:
        raise ValueError(f"scancode_processes must be >= -1, got {scancode_processes}")
    scanner_backend = _normalized_scanner_backend(scanner_backend)
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    if progress_every < 1:
        raise ValueError(f"progress_every must be >= 1, got {progress_every}")
    if max_shards is not None and max_shards < 1:
        raise ValueError(f"max_shards must be >= 1, got {max_shards}")
    if runner is not None and workers > 1:
        raise ValueError("parallel Hugging Face cache prewarm does not support an injected runner")

    input_directory = input_directory.resolve()
    if not input_directory.exists() or not input_directory.is_dir():
        raise ValueError(f"Input Hugging Face dataset directory does not exist: {input_directory}")

    dataset_filter = set(dataset_names or [])
    language_filter = set(languages or [])
    input_shards = _hf_parquet_shards(
        input_directory,
        dataset_filter=dataset_filter,
        language_filter=language_filter,
    )
    if max_shards is not None:
        input_shards = input_shards[:max_shards]
    if not input_shards:
        raise ValueError(f"No Hugging Face Parquet shards found in: {input_directory}")
    if scanner_backend == "api" and runner is None:
        _warm_scancode_api(min_license_score=min_license_score)

    detection_cache_path = detection_cache_path.resolve()
    _ensure_license_detection_cache(detection_cache_path)
    temp_root = detection_cache_path.parent / ".scancode-cache-prewarm-tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    stats = LicenseCachePrewarmStats(
        input_directory=input_directory,
        detection_cache_path=detection_cache_path,
    )
    scancode_headers: list[dict[str, Any]] = []

    _LOGGER.info(
        "Starting Hugging Face license detection cache prewarm input_directory=%s shards=%s workers=%s batch_size=%s scanner_backend=%s scancode_command=%s scancode_processes=%s cache=%s",
        input_directory,
        len(input_shards),
        workers,
        batch_size,
        scanner_backend,
        scancode_command,
        scancode_processes,
        detection_cache_path,
    )

    try:
        if workers == 1:
            for input_shard in input_shards:
                result = _prewarm_hf_license_cache_shard(
                    input_shard,
                    input_directory,
                    temp_root,
                    detection_cache_path,
                    scancode_command=scancode_command,
                    scancode_processes=scancode_processes,
                    scanner_backend=scanner_backend,
                    batch_size=batch_size,
                    min_license_score=min_license_score,
                    min_match_coverage=min_match_coverage,
                    runner=runner,
                )
                _apply_prewarmed_parquet_shard(
                    stats,
                    result,
                    progress_every=progress_every,
                )
                if result.headers and not scancode_headers:
                    scancode_headers = result.headers
        else:
            if scanner_backend == "api":
                _warm_scancode_api(min_license_score=min_license_score)
            executor_kwargs: dict[str, Any] = {
                "max_workers": min(workers, len(input_shards) or 1)
            }
            if scanner_backend == "api":
                fork_context = _fork_multiprocessing_context()
                if fork_context is not None:
                    executor_kwargs["mp_context"] = fork_context
            with ProcessPoolExecutor(**executor_kwargs) as executor:
                futures = {
                    executor.submit(
                        _prewarm_hf_license_cache_shard,
                        input_shard,
                        input_directory,
                        temp_root,
                        detection_cache_path,
                        scancode_command=scancode_command,
                        scancode_processes=scancode_processes,
                        scanner_backend=scanner_backend,
                        batch_size=batch_size,
                        min_license_score=min_license_score,
                        min_match_coverage=min_match_coverage,
                        runner=None,
                    ): input_shard
                    for input_shard in input_shards
                }
                for future in as_completed(futures):
                    result = future.result()
                    _apply_prewarmed_parquet_shard(
                        stats,
                        result,
                        progress_every=progress_every,
                    )
                    if result.headers and not scancode_headers:
                        scancode_headers = result.headers
    finally:
        if temp_root.exists() and not any(temp_root.iterdir()):
            shutil.rmtree(temp_root)

    _LOGGER.info(
        "Finished Hugging Face license detection cache prewarm records_seen=%s unique_comments_seen=%s cached_comments=%s comments_scanned=%s unique_comments_with_detected_license=%s shards_processed=%s batches_run=%s",
        stats.records_seen,
        stats.unique_comments_seen,
        stats.cached_comments,
        stats.comments_scanned,
        stats.unique_comments_with_detected_license,
        stats.shards_processed,
        stats.batches_run,
    )
    return stats


def build_license_score_histogram(
    input_directory: Path,
    *,
    bins: int = 20,
    lower_score: float = 0.0,
    upper_score: float = 100.0,
    dataset_names: Iterable[str] | None = None,
    languages: Iterable[str] | None = None,
    max_shards: int | None = None,
) -> LicenseScoreHistogram:
    if bins < 1:
        raise ValueError(f"bins must be >= 1, got {bins}")
    if not math.isfinite(lower_score) or not math.isfinite(upper_score):
        raise ValueError("score range bounds must be finite")
    if upper_score <= lower_score:
        raise ValueError("upper_score must be greater than lower_score")
    if max_shards is not None and max_shards < 1:
        raise ValueError(f"max_shards must be >= 1, got {max_shards}")

    input_directory = input_directory.resolve()
    if not input_directory.exists() or not input_directory.is_dir():
        raise ValueError(f"Input directory does not exist: {input_directory}")

    dataset_filter = set(dataset_names or [])
    language_filter = set(languages or [])
    bin_edges = _histogram_bin_edges(
        bins,
        lower_score=lower_score,
        upper_score=upper_score,
    )

    parquet_shards = _hf_parquet_shards(
        input_directory,
        dataset_filter=dataset_filter,
        language_filter=language_filter,
    )
    if max_shards is not None:
        parquet_shards = parquet_shards[:max_shards]
    if parquet_shards:
        histogram = LicenseScoreHistogram(
            input_directory=input_directory,
            shard_format="parquet",
            bin_edges=bin_edges,
            bin_counts=[0] * bins,
        )
        for shard in parquet_shards:
            _add_parquet_license_scores_to_histogram(
                histogram,
                shard,
                lower_score=lower_score,
                upper_score=upper_score,
            )
        return histogram

    raise ValueError(f"No ScanCode score shards found in: {input_directory}")


def format_license_score_histogram(
    histogram: LicenseScoreHistogram,
    *,
    width: int = 50,
) -> str:
    if width < 1:
        raise ValueError(f"width must be >= 1, got {width}")

    max_count = max(histogram.bin_counts, default=0)
    lines = [
        f"Input directory: {histogram.input_directory}",
        f"Shard format: {histogram.shard_format}",
        f"Shards read: {histogram.shards_read}",
        f"Records seen: {histogram.records_seen}",
        f"Scores read: {histogram.scores_seen}",
        f"Scores binned: {histogram.scores_binned}",
    ]
    if histogram.min_score is not None and histogram.max_score is not None:
        lines.extend(
            [
                f"Min score: {_format_score(histogram.min_score)}",
                f"Max score: {_format_score(histogram.max_score)}",
                f"Mean score: {_format_score(histogram.mean_score or 0.0)}",
            ]
        )
    if histogram.shards_without_score_column:
        lines.append(
            "Shards without comment_license_score: "
            f"{histogram.shards_without_score_column}"
        )
    if histogram.missing_scores:
        lines.append(f"Missing scores: {histogram.missing_scores}")
    if histogram.invalid_scores:
        lines.append(f"Invalid scores: {histogram.invalid_scores}")
    if histogram.scores_outside_range:
        lines.append(f"Scores outside histogram range: {histogram.scores_outside_range}")

    lines.append("")
    lines.append("ScanCode score histogram")
    for index, ((lower, upper), count) in enumerate(
        zip(histogram.bin_edges, histogram.bin_counts, strict=True)
    ):
        label = _histogram_bin_label(
            lower,
            upper,
            is_last=index == len(histogram.bin_edges) - 1,
        )
        bar_length = 0 if max_count == 0 else round((count / max_count) * width)
        bar = "#" * bar_length
        lines.append(f"{label} | {bar:<{width}} {count}")
    return "\n".join(lines)


def _histogram_bin_edges(
    bins: int,
    *,
    lower_score: float,
    upper_score: float,
) -> list[tuple[float, float]]:
    step = (upper_score - lower_score) / bins
    return [
        (lower_score + index * step, lower_score + (index + 1) * step)
        for index in range(bins)
    ]


def _add_parquet_license_scores_to_histogram(
    histogram: LicenseScoreHistogram,
    shard: Path,
    *,
    lower_score: float,
    upper_score: float,
) -> None:
    histogram.shards_read += 1
    metadata = pq.read_metadata(shard)
    histogram.records_seen += metadata.num_rows
    schema_names = set(metadata.schema.names)
    if "comment_license_score" not in schema_names:
        histogram.shards_without_score_column += 1
        histogram.missing_scores += metadata.num_rows
        return

    parquet_file = pq.ParquetFile(shard)
    for batch in parquet_file.iter_batches(
        batch_size=_HISTOGRAM_PARQUET_BATCH_SIZE,
        columns=["comment_license_score"],
        use_threads=False,
    ):
        _add_score_array_to_histogram(
            histogram,
            batch.column(0),
            lower_score=lower_score,
            upper_score=upper_score,
        )


def _add_score_array_to_histogram(
    histogram: LicenseScoreHistogram,
    values_array: pa.Array,
    *,
    lower_score: float,
    upper_score: float,
) -> None:
    if len(values_array) == 0:
        return

    try:
        values = np.asarray(values_array.to_numpy(zero_copy_only=False), dtype=np.float64)
    except (pa.ArrowInvalid, TypeError, ValueError):
        for value in values_array.to_pylist():
            _add_score_to_histogram(
                histogram,
                value,
                lower_score=lower_score,
                upper_score=upper_score,
            )
        return

    finite_mask = np.isfinite(values)
    non_finite_count = int(values.size - np.count_nonzero(finite_mask))
    missing_count = int(values_array.null_count)
    histogram.missing_scores += missing_count
    histogram.invalid_scores += max(0, non_finite_count - missing_count)

    finite_values = values[finite_mask]
    if finite_values.size == 0:
        return

    histogram.scores_seen += int(finite_values.size)
    histogram.score_sum += float(finite_values.sum())
    min_score = float(finite_values.min())
    max_score = float(finite_values.max())
    histogram.min_score = (
        min_score if histogram.min_score is None else min(histogram.min_score, min_score)
    )
    histogram.max_score = (
        max_score if histogram.max_score is None else max(histogram.max_score, max_score)
    )

    in_range_mask = (finite_values >= lower_score) & (finite_values <= upper_score)
    in_range_count = int(np.count_nonzero(in_range_mask))
    histogram.scores_outside_range += int(finite_values.size) - in_range_count
    if in_range_count == 0:
        return

    counts, _ = np.histogram(
        finite_values[in_range_mask],
        bins=len(histogram.bin_counts),
        range=(lower_score, upper_score),
    )
    histogram.bin_counts = [
        current + int(addition)
        for current, addition in zip(histogram.bin_counts, counts, strict=True)
    ]
    histogram.scores_binned += int(counts.sum())


def _add_score_to_histogram(
    histogram: LicenseScoreHistogram,
    value: Any,
    *,
    lower_score: float,
    upper_score: float,
) -> None:
    if value is None:
        histogram.missing_scores += 1
        return
    try:
        score = float(value)
    except (TypeError, ValueError):
        histogram.invalid_scores += 1
        return
    if not math.isfinite(score):
        histogram.invalid_scores += 1
        return

    histogram.scores_seen += 1
    histogram.score_sum += score
    histogram.min_score = score if histogram.min_score is None else min(histogram.min_score, score)
    histogram.max_score = score if histogram.max_score is None else max(histogram.max_score, score)

    if score < lower_score or score > upper_score:
        histogram.scores_outside_range += 1
        return

    step = (upper_score - lower_score) / len(histogram.bin_counts)
    bin_index = min(int((score - lower_score) / step), len(histogram.bin_counts) - 1)
    histogram.bin_counts[bin_index] += 1
    histogram.scores_binned += 1


def _histogram_bin_label(lower: float, upper: float, *, is_last: bool) -> str:
    right = "]" if is_last else ")"
    return f"[{_format_score(lower):>6}, {_format_score(upper):>6}{right}"


def _format_score(score: float) -> str:
    if score.is_integer():
        return str(int(score))
    return f"{score:.2f}".rstrip("0").rstrip(".")


def _hf_parquet_shards(
    input_directory: Path,
    *,
    dataset_filter: set[str],
    language_filter: set[str],
) -> list[Path]:
    shards: list[Path] = []
    for path in sorted(input_directory.glob("*/*/part-*.parquet")):
        relative = path.relative_to(input_directory)
        if relative.parts[0].startswith("."):
            continue
        dataset, language = relative.parts[0], relative.parts[1]
        if dataset_filter and dataset not in dataset_filter:
            continue
        if language_filter and language not in language_filter:
            continue
        shards.append(path)
    return shards


def _unique_opening_comments(records: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    unique_comments: list[str] = []
    for payload in records:
        comment = _comment_text(payload.get("opening_comment"))
        if comment in seen:
            continue
        seen.add(comment)
        unique_comments.append(comment)
    return unique_comments


def _ensure_license_detection_cache(cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(cache_path, timeout=60.0) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS license_detection_cache (
                cache_key TEXT PRIMARY KEY,
                detection_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.commit()


def _license_detection_cache_key(
    comment: str,
    *,
    scanner_backend: str,
    scancode_identity: str,
    min_license_score: float,
    min_match_coverage: float,
) -> str:
    payload = json.dumps(
        {
            "cache_version": _LICENSE_DETECTION_CACHE_VERSION,
            "comment": comment,
            "scanner_backend": scanner_backend,
            "scancode_identity": scancode_identity,
            "min_license_score": min_license_score,
            "min_match_coverage": min_match_coverage,
            "scan_min_license_score": _LICENSE_SCAN_MIN_SCORE,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_cached_license_detections(
    cache_path: Path,
    comments: list[str],
    *,
    scanner_backend: str,
    scancode_command: str,
    min_license_score: float,
    min_match_coverage: float,
) -> dict[str, dict[str, Any]]:
    if not comments:
        return {}
    _ensure_license_detection_cache(cache_path)
    scancode_identity = _scancode_identity(
        scanner_backend=scanner_backend,
        scancode_command=scancode_command,
    )
    comment_by_key = {
        _license_detection_cache_key(
            comment,
            scanner_backend=scanner_backend,
            scancode_identity=scancode_identity,
            min_license_score=min_license_score,
            min_match_coverage=min_match_coverage,
        ): comment
        for comment in comments
    }
    detections: dict[str, dict[str, Any]] = {}
    keys = list(comment_by_key)
    with sqlite3.connect(cache_path, timeout=60.0) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        for start in range(0, len(keys), _SQLITE_MAX_VARIABLES):
            chunk = keys[start : start + _SQLITE_MAX_VARIABLES]
            placeholders = ",".join("?" for _ in chunk)
            rows = connection.execute(
                f"SELECT cache_key, detection_json FROM license_detection_cache WHERE cache_key IN ({placeholders})",
                chunk,
            )
            for cache_key, detection_json in rows:
                try:
                    detection = json.loads(detection_json)
                except json.JSONDecodeError:
                    continue
                if isinstance(detection, dict):
                    detections[comment_by_key[cache_key]] = detection
    return detections


def _store_cached_license_detections(
    cache_path: Path,
    detections_by_comment: dict[str, dict[str, Any]],
    *,
    scanner_backend: str,
    scancode_command: str,
    min_license_score: float,
    min_match_coverage: float,
) -> None:
    if not detections_by_comment:
        return
    _ensure_license_detection_cache(cache_path)
    scancode_identity = _scancode_identity(
        scanner_backend=scanner_backend,
        scancode_command=scancode_command,
    )
    updated_at = _utc_now()
    rows = [
        (
            _license_detection_cache_key(
                comment,
                scanner_backend=scanner_backend,
                scancode_identity=scancode_identity,
                min_license_score=min_license_score,
                min_match_coverage=min_match_coverage,
            ),
            json.dumps(detection, ensure_ascii=False, sort_keys=True),
            updated_at,
        )
        for comment, detection in detections_by_comment.items()
    ]
    with sqlite3.connect(cache_path, timeout=60.0) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.executemany(
            """
            INSERT OR REPLACE INTO license_detection_cache
                (cache_key, detection_json, updated_at)
            VALUES (?, ?, ?)
            """,
            rows,
        )
        connection.commit()


def _scan_hf_parquet_shard(
    input_shard: Path,
    input_directory: Path,
    output_directory: Path,
    temp_root: Path,
    detection_cache_path: Path,
    *,
    scancode_command: str,
    scancode_processes: int,
    scanner_backend: str,
    batch_size: int,
    min_license_score: float,
    min_match_coverage: float,
    runner: Callable[..., dict[str, Any]] | None,
) -> _ScannedParquetShard:
    input_shard = Path(input_shard)
    input_directory = Path(input_directory)
    output_directory = Path(output_directory)
    temp_root = Path(temp_root)
    detection_cache_path = Path(detection_cache_path)
    relative_path = input_shard.relative_to(input_directory)
    output_shard = output_directory / relative_path
    output_shard.parent.mkdir(parents=True, exist_ok=True)

    source_table = pq.read_table(input_shard)
    comments = _opening_comment_values(source_table)
    headers: list[dict[str, Any]] = []
    detections = 0
    batches_run = 0
    unique_comments = _unique_comments(comments)
    detection_by_comment = _load_cached_license_detections(
        detection_cache_path,
        unique_comments,
        scanner_backend=scanner_backend,
        scancode_command=scancode_command,
        min_license_score=min_license_score,
        min_match_coverage=min_match_coverage,
    )

    unique_batch: list[dict[str, Any]] = []
    queued_comments: set[str] = set()
    for comment in unique_comments:
        if comment in detection_by_comment or comment in queued_comments:
            continue
        unique_batch.append({"opening_comment": comment})
        queued_comments.add(comment)
        if len(unique_batch) < batch_size:
            continue
        batch_result = _scan_batch_records(
            unique_batch,
            temp_root,
            scancode_command=scancode_command,
            scancode_processes=scancode_processes,
            scanner_backend=scanner_backend,
            min_license_score=min_license_score,
            min_match_coverage=min_match_coverage,
            runner=runner,
        )
        if batch_result.headers and not headers:
            headers = batch_result.headers
        batches_run += 1
        scanned_detections: dict[str, dict[str, Any]] = {}
        for scanned_payload in unique_batch:
            scanned_detections[_comment_text(scanned_payload.get("opening_comment"))] = scanned_payload[
                "comment_license_detection"
            ]
        _store_cached_license_detections(
            detection_cache_path,
            scanned_detections,
            scanner_backend=scanner_backend,
            scancode_command=scancode_command,
            min_license_score=min_license_score,
            min_match_coverage=min_match_coverage,
        )
        detection_by_comment.update(scanned_detections)
        unique_batch = []
        queued_comments = set()

    if unique_batch:
        batch_result = _scan_batch_records(
            unique_batch,
            temp_root,
            scancode_command=scancode_command,
            scancode_processes=scancode_processes,
            scanner_backend=scanner_backend,
            min_license_score=min_license_score,
            min_match_coverage=min_match_coverage,
            runner=runner,
        )
        if batch_result.headers and not headers:
            headers = batch_result.headers
        batches_run += 1
        scanned_detections = {}
        for scanned_payload in unique_batch:
            scanned_detections[_comment_text(scanned_payload.get("opening_comment"))] = scanned_payload[
                "comment_license_detection"
            ]
        _store_cached_license_detections(
            detection_cache_path,
            scanned_detections,
            scanner_backend=scanner_backend,
            scancode_command=scancode_command,
            min_license_score=min_license_score,
            min_match_coverage=min_match_coverage,
        )
        detection_by_comment.update(scanned_detections)

    detection_jsons: list[str] = []
    license_scores: list[float] = []
    for comment in comments:
        detection = detection_by_comment.get(comment) or _extract_license_detection(
            None,
            min_license_score=min_license_score,
            min_match_coverage=min_match_coverage,
        )
        if detection["contains_license_notice"]:
            detections += 1
        license_scores.append(_license_score_from_detection(detection))
        detection_jsons.append(json.dumps(detection, ensure_ascii=False, sort_keys=True))

    output_table = _table_with_license_detection_columns(
        source_table,
        detection_jsons=detection_jsons,
        license_scores=license_scores,
    )
    temp_output = output_shard.with_name(
        f"{output_shard.name}.tmp.{os.getpid()}"
    )
    pq.write_table(output_table, temp_output)
    temp_output.replace(output_shard)
    return _ScannedParquetShard(
        relative_path=relative_path.as_posix(),
        output_path=str(output_shard),
        records_scanned=len(comments),
        detections=detections,
        batches_run=batches_run,
        headers=headers,
        **_source_shard_fingerprint(input_shard),
    )


def _prewarm_hf_license_cache_shard(
    input_shard: Path,
    input_directory: Path,
    temp_root: Path,
    detection_cache_path: Path,
    *,
    scancode_command: str,
    scancode_processes: int,
    scanner_backend: str,
    batch_size: int,
    min_license_score: float,
    min_match_coverage: float,
    runner: Callable[..., dict[str, Any]] | None,
) -> _PrewarmedParquetShard:
    input_shard = Path(input_shard)
    input_directory = Path(input_directory)
    temp_root = Path(temp_root)
    detection_cache_path = Path(detection_cache_path)
    relative_path = input_shard.relative_to(input_directory)

    comments = _opening_comments_from_hf_shard(input_shard)
    unique_comments = _unique_comments(comments)
    detection_by_comment = _load_cached_license_detections(
        detection_cache_path,
        unique_comments,
        scanner_backend=scanner_backend,
        scancode_command=scancode_command,
        min_license_score=min_license_score,
        min_match_coverage=min_match_coverage,
    )
    headers: list[dict[str, Any]] = []
    detections = 0
    comments_scanned = 0
    batches_run = 0
    unique_batch: list[dict[str, Any]] = []

    def scan_and_store_batch() -> None:
        nonlocal detections, comments_scanned, batches_run, headers, unique_batch
        batch_result = _scan_batch_records(
            unique_batch,
            temp_root,
            scancode_command=scancode_command,
            scancode_processes=scancode_processes,
            scanner_backend=scanner_backend,
            min_license_score=min_license_score,
            min_match_coverage=min_match_coverage,
            runner=runner,
        )
        if batch_result.headers and not headers:
            headers = batch_result.headers
        batches_run += 1
        comments_scanned += batch_result.records_scanned
        detections += batch_result.detections
        scanned_detections: dict[str, dict[str, Any]] = {}
        for scanned_payload in unique_batch:
            scanned_detections[_comment_text(scanned_payload.get("opening_comment"))] = (
                scanned_payload["comment_license_detection"]
            )
        _store_cached_license_detections(
            detection_cache_path,
            scanned_detections,
            scanner_backend=scanner_backend,
            scancode_command=scancode_command,
            min_license_score=min_license_score,
            min_match_coverage=min_match_coverage,
        )
        detection_by_comment.update(scanned_detections)
        unique_batch = []

    for comment in unique_comments:
        if comment in detection_by_comment:
            continue
        unique_batch.append({"opening_comment": comment})
        if len(unique_batch) >= batch_size:
            scan_and_store_batch()

    if unique_batch:
        scan_and_store_batch()

    return _PrewarmedParquetShard(
        relative_path=relative_path.as_posix(),
        records_seen=len(comments),
        unique_comments_seen=len(unique_comments),
        cached_comments=len(unique_comments) - comments_scanned,
        comments_scanned=comments_scanned,
        detections=detections,
        batches_run=batches_run,
        headers=headers,
    )


def _opening_comments_from_hf_shard(input_shard: Path) -> list[str]:
    schema = pq.read_schema(input_shard)
    if "opening_comment" not in schema.names:
        raise ValueError(f"Hugging Face Parquet shard is missing opening_comment: {input_shard}")
    source_table = pq.read_table(input_shard, columns=["opening_comment"])
    return _opening_comment_values(source_table)


def _apply_prewarmed_parquet_shard(
    stats: LicenseCachePrewarmStats,
    result: _PrewarmedParquetShard,
    *,
    progress_every: int,
) -> None:
    stats.records_seen += result.records_seen
    stats.unique_comments_seen += result.unique_comments_seen
    stats.cached_comments += result.cached_comments
    stats.comments_scanned += result.comments_scanned
    stats.unique_comments_with_detected_license += result.detections
    stats.batches_run += result.batches_run
    stats.shards_processed += 1
    if stats.shards_processed % progress_every == 0:
        _LOGGER.info(
            "Hugging Face license cache prewarm progress shards_processed=%s records_seen=%s unique_comments_seen=%s cached_comments=%s comments_scanned=%s latest_shard=%s",
            stats.shards_processed,
            stats.records_seen,
            stats.unique_comments_seen,
            stats.cached_comments,
            stats.comments_scanned,
            result.relative_path,
        )


def _opening_comment_values(source_table: pa.Table) -> list[str]:
    if "opening_comment" not in source_table.column_names:
        raise ValueError("Hugging Face Parquet input is missing the opening_comment column")
    return [_comment_text(value) for value in source_table.column("opening_comment").to_pylist()]


def _comment_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _unique_comments(comments: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique_comments: list[str] = []
    for comment in comments:
        if comment in seen:
            continue
        seen.add(comment)
        unique_comments.append(comment)
    return unique_comments


def _table_with_license_detection_columns(
    source_table: pa.Table,
    *,
    detection_jsons: list[str],
    license_scores: list[float],
) -> pa.Table:
    existing_license_columns = [
        column for column in _HF_LICENSE_SCAN_COLUMNS if column in source_table.column_names
    ]
    if existing_license_columns:
        source_table = source_table.drop_columns(existing_license_columns)
    return (
        source_table
        .append_column(
            "comment_license_detection",
            pa.array(detection_jsons, type=pa.string()),
        )
        .append_column(
            "comment_license_score",
            pa.array(license_scores, type=pa.float64()),
        )
    )


def _schema_with_license_detection(source_schema: pa.Schema) -> pa.Schema:
    fields = [
        source_schema.field(index)
        for index in range(len(source_schema))
        if source_schema.field(index).name not in _HF_LICENSE_SCAN_COLUMNS
    ]
    fields.append(pa.field("comment_license_detection", pa.string()))
    fields.append(pa.field("comment_license_score", pa.float64()))
    return pa.schema(fields)


def _hf_output_shard_is_current(
    output_shard: Path,
    relative_path: str,
    *,
    completed_shards: set[str],
) -> bool:
    if relative_path not in completed_shards or not output_shard.exists():
        return False
    try:
        schema_names = set(pq.read_schema(output_shard).names)
    except Exception:
        return False
    return _HF_LICENSE_SCAN_COLUMNS <= schema_names


def _pending_hf_shards_after_pool_failure(
    candidate_shards: Iterable[Path],
    *,
    input_directory: Path,
    output_directory: Path,
    completed_shards: set[str],
) -> list[Path]:
    pending: list[Path] = []
    for input_shard in candidate_shards:
        relative_path = input_shard.relative_to(input_directory).as_posix()
        if _hf_output_shard_is_current(
            output_directory / relative_path,
            relative_path,
            completed_shards=completed_shards,
        ):
            continue
        pending.append(input_shard)
    return pending


def _prune_stale_hf_checkpoint_entries(
    checkpoint: LicenseScanCheckpoint,
    *,
    completed_shards: set[str],
    input_directory: Path,
    output_directory: Path,
) -> bool:
    current_completed: set[str] = set()
    for relative_path in completed_shards:
        input_shard = input_directory / relative_path
        output_shard = output_directory / relative_path
        if _checkpoint_source_matches(
            input_shard,
            checkpoint.shard_stats.get(relative_path),
        ) and _hf_output_shard_is_current(
            output_shard,
            relative_path,
            completed_shards=completed_shards,
        ):
            current_completed.add(relative_path)
    if current_completed == completed_shards:
        return False
    checkpoint.completed_shards = sorted(current_completed)
    checkpoint.shard_stats = {
        relative_path: stats
        for relative_path, stats in checkpoint.shard_stats.items()
        if relative_path in current_completed
    }
    return True


def _hydrate_hf_checkpoint_stats(
    checkpoint: LicenseScanCheckpoint,
    *,
    completed_shards: set[str],
    input_directory: Path,
    output_directory: Path,
) -> bool:
    changed = False
    for relative_path in sorted(completed_shards):
        if relative_path in checkpoint.shard_stats:
            continue
        output_shard = output_directory / relative_path
        if not output_shard.exists():
            continue
        try:
            schema_names = set(pq.read_schema(output_shard).names)
        except Exception:
            continue
        if not _HF_LICENSE_SCAN_COLUMNS <= schema_names:
            continue
        checkpoint.shard_stats[relative_path] = _parquet_license_detection_stats(output_shard)
        changed = True
    return changed


def _parquet_license_detection_stats(output_shard: Path) -> dict[str, int]:
    metadata = pq.read_metadata(output_shard)
    records_scanned = metadata.num_rows
    detections = 0
    schema_names = set(metadata.schema.names)
    if "comment_license_detection" in schema_names:
        detections = _count_parquet_license_detections(output_shard)
    return {
        "records_scanned": records_scanned,
        "records_with_detected_license": detections,
        "batches_run": 0,
    }


def _count_parquet_license_detections(output_shard: Path) -> int:
    detections = 0
    table = pq.read_table(output_shard, columns=["comment_license_detection"])
    for detection_json in table.column("comment_license_detection").to_pylist():
        if detection_json is None:
            continue
        try:
            detection = json.loads(detection_json)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(detection, dict) and detection.get("contains_license_notice"):
            detections += 1
    return detections


def _apply_scanned_parquet_shard(
    stats: LicenseScanStats,
    checkpoint: LicenseScanCheckpoint,
    checkpoint_path: Path,
    completed_shards: set[str],
    result: _ScannedParquetShard,
    *,
    progress_every: int,
) -> None:
    stats.records_scanned += result.records_scanned
    stats.records_with_detected_license += result.detections
    stats.records_without_detected_license += result.records_scanned - result.detections
    stats.batches_run += result.batches_run
    stats.shards_processed += 1
    completed_shards.add(result.relative_path)
    checkpoint.completed_shards = sorted(completed_shards)
    checkpoint.shard_stats[result.relative_path] = {
        "records_scanned": result.records_scanned,
        "records_with_detected_license": result.detections,
        "batches_run": result.batches_run,
        "source_size": result.source_size,
        "source_mtime_ns": result.source_mtime_ns,
    }
    checkpoint.save(checkpoint_path)
    if stats.shards_processed % progress_every == 0:
        _LOGGER.info(
            "Hugging Face license scan progress shards_processed=%s records_scanned=%s records_with_detected_license=%s latest_shard=%s",
            stats.shards_processed,
            stats.records_scanned,
            stats.records_with_detected_license,
            result.relative_path,
        )


def _scan_batch_records(
    batch: list[dict[str, Any]],
    temp_root: Path,
    *,
    scancode_command: str,
    scancode_processes: int,
    scanner_backend: str,
    min_license_score: float,
    min_match_coverage: float,
    runner: Callable[..., dict[str, Any]] | None,
) -> _ScannedBatch:
    if runner is None and scanner_backend == "api":
        return _scan_batch_records_with_api(
            batch,
            min_license_score=min_license_score,
            min_match_coverage=min_match_coverage,
        )

    with TemporaryDirectory(dir=temp_root) as temp_dir:
        batch_root = Path(temp_dir)
        inputs_dir = batch_root / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        mapping: dict[str, dict[str, Any]] = {}

        for index, payload in enumerate(batch):
            file_name = f"comment-{index:06d}.txt"
            mapping[file_name] = payload
            comment = _comment_text(payload.get("opening_comment"))
            (inputs_dir / file_name).write_text(comment, encoding="utf-8")

        scan_result = _run_scancode(
            inputs_dir,
            batch_root / "scancode-result.json",
            scancode_command=scancode_command,
            scancode_processes=scancode_processes,
            min_license_score=min_license_score,
            runner=runner,
        )
        resource_results = _resource_results_by_filename(scan_result)

        for file_name, payload in mapping.items():
            detection = _extract_license_detection(
                resource_results.get(file_name),
                min_license_score=min_license_score,
                min_match_coverage=min_match_coverage,
            )
            payload["comment_license_detection"] = detection
            payload["comment_license_score"] = _license_score_from_detection(detection)

        detections = sum(
            1
            for payload in batch
            if payload["comment_license_detection"]["contains_license_notice"]
        )
        headers = scan_result.get("headers", [])
        return _ScannedBatch(
            headers=headers if isinstance(headers, list) else [],
            records_scanned=len(batch),
            detections=detections,
        )


def _scan_batch_records_with_api(
    batch: list[dict[str, Any]],
    *,
    min_license_score: float,
    min_match_coverage: float,
) -> _ScannedBatch:
    for payload in batch:
        resource = _scan_comment_with_scancode_api(
            _comment_text(payload.get("opening_comment")),
            min_license_score=min_license_score,
        )
        payload["comment_license_detection"] = _extract_license_detection(
            resource,
            min_license_score=min_license_score,
            min_match_coverage=min_match_coverage,
        )
        payload["comment_license_score"] = _license_score_from_detection(
            payload["comment_license_detection"]
        )

    detections = sum(
        1
        for payload in batch
        if payload["comment_license_detection"]["contains_license_notice"]
    )
    return _ScannedBatch(headers=[], records_scanned=len(batch), detections=detections)


def _scan_comment_with_scancode_api(
    comment: str,
    *,
    min_license_score: float,
) -> dict[str, Any]:
    scan_errors: list[str] = []
    query_string = comment
    if len(comment) > _MAX_SCANCODE_API_QUERY_CHARS:
        scan_errors.append(
            "Opening comment truncated from "
            f"{len(comment)} to {_MAX_SCANCODE_API_QUERY_CHARS} characters "
            "before ScanCode API scan."
        )
        query_string = comment[:_MAX_SCANCODE_API_QUERY_CHARS]

    try:
        from scancode.api import get_licenses
    except ImportError as exc:  # pragma: no cover - dependency is declared by the project.
        raise RuntimeError(
            "ScanCode Python API is unavailable. Run `uv sync` or use --scanner-backend cli."
        ) from exc
    try:
        result = get_licenses(None, min_score=_LICENSE_SCAN_MIN_SCORE, query_string=query_string)
    except Exception as exc:
        raise RuntimeError(f"ScanCode Python API failed: {type(exc).__name__}: {exc}") from exc
    if not isinstance(result, dict):
        raise RuntimeError(
            f"ScanCode Python API returned {type(result).__name__}, expected a mapping"
        )
    existing_scan_errors = result.get("scan_errors") or []
    if existing_scan_errors:
        detail = "; ".join(str(item) for item in existing_scan_errors)
        raise RuntimeError(f"ScanCode Python API reported scan errors: {detail}")
    result["scan_errors"] = scan_errors
    return result


def _warm_scancode_api(*, min_license_score: float) -> None:
    resource = _scan_comment_with_scancode_api(
        "MIT License",
        min_license_score=min_license_score,
    )
    detection = _extract_license_detection(
        resource,
        min_license_score=_STACK_V2_MIN_LICENSE_SCORE,
        min_match_coverage=_STACK_V2_MIN_MATCH_COVERAGE,
    )
    if (
        not detection["contains_license_notice"]
        or _license_score_from_detection(detection) < _STACK_V2_MIN_LICENSE_SCORE
    ):
        raise RuntimeError(
            "ScanCode Python API canary failed to detect the MIT License at score 95 or higher"
        )


def _fork_multiprocessing_context() -> multiprocessing.context.BaseContext | None:
    try:
        return multiprocessing.get_context("fork")
    except ValueError:
        return None


def _run_scancode(
    inputs_dir: Path,
    output_path: Path,
    *,
    scancode_command: str,
    scancode_processes: int,
    min_license_score: float,
    runner: Callable[..., dict[str, Any]] | None,
) -> dict[str, Any]:
    if runner is not None:
        return runner(
            inputs_dir=inputs_dir,
            output_path=output_path,
            scancode_command=scancode_command,
            scancode_processes=scancode_processes,
            min_license_score=min_license_score,
        )

    command = [scancode_command, "--quiet", "--license", "--json-pp", str(output_path)]
    command.extend(["--processes", str(scancode_processes)])
    command.append(str(inputs_dir))

    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ScanCode CLI was not found. Install 'scancode-toolkit' or pass --scancode with the executable path."
        ) from exc

    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        detail = stderr or stdout or "no output"
        raise RuntimeError(f"ScanCode failed with exit code {completed.returncode}: {detail}")

    return json.loads(output_path.read_text(encoding="utf-8"))


def _resource_results_by_filename(scan_result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    resources = scan_result.get("files")
    if resources is None:
        resources = scan_result.get("resources", [])
    result: dict[str, dict[str, Any]] = {}
    for resource in resources or []:
        if not isinstance(resource, dict):
            continue
        path = resource.get("path")
        if path is None:
            continue
        result[Path(str(path)).name] = resource
    return result


def _extract_license_detection(
    resource: dict[str, Any] | None,
    *,
    min_license_score: float,
    min_match_coverage: float,
) -> dict[str, Any]:
    if resource is None:
        return {
            "contains_license_notice": False,
            "detected_license_expression": None,
            "detected_license_expression_spdx": None,
            "percentage_of_license_text": 0,
            "best_license_score": 0.0,
            "license_matches": [],
            "scan_errors": [],
        }

    all_matches = list(_iter_scancode_matches(resource))
    best_license_score = _best_license_score(_iter_scancode_score_matches(resource))
    matching_licenses = _filtered_license_matches(
        all_matches,
        min_license_score=min_license_score,
        min_match_coverage=min_match_coverage,
    )
    contains_license_notice = bool(matching_licenses)
    detected_expression = _resource_field(
        resource,
        "detected_license_expression",
        "detected_license_expressions",
    )
    detected_expression_spdx = _resource_field(
        resource,
        "detected_license_expression_spdx",
        "detected_license_expressions_spdx",
    )

    return {
        "contains_license_notice": contains_license_notice,
        "detected_license_expression": detected_expression if contains_license_notice else None,
        "detected_license_expression_spdx": detected_expression_spdx if contains_license_notice else None,
        "percentage_of_license_text": resource.get("percentage_of_license_text", 0) if contains_license_notice else 0,
        "best_license_score": best_license_score,
        "license_matches": matching_licenses,
        "scan_errors": resource.get("scan_errors", []),
    }


def _filtered_license_matches(
    matches_to_filter: Iterable[dict[str, Any]],
    *,
    min_license_score: float,
    min_match_coverage: float,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for match in matches_to_filter:
        score = float(match.get("score", 0.0) or 0.0)
        coverage = float(match.get("match_coverage", 0.0) or 0.0)
        if score < min_license_score or coverage < min_match_coverage:
            continue
        matches.append(match)
    return matches


def _best_license_score(matches: Iterable[dict[str, Any]]) -> float:
    best_score = 0.0
    for match in matches:
        try:
            score = float(match.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        if score > best_score:
            best_score = score
    return best_score


def _license_score_from_detection(detection: dict[str, Any]) -> float:
    try:
        return float(detection.get("best_license_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _iter_scancode_matches(resource: dict[str, Any]) -> Iterable[dict[str, Any]]:
    legacy_licenses = resource.get("licenses") or []
    for lic in legacy_licenses:
        if not isinstance(lic, dict):
            continue
        matched_rule = lic.get("matched_rule", {}) or {}
        yield {
            "license_expression": lic.get("spdx_license_key") or lic.get("key"),
            "score": float(lic.get("score", 0.0) or 0.0),
            "match_coverage": float(
                lic.get("match_coverage", 0.0) or matched_rule.get("coverage", 0.0) or 0.0
            ),
            "rule_identifier": matched_rule.get("identifier"),
            "matcher": lic.get("matcher"),
            "start_line": lic.get("start_line"),
            "end_line": lic.get("end_line"),
        }

    for detection in resource.get("license_detections", []) or []:
        if not isinstance(detection, dict):
            continue
        detection_expression = detection.get("license_expression")
        detection_spdx = detection.get("license_expression_spdx")
        for match in detection.get("matches", []) or []:
            if not isinstance(match, dict):
                continue
            yield {
                "license_expression": match.get("license_expression") or detection_expression,
                "license_expression_spdx": match.get("license_expression_spdx") or detection_spdx,
                "score": float(match.get("score", 0.0) or 0.0),
                "match_coverage": float(match.get("match_coverage", 0.0) or 0.0),
                "rule_identifier": match.get("rule_identifier"),
                "matcher": match.get("matcher"),
                "start_line": match.get("start_line"),
                "end_line": match.get("end_line"),
            }


def _iter_scancode_score_matches(resource: dict[str, Any]) -> Iterable[dict[str, Any]]:
    yield from _iter_scancode_matches(resource)
    for clue in resource.get("license_clues", []) or []:
        if isinstance(clue, dict):
            yield clue


def _resource_field(resource: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = resource.get(key)
        if value is not None:
            return value
    return None


def _write_hf_license_scan_manifest(
    *,
    input_directory: Path,
    output_directory: Path,
    checkpoint_path: Path,
    stats: LicenseScanStats,
    input_shards: list[Path],
    batch_size: int,
    min_license_score: float,
    min_match_coverage: float,
    scancode_command: str,
    scancode_processes: int,
    scanner_backend: str,
    scan_configuration: dict[str, Any],
    detection_cache_path: Path,
    scancode_headers: list[dict[str, Any]],
    workers: int,
    dataset_filter: list[str],
    language_filter: list[str],
) -> None:
    source_manifest_path = input_directory / "manifest.json"
    source_manifest = None
    if source_manifest_path.exists():
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    manifest_path = output_directory / "manifest.json"
    if not scancode_headers and manifest_path.exists():
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        previous_headers = previous_manifest.get("scancode_headers")
        if (
            previous_manifest.get("scan_configuration") == scan_configuration
            and isinstance(previous_headers, list)
        ):
            scancode_headers = previous_headers

    manifest = {
        "source_directory": str(input_directory),
        "created_at": _utc_now(),
        "records_scanned": stats.records_scanned,
        "records_with_detected_license": stats.records_with_detected_license,
        "records_without_detected_license": stats.records_without_detected_license,
        "shards_processed": stats.shards_processed,
        "shards_skipped": stats.shards_skipped,
        "batches_run": stats.batches_run,
        "batch_size": batch_size,
        "min_license_score": min_license_score,
        "min_match_coverage": min_match_coverage,
        "scancode_command": scancode_command,
        "scancode_processes": scancode_processes,
        "scanner_backend": scanner_backend,
        "scan_configuration": scan_configuration,
        "detection_cache_path": str(detection_cache_path),
        "scancode_headers": scancode_headers,
        "checkpoint_path": str(checkpoint_path),
        "workers": workers,
        "dataset_filter": dataset_filter,
        "language_filter": language_filter,
        "input_shards": sorted(path.relative_to(input_directory).as_posix() for path in input_shards),
        "output_shards": sorted(
            path.relative_to(output_directory).as_posix()
            for path in output_directory.glob("*/*/part-*.parquet")
        ),
        "source_manifest": source_manifest,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
