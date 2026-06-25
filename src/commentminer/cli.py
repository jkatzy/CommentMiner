from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import shutil
from typing import Callable, Sequence

from tqdm.contrib.logging import logging_redirect_tqdm

from .aggregation import aggregate_comment_runs
from .config import PipelineConfig
from .downloader import HuggingFaceDownloader
from .deduplication import deduplicate_comment_run
from .export_hf import export_huggingface_dataset
from .extractors import ML4SEOpeningCommentExtractor
from .license_scanner import scan_comment_licenses
from .logging_utils import configure_logging
from .pipeline import run_dataset
from .sources import HuggingFaceParquetSource, StackV2SWHContentSource, UrlListJsonlSource


_DEFAULT_EXTRACTION_WORKERS = 4
_DEFAULT_PREFETCH_FILES = 4
_DEFAULT_DOWNLOAD_WORKERS = 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="commentminer",
        description="Utilities for inspecting CommentMiner pipeline configuration.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging verbosity for long-running operations.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate-config",
        help="Load a config file and print a short summary.",
    )
    validate.add_argument("config", type=Path)

    dump = subparsers.add_parser(
        "dump-config",
        help="Load a config file and print the resolved configuration as JSON.",
    )
    dump.add_argument("config", type=Path)

    list_languages = subparsers.add_parser(
        "list-languages",
        help="List configured language choices for a dataset.",
    )
    list_languages.add_argument("config", type=Path)
    list_languages.add_argument("dataset")
    list_languages.add_argument("--token-env")

    plan_download = subparsers.add_parser(
        "plan-download",
        help="Resolve which Hugging Face files would be downloaded.",
    )
    plan_download.add_argument("config", type=Path)
    plan_download.add_argument("dataset")
    plan_download.add_argument("--language")
    plan_download.add_argument("--max-files", type=int)
    plan_download.add_argument("--show-files", type=int, default=20)
    plan_download.add_argument("--token-env")

    download = subparsers.add_parser(
        "download",
        help="Download files for a configured Hugging Face dataset with checkpointed resume.",
    )
    download.add_argument("config", type=Path)
    download.add_argument("dataset")
    download.add_argument("--language")
    download.add_argument("--max-files", type=int)
    download.add_argument(
        "--download-workers",
        type=int,
        help="Concurrent file download workers.",
    )
    download.add_argument("--token-env")

    mine = subparsers.add_parser(
        "mine-dataset",
        help="Run comment mining for a supported configured dataset.",
    )
    mine.add_argument("config", type=Path)
    mine.add_argument("dataset")
    mine.add_argument("--language")
    mine.add_argument("--token-env")
    mine.add_argument("--max-records", type=int)
    mine.add_argument("--max-files", type=int)
    mine.add_argument("--max-comment-start-row", type=int, default=10)
    mine.add_argument(
        "--prefetch-files",
        type=int,
        help="Maximum parquet source files kept downloaded or in flight for this run.",
    )
    mine.add_argument(
        "--download-workers",
        type=int,
        help="Concurrent parquet file download workers.",
    )
    mine.add_argument(
        "--content-download-workers",
        type=int,
        help="Concurrent Stack v2 S3 content download threads.",
    )
    mine.add_argument(
        "--content-prefetch-records",
        type=int,
        help="Maximum Stack v2 rows with queued or in-flight S3 content downloads.",
    )
    mine.add_argument(
        "--extraction-workers",
        type=int,
        help="Concurrent comment extraction worker threads.",
    )
    mine.add_argument(
        "--extraction-buffer",
        type=int,
        help="Maximum records queued for comment extraction.",
    )
    mine.add_argument(
        "--cache-source-files",
        action="store_true",
        help="Use the Hugging Face cache for source files instead of direct scratch downloads.",
    )
    mine.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Emit a mining progress log every N records.",
    )
    mine.add_argument(
        "--no-tqdm",
        action="store_true",
        help="Disable per-shard tqdm progress bars during parquet streaming.",
    )

    aggregate = subparsers.add_parser(
        "aggregate-comment-runs",
        help="Combine extracted comment runs into one aggregated dataset before downstream processing.",
    )
    aggregate.add_argument("input_directories", type=Path, nargs="+")
    aggregate.add_argument("--output-root", type=Path)
    aggregate.add_argument(
        "--dataset-name",
        default="combined-comments",
        help="Dataset name to assign to the aggregated output run.",
    )
    aggregate.add_argument(
        "--source-field",
        default="source_dataset",
        help="Field name used to store the original source dataset on each aggregated record.",
    )
    aggregate.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Emit an aggregation progress log every N records.",
    )

    deduplicate = subparsers.add_parser(
        "deduplicate-comment-run",
        help="Deduplicate an aggregated comment run before downstream processing.",
    )
    deduplicate.add_argument("input_directory", type=Path)
    deduplicate.add_argument("--output-root", type=Path)
    deduplicate.add_argument(
        "--dataset-name",
        help="Dataset name to assign to the deduplicated output run. Defaults to <input-dataset>-deduplicated.",
    )
    deduplicate.add_argument(
        "--source-field",
        default="source_dataset",
        help="Field name used to identify the original source dataset in each occurrence record.",
    )
    deduplicate.add_argument(
        "--hash-workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Number of worker threads to use while normalizing and hashing comments.",
    )
    deduplicate.add_argument(
        "--hash-batch-size",
        type=int,
        default=1000,
        help="Number of comments to hash per batch before writing to the temporary sort input.",
    )
    deduplicate.add_argument(
        "--sort-parallelism",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Parallelism hint passed to the external sort command.",
    )
    deduplicate.add_argument(
        "--sort-command",
        default="sort",
        help="External sort executable used to group identical comment hashes.",
    )
    deduplicate.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Emit a deduplication progress log every N records or groups.",
    )

    scan_licenses = subparsers.add_parser(
        "scan-comment-licenses",
        help="Scan previously extracted comment shards for license notices using ScanCode.",
    )
    scan_licenses.add_argument("input_directory", type=Path)
    scan_licenses.add_argument("--output-directory", type=Path)
    scan_licenses.add_argument("--scancode", default="scancode")
    scan_licenses.add_argument("--batch-size", type=int, default=500)
    scan_licenses.add_argument(
        "--min-license-score",
        type=float,
        default=95.0,
        help="Minimum ScanCode score for a match to count as a license hit. Matches the Stack v2 pipeline default.",
    )
    scan_licenses.add_argument(
        "--min-match-coverage",
        type=float,
        default=95.0,
        help="Minimum ScanCode match coverage for a match to count as a license hit. Matches the Stack v2 pipeline default.",
    )
    scan_licenses.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Emit a license scan progress log every N records.",
    )

    mine_config = subparsers.add_parser(
        "mine-config",
        help="Mine every enabled dataset/language slice in a bounded config-driven matrix.",
    )
    mine_config.add_argument("config", type=Path)
    mine_config.add_argument("--datasets", help="Comma-separated dataset names to include.")
    mine_config.add_argument("--languages", help="Comma-separated language names to include.")
    mine_config.add_argument("--token-env")
    mine_config.add_argument("--max-languages", type=int)
    mine_config.add_argument("--max-files-per-language", type=int, default=1)
    mine_config.add_argument("--max-records-per-language", type=int)
    mine_config.add_argument("--max-comment-start-row", type=int, default=10)
    mine_config.add_argument("--prefetch-files", type=int, default=_DEFAULT_PREFETCH_FILES)
    mine_config.add_argument("--download-workers", type=int, default=_DEFAULT_DOWNLOAD_WORKERS)
    mine_config.add_argument("--content-download-workers", type=int)
    mine_config.add_argument("--content-prefetch-records", type=int)
    mine_config.add_argument("--extraction-workers", type=int, default=_DEFAULT_EXTRACTION_WORKERS)
    mine_config.add_argument("--extraction-buffer", type=int)
    mine_config.add_argument("--cache-source-files", action="store_true")
    mine_config.add_argument("--progress-every", type=int, default=1000)
    mine_config.add_argument("--no-tqdm", action="store_true")
    mine_config.add_argument(
        "--skip-errors",
        action="store_true",
        help="Continue with the next slice when one dataset/language fails.",
    )

    export_hf = subparsers.add_parser(
        "export-hf-dataset",
        help="Materialize mined comment shards as a single Hugging Face upload tree.",
    )
    export_hf.add_argument("config", type=Path)
    export_hf.add_argument(
        "output_directory",
        type=Path,
        help="Destination root, for example var/comment-dataset.",
    )
    export_hf.add_argument(
        "--input-directory",
        type=Path,
        help="Mined shard root. Defaults to storage.output_directory from the config.",
    )
    export_hf.add_argument(
        "--dedupe-record-ids",
        action="store_true",
        help="Skip duplicate dataset+record_id rows while exporting.",
    )
    export_hf.add_argument(
        "--format",
        choices=["parquet", "jsonl"],
        default="parquet",
        help="Final shard format. Defaults to parquet for Hugging Face upload.",
    )
    export_hf.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the destination directory before exporting.",
    )
    export_hf.add_argument("--max-records-per-shard", type=int)
    export_hf.add_argument("--max-bytes-per-shard", type=int)
    export_hf.add_argument("--max-open-writers", type=int, default=64)

    return parser


def _validate_config(config_path: Path) -> int:
    config = PipelineConfig.from_path(config_path)
    enabled_sources = [spec.name for spec in config.datasets if spec.enabled]
    print(f"Config: {config_path}")
    print(f"Datasets: {len(config.datasets)} total, {len(enabled_sources)} enabled")
    print(f"Output directory: {config.storage.output_directory}")
    print(f"Checkpoint directory: {config.storage.checkpoint_directory}")
    print(f"Download directory: {config.storage.download_directory}")
    print(f"Hugging Face cache: {config.storage.huggingface_cache_directory}")
    if enabled_sources:
        print("Enabled sources:")
        for name in enabled_sources:
            print(f"- {name}")
    return 0


def _dump_config(config_path: Path) -> int:
    config = PipelineConfig.from_path(config_path)
    print(json.dumps(config.to_dict(), indent=2))
    return 0


def _resolve_token(token_env: str | None) -> str | bool | None:
    if token_env is None:
        return None
    token = os.environ.get(token_env)
    if not token:
        raise ValueError(f"Environment variable '{token_env}' is not set")
    return token


def _positive_int_option(*values: object, default: int) -> int:
    for value in values:
        if value is None:
            continue
        result = int(value)
        if result < 1:
            raise ValueError(f"Expected positive integer, got {result}")
        return result
    return default


def _format_size(size_bytes: int | None) -> str:
    if size_bytes is None:
        return "unknown"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{size_bytes} B"


def _list_languages(
    config_path: Path,
    dataset_name: str,
    *,
    token_env: str | None,
) -> int:
    config = PipelineConfig.from_path(config_path)
    dataset = config.require_dataset(dataset_name)
    if not dataset.supports_language_selection():
        print(f"Dataset '{dataset.name}' does not define language-aware download selection.")
        return 0
    languages = dataset.available_languages()
    if not languages and dataset.source == "huggingface_hub":
        downloader = HuggingFaceDownloader()
        languages = downloader.list_languages(
            dataset,
            token=_resolve_token(token_env),
            cache_directory=downloader.remote_file_cache_directory(config),
        )
    if not languages:
        print(f"Dataset '{dataset.name}' supports language templating, but no fixed language list is configured.")
        return 0
    print(f"Dataset: {dataset.name}")
    print("Languages:")
    for language in languages:
        print(f"- {language}")
    return 0


def _plan_download(
    config_path: Path,
    dataset_name: str,
    *,
    language: str | None,
    max_files: int | None,
    show_files: int,
    token_env: str | None,
) -> int:
    config = PipelineConfig.from_path(config_path)
    dataset = config.require_dataset(dataset_name)
    downloader = HuggingFaceDownloader()
    token = _resolve_token(token_env)
    plan = downloader.plan_download(
        config,
        dataset,
        language=language,
        token=token,
        max_files=max_files,
    )
    print(f"Dataset: {plan.dataset}")
    print(f"Repo: {plan.repo_id} @ {plan.revision}")
    print(f"Language: {plan.language or 'all'}")
    print(f"Download root: {plan.download_root}")
    print(f"Cache directory: {plan.cache_directory}")
    print(f"Checkpoint: {plan.checkpoint_path}")
    print(f"Matched files: {plan.matched_count} ({_format_size(plan.matched_bytes)})")
    print(f"Already downloaded: {plan.completed_count}")
    print(f"Pending download: {plan.pending_count} ({_format_size(plan.pending_bytes)})")
    if plan.allow_patterns:
        print("Allow patterns:")
        for pattern in plan.allow_patterns:
            print(f"- {pattern}")
    if plan.ignore_patterns:
        print("Ignore patterns:")
        for pattern in plan.ignore_patterns:
            print(f"- {pattern}")
    if plan.pending_files and show_files > 0:
        print("Pending files:")
        for remote in plan.pending_files[:show_files]:
            print(f"- {remote.path}")
    return 0


def _download_dataset(
    config_path: Path,
    dataset_name: str,
    *,
    language: str | None,
    max_files: int | None,
    download_workers: int | None,
    token_env: str | None,
) -> int:
    config = PipelineConfig.from_path(config_path)
    dataset = config.require_dataset(dataset_name)
    downloader = HuggingFaceDownloader()
    token = _resolve_token(token_env)
    summary = downloader.download(
        config,
        dataset,
        language=language,
        token=token,
        max_files=max_files,
        download_workers=download_workers,
    )
    print(f"Dataset: {summary.dataset}")
    print(f"Repo: {summary.repo_id} @ {summary.revision}")
    print(f"Language: {summary.language or 'all'}")
    print(f"Download root: {summary.download_root}")
    print(f"Checkpoint: {summary.checkpoint_path}")
    print(f"Matched files: {summary.matched_count}")
    print(f"Already downloaded: {summary.already_downloaded_count}")
    print(f"Downloaded now: {summary.downloaded_count}")
    return 0


def _build_source(
    config: PipelineConfig,
    dataset_name: str,
    *,
    language: str | None,
    show_progress: bool,
    token_env: str | None,
    max_files: int | None = None,
    prefetch_files: int | None = None,
    download_workers: int | None = None,
    content_download_workers: int | None = None,
    content_prefetch_records: int | None = None,
    content_language_filter: Callable[[str], bool] | None = None,
    cache_source_files: bool | None = None,
):
    dataset = config.require_dataset(dataset_name)
    token = _resolve_token(token_env)
    if dataset.source == "huggingface_hub":
        source_class = (
            StackV2SWHContentSource
            if dataset.extra.get("content_backend") == "softwareheritage_s3"
            else HuggingFaceParquetSource
        )
        source_kwargs = {
            "language": language,
            "show_progress": show_progress,
            "token": token,
            "max_files": max_files,
            "prefetch_files": prefetch_files,
            "download_workers": download_workers,
            "cache_source_files": cache_source_files,
        }
        if source_class is StackV2SWHContentSource:
            source_kwargs["content_download_workers"] = content_download_workers
            source_kwargs["content_prefetch_records"] = content_prefetch_records
            source_kwargs["content_language_filter"] = content_language_filter
        return dataset, source_class(config, dataset, **source_kwargs)
    if dataset.source == "url_list_jsonl":
        return dataset, UrlListJsonlSource(
            config,
            dataset,
            language=language,
            show_progress=show_progress,
            token=token,
            max_files=max_files,
        )
    raise ValueError(
        f"Dataset '{dataset.name}' is not supported by the mining command yet"
    )


def _mine_dataset(
    config_path: Path,
    dataset_name: str,
    *,
    language: str | None,
    show_progress: bool,
    token_env: str | None,
    max_records: int | None,
    max_files: int | None,
    max_comment_start_row: int,
    prefetch_files: int | None,
    download_workers: int | None,
    content_download_workers: int | None,
    content_prefetch_records: int | None,
    extraction_workers: int | None,
    extraction_buffer: int | None,
    cache_source_files: bool,
    progress_every: int,
) -> int:
    config = PipelineConfig.from_path(config_path)
    extractor = ML4SEOpeningCommentExtractor(max_start_row=max_comment_start_row)
    dataset, source = _build_source(
        config,
        dataset_name,
        language=language,
        show_progress=show_progress,
        token_env=token_env,
        max_files=max_files,
        prefetch_files=prefetch_files,
        download_workers=download_workers,
        content_download_workers=content_download_workers,
        content_prefetch_records=content_prefetch_records,
        content_language_filter=extractor.supports_language_value,
        cache_source_files=cache_source_files,
    )
    resolved_extraction_workers = _positive_int_option(
        extraction_workers,
        dataset.extra.get("extraction_workers"),
        default=_DEFAULT_EXTRACTION_WORKERS,
    )
    resolved_extraction_buffer = _positive_int_option(
        extraction_buffer,
        dataset.extra.get("extraction_buffer"),
        default=resolved_extraction_workers * 4,
    )
    logging_context = logging_redirect_tqdm() if show_progress else nullcontext()
    with logging_context:
        stats = run_dataset(
            source,
            extractor,
            config,
            max_records=max_records,
            progress_every=progress_every,
            extraction_workers=resolved_extraction_workers,
            extraction_buffer=resolved_extraction_buffer,
        )
    print(f"Dataset: {dataset.name}")
    print(f"Language: {language or 'all'}")
    print(f"Records seen: {stats.records_seen}")
    print(f"Comments written: {stats.comments_written}")
    print(f"Skipped without comment: {stats.skipped_without_comment}")
    print(f"Shards written: {stats.shards_written}")
    return 0


def _scan_comment_licenses(
    input_directory: Path,
    *,
    output_directory: Path | None,
    scancode: str,
    batch_size: int,
    min_license_score: float,
    min_match_coverage: float,
    progress_every: int,
) -> int:
    stats = scan_comment_licenses(
        input_directory,
        output_directory=output_directory,
        scancode_command=scancode,
        batch_size=batch_size,
        min_license_score=min_license_score,
        min_match_coverage=min_match_coverage,
        progress_every=progress_every,
    )
    print(f"Input directory: {stats.input_directory}")
    print(f"Output directory: {stats.output_directory}")
    print(f"Records scanned: {stats.records_scanned}")
    print(f"Records with detected license: {stats.records_with_detected_license}")
    print(f"Records without detected license: {stats.records_without_detected_license}")
    print(f"Shards processed: {stats.shards_processed}")
    print(f"Shards skipped: {stats.shards_skipped}")
    print(f"Batches run: {stats.batches_run}")
    return 0


def _aggregate_comment_runs(
    input_directories: Sequence[Path],
    *,
    output_root: Path | None,
    dataset_name: str,
    source_field: str,
    progress_every: int,
) -> int:
    stats = aggregate_comment_runs(
        input_directories,
        output_root=output_root,
        dataset_name=dataset_name,
        source_field=source_field,
        progress_every=progress_every,
    )
    print(f"Dataset: {stats.dataset_name}")
    print(f"Output directory: {stats.output_directory}")
    print(f"Source datasets: {', '.join(stats.source_datasets)}")
    print(f"Input runs: {len(stats.input_directories)}")
    print(f"Records aggregated: {stats.records_aggregated}")
    print(f"Shards written: {stats.shards_written}")
    return 0


def _deduplicate_comment_run(
    input_directory: Path,
    *,
    output_root: Path | None,
    dataset_name: str | None,
    source_field: str,
    hash_workers: int,
    hash_batch_size: int,
    sort_parallelism: int,
    sort_command: str,
    progress_every: int,
) -> int:
    stats = deduplicate_comment_run(
        input_directory,
        output_root=output_root,
        dataset_name=dataset_name,
        source_field=source_field,
        hash_workers=hash_workers,
        hash_batch_size=hash_batch_size,
        sort_parallelism=sort_parallelism,
        sort_command=sort_command,
        progress_every=progress_every,
    )
    print(f"Input directory: {stats.input_directory}")
    print(f"Input dataset: {stats.input_dataset_name}")
    print(f"Dataset: {stats.dataset_name}")
    print(f"Output directory: {stats.output_directory}")
    print(f"Records seen: {stats.records_seen}")
    print(f"Unique comments: {stats.unique_comments}")
    print(f"Duplicate occurrences: {stats.duplicate_occurrences}")
    print(f"Shards written: {stats.shards_written}")
    return 0


def _split_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _mine_config(
    config_path: Path,
    *,
    dataset_names: list[str] | None,
    languages: list[str] | None,
    token_env: str | None,
    max_languages: int | None,
    max_files_per_language: int,
    max_records_per_language: int | None,
    max_comment_start_row: int,
    prefetch_files: int,
    download_workers: int,
    content_download_workers: int | None,
    content_prefetch_records: int | None,
    extraction_workers: int,
    extraction_buffer: int | None,
    cache_source_files: bool,
    progress_every: int,
    show_progress: bool,
    skip_errors: bool,
) -> int:
    config = PipelineConfig.from_path(config_path)
    downloader = HuggingFaceDownloader()
    extractor = ML4SEOpeningCommentExtractor(max_start_row=max_comment_start_row)
    requested_datasets = set(dataset_names) if dataset_names is not None else None
    summaries: list[tuple[str, str | None, int, int]] = []

    for dataset in config.datasets:
        if not dataset.enabled:
            continue
        if requested_datasets is not None and dataset.name not in requested_datasets:
            continue

        dataset_languages: list[str | None]
        if languages is not None:
            dataset_languages = list(languages)
        elif dataset.supports_language_selection():
            configured_languages = dataset.available_languages()
            dataset_languages = configured_languages or downloader.list_languages(
                dataset,
                token=_resolve_token(token_env),
                cache_directory=downloader.remote_file_cache_directory(config),
            )
        else:
            dataset_languages = [None]

        if max_languages is not None:
            dataset_languages = dataset_languages[:max_languages]
        if not dataset_languages:
            dataset_languages = [None]

        for language in dataset_languages:
            try:
                _, source = _build_source(
                    config,
                    dataset.name,
                    language=language,
                    show_progress=show_progress,
                    token_env=token_env,
                    max_files=max_files_per_language,
                    prefetch_files=prefetch_files,
                    download_workers=download_workers,
                    content_download_workers=content_download_workers,
                    content_prefetch_records=content_prefetch_records,
                    content_language_filter=extractor.supports_language_value,
                    cache_source_files=cache_source_files,
                )
                logging_context = logging_redirect_tqdm() if show_progress else nullcontext()
                with logging_context:
                    resolved_extraction_workers = _positive_int_option(
                        extraction_workers,
                        dataset.extra.get("extraction_workers"),
                        default=_DEFAULT_EXTRACTION_WORKERS,
                    )
                    resolved_extraction_buffer = _positive_int_option(
                        extraction_buffer,
                        dataset.extra.get("extraction_buffer"),
                        default=resolved_extraction_workers * 4,
                    )
                    stats = run_dataset(
                        source,
                        extractor,
                        config,
                        max_records=max_records_per_language,
                        progress_every=progress_every,
                        extraction_workers=resolved_extraction_workers,
                        extraction_buffer=resolved_extraction_buffer,
                    )
                summaries.append(
                    (
                        dataset.name,
                        language,
                        stats.records_seen,
                        stats.comments_written,
                    )
                )
            except Exception:
                if not skip_errors:
                    raise
                summaries.append((dataset.name, language, -1, -1))

    print("Mining summary:")
    for dataset_name, language, records_seen, comments_written in summaries:
        status = "failed" if records_seen < 0 else f"{records_seen} records, {comments_written} comments"
        print(f"- {dataset_name} / {language or 'all'}: {status}")
    return 0


def _export_hf_dataset(
    config_path: Path,
    output_directory: Path,
    *,
    input_directory: Path | None,
    dedupe_record_ids: bool,
    output_format: str,
    overwrite: bool,
    max_records_per_shard: int | None,
    max_bytes_per_shard: int | None,
    max_open_writers: int,
) -> int:
    config = PipelineConfig.from_path(config_path)
    source_directory = input_directory or config.storage.output_directory
    if overwrite and output_directory.exists():
        shutil.rmtree(output_directory)
    if output_directory.exists() and any(output_directory.iterdir()):
        raise ValueError(
            f"Output directory '{output_directory}' already exists and is not empty. Use --overwrite to replace it."
        )
    stats = export_huggingface_dataset(
        source_directory,
        output_directory,
        output_format=output_format,
        max_records_per_shard=max_records_per_shard or config.storage.max_records_per_shard,
        max_bytes_per_shard=max_bytes_per_shard or config.storage.max_bytes_per_shard,
        dedupe_record_ids=dedupe_record_ids,
        max_open_writers=max_open_writers,
    )
    print(f"Input directory: {source_directory}")
    print(f"Output directory: {stats.output_directory}")
    print(f"Format: {output_format}")
    print(f"Records written: {stats.records_written}")
    print(f"Duplicates skipped: {stats.records_skipped_duplicate}")
    print(f"Groups: {len(stats.groups)}")
    for group in sorted(stats.groups.values(), key=lambda item: (item.dataset, item.language)):
        print(f"- {group.dataset} / {group.language}: {group.records} records")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    try:
        if args.command == "validate-config":
            return _validate_config(args.config)
        if args.command == "dump-config":
            return _dump_config(args.config)
        if args.command == "list-languages":
            return _list_languages(
                args.config,
                args.dataset,
                token_env=args.token_env,
            )
        if args.command == "plan-download":
            return _plan_download(
                args.config,
                args.dataset,
                language=args.language,
                max_files=args.max_files,
                show_files=args.show_files,
                token_env=args.token_env,
            )
        if args.command == "download":
            return _download_dataset(
                args.config,
                args.dataset,
                language=args.language,
                max_files=args.max_files,
                download_workers=args.download_workers,
                token_env=args.token_env,
            )
        if args.command == "mine-dataset":
            return _mine_dataset(
                args.config,
                args.dataset,
                language=args.language,
                show_progress=not args.no_tqdm,
                token_env=args.token_env,
                max_records=args.max_records,
                max_files=args.max_files,
                max_comment_start_row=args.max_comment_start_row,
                prefetch_files=args.prefetch_files,
                download_workers=args.download_workers,
                content_download_workers=args.content_download_workers,
                content_prefetch_records=args.content_prefetch_records,
                extraction_workers=args.extraction_workers,
                extraction_buffer=args.extraction_buffer,
                cache_source_files=args.cache_source_files,
                progress_every=args.progress_every,
            )
        if args.command == "aggregate-comment-runs":
            return _aggregate_comment_runs(
                args.input_directories,
                output_root=args.output_root,
                dataset_name=args.dataset_name,
                source_field=args.source_field,
                progress_every=args.progress_every,
            )
        if args.command == "deduplicate-comment-run":
            return _deduplicate_comment_run(
                args.input_directory,
                output_root=args.output_root,
                dataset_name=args.dataset_name,
                source_field=args.source_field,
                hash_workers=args.hash_workers,
                hash_batch_size=args.hash_batch_size,
                sort_parallelism=args.sort_parallelism,
                sort_command=args.sort_command,
                progress_every=args.progress_every,
            )
        if args.command == "scan-comment-licenses":
            return _scan_comment_licenses(
                args.input_directory,
                output_directory=args.output_directory,
                scancode=args.scancode,
                batch_size=args.batch_size,
                min_license_score=args.min_license_score,
                min_match_coverage=args.min_match_coverage,
                progress_every=args.progress_every,
            )
        if args.command == "mine-config":
            return _mine_config(
                args.config,
                dataset_names=_split_csv(args.datasets),
                languages=_split_csv(args.languages),
                token_env=args.token_env,
                max_languages=args.max_languages,
                max_files_per_language=args.max_files_per_language,
                max_records_per_language=args.max_records_per_language,
                max_comment_start_row=args.max_comment_start_row,
                prefetch_files=args.prefetch_files,
                download_workers=args.download_workers,
                content_download_workers=args.content_download_workers,
                content_prefetch_records=args.content_prefetch_records,
                extraction_workers=args.extraction_workers,
                extraction_buffer=args.extraction_buffer,
                cache_source_files=args.cache_source_files,
                progress_every=args.progress_every,
                show_progress=not args.no_tqdm,
                skip_errors=args.skip_errors,
            )
        if args.command == "export-hf-dataset":
            return _export_hf_dataset(
                args.config,
                args.output_directory,
                input_directory=args.input_directory,
                dedupe_record_ids=args.dedupe_record_ids,
                output_format=args.format,
                overwrite=args.overwrite,
                max_records_per_shard=args.max_records_per_shard,
                max_bytes_per_shard=args.max_bytes_per_shard,
                max_open_writers=args.max_open_writers,
            )
    except (KeyError, ValueError) as exc:
        parser.exit(status=2, message=f"{exc}\n")
    except RuntimeError as exc:
        parser.exit(status=1, message=f"{exc}\n")

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
