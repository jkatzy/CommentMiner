from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import logging
import os
from pathlib import Path
import sys
from typing import Sequence

from tqdm.contrib.logging import logging_redirect_tqdm

from .config import PipelineConfig
from .downloader import HuggingFaceDownloader, RedPajamaManifestDownloader
from .extractors import ML4SEOpeningCommentExtractor
from .license_scanner import scan_comment_licenses
from .logging_utils import configure_logging
from .models import ShardedDatasetSource
from .pipeline import run_dataset, run_sharded_dataset
from .sources import RedPajamaGithubSource, TheStackParquetSource

_DEFAULT_WORKERS = max(1, min(2, os.cpu_count() or 1))
_LOGGER = logging.getLogger(__name__)


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
    mine.add_argument("--max-comment-start-row", type=int, default=3)
    mine.add_argument(
        "--max-parser-characters",
        type=int,
        default=None,
        help="Limit the parser input to the first N characters of each file.",
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
        help="Disable per-shard tqdm progress bars during streaming.",
    )
    mine.add_argument(
        "--workers",
        type=int,
        default=_DEFAULT_WORKERS,
        help="Number of shard workers for shard-atomic mining. Values above 1 allow out-of-order shard completion.",
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
        languages = downloader.list_languages(dataset, token=_resolve_token(token_env))
    if not languages:
        print(f"Dataset '{dataset.name}' supports language templating, but no fixed language list is configured.")
        return 0
    print(f"Dataset: {dataset.name}")
    print("Languages:")
    for language in languages:
        print(f"- {language}")
    return 0


def _build_downloader(dataset):
    if dataset.source == "redpajama_manifest":
        return RedPajamaManifestDownloader()
    if dataset.source == "huggingface_hub":
        return HuggingFaceDownloader()
    raise ValueError(f"Dataset '{dataset.name}' source '{dataset.source}' is not supported yet")


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
    downloader = _build_downloader(dataset)
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
    token_env: str | None,
) -> int:
    config = PipelineConfig.from_path(config_path)
    dataset = config.require_dataset(dataset_name)
    downloader = _build_downloader(dataset)
    token = _resolve_token(token_env)
    summary = downloader.download(
        config,
        dataset,
        language=language,
        token=token,
        max_files=max_files,
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
):
    dataset = config.require_dataset(dataset_name)
    token = _resolve_token(token_env)
    if dataset.source == "huggingface_hub" and dataset.resolve_repo_id() == "bigcode/the-stack":
        return dataset, TheStackParquetSource(
            config,
            dataset,
            language=language,
            show_progress=show_progress,
            token=token,
        )
    if dataset.source == "redpajama_manifest":
        return dataset, RedPajamaGithubSource(
            config,
            dataset,
            language=language,
            show_progress=show_progress,
            token=token,
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
    max_comment_start_row: int,
    max_parser_characters: int | None,
    progress_every: int,
    workers: int,
) -> int:
    config = PipelineConfig.from_path(config_path)
    dataset, source = _build_source(
        config,
        dataset_name,
        language=language,
        show_progress=show_progress,
        token_env=token_env,
    )
    logging_context = logging_redirect_tqdm() if show_progress else nullcontext()
    if max_records is None and isinstance(source, ShardedDatasetSource):
        with logging_context:
            stats = run_sharded_dataset(
                source,
                lambda: ML4SEOpeningCommentExtractor(
                    max_start_row=max_comment_start_row,
                    max_input_characters=max_parser_characters,
                ),
                config,
                max_workers=workers,
                progress_every=progress_every,
            )
    else:
        if max_records is not None:
            warning = (
                f"WARNING: --max-records={max_records} uses sequential record-level mining"
                f"{' and ignores --workers' if workers != 1 else ''}"
                " to preserve an exact sample size."
            )
            print(warning, file=sys.stderr)
        extractor = ML4SEOpeningCommentExtractor(
            max_start_row=max_comment_start_row,
            max_input_characters=max_parser_characters,
        )
        with logging_context:
            stats = run_dataset(
                source,
                extractor,
                config,
                max_records=max_records,
                progress_every=progress_every,
            )
    print(f"Dataset: {dataset.name}")
    print(f"Language: {language or 'all'}")
    print(f"Records seen: {stats.records_seen}")
    print(f"Comments written: {stats.comments_written}")
    print(f"Skipped without comment: {stats.skipped_without_comment}")
    print(f"Shards written: {stats.shards_written}")
    print(f"Failed shards: {stats.failed_shards}")
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
                max_comment_start_row=args.max_comment_start_row,
                max_parser_characters=args.max_parser_characters,
                progress_every=args.progress_every,
                workers=args.workers,
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
    except (KeyError, ValueError) as exc:
        parser.exit(status=2, message=f"{exc}\n")
    except RuntimeError as exc:
        parser.exit(status=1, message=f"{exc}\n")

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
