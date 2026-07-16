from __future__ import annotations

import argparse
import csv
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import time
from typing import Sequence

import pyarrow.parquet as pq

from commentminer.config import PipelineConfig, StorageConfig
from commentminer.downloader import HuggingFaceDownloader, RemoteFile
from commentminer.sources import (
    AiohttpSoftwareHeritageContentFetcher,
    _raise_fd_limit_for_stack_v2_content_workers,
    _stack_v2_content_fetcher_from_dataset,
)
from commentminer.stackv2_packages import (
    StackV2IdPackage,
    StackV2IdSegment,
    StackV2SWHContentPackageSource,
)


DEFAULT_LANGUAGES = [
    "Python",
    "JavaScript",
    "TypeScript",
    "Java",
    "C++",
    "Rust",
    "Go",
    "Dockerfile",
    "AMPL",
    "Procfile",
]


@dataclass(frozen=True, slots=True)
class LanguagePlan:
    language: str
    files: int
    ids: int


@dataclass(frozen=True, slots=True)
class MatrixResult:
    package_workers: int
    content_download_workers: int
    content_prefetch_records: int
    packages: int
    ids_planned: int
    records_seen: int
    fetched: int
    missing: int
    unsupported: int
    failed: int
    seconds: float
    fetched_per_second: float
    records_per_second: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Stack v2 package workers and S3 content download workers.",
    )
    parser.add_argument("--config", type=Path, default=Path("config/pipeline.example.json"))
    parser.add_argument("--dataset", default="the-stack-v2")
    parser.add_argument(
        "--languages",
        default=",".join(DEFAULT_LANGUAGES),
        help="Comma-separated Stack v2 languages to sample.",
    )
    parser.add_argument("--package-workers", default="1,2,4,8")
    parser.add_argument(
        "--package-worker-backend",
        choices=["thread", "process"],
        default="thread",
    )
    parser.add_argument("--content-download-workers", default="64,128,256,512,1024,2048")
    parser.add_argument("--package-size", type=int, default=1_000)
    parser.add_argument("--max-rows-per-language", type=int, default=1_000)
    parser.add_argument("--max-files-per-language", type=int, default=2)
    parser.add_argument("--metadata-download-workers", type=int, default=8)
    parser.add_argument(
        "--swh-content-url-template",
        help="Override Stack v2 content URL template, e.g. https://softwareheritage.s3.amazonaws.com/content/{blob_id}.",
    )
    parser.add_argument(
        "--swh-content-compression",
        help="Override Stack v2 content compression. Defaults to dataset config.",
    )
    parser.add_argument(
        "--swh-content-decode-workers",
        type=int,
        help="Override Stack v2 aiohttp gzip/decode worker count.",
    )
    parser.add_argument(
        "--swh-content-decode-executor",
        choices=["inline", "thread", "process"],
        help="Override Stack v2 aiohttp gzip/decode executor.",
    )
    parser.add_argument(
        "--content-prefetch-multiplier",
        type=int,
        default=1,
        help="Per-package queued rows multiplier relative to content workers.",
    )
    parser.add_argument("--token-env")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--skip-errors", action="store_true")
    parser.add_argument(
        "--shuffle-packages",
        action="store_true",
        help="Interleave package order by taking every Nth package; useful for long partial sweeps.",
    )
    parser.add_argument(
        "--limit-combos",
        type=int,
        help="Run only the first N matrix combinations after parsing worker lists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_config = PipelineConfig.from_path(args.config)
    dataset = base_config.require_dataset(args.dataset)
    if (
        args.swh_content_url_template
        or args.swh_content_compression
        or args.swh_content_decode_workers is not None
        or args.swh_content_decode_executor
    ):
        dataset.extra = dict(dataset.extra)
        if args.swh_content_url_template:
            dataset.extra["swh_content_url_template"] = args.swh_content_url_template
        if args.swh_content_compression:
            dataset.extra["swh_content_compression"] = args.swh_content_compression
        if args.swh_content_decode_workers is not None:
            dataset.extra["swh_content_decode_workers"] = args.swh_content_decode_workers
        if args.swh_content_decode_executor:
            dataset.extra["swh_content_decode_executor"] = args.swh_content_decode_executor
    root = args.root or (
        Path("var")
        / "benchmarks"
        / "stack-v2-worker-matrix"
        / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    root.mkdir(parents=True, exist_ok=True)
    config = _benchmark_config(base_config, root)
    token = _resolve_token(args.token_env)
    downloader = HuggingFaceDownloader()
    languages = _csv_strings(args.languages)
    package_worker_values = _csv_ints(args.package_workers)
    content_worker_values = _csv_ints(args.content_download_workers)
    combos = [
        (package_workers, content_workers)
        for package_workers in package_worker_values
        for content_workers in content_worker_values
    ]
    if args.limit_combos is not None:
        combos = combos[: args.limit_combos]

    packages, language_plans = _plan_packages(
        config,
        dataset,
        downloader=downloader,
        token=token,
        languages=languages,
        package_size=args.package_size,
        max_rows_per_language=args.max_rows_per_language,
        max_files_per_language=args.max_files_per_language,
        metadata_download_workers=args.metadata_download_workers,
    )
    if args.shuffle_packages:
        packages = _interleave_packages(packages, max(package_worker_values))
    plan_path = root / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
                "config": str(args.config),
                "dataset": dataset.name,
                "repo_id": dataset.resolve_repo_id(),
                "revision": dataset.revision,
                "languages": [asdict(item) for item in language_plans],
                "package_size": args.package_size,
                "packages": len(packages),
                "ids": sum(package.id_count for package in packages),
                "package_workers": package_worker_values,
                "package_worker_backend": args.package_worker_backend,
                "content_download_workers": content_worker_values,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    results_path = root / "results.csv"
    with results_path.open("w", newline="", encoding="utf-8") as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=list(MatrixResult.__dataclass_fields__))
        writer.writeheader()
        for package_workers, content_workers in combos:
            content_prefetch_records = max(
                content_workers,
                content_workers * args.content_prefetch_multiplier,
            )
            print(
                "RUN",
                f"package_workers={package_workers}",
                f"content_download_workers={content_workers}",
                f"content_prefetch_records={content_prefetch_records}",
                f"packages={len(packages)}",
                f"ids={sum(package.id_count for package in packages)}",
                flush=True,
            )
            result = _run_matrix_item(
                config,
                dataset,
                packages,
                downloader=downloader,
                token=token,
                package_workers=package_workers,
                content_download_workers=content_workers,
                content_prefetch_records=content_prefetch_records,
                package_worker_backend=args.package_worker_backend,
                skip_errors=args.skip_errors,
            )
            writer.writerow(asdict(result))
            csv_handle.flush()
            print(
                "RESULT",
                f"package_workers={result.package_workers}",
                f"content_download_workers={result.content_download_workers}",
                f"fetched_per_second={result.fetched_per_second:.2f}",
                f"records_per_second={result.records_per_second:.2f}",
                f"seconds={result.seconds:.3f}",
                f"fetched={result.fetched}",
                f"missing={result.missing}",
                f"failed={result.failed}",
                flush=True,
            )

    best = _best_result(results_path)
    print(f"Plan: {plan_path}")
    print(f"Results: {results_path}")
    if best is not None:
        print(
            "Best:",
            f"package_workers={best.package_workers}",
            f"content_download_workers={best.content_download_workers}",
            f"fetched_per_second={best.fetched_per_second:.2f}",
        )
    return 0


def _benchmark_config(base_config: PipelineConfig, root: Path) -> PipelineConfig:
    return PipelineConfig(
        storage=StorageConfig(
            working_directory=base_config.storage.working_directory,
            output_directory=root / "output",
            checkpoint_directory=root / "checkpoints",
            download_directory=root / "downloads",
            huggingface_cache_directory=base_config.storage.huggingface_cache_directory,
            max_records_per_shard=base_config.storage.max_records_per_shard,
            max_bytes_per_shard=base_config.storage.max_bytes_per_shard,
        ),
        datasets=base_config.datasets,
        checkpoint_interval_records=base_config.checkpoint_interval_records,
    )


def _plan_packages(
    config: PipelineConfig,
    dataset,
    *,
    downloader: HuggingFaceDownloader,
    token: str | bool | None,
    languages: Sequence[str],
    package_size: int,
    max_rows_per_language: int,
    max_files_per_language: int,
    metadata_download_workers: int,
) -> tuple[list[StackV2IdPackage], list[LanguagePlan]]:
    segments: list[StackV2IdSegment] = []
    language_plans: list[LanguagePlan] = []

    for language in languages:
        plan = downloader.plan_download(
            config,
            dataset,
            language=language,
            token=token,
            max_files=max_files_per_language,
            checkpoint_namespace="benchmark-stack-v2-metadata",
        )
        refs = [(language, remote) for remote in plan.pending_files]
        row_counts = _metadata_row_counts(
            config,
            dataset,
            downloader=downloader,
            token=token,
            refs=refs,
            workers=metadata_download_workers,
        )
        remaining = max_rows_per_language
        files_used = 0
        ids_used = 0
        for remote, rows in row_counts:
            if remaining <= 0:
                break
            take = min(rows, remaining)
            if take <= 0:
                continue
            segments.append(
                StackV2IdSegment(
                    language=language,
                    remote=remote,
                    start_row=0,
                    end_row=take,
                )
            )
            files_used += 1
            ids_used += take
            remaining -= take
        language_plans.append(LanguagePlan(language=language, files=files_used, ids=ids_used))
        print(
            "LANGUAGE",
            f"language={language}",
            f"files={files_used}",
            f"ids={ids_used}",
            flush=True,
        )

    return _packages_from_segments(segments, package_size=package_size), language_plans


def _metadata_row_counts(
    config: PipelineConfig,
    dataset,
    *,
    downloader: HuggingFaceDownloader,
    token: str | bool | None,
    refs: Sequence[tuple[str, RemoteFile]],
    workers: int,
) -> list[tuple[RemoteFile, int]]:
    def load(ref: tuple[str, RemoteFile]) -> tuple[RemoteFile, int]:
        _, remote = ref
        local_path = downloader.download_remote_file(
            config,
            dataset,
            remote.path,
            language=None,
            token=token,
            use_cache=False,
        )
        parquet_file = pq.ParquetFile(local_path)
        if parquet_file.metadata is None:
            raise ValueError(f"Missing parquet row metadata for {remote.path}")
        return remote, parquet_file.metadata.num_rows

    if workers <= 1 or len(refs) <= 1:
        return [load(ref) for ref in refs]
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="stack-v2-metadata-bench") as executor:
        futures = [executor.submit(load, ref) for ref in refs]
        return [future.result() for future in futures]


def _packages_from_segments(
    segments: Sequence[StackV2IdSegment],
    *,
    package_size: int,
) -> list[StackV2IdPackage]:
    packages: list[StackV2IdPackage] = []
    current: list[StackV2IdSegment] = []
    current_count = 0

    def flush() -> None:
        nonlocal current, current_count
        if current:
            packages.append(StackV2IdPackage(index=len(packages), segments=tuple(current)))
        current = []
        current_count = 0

    for segment in segments:
        start = segment.start_row
        while start < segment.end_row:
            available = package_size - current_count
            take = min(available, segment.end_row - start)
            current.append(
                StackV2IdSegment(
                    language=segment.language,
                    remote=segment.remote,
                    start_row=start,
                    end_row=start + take,
                )
            )
            current_count += take
            start += take
            if current_count >= package_size:
                flush()
    flush()
    return packages


def _run_matrix_item(
    config: PipelineConfig,
    dataset,
    packages: Sequence[StackV2IdPackage],
    *,
    downloader: HuggingFaceDownloader,
    token: str | bool | None,
    package_workers: int,
    content_download_workers: int,
    content_prefetch_records: int,
    package_worker_backend: str,
    skip_errors: bool,
) -> MatrixResult:
    counts = {
        "records_seen": 0,
        "fetched": 0,
        "missing": 0,
        "unsupported": 0,
        "failed": 0,
    }
    _raise_fd_limit_for_stack_v2_content_workers(content_download_workers)
    if package_worker_backend == "process":
        executor_class = ProcessPoolExecutor
        fetcher = None
        per_package_content_workers = max(
            1,
            (content_download_workers + package_workers - 1) // package_workers,
        )
        content_executor = None
    else:
        executor_class = ThreadPoolExecutor
        fetcher = _stack_v2_content_fetcher_from_dataset(
            dataset,
            min_pool_connections=content_download_workers,
        )
        if isinstance(fetcher, AiohttpSoftwareHeritageContentFetcher):
            per_package_content_workers = content_download_workers
            content_executor = None
            fetcher.warm(concurrency=content_download_workers)
        else:
            per_package_content_workers = content_download_workers
            content_executor = ThreadPoolExecutor(
                max_workers=content_download_workers,
                thread_name_prefix="stack-v2-content-bench",
            )
    started = time.perf_counter()
    try:
        pending_packages = list(packages)
        with executor_class(max_workers=package_workers) as executor:
            futures: dict[Future[dict[str, int]], StackV2IdPackage] = {}

            def fill() -> None:
                while len(futures) < package_workers and pending_packages:
                    package = pending_packages.pop(0)
                    futures[
                        executor.submit(
                            _run_package,
                            config,
                            dataset,
                            package,
                            downloader=downloader if package_worker_backend == "thread" else None,
                            token=token,
                            content_fetcher=fetcher,
                            content_executor=content_executor,
                            content_download_workers=per_package_content_workers,
                            content_prefetch_records=content_prefetch_records,
                        )
                    ] = package

            fill()
            while futures:
                completed, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in completed:
                    package = futures.pop(future)
                    try:
                        result = future.result()
                    except Exception:
                        counts["failed"] += package.id_count
                        if not skip_errors:
                            raise
                    else:
                        for key, value in result.items():
                            counts[key] += value
                        elapsed_so_far = time.perf_counter() - started
                        print(
                            "PACKAGE_RESULT",
                            f"package={package.index}",
                            f"records={result['records_seen']}",
                            f"fetched={result['fetched']}",
                            f"elapsed={elapsed_so_far:.3f}",
                            flush=True,
                        )
                    fill()
    finally:
        if content_executor is not None:
            content_executor.shutdown(wait=True, cancel_futures=True)
        if isinstance(fetcher, AiohttpSoftwareHeritageContentFetcher):
            fetcher.close()
    elapsed = time.perf_counter() - started
    fetched_per_second = counts["fetched"] / elapsed if elapsed > 0 else 0.0
    records_per_second = counts["records_seen"] / elapsed if elapsed > 0 else 0.0
    return MatrixResult(
        package_workers=package_workers,
        content_download_workers=content_download_workers,
        content_prefetch_records=content_prefetch_records,
        packages=len(packages),
        ids_planned=sum(package.id_count for package in packages),
        records_seen=counts["records_seen"],
        fetched=counts["fetched"],
        missing=counts["missing"],
        unsupported=counts["unsupported"],
        failed=counts["failed"],
        seconds=elapsed,
        fetched_per_second=fetched_per_second,
        records_per_second=records_per_second,
    )


def _run_package(
    config: PipelineConfig,
    dataset,
    package: StackV2IdPackage,
    *,
    downloader: HuggingFaceDownloader,
    token: str | bool | None,
    content_fetcher,
    content_executor: ThreadPoolExecutor | None,
    content_download_workers: int,
    content_prefetch_records: int,
) -> dict[str, int]:
    downloader = downloader or HuggingFaceDownloader()
    owns_fetcher = content_fetcher is None
    if content_fetcher is None:
        content_fetcher = _stack_v2_content_fetcher_from_dataset(
            dataset,
            min_pool_connections=content_download_workers,
        )
        if isinstance(content_fetcher, AiohttpSoftwareHeritageContentFetcher):
            content_fetcher.warm(concurrency=content_download_workers)
    source = StackV2SWHContentPackageSource(
        config,
        dataset,
        package,
        show_progress=False,
        token=token,
        downloader=downloader,
        cache_source_files=False,
        content_fetcher=content_fetcher,
        content_executor=content_executor,
        content_download_workers=content_download_workers,
        content_prefetch_records=content_prefetch_records,
        content_language_filter=None,
    )
    counts = {
        "records_seen": 0,
        "fetched": 0,
        "missing": 0,
        "unsupported": 0,
    }
    try:
        for record in source.iter_records():
            counts["records_seen"] += 1
            status = str(record.metadata.get("content_fetch_status") or "")
            if status == "fetched":
                counts["fetched"] += 1
            elif status == "missing":
                counts["missing"] += 1
            elif status == "unsupported_language":
                counts["unsupported"] += 1
    finally:
        if owns_fetcher and isinstance(content_fetcher, AiohttpSoftwareHeritageContentFetcher):
            content_fetcher.close()
    return counts


def _best_result(results_path: Path) -> MatrixResult | None:
    if not results_path.exists():
        return None
    best: MatrixResult | None = None
    with results_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            result = MatrixResult(
                package_workers=int(row["package_workers"]),
                content_download_workers=int(row["content_download_workers"]),
                content_prefetch_records=int(row["content_prefetch_records"]),
                packages=int(row["packages"]),
                ids_planned=int(row["ids_planned"]),
                records_seen=int(row["records_seen"]),
                fetched=int(row["fetched"]),
                missing=int(row["missing"]),
                unsupported=int(row["unsupported"]),
                failed=int(row["failed"]),
                seconds=float(row["seconds"]),
                fetched_per_second=float(row["fetched_per_second"]),
                records_per_second=float(row["records_per_second"]),
            )
            if best is None or result.fetched_per_second > best.fetched_per_second:
                best = result
    return best


def _interleave_packages(
    packages: Sequence[StackV2IdPackage],
    stride: int,
) -> list[StackV2IdPackage]:
    if stride <= 1:
        return list(packages)
    result: list[StackV2IdPackage] = []
    for offset in range(stride):
        result.extend(packages[offset::stride])
    return [
        StackV2IdPackage(index=index, segments=package.segments)
        for index, package in enumerate(result)
    ]


def _resolve_token(token_env: str | None) -> str | bool | None:
    if token_env is None:
        return None
    token = os.environ.get(token_env)
    if not token:
        raise ValueError(f"Environment variable '{token_env}' is not set")
    return token


def _csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _csv_ints(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if any(item < 1 for item in result):
        raise ValueError(f"Expected positive integers, got {value}")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
