from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import time

from commentminer.config import DatasetSpec, PipelineConfig, StorageConfig
from commentminer.downloader import DownloadPlan, HuggingFaceDownloader, RemoteFile
from commentminer.extractors import ML4SEOpeningCommentExtractor
from commentminer.models import ExtractedComment
from commentminer.pipeline import run_dataset
from commentminer.sources import StackV2SWHContentSource


class SingleFileDownloader(HuggingFaceDownloader):
    def __init__(self, parquet_path: Path, remote_path: str) -> None:
        super().__init__()
        self.parquet_path = parquet_path
        self.remote = RemoteFile(remote_path, parquet_path.stat().st_size)

    def plan_download(self, *args: object, **kwargs: object) -> DownloadPlan:
        return DownloadPlan(
            dataset="the-stack-v2",
            repo_id="bigcode/the-stack-v2",
            revision="main",
            language=None,
            download_root=self.parquet_path.parent,
            checkpoint_path=Path("/tmp/commentminer-stack-v2-benchmark-download.json"),
            cache_directory=Path("/tmp/commentminer-stack-v2-benchmark-cache"),
            allow_patterns=[],
            ignore_patterns=[],
            matched_files=[self.remote],
            pending_files=[self.remote],
            completed_files=[],
        )

    def download_remote_file(self, *args: object, **kwargs: object) -> Path:
        return self.parquet_path

    def remove_local_file(self, *args: object, **kwargs: object) -> None:
        return None

    def mark_file_completed(self, *args: object, **kwargs: object) -> None:
        return None


class StubContentFetcher:
    url_template = "benchmark-stub://{blob_id}"

    def fetch(self, blob_id: str, src_encoding: str | None) -> str:
        return "// benchmark header\nint main() { return 0; }\n"


class BenchmarkExtractor:
    def extract_opening_comments(self, record) -> list[ExtractedComment]:
        if not record.content:
            return []
        return [ExtractedComment(text="// benchmark header", start_line=1, index=0)]

    def extract_opening_comment(self, record) -> str | None:
        return "// benchmark header" if record.content else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark CommentMiner Stack v2 parquet processing throughput.",
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=Path("var/downloads/the-stack-v2/c/data/C++/train-00000-of-00007.parquet"),
        help="Local Stack v2 parquet shard to read.",
    )
    parser.add_argument(
        "--remote-path",
        default="data/C++/train-00000-of-00007.parquet",
        help="Remote path label used for generated Stack v2 record ids.",
    )
    parser.add_argument("--language", default="C++")
    parser.add_argument("--max-records", type=int, default=100_000)
    parser.add_argument("--target-files-per-second", type=float, default=10_000.0)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--checkpoint-interval", type=int, default=1_000_000)
    parser.add_argument("--extraction-workers", type=int, default=1)
    parser.add_argument("--extraction-buffer", type=int)
    parser.add_argument(
        "--content-mode",
        choices=["s3", "stub"],
        default="s3",
        help="s3 measures Software Heritage object reads; stub isolates local processing only.",
    )
    parser.add_argument("--content-download-workers", type=int, default=32)
    parser.add_argument("--content-prefetch-records", type=int, default=128)
    parser.add_argument(
        "--swh-content-url-template",
        default="s3://softwareheritage/content/{blob_id}",
        help="Stack v2 content URL template. Use https://softwareheritage.s3.amazonaws.com/content/{blob_id} to benchmark pooled HTTPS.",
    )
    parser.add_argument(
        "--extractor",
        choices=["benchmark", "ml4se"],
        default="benchmark",
        help="Extractor used during the benchmark.",
    )
    return parser.parse_args()


def build_config(root: Path, checkpoint_interval: int) -> PipelineConfig:
    return PipelineConfig(
        storage=StorageConfig(
            working_directory=root / "work",
            output_directory=root / "out",
            checkpoint_directory=root / "checkpoints",
            download_directory=root / "downloads",
            huggingface_cache_directory=root / "hf-cache",
            max_records_per_shard=1_000_000,
            max_bytes_per_shard=1_000_000_000,
        ),
        datasets=[],
        checkpoint_interval_records=checkpoint_interval,
    )


def build_dataset(args: argparse.Namespace) -> DatasetSpec:
    extra = {
        "content_backend": "softwareheritage_s3",
        "language_columns": ["language", "gha_language", "extension"],
        "path_columns": ["path"],
        "repo_columns": ["repo_name"],
        "metadata_columns": [
            "blob_id",
            "src_encoding",
            "language",
            "gha_language",
            "extension",
            "path",
            "repo_name",
        ],
    }
    if args.content_mode == "s3":
        extra["swh_content_url_template"] = args.swh_content_url_template
        extra["swh_s3_max_pool_connections"] = max(args.content_download_workers * 4, 64)
    return DatasetSpec(
        name="the-stack-v2",
        source="huggingface_hub",
        repo_id="bigcode/the-stack-v2",
        streaming=True,
        batch_size=args.batch_size,
        extra=extra,
    )


def main() -> int:
    args = parse_args()
    parquet_path = args.parquet.resolve()
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)

    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        config = build_config(root, args.checkpoint_interval)
        dataset = build_dataset(args)
        if args.content_mode == "stub":
            content_fetcher = StubContentFetcher()
        else:
            content_fetcher = None
        source = StackV2SWHContentSource(
            config,
            dataset,
            language=args.language,
            show_progress=False,
            downloader=SingleFileDownloader(parquet_path, args.remote_path),
            content_fetcher=content_fetcher,
            content_download_workers=args.content_download_workers,
            content_prefetch_records=args.content_prefetch_records,
        )
        extractor = (
            ML4SEOpeningCommentExtractor(max_start_row=10)
            if args.extractor == "ml4se"
            else BenchmarkExtractor()
        )

        started_at = time.perf_counter()
        stats = run_dataset(
            source,
            extractor,
            config,
            max_records=args.max_records,
            progress_every=0,
            extraction_workers=args.extraction_workers,
            extraction_buffer=args.extraction_buffer,
        )
        elapsed = time.perf_counter() - started_at

    rate = stats.records_seen / elapsed if elapsed > 0 else 0.0
    print(f"content_mode={args.content_mode}")
    print(f"records={stats.records_seen}")
    print(f"seconds={elapsed:.6f}")
    print(f"files_per_second={rate:.2f}")
    print(f"target_files_per_second={args.target_files_per_second:.2f}")
    print(f"comments_written={stats.comments_written}")
    if rate < args.target_files_per_second:
        print("result=FAIL")
        return 1
    print("result=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
