from __future__ import annotations

import gzip
import json
import shutil
import tempfile
import unittest
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

import commentminer.sources as sources_module
from commentminer.config import DatasetSpec, PipelineConfig, StorageConfig
from commentminer.downloader import HuggingFaceDownloader
from commentminer.sources import (
    ShardRowCursor,
    SoftwareHeritageS3ContentFetcher,
    StackV2SWHContentSource,
    TheStackParquetSource,
    UrlLineCursor,
    UrlListJsonlSource,
)


@dataclass
class FakeRepoFile:
    path: str
    size: int


class FakeApi:
    def __init__(self, files: list[FakeRepoFile]) -> None:
        self._files = files

    def list_repo_tree(self, **_: object):
        return list(self._files)


def _write_fixture(path: Path) -> None:
    table = pa.Table.from_pylist(
        [
            {
                "content": "# first\nprint('a')\n",
                "lang": "Python",
                "language": "Python",
                "max_stars_repo_name": "repo-a",
                "max_stars_repo_path": "src/a.py",
                "hexsha": "a1",
                "created_at": datetime(2026, 1, 2, tzinfo=timezone.utc),
            },
            {
                "content": "print('b')\n",
                "lang": "Python",
                "max_stars_repo_name": "repo-a",
                "max_stars_repo_path": "src/b.py",
                "hexsha": "b2",
            },
            {
                "content": "# third\nprint('c')\n",
                "lang": "Python",
                "max_stars_repo_name": "repo-b",
                "max_stars_repo_path": "src/c.py",
                "hexsha": "c3",
            },
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _write_stack_v2_fixture(path: Path) -> None:
    table = pa.Table.from_pylist(
        [
            {
                "blob_id": "blob-a",
                "path": "/src/a.py",
                "repo_name": "org/repo-a",
                "src_encoding": "utf-8",
                "language": "Python",
                "gha_language": None,
                "extension": "py",
            },
            {
                "blob_id": "blob-b",
                "path": "/src/b.py",
                "repo_name": "org/repo-b",
                "src_encoding": "latin-1",
                "language": "Python",
                "gha_language": None,
                "extension": "py",
            },
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _copy_fixture_download(**kwargs: object) -> str:
    fixture_path = Path(str(kwargs["token"]))
    local_dir = Path(str(kwargs["local_dir"]))
    filename = Path(str(kwargs["filename"]))
    target_path = local_dir / filename
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(fixture_path, target_path)
    return str(target_path)


class FakeStackV2ContentFetcher:
    def __init__(self, content_by_blob: dict[str, str]) -> None:
        self.content_by_blob = content_by_blob
        self.calls: list[tuple[str, str | None]] = []

    def fetch(self, blob_id: str, src_encoding: str | None) -> str:
        self.calls.append((blob_id, src_encoding))
        return self.content_by_blob[blob_id]


class TheStackParquetSourceTests(unittest.TestCase):
    def test_source_streams_rows_and_deletes_processed_shard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            fixture_path = root / "fixtures" / "train.parquet"
            _write_fixture(fixture_path)

            config = PipelineConfig(
                storage=StorageConfig(
                    working_directory=root / "work",
                    output_directory=root / "output",
                    checkpoint_directory=root / "checkpoints",
                    download_directory=root / "downloads",
                    huggingface_cache_directory=root / "hf-cache",
                    max_records_per_shard=10,
                    max_bytes_per_shard=1024,
                ),
                datasets=[],
                checkpoint_interval_records=1,
            )
            dataset = DatasetSpec(
                name="the-stack",
                source="huggingface_hub",
                repo_id="bigcode/the-stack",
                allow_patterns=["data/{language}/**"],
                streaming=True,
                batch_size=2,
            )
            downloader = HuggingFaceDownloader(
                api=FakeApi([FakeRepoFile("data/python/train-00000-of-00001.parquet", 10)]),
                download_file=_copy_fixture_download,
            )
            source = TheStackParquetSource(
                config,
                dataset,
                language=None,
                show_progress=False,
                token=str(fixture_path),
                downloader=downloader,
            )

            records = list(source.iter_records())

            self.assertEqual(source.name, "the-stack")
            self.assertEqual(len(records), 3)
            self.assertEqual(records[0].record_id, "data/python/train-00000-of-00001.parquet::row::0")
            self.assertEqual(records[0].path, "src/a.py")
            self.assertEqual(records[2].repo, "repo-b")

            downloaded_path = root / "downloads" / "the-stack" / "data/python/train-00000-of-00001.parquet"
            self.assertFalse(downloaded_path.exists())

            checkpoint_path = root / "checkpoints" / "processed-shards" / "the-stack.json"
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            self.assertEqual(
                checkpoint["completed_files"],
                ["data/python/train-00000-of-00001.parquet"],
            )

    def test_source_resumes_inside_shard_after_crash_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            fixture_path = root / "fixtures" / "train.parquet"
            _write_fixture(fixture_path)

            config = PipelineConfig(
                storage=StorageConfig(
                    working_directory=root / "work",
                    output_directory=root / "output",
                    checkpoint_directory=root / "checkpoints",
                    download_directory=root / "downloads",
                    huggingface_cache_directory=root / "hf-cache",
                    max_records_per_shard=10,
                    max_bytes_per_shard=1024,
                ),
                datasets=[],
                checkpoint_interval_records=1,
            )
            dataset = DatasetSpec(
                name="the-stack",
                source="huggingface_hub",
                repo_id="bigcode/the-stack",
                allow_patterns=["data/{language}/**"],
                streaming=True,
                batch_size=2,
            )
            downloader = HuggingFaceDownloader(
                api=FakeApi([FakeRepoFile("data/python/train-00000-of-00001.parquet", 10)]),
                download_file=_copy_fixture_download,
            )
            source = TheStackParquetSource(
                config,
                dataset,
                language="python",
                show_progress=False,
                token=str(fixture_path),
                downloader=downloader,
            )

            resume_after = ShardRowCursor("data/python/train-00000-of-00001.parquet", 0).to_record_id()
            records = list(source.iter_records(start_after=resume_after))

            self.assertEqual([record.path for record in records], ["src/b.py", "src/c.py"])
            self.assertEqual(source.name, "the-stack__python")
            downloaded_path = root / "downloads" / "the-stack" / "data/python/train-00000-of-00001.parquet"
            self.assertFalse(downloaded_path.exists())

    def test_stack_v2_source_fetches_content_from_blob_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            fixture_path = root / "fixtures" / "train.parquet"
            _write_stack_v2_fixture(fixture_path)

            config = PipelineConfig(
                storage=StorageConfig(
                    working_directory=root / "work",
                    output_directory=root / "output",
                    checkpoint_directory=root / "checkpoints",
                    download_directory=root / "downloads",
                    huggingface_cache_directory=root / "hf-cache",
                    max_records_per_shard=10,
                    max_bytes_per_shard=1024,
                ),
                datasets=[],
                checkpoint_interval_records=1,
            )
            dataset = DatasetSpec(
                name="the-stack-v2",
                source="huggingface_hub",
                repo_id="bigcode/the-stack-v2",
                allow_patterns=["data/{language}/**"],
                streaming=True,
                batch_size=2,
                extra={
                    "content_backend": "softwareheritage_s3",
                    "language_columns": ["language", "gha_language", "extension"],
                    "path_columns": ["path"],
                    "repo_columns": ["repo_name"],
                },
            )
            downloader = HuggingFaceDownloader(
                api=FakeApi([FakeRepoFile("data/Python/train-00000-of-00001.parquet", 10)]),
                download_file=_copy_fixture_download,
            )
            content_fetcher = FakeStackV2ContentFetcher(
                {
                    "blob-a": "# first\nprint('a')\n",
                    "blob-b": "# second\nprint('b')\n",
                }
            )
            source = StackV2SWHContentSource(
                config,
                dataset,
                language="Python",
                show_progress=False,
                token=str(fixture_path),
                downloader=downloader,
                content_fetcher=content_fetcher,
            )

            records = list(source.iter_records())

            self.assertEqual(source.name, "the-stack-v2__Python")
            self.assertEqual(
                [record.content for record in records],
                ["# first\nprint('a')\n", "# second\nprint('b')\n"],
            )
            self.assertEqual([record.path for record in records], ["/src/a.py", "/src/b.py"])
            self.assertEqual([record.repo for record in records], ["org/repo-a", "org/repo-b"])
            self.assertEqual(
                content_fetcher.calls,
                [("blob-a", "utf-8"), ("blob-b", "latin-1")],
            )
            self.assertEqual(records[0].metadata["blob_id"], "blob-a")
            self.assertEqual(records[0].metadata["content_backend"], "softwareheritage_s3")

    def test_stack_v2_source_can_fetch_content_with_process_workers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            fixture_path = root / "fixtures" / "train.parquet"
            content_root = root / "content"
            _write_stack_v2_fixture(fixture_path)
            content_root.mkdir(parents=True)
            for blob_id, payload in {
                "blob-a": "# first\nprint('a')\n",
                "blob-b": "# second\nprint('b')\n",
            }.items():
                with gzip.open(content_root / f"{blob_id}.gz", "wb") as handle:
                    handle.write(payload.encode("utf-8"))

            config = PipelineConfig(
                storage=StorageConfig(
                    working_directory=root / "work",
                    output_directory=root / "output",
                    checkpoint_directory=root / "checkpoints",
                    download_directory=root / "downloads",
                    huggingface_cache_directory=root / "hf-cache",
                    max_records_per_shard=10,
                    max_bytes_per_shard=1024,
                ),
                datasets=[],
                checkpoint_interval_records=1,
            )
            dataset = DatasetSpec(
                name="the-stack-v2",
                source="huggingface_hub",
                repo_id="bigcode/the-stack-v2",
                allow_patterns=["data/{language}/**"],
                streaming=True,
                batch_size=2,
                extra={
                    "content_backend": "softwareheritage_s3",
                    "language_columns": ["language", "gha_language", "extension"],
                    "path_columns": ["path"],
                    "repo_columns": ["repo_name"],
                    "swh_content_url_template": str(content_root / "{blob_id}.gz"),
                    "swh_content_compression": ".gz",
                },
            )
            downloader = HuggingFaceDownloader(
                api=FakeApi([FakeRepoFile("data/Python/train-00000-of-00001.parquet", 10)]),
                download_file=_copy_fixture_download,
            )
            source = StackV2SWHContentSource(
                config,
                dataset,
                language="Python",
                show_progress=False,
                token=str(fixture_path),
                downloader=downloader,
                content_download_workers=2,
                content_prefetch_records=2,
            )

            records = list(source.iter_records())

            self.assertEqual(source.content_download_workers, 2)
            self.assertEqual(
                [record.content for record in records],
                ["# first\nprint('a')\n", "# second\nprint('b')\n"],
            )
            self.assertEqual([record.metadata["row_index"] for record in records], [0, 1])

    def test_stack_v2_source_skips_content_for_unsupported_languages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            fixture_path = root / "fixtures" / "train.parquet"
            _write_stack_v2_fixture(fixture_path)

            config = PipelineConfig(
                storage=StorageConfig(
                    working_directory=root / "work",
                    output_directory=root / "output",
                    checkpoint_directory=root / "checkpoints",
                    download_directory=root / "downloads",
                    huggingface_cache_directory=root / "hf-cache",
                    max_records_per_shard=10,
                    max_bytes_per_shard=1024,
                ),
                datasets=[],
                checkpoint_interval_records=1,
            )
            dataset = DatasetSpec(
                name="the-stack-v2",
                source="huggingface_hub",
                repo_id="bigcode/the-stack-v2",
                allow_patterns=["data/{language}/**"],
                streaming=True,
                batch_size=2,
                extra={
                    "content_backend": "softwareheritage_s3",
                    "language_columns": ["language", "gha_language", "extension"],
                    "path_columns": ["path"],
                    "repo_columns": ["repo_name"],
                },
            )
            downloader = HuggingFaceDownloader(
                api=FakeApi([FakeRepoFile("data/Python/train-00000-of-00001.parquet", 10)]),
                download_file=_copy_fixture_download,
            )
            content_fetcher = FakeStackV2ContentFetcher(
                {
                    "blob-a": "# first\nprint('a')\n",
                    "blob-b": "# second\nprint('b')\n",
                }
            )
            source = StackV2SWHContentSource(
                config,
                dataset,
                language="Python",
                show_progress=False,
                token=str(fixture_path),
                downloader=downloader,
                content_fetcher=content_fetcher,
                content_download_workers=2,
                content_prefetch_records=2,
                content_language_filter=lambda _: False,
            )

            records = list(source.iter_records())

            self.assertEqual([record.content for record in records], ["", ""])
            self.assertEqual(content_fetcher.calls, [])
            self.assertEqual(
                [record.metadata["content_fetch_status"] for record in records],
                ["unsupported_language", "unsupported_language"],
            )

    def test_stack_v2_source_skips_missing_content_by_default(self) -> None:
        class MissingContentFetcher:
            url_template = "custom"

            def fetch(self, blob_id: str, src_encoding: str | None) -> str:
                raise FileNotFoundError(blob_id)

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            fixture_path = root / "fixtures" / "train.parquet"
            _write_stack_v2_fixture(fixture_path)

            config = PipelineConfig(
                storage=StorageConfig(
                    working_directory=root / "work",
                    output_directory=root / "output",
                    checkpoint_directory=root / "checkpoints",
                    download_directory=root / "downloads",
                    huggingface_cache_directory=root / "hf-cache",
                    max_records_per_shard=10,
                    max_bytes_per_shard=1024,
                ),
                datasets=[],
                checkpoint_interval_records=1,
            )
            dataset = DatasetSpec(
                name="the-stack-v2",
                source="huggingface_hub",
                repo_id="bigcode/the-stack-v2",
                allow_patterns=["data/{language}/**"],
                streaming=True,
                batch_size=2,
                extra={
                    "content_backend": "softwareheritage_s3",
                    "language_columns": ["language", "gha_language", "extension"],
                    "path_columns": ["path"],
                    "repo_columns": ["repo_name"],
                },
            )
            downloader = HuggingFaceDownloader(
                api=FakeApi([FakeRepoFile("data/Python/train-00000-of-00001.parquet", 10)]),
                download_file=_copy_fixture_download,
            )
            source = StackV2SWHContentSource(
                config,
                dataset,
                language="Python",
                show_progress=False,
                token=str(fixture_path),
                downloader=downloader,
                content_fetcher=MissingContentFetcher(),
                content_download_workers=2,
                content_prefetch_records=2,
            )

            records = list(source.iter_records())

            self.assertEqual([record.content for record in records], ["", ""])
            self.assertEqual(
                [record.metadata["content_fetch_status"] for record in records],
                ["missing", "missing"],
            )

    def test_software_heritage_fetcher_decodes_local_gzip_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            content_path = root / "blob-a.gz"
            with gzip.open(content_path, "wb") as handle:
                handle.write("# café\nprint('a')\n".encode("latin-1"))

            fetcher = SoftwareHeritageS3ContentFetcher(
                url_template=str(root / "{blob_id}.gz"),
                compression=".gz",
            )

            content = fetcher.fetch("blob-a", "latin-1")

            self.assertEqual(content, "# café\nprint('a')\n")

    def test_software_heritage_fetcher_defaults_to_unsigned_s3(self) -> None:
        dataset = DatasetSpec(
            name="the-stack-v2",
            source="huggingface_hub",
            repo_id="bigcode/the-stack-v2",
        )

        fetcher = SoftwareHeritageS3ContentFetcher.from_dataset(dataset)

        self.assertTrue(fetcher.aws_unsigned)

    def test_software_heritage_fetcher_can_use_signed_s3(self) -> None:
        dataset = DatasetSpec(
            name="the-stack-v2",
            source="huggingface_hub",
            repo_id="bigcode/the-stack-v2",
            extra={"aws_unsigned": False},
        )

        fetcher = SoftwareHeritageS3ContentFetcher.from_dataset(dataset)

        self.assertFalse(fetcher.aws_unsigned)

    def test_software_heritage_fetcher_retries_retryable_read_errors(self) -> None:
        class RetryFetcher(SoftwareHeritageS3ContentFetcher):
            def __init__(self) -> None:
                super().__init__(
                    url_template="s3://softwareheritage/content/{blob_id}",
                    compression=None,
                    s3_read_retries=2,
                    s3_retry_backoff_seconds=0,
                )
                self.calls = 0

            def _read_bytes(self, url: str) -> bytes:
                self.calls += 1
                if self.calls == 1:
                    raise OSError("temporary failure in name resolution")
                return b"# recovered\n"

        fetcher = RetryFetcher()

        content = fetcher.fetch("blob-a", "utf-8")

        self.assertEqual(content, "# recovered\n")
        self.assertEqual(fetcher.calls, 2)

    def test_stack_v2_source_defaults_to_measured_s3_content_concurrency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = PipelineConfig(
                storage=StorageConfig(
                    working_directory=root / "work",
                    output_directory=root / "output",
                    checkpoint_directory=root / "checkpoints",
                    download_directory=root / "downloads",
                    huggingface_cache_directory=root / "hf-cache",
                    max_records_per_shard=10,
                    max_bytes_per_shard=1024,
                ),
                datasets=[],
                checkpoint_interval_records=1,
            )
            dataset = DatasetSpec(
                name="the-stack-v2",
                source="huggingface_hub",
                repo_id="bigcode/the-stack-v2",
                streaming=True,
                extra={"content_backend": "softwareheritage_s3"},
            )

            source = StackV2SWHContentSource(config, dataset, show_progress=False)

            self.assertEqual(source.content_download_workers, 32)
            self.assertEqual(source.content_prefetch_records, 128)
            self.assertIsInstance(source.content_fetcher, SoftwareHeritageS3ContentFetcher)
            self.assertGreaterEqual(
                source.content_fetcher.s3_max_pool_connections,
                source.content_download_workers,
            )

    def test_stack_v2_source_sizes_s3_pool_to_requested_content_workers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = PipelineConfig(
                storage=StorageConfig(
                    working_directory=root / "work",
                    output_directory=root / "output",
                    checkpoint_directory=root / "checkpoints",
                    download_directory=root / "downloads",
                    huggingface_cache_directory=root / "hf-cache",
                    max_records_per_shard=10,
                    max_bytes_per_shard=1024,
                ),
                datasets=[],
                checkpoint_interval_records=1,
            )
            dataset = DatasetSpec(
                name="the-stack-v2",
                source="huggingface_hub",
                repo_id="bigcode/the-stack-v2",
                streaming=True,
                extra={"content_backend": "softwareheritage_s3"},
            )

            source = StackV2SWHContentSource(
                config,
                dataset,
                show_progress=False,
                content_download_workers=1536,
                content_prefetch_records=1536,
            )

            self.assertIsInstance(source.content_fetcher, SoftwareHeritageS3ContentFetcher)
            self.assertEqual(source.content_download_workers, 1536)
            self.assertGreaterEqual(source.content_fetcher.s3_max_pool_connections, 1536)

    def test_stack_v2_source_supports_1024_requested_content_workers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = PipelineConfig(
                storage=StorageConfig(
                    working_directory=root / "work",
                    output_directory=root / "output",
                    checkpoint_directory=root / "checkpoints",
                    download_directory=root / "downloads",
                    huggingface_cache_directory=root / "hf-cache",
                    max_records_per_shard=10,
                    max_bytes_per_shard=1024,
                ),
                datasets=[],
                checkpoint_interval_records=1,
            )
            dataset = DatasetSpec(
                name="the-stack-v2",
                source="huggingface_hub",
                repo_id="bigcode/the-stack-v2",
                streaming=True,
                extra={"content_backend": "softwareheritage_s3"},
            )

            source = StackV2SWHContentSource(
                config,
                dataset,
                show_progress=False,
                content_download_workers=1024,
                content_prefetch_records=4096,
            )

            self.assertIsInstance(source.content_fetcher, SoftwareHeritageS3ContentFetcher)
            self.assertEqual(source.content_download_workers, 1024)
            self.assertEqual(source.content_prefetch_records, 4096)
            self.assertGreaterEqual(source.content_fetcher.s3_max_pool_connections, 1024)

    def test_url_list_jsonl_source_streams_records_without_source_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            jsonl_path = root / "redpajama.jsonl"
            jsonl_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "text": "/** header */\nclass A {}\n",
                                "meta": {
                                    "language": [{"name": "Java", "bytes": "20"}],
                                    "path": "src/A.java",
                                    "repo_name": "org/repo",
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "text": "print('x')\n",
                                "meta": {
                                    "language": [{"name": "Python", "bytes": "11"}],
                                    "path": "x.py",
                                    "repo_name": "org/repo",
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "text": "/* should not be treated as Java */\nint main() {}\n",
                                "meta": {
                                    "language": [],
                                    "path": "main.c",
                                    "repo_name": "org/repo",
                                },
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )
            config = PipelineConfig(
                storage=StorageConfig(
                    working_directory=root / "work",
                    output_directory=root / "output",
                    checkpoint_directory=root / "checkpoints",
                    download_directory=root / "downloads",
                    huggingface_cache_directory=root / "hf-cache",
                    max_records_per_shard=10,
                    max_bytes_per_shard=1024,
                ),
                datasets=[],
                checkpoint_interval_records=1,
            )
            dataset = DatasetSpec(
                name="redpajama-github",
                source="url_list_jsonl",
                repo_id="togethercomputer/RedPajama-Data-1T",
                extra={
                    "urls": [str(jsonl_path)],
                    "content_columns": ["text"],
                    "language_columns": [],
                    "language_hint_columns": ["meta.language"],
                    "infer_language_from_path": True,
                    "path_columns": ["meta.path"],
                    "repo_columns": ["meta.repo_name"],
                },
            )
            source = UrlListJsonlSource(
                config,
                dataset,
                language="java",
                show_progress=False,
                max_files=1,
            )

            records = list(source.iter_records())

            self.assertEqual(len(records), 1)
            self.assertEqual(source.name, "redpajama-github__java")
            self.assertEqual(records[0].language, "java")
            self.assertEqual(records[0].path, "src/A.java")
            self.assertEqual(records[0].repo, "org/repo")
            self.assertEqual(records[0].metadata["ext"], "java")
            self.assertEqual(records[0].metadata["path_language"], "java")
            self.assertEqual(records[0].metadata["line_index"], 0)

    def test_url_list_jsonl_source_can_infer_redpajama_file_language_from_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            jsonl_path = root / "redpajama.jsonl"
            jsonl_path.write_text(
                json.dumps(
                    {
                        "text": "// header\nclass A {}\n",
                        "meta": {
                            "language": [
                                {"name": "1C Enterprise", "bytes": "10"},
                                {"name": "C#", "bytes": "100"},
                            ],
                            "path": "src/A.cs",
                            "repo_name": "org/repo",
                        },
                    }
                ),
                encoding="utf-8",
            )
            config = PipelineConfig(
                storage=StorageConfig(
                    working_directory=root / "work",
                    output_directory=root / "output",
                    checkpoint_directory=root / "checkpoints",
                    download_directory=root / "downloads",
                    huggingface_cache_directory=root / "hf-cache",
                    max_records_per_shard=10,
                    max_bytes_per_shard=1024,
                ),
                datasets=[],
                checkpoint_interval_records=1,
            )
            dataset = DatasetSpec(
                name="redpajama-github",
                source="url_list_jsonl",
                repo_id="togethercomputer/RedPajama-Data-1T",
                extra={
                    "urls": [str(jsonl_path)],
                    "content_columns": ["text"],
                    "language_columns": [],
                    "language_hint_columns": ["meta.language"],
                    "infer_language_from_path": True,
                    "path_columns": ["meta.path"],
                    "repo_columns": ["meta.repo_name"],
                },
            )
            source = UrlListJsonlSource(config, dataset, show_progress=False, max_files=1)

            records = list(source.iter_records())

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].language, "c#")
            self.assertEqual(records[0].metadata["ext"], "cs")
            self.assertEqual(records[0].metadata["path_language"], "c#")

    def test_url_list_jsonl_source_retries_stream_from_last_byte_offset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = PipelineConfig(
                storage=StorageConfig(
                    working_directory=root / "work",
                    output_directory=root / "output",
                    checkpoint_directory=root / "checkpoints",
                    download_directory=root / "downloads",
                    huggingface_cache_directory=root / "hf-cache",
                    max_records_per_shard=10,
                    max_bytes_per_shard=1024,
                ),
                datasets=[],
                checkpoint_interval_records=1,
            )
            dataset = DatasetSpec(
                name="redpajama-github",
                source="url_list_jsonl",
                repo_id="togethercomputer/RedPajama-Data-1T",
                extra={
                    "urls": ["https://example.test/redpajama.jsonl"],
                    "content_columns": ["text"],
                    "stream_retries": 1,
                    "stream_retry_backoff_seconds": 0,
                },
            )
            first_line = json.dumps({"text": "# first\nprint('a')\n"})
            second_line = json.dumps({"text": "# second\nprint('b')\n"})
            start_bytes: list[int] = []

            def flaky_text_lines(url: str, *, start_byte: int = 0):
                self.assertEqual(url, "https://example.test/redpajama.jsonl")
                start_bytes.append(start_byte)
                if len(start_bytes) == 1:
                    yield sources_module._TextLine(
                        text=first_line,
                        next_byte_offset=100,
                        resumed_from_byte=False,
                    )
                    raise httpx.RemoteProtocolError("peer closed connection")
                yield sources_module._TextLine(
                    text=second_line,
                    next_byte_offset=200,
                    resumed_from_byte=True,
                )

            original_iter_text_lines = sources_module._iter_text_lines
            sources_module._iter_text_lines = flaky_text_lines
            try:
                source = UrlListJsonlSource(
                    config,
                    dataset,
                    show_progress=False,
                    max_files=1,
                )

                records = list(source.iter_records())
            finally:
                sources_module._iter_text_lines = original_iter_text_lines

            self.assertEqual(start_bytes, [0, 100])
            self.assertEqual(
                [record.content for record in records],
                ["# first\nprint('a')\n", "# second\nprint('b')\n"],
            )
            self.assertEqual([record.metadata["line_index"] for record in records], [0, 1])

    def test_url_list_jsonl_source_retries_while_resuming_to_line_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = PipelineConfig(
                storage=StorageConfig(
                    working_directory=root / "work",
                    output_directory=root / "output",
                    checkpoint_directory=root / "checkpoints",
                    download_directory=root / "downloads",
                    huggingface_cache_directory=root / "hf-cache",
                    max_records_per_shard=10,
                    max_bytes_per_shard=1024,
                ),
                datasets=[],
                checkpoint_interval_records=1,
            )
            url = "https://example.test/redpajama.jsonl"
            dataset = DatasetSpec(
                name="redpajama-github",
                source="url_list_jsonl",
                repo_id="togethercomputer/RedPajama-Data-1T",
                extra={
                    "urls": [url],
                    "content_columns": ["text"],
                    "stream_retries": 1,
                    "stream_retry_backoff_seconds": 0,
                },
            )
            first_line = json.dumps({"text": "# first\nprint('a')\n"})
            second_line = json.dumps({"text": "# second\nprint('b')\n"})
            third_line = json.dumps({"text": "# third\nprint('c')\n"})
            start_bytes: list[int] = []

            def flaky_text_lines(stream_url: str, *, start_byte: int = 0):
                self.assertEqual(stream_url, url)
                start_bytes.append(start_byte)
                if len(start_bytes) == 1:
                    yield sources_module._TextLine(
                        text=first_line,
                        next_byte_offset=100,
                        resumed_from_byte=False,
                    )
                    raise httpx.RemoteProtocolError("peer closed connection")
                yield sources_module._TextLine(
                    text=second_line,
                    next_byte_offset=200,
                    resumed_from_byte=True,
                )
                yield sources_module._TextLine(
                    text=third_line,
                    next_byte_offset=300,
                    resumed_from_byte=True,
                )

            original_iter_text_lines = sources_module._iter_text_lines
            sources_module._iter_text_lines = flaky_text_lines
            try:
                source = UrlListJsonlSource(
                    config,
                    dataset,
                    show_progress=False,
                    max_files=1,
                )

                records = list(
                    source.iter_records(
                        start_after=UrlLineCursor(url, 1).to_record_id(),
                    )
                )
            finally:
                sources_module._iter_text_lines = original_iter_text_lines

            self.assertEqual(start_bytes, [0, 100])
            self.assertEqual(
                [record.content for record in records],
                ["# third\nprint('c')\n"],
            )
            self.assertEqual([record.metadata["line_index"] for record in records], [2])


if __name__ == "__main__":
    unittest.main()
