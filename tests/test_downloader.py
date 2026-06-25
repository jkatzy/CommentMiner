from __future__ import annotations

import json
import tempfile
import threading
import unittest
from dataclasses import dataclass
from pathlib import Path

import httpx

import commentminer.downloader as downloader_module
from commentminer.config import DatasetSpec, PipelineConfig, StorageConfig
from commentminer.downloader import HuggingFaceDownloader


@dataclass
class FakeRepoFile:
    path: str
    size: int


class FakeApi:
    def __init__(self, files: list[FakeRepoFile]) -> None:
        self._files = files
        self.calls = 0

    def list_repo_tree(self, **_: object):
        self.calls += 1
        return list(self._files)


def fake_download_file(**kwargs: object) -> str:
    local_dir = Path(str(kwargs["local_dir"]))
    filename = Path(str(kwargs["filename"]))
    target_path = local_dir / filename
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text("payload", encoding="utf-8")
    return str(target_path)


class FakeStreamResponse:
    def __init__(self, *, fail: bool) -> None:
        self.fail = fail

    def __enter__(self) -> "FakeStreamResponse":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_bytes(self):
        if self.fail:
            yield b"partial"
            raise httpx.RemoteProtocolError("peer closed connection")
        yield b"payload"


class HuggingFaceDownloaderTests(unittest.TestCase):
    def test_list_languages_discovers_languages_from_remote_paths(self) -> None:
        dataset = DatasetSpec(
            name="the-stack",
            source="huggingface_hub",
            repo_id="bigcode/the-stack",
            repo_type="dataset",
            revision="main",
            allow_patterns=["data/{language}/**"],
        )
        downloader = HuggingFaceDownloader(
            api=FakeApi(
                [
                    FakeRepoFile("data/python/train-00000-of-00001.parquet", 10),
                    FakeRepoFile("data/java/train-00000-of-00001.parquet", 12),
                    FakeRepoFile("data/python/train-00001-of-00001.parquet", 14),
                ]
            )
        )

        languages = downloader.list_languages(dataset)

        self.assertEqual(languages, ["java", "python"])

    def test_plan_download_uses_language_filter_and_checkpoint(self) -> None:
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
            config.datasets.append(
                DatasetSpec(
                    name="the-stack-v2",
                    source="huggingface_hub",
                    repo_id="bigcode/the-stack-v2",
                    repo_type="dataset",
                    revision="main",
                    allow_patterns=["data/{language}/**"],
                    languages=["python", "java"],
                )
            )

            checkpoint_root = config.storage.checkpoint_directory / "downloads"
            checkpoint_root.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_root / "the-stack-v2-python.json"
            checkpoint_path.write_text(
                json.dumps(
                    {
                        "dataset": "the-stack-v2",
                        "repo_id": "bigcode/the-stack-v2",
                        "revision": "main",
                        "language": "python",
                        "completed_files": ["data/python/part-00000.parquet"],
                    }
                ),
                encoding="utf-8",
            )

            downloader = HuggingFaceDownloader(
                api=FakeApi(
                    [
                        FakeRepoFile("data/python/part-00000.parquet", 10),
                        FakeRepoFile("data/python/part-00001.parquet", 12),
                        FakeRepoFile("data/java/part-00000.parquet", 14),
                    ]
                )
            )

            plan = downloader.plan_download(
                config,
                config.require_dataset("the-stack-v2"),
                language="python",
            )

            self.assertEqual(plan.matched_count, 2)
            self.assertEqual(plan.completed_count, 1)
            self.assertEqual(plan.pending_count, 1)
            self.assertEqual(plan.pending_files[0].path, "data/python/part-00001.parquet")

    def test_plan_download_reuses_cached_remote_file_listing(self) -> None:
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
            config.datasets.append(
                DatasetSpec(
                    name="the-stack-v2",
                    source="huggingface_hub",
                    repo_id="bigcode/the-stack-v2",
                    repo_type="dataset",
                    revision="main",
                    allow_patterns=["data/{language}/**"],
                    languages=["python", "java"],
                )
            )
            fake_api = FakeApi(
                [
                    FakeRepoFile("data/python/part-00000.parquet", 10),
                    FakeRepoFile("data/java/part-00000.parquet", 14),
                ]
            )
            downloader = HuggingFaceDownloader(api=fake_api)
            dataset = config.require_dataset("the-stack-v2")

            python_plan = downloader.plan_download(config, dataset, language="python")
            java_plan = downloader.plan_download(config, dataset, language="java")

            self.assertEqual(fake_api.calls, 1)
            self.assertEqual(python_plan.matched_count, 1)
            self.assertEqual(java_plan.matched_count, 1)

    def test_download_writes_files_and_updates_checkpoint(self) -> None:
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
            config.datasets.append(
                DatasetSpec(
                    name="the-stack",
                    source="huggingface_hub",
                    repo_id="bigcode/the-stack",
                    repo_type="dataset",
                    revision="main",
                    allow_patterns=["data/**"],
                )
            )

            downloader = HuggingFaceDownloader(
                api=FakeApi(
                    [
                        FakeRepoFile("data/a.parquet", 10),
                        FakeRepoFile("data/b.parquet", 12),
                    ]
                ),
                download_file=fake_download_file,
            )

            summary = downloader.download(config, config.require_dataset("the-stack"))

            self.assertEqual(summary.matched_count, 2)
            self.assertEqual(summary.already_downloaded_count, 0)
            self.assertEqual(summary.downloaded_count, 2)
            self.assertTrue((summary.download_root / "data/a.parquet").exists())
            self.assertTrue((summary.download_root / "data/b.parquet").exists())

            checkpoint = json.loads(summary.checkpoint_path.read_text(encoding="utf-8"))
            self.assertEqual(
                checkpoint["completed_files"],
                ["data/a.parquet", "data/b.parquet"],
            )

            second_summary = downloader.download(config, config.require_dataset("the-stack"))
            self.assertEqual(second_summary.already_downloaded_count, 2)
            self.assertEqual(second_summary.downloaded_count, 0)

    def test_download_can_use_multiple_workers(self) -> None:
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
            config.datasets.append(
                DatasetSpec(
                    name="the-stack",
                    source="huggingface_hub",
                    repo_id="bigcode/the-stack",
                    repo_type="dataset",
                    revision="main",
                    allow_patterns=["data/**"],
                )
            )
            condition = threading.Condition()
            active = 0
            max_active = 0

            def blocking_download_file(**kwargs: object) -> str:
                nonlocal active, max_active
                with condition:
                    active += 1
                    max_active = max(max_active, active)
                    condition.notify_all()
                    condition.wait_for(lambda: max_active >= 2, timeout=1.0)
                try:
                    return fake_download_file(**kwargs)
                finally:
                    with condition:
                        active -= 1
                        condition.notify_all()

            downloader = HuggingFaceDownloader(
                api=FakeApi(
                    [
                        FakeRepoFile("data/a.parquet", 10),
                        FakeRepoFile("data/b.parquet", 12),
                        FakeRepoFile("data/c.parquet", 14),
                    ]
                ),
                download_file=blocking_download_file,
            )

            summary = downloader.download(
                config,
                config.require_dataset("the-stack"),
                download_workers=3,
            )

            self.assertEqual(summary.downloaded_count, 3)
            self.assertGreaterEqual(max_active, 2)
            self.assertTrue((summary.download_root / "data/a.parquet").exists())
            self.assertTrue((summary.download_root / "data/b.parquet").exists())
            self.assertTrue((summary.download_root / "data/c.parquet").exists())

    def test_direct_download_retries_truncated_stream(self) -> None:
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
                name="the-stack",
                source="huggingface_hub",
                repo_id="bigcode/the-stack",
                allow_patterns=["data/**"],
                extra={
                    "download_retries": 1,
                    "download_retry_backoff_seconds": 0,
                },
            )
            downloader = HuggingFaceDownloader(api=FakeApi([]))
            calls: list[object] = []

            def flaky_stream(*args: object, **kwargs: object) -> FakeStreamResponse:
                calls.append((args, kwargs))
                return FakeStreamResponse(fail=len(calls) == 1)

            original_stream = downloader_module.httpx.stream
            downloader_module.httpx.stream = flaky_stream
            try:
                local_path = downloader.download_remote_file(
                    config,
                    dataset,
                    "data/a.parquet",
                    use_cache=False,
                )
            finally:
                downloader_module.httpx.stream = original_stream

            self.assertEqual(len(calls), 2)
            self.assertEqual(local_path.read_bytes(), b"payload")
            self.assertFalse(local_path.with_name(f"{local_path.name}.incomplete").exists())


if __name__ == "__main__":
    unittest.main()
