from __future__ import annotations

import json
import shutil
import threading
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from commentminer.config import DatasetSpec, PipelineConfig, StorageConfig
from commentminer.downloader import HuggingFaceDownloader
from commentminer.extractors import ML4SEOpeningCommentExtractor
from commentminer.pipeline import run_sharded_dataset
from commentminer.sources import TheStackParquetSource


@dataclass
class FakeRepoFile:
    path: str
    size: int


class FakeApi:
    def __init__(self, files: list[FakeRepoFile]) -> None:
        self._files = files

    def list_repo_tree(self, **_: object):
        return list(self._files)


class StreamingApi:
    def __init__(self, files: list[FakeRepoFile], download_started: threading.Event) -> None:
        self._files = files
        self._download_started = download_started

    def list_repo_tree(self, **_: object):
        def _iter():
            first = True
            for item in self._files:
                if not first and not self._download_started.wait(timeout=1):
                    raise AssertionError("Shard discovery did not stream ahead of worker downloads")
                yield item
                first = False

        return _iter()


def _write_fixture(path: Path) -> None:
    table = pa.Table.from_pylist(
        [
            {
                "content": "# first comment\n# second line\n\nprint('a')\n",
                "lang": "Python",
                "ext": "python",
                "max_stars_repo_name": "repo-a",
                "max_stars_repo_path": "src/a.py",
                "hexsha": "a1",
            },
            {
                "content": "print('b')\n",
                "lang": "Python",
                "ext": "python",
                "max_stars_repo_name": "repo-b",
                "max_stars_repo_path": "src/b.py",
                "hexsha": "b2",
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


class FixtureRouterDownload:
    def __init__(self, mapping: dict[str, Path]) -> None:
        self.mapping = mapping

    def __call__(self, **kwargs: object) -> str:
        filename = str(kwargs["filename"])
        fixture_path = self.mapping[filename]
        local_dir = Path(str(kwargs["local_dir"]))
        target_path = local_dir / filename
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(fixture_path, target_path)
        return str(target_path)


class MiningTests(unittest.TestCase):
    def test_run_sharded_dataset_writes_comment_without_content_and_keeps_metadata(self) -> None:
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
                    max_bytes_per_shard=1024 * 1024,
                ),
                datasets=[],
                checkpoint_interval_records=1,
            )
            dataset = DatasetSpec(
                name="the-stack",
                source="huggingface_hub",
                repo_id="bigcode/the-stack",
                allow_patterns=["data/python/**"],
                streaming=True,
                batch_size=50,
            )
            source = TheStackParquetSource(
                config,
                dataset,
                show_progress=False,
                token=str(fixture_path),
                downloader=HuggingFaceDownloader(
                    api=FakeApi([FakeRepoFile("data/python/train-00000-of-00001.parquet", 10)]),
                    download_file=_copy_fixture_download,
                ),
            )

            stats = run_sharded_dataset(
                source,
                ML4SEOpeningCommentExtractor,
                config,
                max_workers=1,
            )

            self.assertEqual(stats.records_seen, 2)
            self.assertEqual(stats.comments_written, 1)
            self.assertEqual(stats.failed_shards, 0)
            shard_path = next(config.storage.output_directory.rglob("part-00000.jsonl"))
            payload = json.loads(shard_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(payload["opening_comment"], "# first comment\n# second line")
            self.assertNotIn("content", payload)
            self.assertEqual(payload["metadata"]["hexsha"], "a1")
            self.assertEqual(payload["metadata"]["ext"], "python")
            self.assertEqual(payload["path"], "src/a.py")

    def test_run_sharded_dataset_skips_failed_shard_and_reruns_missing_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            good_fixture = root / "fixtures" / "good.parquet"
            repaired_fixture = root / "fixtures" / "repaired.parquet"
            bad_fixture = root / "fixtures" / "bad.parquet"
            _write_fixture(good_fixture)
            _write_fixture(repaired_fixture)
            bad_fixture.parent.mkdir(parents=True, exist_ok=True)
            bad_fixture.write_text("not parquet", encoding="utf-8")

            config = PipelineConfig(
                storage=StorageConfig(
                    working_directory=root / "work",
                    output_directory=root / "output",
                    checkpoint_directory=root / "checkpoints",
                    download_directory=root / "downloads",
                    huggingface_cache_directory=root / "hf-cache",
                    max_records_per_shard=10,
                    max_bytes_per_shard=1024 * 1024,
                ),
                datasets=[],
                checkpoint_interval_records=1,
            )
            dataset = DatasetSpec(
                name="the-stack",
                source="huggingface_hub",
                repo_id="bigcode/the-stack",
                allow_patterns=["data/python/**"],
                streaming=True,
                batch_size=50,
            )
            files = [
                FakeRepoFile("data/python/bad.parquet", 10),
                FakeRepoFile("data/python/good.parquet", 10),
            ]

            failing_source = TheStackParquetSource(
                config,
                dataset,
                show_progress=False,
                downloader=HuggingFaceDownloader(
                    api=FakeApi(files),
                    download_file=FixtureRouterDownload(
                        {
                            "data/python/bad.parquet": bad_fixture,
                            "data/python/good.parquet": good_fixture,
                        }
                    ),
                ),
            )

            first_stats = run_sharded_dataset(
                failing_source,
                ML4SEOpeningCommentExtractor,
                config,
                max_workers=2,
            )

            self.assertEqual(first_stats.records_seen, 2)
            self.assertEqual(first_stats.comments_written, 1)
            self.assertEqual(first_stats.failed_shards, 1)
            checkpoint_path = root / "checkpoints" / "processed-shards" / "the-stack.json"
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            self.assertEqual(checkpoint["completed_files"], ["data/python/good.parquet"])

            repaired_source = TheStackParquetSource(
                config,
                dataset,
                show_progress=False,
                downloader=HuggingFaceDownloader(
                    api=FakeApi(files),
                    download_file=FixtureRouterDownload(
                        {
                            "data/python/bad.parquet": repaired_fixture,
                            "data/python/good.parquet": good_fixture,
                        }
                    ),
                ),
            )

            second_stats = run_sharded_dataset(
                repaired_source,
                ML4SEOpeningCommentExtractor,
                config,
                max_workers=2,
            )

            self.assertEqual(second_stats.records_seen, 2)
            self.assertEqual(second_stats.comments_written, 1)
            self.assertEqual(second_stats.failed_shards, 0)
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            self.assertEqual(
                checkpoint["completed_files"],
                ["data/python/bad.parquet", "data/python/good.parquet"],
            )

    def test_run_sharded_dataset_starts_workers_before_full_shard_discovery_finishes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            first_fixture = root / "fixtures" / "first.parquet"
            second_fixture = root / "fixtures" / "second.parquet"
            _write_fixture(first_fixture)
            _write_fixture(second_fixture)
            config = PipelineConfig(
                storage=StorageConfig(
                    working_directory=root / "work",
                    output_directory=root / "output",
                    checkpoint_directory=root / "checkpoints",
                    download_directory=root / "downloads",
                    huggingface_cache_directory=root / "hf-cache",
                    max_records_per_shard=10,
                    max_bytes_per_shard=1024 * 1024,
                ),
                datasets=[],
                checkpoint_interval_records=1,
            )
            dataset = DatasetSpec(
                name="the-stack",
                source="huggingface_hub",
                repo_id="bigcode/the-stack",
                allow_patterns=["data/python/**"],
                streaming=True,
                batch_size=50,
            )
            files = [
                FakeRepoFile("data/python/first.parquet", 10),
                FakeRepoFile("data/python/second.parquet", 10),
            ]
            download_started = threading.Event()
            router = FixtureRouterDownload(
                {
                    "data/python/first.parquet": first_fixture,
                    "data/python/second.parquet": second_fixture,
                }
            )

            def streaming_download(**kwargs: object) -> str:
                download_started.set()
                return router(**kwargs)

            source = TheStackParquetSource(
                config,
                dataset,
                show_progress=False,
                downloader=HuggingFaceDownloader(
                    api=StreamingApi(files, download_started),
                    download_file=streaming_download,
                ),
            )

            stats = run_sharded_dataset(
                source,
                ML4SEOpeningCommentExtractor,
                config,
                max_workers=1,
            )

            self.assertEqual(stats.records_seen, 4)
            self.assertEqual(stats.comments_written, 2)
            self.assertEqual(stats.failed_shards, 0)


if __name__ == "__main__":
    unittest.main()
