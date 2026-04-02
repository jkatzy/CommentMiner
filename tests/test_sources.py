from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from commentminer.config import DatasetSpec, PipelineConfig, StorageConfig
from commentminer.downloader import HuggingFaceDownloader
from commentminer.sources import ShardRowCursor, TheStackParquetSource


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
                "max_stars_repo_name": "repo-a",
                "max_stars_repo_path": "src/a.py",
                "hexsha": "a1",
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


def _copy_fixture_download(**kwargs: object) -> str:
    fixture_path = Path(str(kwargs["token"]))
    local_dir = Path(str(kwargs["local_dir"]))
    filename = Path(str(kwargs["filename"]))
    target_path = local_dir / filename
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(fixture_path, target_path)
    return str(target_path)


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
                allow_patterns=["data/python/**"],
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
                token=str(fixture_path),
                downloader=downloader,
            )

            records = list(source.iter_records())

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
                allow_patterns=["data/python/**"],
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
                token=str(fixture_path),
                downloader=downloader,
            )

            resume_after = ShardRowCursor("data/python/train-00000-of-00001.parquet", 0).to_record_id()
            records = list(source.iter_records(start_after=resume_after))

            self.assertEqual([record.path for record in records], ["src/b.py", "src/c.py"])
            downloaded_path = root / "downloads" / "the-stack" / "data/python/train-00000-of-00001.parquet"
            self.assertFalse(downloaded_path.exists())


if __name__ == "__main__":
    unittest.main()
