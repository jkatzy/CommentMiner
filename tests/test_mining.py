from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from commentminer.config import DatasetSpec, PipelineConfig, StorageConfig
from commentminer.downloader import HuggingFaceDownloader
from commentminer.extractors import ML4SEOpeningCommentExtractor
from commentminer.pipeline import run_dataset
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
                "created_at": datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
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


class MiningTests(unittest.TestCase):
    def test_run_dataset_writes_comment_without_content_and_keeps_metadata(self) -> None:
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
            extractor = ML4SEOpeningCommentExtractor()

            stats = run_dataset(source, extractor, config)

            self.assertEqual(stats.records_seen, 2)
            self.assertEqual(stats.comments_written, 1)
            shard_path = next(config.storage.output_directory.rglob("part-00000.parquet"))
            payload = pq.read_table(shard_path).to_pylist()[0]
            metadata = json.loads(payload["metadata"])
            self.assertEqual(payload["opening_comment"], "# first comment\n# second line")
            self.assertNotIn("content", payload)
            self.assertEqual(metadata["hexsha"], "a1")
            self.assertEqual(metadata["ext"], "python")
            self.assertEqual(metadata["created_at"], "2026-01-02T03:04:05+00:00")
            self.assertEqual(payload["path"], "src/a.py")


if __name__ == "__main__":
    unittest.main()
