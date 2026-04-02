from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from commentminer.config import PipelineConfig, StorageConfig
from commentminer.models import InputRecord
from commentminer.pipeline import run_dataset


class FakeSource:
    name = "toy-source"

    def __init__(self, records: list[InputRecord]) -> None:
        self._records = records

    def iter_records(self, start_after: str | None = None):
        started = start_after is None
        for record in self._records:
            if not started:
                if record.record_id == start_after:
                    started = True
                continue
            yield record


class FakeExtractor:
    def extract_opening_comment(self, record: InputRecord) -> str | None:
        if record.content.startswith("comment:"):
            return record.content.split(":", maxsplit=1)[1].strip()
        return None


class PipelineTests(unittest.TestCase):
    def test_run_dataset_writes_comment_shards_and_checkpoint(self) -> None:
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

            records = [
                InputRecord(dataset="toy-source", record_id="1", content="comment: first"),
                InputRecord(dataset="toy-source", record_id="2", content="code only"),
                InputRecord(dataset="toy-source", record_id="3", content="comment: second"),
            ]
            stats = run_dataset(FakeSource(records), FakeExtractor(), config)

            self.assertEqual(stats.records_seen, 3)
            self.assertEqual(stats.comments_written, 2)
            self.assertEqual(stats.skipped_without_comment, 1)
            self.assertEqual(stats.shards_written, 1)

            checkpoint_path = config.storage.checkpoint_directory / "toy-source.json"
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            self.assertEqual(checkpoint["last_record_id"], "3")
            self.assertEqual(checkpoint["comments_written"], 2)

            shard_paths = list(config.storage.output_directory.rglob("part-00000.jsonl"))
            self.assertEqual(len(shard_paths), 1)
            lines = shard_paths[0].read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)

            payloads = [json.loads(line) for line in lines]
            self.assertEqual(payloads[0]["opening_comment"], "first")
            self.assertEqual(payloads[1]["opening_comment"], "second")


if __name__ == "__main__":
    unittest.main()
