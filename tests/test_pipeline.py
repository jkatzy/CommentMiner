from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path

import pyarrow.parquet as pq

from commentminer.config import PipelineConfig, StorageConfig
from commentminer.models import ExtractedComment, InputRecord
from commentminer.parquet_io import COMMENT_SCHEMA
from commentminer.pipeline import run_dataset


class FakeSource:
    name = "toy-source"

    def __init__(self, records: list[InputRecord]) -> None:
        self._records = records
        self.records_requested = 0

    def iter_records(self, start_after: str | None = None):
        started = start_after is None
        for record in self._records:
            if not started:
                if record.record_id == start_after:
                    started = True
                continue
            self.records_requested += 1
            yield record


class FakeExtractor:
    def extract_opening_comment(self, record: InputRecord) -> str | None:
        if record.content.startswith("comment:"):
            return record.content.split(":", maxsplit=1)[1].strip()
        return None


class FakeMultiExtractor:
    def extract_opening_comments(self, record: InputRecord) -> list[ExtractedComment]:
        return [
            ExtractedComment(text="first", start_line=2, index=0),
            ExtractedComment(text="second", start_line=7, index=1),
        ]

    def extract_opening_comment(self, record: InputRecord) -> str | None:
        return "first"


class BlockingThreadExtractor:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self.active = 0
        self.max_active = 0
        self.thread_names: set[str] = set()

    def extract_opening_comment(self, record: InputRecord) -> str | None:
        with self._condition:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.thread_names.add(threading.current_thread().name)
            self._condition.notify_all()
            self._condition.wait_for(lambda: self.max_active >= 2, timeout=1.0)
        try:
            return f"comment {record.record_id}"
        finally:
            with self._condition:
                self.active -= 1
                self._condition.notify_all()


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

            shard_paths = list(config.storage.output_directory.rglob("part-00000.parquet"))
            self.assertEqual(len(shard_paths), 1)
            self.assertEqual(pq.read_schema(shard_paths[0]), COMMENT_SCHEMA)
            parquet_file = pq.ParquetFile(shard_paths[0])
            self.assertEqual(parquet_file.metadata.num_rows, 2)
            payloads = parquet_file.read().to_pylist()
            self.assertEqual(payloads[0]["opening_comment"], "first")
            self.assertEqual(payloads[1]["opening_comment"], "second")
            self.assertFalse(list(config.storage.output_directory.rglob(".*.tmp.*")))

    def test_run_dataset_writes_multiple_comments_from_one_record(self) -> None:
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
                InputRecord(dataset="toy-source", record_id="1", content="code"),
            ]
            stats = run_dataset(FakeSource(records), FakeMultiExtractor(), config)

            self.assertEqual(stats.records_seen, 1)
            self.assertEqual(stats.comments_written, 2)
            self.assertEqual(stats.skipped_without_comment, 0)

            shard_path = next(config.storage.output_directory.rglob("part-00000.parquet"))
            payloads = pq.read_table(shard_path).to_pylist()
            self.assertEqual(
                [payload["record_id"] for payload in payloads],
                ["1::comment::0", "1::comment::1"],
            )
            self.assertEqual(
                [payload["opening_comment"] for payload in payloads],
                ["first", "second"],
            )
            first_metadata = json.loads(payloads[0]["metadata"])
            second_metadata = json.loads(payloads[1]["metadata"])
            self.assertEqual(first_metadata["comment_start_line"], 2)
            self.assertEqual(second_metadata["comment_index"], 1)

    def test_run_dataset_extracts_comments_with_multiple_workers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = PipelineConfig(
                storage=StorageConfig(
                    working_directory=root / "work",
                    output_directory=root / "output",
                    checkpoint_directory=root / "checkpoints",
                    download_directory=root / "downloads",
                    huggingface_cache_directory=root / "hf-cache",
                    max_records_per_shard=20,
                    max_bytes_per_shard=1024,
                ),
                datasets=[],
                checkpoint_interval_records=1,
            )
            records = [
                InputRecord(dataset="toy-source", record_id=str(index), content="code")
                for index in range(8)
            ]
            extractor = BlockingThreadExtractor()

            stats = run_dataset(
                FakeSource(records),
                extractor,
                config,
                extraction_workers=4,
                extraction_buffer=4,
            )

            self.assertEqual(stats.records_seen, 8)
            self.assertEqual(stats.comments_written, 8)
            self.assertGreaterEqual(extractor.max_active, 2)
            self.assertGreaterEqual(len(extractor.thread_names), 2)

            payloads = []
            for shard_path in sorted(config.storage.output_directory.rglob("part-*.parquet")):
                payloads.extend(pq.read_table(shard_path).to_pylist())
            self.assertEqual(
                [payload["record_id"] for payload in payloads],
                [str(index) for index in range(8)],
            )

    def test_run_dataset_does_not_read_past_max_records_with_single_worker(self) -> None:
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
                checkpoint_interval_records=10,
            )
            source = FakeSource(
                [
                    InputRecord(dataset="toy-source", record_id=str(index), content="comment: x")
                    for index in range(3)
                ]
            )

            stats = run_dataset(
                source,
                FakeExtractor(),
                config,
                max_records=2,
                progress_every=0,
                extraction_workers=1,
            )

            self.assertEqual(stats.records_seen, 2)
            self.assertEqual(source.records_requested, 2)


if __name__ == "__main__":
    unittest.main()
