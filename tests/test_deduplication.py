from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from commentminer.deduplication import deduplicate_comment_run


def _write_aggregated_run(root: Path, dataset_name: str, run_name: str, records: list[dict[str, object]]) -> Path:
    run_directory = root / dataset_name / run_name
    run_directory.mkdir(parents=True, exist_ok=True)
    (run_directory / "manifest.json").write_text(
        json.dumps({"dataset": dataset_name, "run_id": run_name}, indent=2),
        encoding="utf-8",
    )
    (run_directory / "part-00000.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    return run_directory


class DeduplicationTests(unittest.TestCase):
    def test_deduplicate_comment_run_groups_comments_by_normalized_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            aggregated_run = _write_aggregated_run(
                root / "output",
                "combined-comments",
                "run-1",
                [
                    {
                        "dataset": "combined-comments",
                        "record_id": "stack-1",
                        "opening_comment": "Licensed under the Apache License, Version 2.0",
                        "language": "python",
                        "path": "a.py",
                        "repo": "repo-a",
                        "extracted_at": "2026-04-07T00:00:00+00:00",
                        "metadata": {"stars": 5},
                        "source_dataset": "the-stack",
                    },
                    {
                        "dataset": "combined-comments",
                        "record_id": "rp-1",
                        "opening_comment": "Licensed under the Apache License Version 2 0",
                        "language": "python",
                        "path": "b.py",
                        "repo": "repo-b",
                        "extracted_at": "2026-04-07T00:00:00+00:00",
                        "metadata": {"license": "apache-2.0"},
                        "source_dataset": "redpajama-github",
                    },
                    {
                        "dataset": "combined-comments",
                        "record_id": "stack-2",
                        "opening_comment": "This file is part of project X",
                        "language": "python",
                        "path": "c.py",
                        "repo": "repo-c",
                        "extracted_at": "2026-04-07T00:00:00+00:00",
                        "metadata": {"stars": 1},
                        "source_dataset": "the-stack",
                    },
                ],
            )

            stats = deduplicate_comment_run(
                aggregated_run,
                hash_workers=2,
                hash_batch_size=2,
                sort_parallelism=2,
            )

            self.assertEqual(stats.input_dataset_name, "combined-comments")
            self.assertEqual(stats.dataset_name, "combined-comments-deduplicated")
            self.assertEqual(stats.records_seen, 3)
            self.assertEqual(stats.unique_comments, 2)
            self.assertEqual(stats.duplicate_occurrences, 1)
            self.assertEqual(stats.shards_written, 1)

            output_shard = stats.output_directory / "part-00000.jsonl"
            payloads = [json.loads(line) for line in output_shard.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(payloads), 2)

            duplicate_group = next(item for item in payloads if item["occurrence_count"] == 2)
            self.assertEqual(duplicate_group["dataset"], "combined-comments-deduplicated")
            self.assertEqual(
                duplicate_group["source_datasets"],
                ["redpajama-github", "the-stack"],
            )
            self.assertEqual(
                [occurrence["record_id"] for occurrence in duplicate_group["occurrences"]],
                ["stack-1", "rp-1"],
            )
            self.assertNotIn("opening_comment", duplicate_group["occurrences"][0])

            manifest = json.loads((stats.output_directory / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["dataset"], "combined-comments-deduplicated")
            self.assertEqual(manifest["records_seen"], 3)
            self.assertEqual(manifest["unique_comments"], 2)
            self.assertEqual(manifest["duplicate_occurrences"], 1)
            self.assertEqual(
                manifest["normalization"],
                "remove all whitespace and non-alphanumeric characters",
            )

    def test_deduplicate_comment_run_rejects_output_root_inside_input_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            aggregated_run = _write_aggregated_run(
                root / "output",
                "combined-comments",
                "run-1",
                [
                    {
                        "dataset": "combined-comments",
                        "record_id": "stack-1",
                        "opening_comment": "hello",
                        "language": "python",
                        "path": "a.py",
                        "repo": "repo-a",
                        "extracted_at": "2026-04-07T00:00:00+00:00",
                        "metadata": {},
                        "source_dataset": "the-stack",
                    }
                ],
            )

            with self.assertRaisesRegex(ValueError, "must not be inside the input run directory"):
                deduplicate_comment_run(
                    aggregated_run,
                    output_root=aggregated_run,
                )

    def test_deduplicate_comment_run_fails_hard_and_cleans_output_on_missing_sort(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            aggregated_run = _write_aggregated_run(
                root / "output",
                "combined-comments",
                "run-1",
                [
                    {
                        "dataset": "combined-comments",
                        "record_id": "stack-1",
                        "opening_comment": "hello",
                        "language": "python",
                        "path": "a.py",
                        "repo": "repo-a",
                        "extracted_at": "2026-04-07T00:00:00+00:00",
                        "metadata": {},
                        "source_dataset": "the-stack",
                    }
                ],
            )
            dedup_output_root = root / "dedup-output"

            with self.assertRaises(FileNotFoundError):
                deduplicate_comment_run(
                    aggregated_run,
                    output_root=dedup_output_root,
                    dataset_name="combined-comments-deduplicated",
                    sort_command="definitely-not-a-real-sort-command",
                )

            self.assertFalse((dedup_output_root / "combined-comments-deduplicated").exists())


if __name__ == "__main__":
    unittest.main()
