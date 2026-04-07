from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from commentminer.aggregation import aggregate_comment_runs


def _write_comment_run(root: Path, dataset_name: str, run_name: str, records: list[dict[str, object]]) -> Path:
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


class AggregationTests(unittest.TestCase):
    def test_aggregate_comment_runs_combines_runs_and_tracks_source_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            stack_run = _write_comment_run(
                root / "output",
                "the-stack",
                "run-1",
                [
                    {
                        "dataset": "the-stack",
                        "record_id": "stack-1",
                        "opening_comment": "stack comment",
                        "language": "python",
                        "path": "a.py",
                        "repo": "repo-a",
                        "extracted_at": "2026-04-07T00:00:00+00:00",
                        "metadata": {"stars": 5},
                    }
                ],
            )
            redpajama_run = _write_comment_run(
                root / "output",
                "redpajama-github",
                "run-2",
                [
                    {
                        "dataset": "redpajama-github",
                        "record_id": "rp-1",
                        "opening_comment": "redpajama comment",
                        "language": "python",
                        "path": "b.py",
                        "repo": "repo-b",
                        "extracted_at": "2026-04-07T00:00:00+00:00",
                        "metadata": {"license": "mit"},
                    }
                ],
            )

            stats = aggregate_comment_runs(
                [stack_run, redpajama_run],
                dataset_name="all-comments",
                source_field="source_dataset",
            )

            self.assertEqual(stats.dataset_name, "all-comments")
            self.assertEqual(stats.records_aggregated, 2)
            self.assertEqual(stats.shards_written, 1)
            self.assertEqual(stats.source_datasets, ["redpajama-github", "the-stack"])

            output_shard = stats.output_directory / "part-00000.jsonl"
            payloads = [json.loads(line) for line in output_shard.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([item["dataset"] for item in payloads], ["all-comments", "all-comments"])
            self.assertEqual(
                [item["source_dataset"] for item in payloads],
                ["the-stack", "redpajama-github"],
            )

            manifest = json.loads((stats.output_directory / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["dataset"], "all-comments")
            self.assertEqual(manifest["source_field"], "source_dataset")
            self.assertEqual(manifest["records_aggregated"], 2)
            self.assertEqual(
                manifest["source_datasets"],
                ["redpajama-github", "the-stack"],
            )

    def test_aggregate_comment_runs_rejects_output_root_inside_input_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            stack_run = _write_comment_run(
                root / "output",
                "the-stack",
                "run-1",
                [
                    {
                        "dataset": "the-stack",
                        "record_id": "stack-1",
                        "opening_comment": "stack comment",
                        "language": "python",
                        "path": "a.py",
                        "repo": "repo-a",
                        "extracted_at": "2026-04-07T00:00:00+00:00",
                        "metadata": {},
                    }
                ],
            )

            with self.assertRaisesRegex(ValueError, "must not be inside an input run directory"):
                aggregate_comment_runs(
                    [stack_run],
                    output_root=stack_run,
                )


if __name__ == "__main__":
    unittest.main()
