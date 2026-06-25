from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq

from commentminer.export_hf import export_huggingface_dataset


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


class HuggingFaceExportTests(unittest.TestCase):
    def test_export_groups_mined_comments_by_dataset_and_language(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            mined = root / "output"
            destination = root / "comment-dataset"
            _write_jsonl(
                mined / "the-heap-antlr" / "20260620T000000Z" / "part-00000.jsonl",
                [
                    {
                        "dataset": "the-heap",
                        "record_id": "heap-1",
                        "opening_comment": "/* header */",
                        "language": "ANTLR",
                        "path": "Grammar.g4",
                        "repo": "owner/repo",
                        "metadata": {"row_index": 0},
                    },
                    {
                        "dataset": "the-heap",
                        "record_id": "heap-1",
                        "opening_comment": "/* duplicate */",
                        "language": "ANTLR",
                        "path": "Grammar.g4",
                        "repo": "owner/repo",
                        "metadata": {"row_index": 0},
                    },
                ],
            )
            _write_jsonl(
                mined / "redpajama-github-java" / "20260620T000000Z" / "part-00000.jsonl",
                [
                    {
                        "dataset": "redpajama-github",
                        "record_id": "rp-1",
                        "opening_comment": "// header",
                        "language": "Java",
                        "path": "A.java",
                        "repo": "owner/repo",
                        "metadata": {"line_index": 1},
                    }
                ],
            )

            stats = export_huggingface_dataset(
                mined,
                destination,
                output_format="parquet",
                max_records_per_shard=100,
                max_bytes_per_shard=1024 * 1024,
                dedupe_record_ids=True,
            )

            self.assertEqual(stats.records_written, 2)
            self.assertEqual(stats.records_skipped_duplicate, 1)
            heap_shard = destination / "the-heap" / "ANTLR" / "part-00000.parquet"
            redpajama_shard = destination / "redpajama-github" / "Java" / "part-00000.parquet"
            self.assertTrue(heap_shard.exists())
            self.assertTrue(redpajama_shard.exists())

            heap_rows = pq.read_table(heap_shard).to_pylist()
            self.assertEqual(len(heap_rows), 1)
            self.assertEqual(heap_rows[0]["record_id"], "heap-1")
            self.assertEqual(json.loads(heap_rows[0]["metadata"]), {"row_index": 0})

            manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["records_written"], 2)
            self.assertEqual(manifest["records_skipped_duplicate"], 1)
            self.assertEqual(manifest["format"], "parquet")
            self.assertEqual(manifest["layout"], "<dataset>/<language>/part-*.parquet")
            self.assertIn("the-heap__ANTLR", (destination / "README.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
