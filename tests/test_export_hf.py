from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq

from commentminer.export_hf import export_huggingface_dataset
from commentminer.parquet_io import normalize_comment_record, write_comment_records


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    write_comment_records(
        path,
        [normalize_comment_record(row) for row in rows],
    )


class HuggingFaceExportTests(unittest.TestCase):
    def test_export_groups_mined_comments_by_dataset_and_language(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            mined = root / "output"
            destination = root / "comment-dataset"
            _write_parquet(
                mined / "the-heap-antlr" / "20260620T000000Z" / "part-00000.parquet",
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
            _write_parquet(
                mined / "the-stack-java" / "20260620T000000Z" / "part-00000.parquet",
                [
                    {
                        "dataset": "the-stack",
                        "record_id": "stack-1",
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
                max_records_per_shard=100,
                max_bytes_per_shard=1024 * 1024,
                dedupe_record_ids=True,
            )

            self.assertEqual(stats.records_written, 2)
            self.assertEqual(stats.records_skipped_duplicate, 1)
            heap_shard = destination / "the-heap" / "ANTLR" / "part-00000.parquet"
            stack_shard = destination / "the-stack" / "Java" / "part-00000.parquet"
            self.assertTrue(heap_shard.exists())
            self.assertTrue(stack_shard.exists())

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

    def test_export_can_write_language_split_dataset_card(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            mined = root / "output"
            destination = root / "comment-dataset"
            _write_parquet(
                mined
                / "the-stack-v2-dedup-package-1"
                / "20260620T000000Z"
                / "part-00000.parquet",
                [
                    {
                        "dataset": "the-stack-v2-dedup",
                        "record_id": "cpp-1",
                        "opening_comment": "// header",
                        "language": "C++",
                        "path": "main.cpp",
                        "repo": "owner/repo",
                        "metadata": {},
                    },
                    {
                        "dataset": "the-stack-v2-dedup",
                        "record_id": "csharp-1",
                        "opening_comment": "// header",
                        "language": "C#",
                        "path": "Program.cs",
                        "repo": "owner/repo",
                        "metadata": {},
                    },
                ],
            )

            export_huggingface_dataset(
                mined,
                destination,
                max_records_per_shard=100,
                max_bytes_per_shard=1024 * 1024,
                dataset_card_layout="language-splits",
            )

            readme = (destination / "README.md").read_text(encoding="utf-8")
            self.assertIn('config_name: "the-stack-v2-dedup"', readme)
            self.assertIn('split: "C_plus_plus"', readme)
            self.assertIn('path: "the-stack-v2-dedup/C++/part-*.parquet"', readme)
            self.assertIn('split: "C_sharp"', readme)
            manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["dataset_card_layout"], "language-splits")

    def test_export_can_dedupe_within_input_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            mined = root / "output"
            destination = root / "comment-dataset"
            row = {
                "dataset": "the-stack-v2-dedup",
                "record_id": "row-1",
                "opening_comment": "// header",
                "language": "Python",
                "path": "a.py",
                "repo": "owner/repo",
                "metadata": {},
            }
            _write_parquet(
                mined / "package-1" / "20260620T000000Z" / "part-00000.parquet",
                [row, row],
            )
            _write_parquet(
                mined / "package-2" / "20260620T000000Z" / "part-00000.parquet",
                [row],
            )

            stats = export_huggingface_dataset(
                mined,
                destination,
                max_records_per_shard=100,
                max_bytes_per_shard=1024 * 1024,
                dedupe_record_ids=True,
                dedupe_scope="input-group",
            )

            self.assertEqual(stats.records_written, 2)
            self.assertEqual(stats.records_skipped_duplicate, 1)
            shard = destination / "the-stack-v2-dedup" / "Python" / "part-00000.parquet"
            self.assertEqual(len(pq.read_table(shard).to_pylist()), 2)

    def test_export_can_write_parallel_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            mined = root / "output"
            destination = root / "comment-dataset"
            for index in range(4):
                _write_parquet(
                    mined
                    / f"package-{index}"
                    / "20260620T000000Z"
                    / "part-00000.parquet",
                    [
                        {
                            "dataset": "the-stack-v2-dedup",
                            "record_id": f"row-{index}",
                            "opening_comment": "// header",
                            "language": "Python",
                            "path": f"{index}.py",
                            "repo": "owner/repo",
                            "metadata": {},
                        }
                    ],
                )

            stats = export_huggingface_dataset(
                mined,
                destination,
                max_records_per_shard=2,
                max_bytes_per_shard=1024 * 1024,
                dedupe_record_ids=True,
                dedupe_scope="input-group",
                workers=2,
                max_open_writers=2,
            )

            self.assertEqual(stats.records_written, 4)
            shards = sorted((destination / "the-stack-v2-dedup" / "Python").glob("part-*.parquet"))
            self.assertGreaterEqual(len(shards), 2)
            self.assertTrue(any(path.name.startswith("part-00000-") for path in shards))
            self.assertTrue(any(path.name.startswith("part-00001-") for path in shards))
            rows = []
            for shard in shards:
                rows.extend(pq.read_table(shard).to_pylist())
            self.assertEqual(sorted(row["record_id"] for row in rows), ["row-0", "row-1", "row-2", "row-3"])


if __name__ == "__main__":
    unittest.main()
