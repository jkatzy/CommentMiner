from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from commentminer.encoding_benchmark import (
    EncodingModelSpec,
    load_encoding_model_specs,
    parse_encoding_model_spec,
    run_encoding_capacity_benchmark,
)


class FakeParameter:
    def __init__(self, count: int) -> None:
        self.count = count

    def numel(self) -> int:
        return self.count


class FakeEncoder:
    def __init__(self, *, parameter_count: int = 22_000_000, fail_above: int | None = None) -> None:
        self.parameter_count = parameter_count
        self.fail_above = fail_above
        self.calls: list[int] = []

    def parameters(self):
        return [FakeParameter(self.parameter_count)]

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ):
        self.calls.append(len(texts))
        if self.fail_above is not None and len(texts) > self.fail_above:
            raise RuntimeError(f"capacity exceeded at {len(texts)}")
        return np.zeros((len(texts), 4), dtype=np.float32)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


class EncodingBenchmarkTests(unittest.TestCase):
    def test_parse_model_specs_validate_supported_parameter_range(self) -> None:
        spec = parse_encoding_model_spec("sentence-transformers/all-MiniLM-L6-v2=22M")
        self.assertEqual(spec.model_id, "sentence-transformers/all-MiniLM-L6-v2")
        self.assertEqual(spec.parameter_count, 22_000_000)

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = Path(tmp_dir) / "models.json"
            config.write_text(
                json.dumps(
                    {
                        "models": [
                            {
                                "model_id": "example/model",
                                "parameters": "0.6B",
                                "revision": "main",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            specs = load_encoding_model_specs(["other/model=2M"], model_config=config)

        self.assertEqual([item.model_id for item in specs], ["example/model", "other/model"])
        self.assertEqual(specs[0].parameter_count, 600_000_000)
        self.assertEqual(specs[0].revision, "main")
        self.assertEqual(specs[1].parameter_count, 2_000_000)

        with self.assertRaisesRegex(ValueError, "outside the supported 2M-8B range"):
            parse_encoding_model_spec("too-small/model=1M")
        with self.assertRaisesRegex(ValueError, "outside the supported 2M-8B range"):
            parse_encoding_model_spec("too-large/model=8.1B")

    def test_benchmarks_jsonl_sample_counts_and_records_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_directory = root / "comments"
            output_directory = root / "benchmark"
            _write_jsonl(
                input_directory / "part-00000.jsonl",
                [
                    {"dataset": "combined", "language": "Python", "opening_comment": "first"},
                    {"dataset": "combined", "language": "Python", "opening_comment": "second"},
                    {"dataset": "combined", "language": "Python", "opening_comment": "third"},
                    {"dataset": "combined", "language": "Python", "opening_comment": "fourth"},
                    {"dataset": "combined", "language": "Python", "opening_comment": ""},
                ],
            )
            encoder = FakeEncoder(fail_above=2)

            def _loader(spec, device, cache_folder, trust_remote_code):
                return encoder

            stats = run_encoding_capacity_benchmark(
                input_directory,
                [EncodingModelSpec("fake/encoder", parameter_count=22_000_000)],
                output_directory=output_directory,
                sample_counts=[1, 2, 4],
                batch_size=2,
                encoder_loader=_loader,
            )

            self.assertEqual(stats.input_format, "jsonl")
            self.assertEqual(stats.records_seen, 5)
            self.assertEqual(stats.records_without_text, 1)
            self.assertEqual(stats.samples_loaded, 4)
            self.assertEqual(encoder.calls, [1, 2, 4])

            report = json.loads((output_directory / "encoding-capacity-report.json").read_text(encoding="utf-8"))
            model = report["models"][0]
            self.assertEqual(model["largest_successful_sample_count"], 2)
            self.assertEqual([step["sample_count"] for step in model["steps"]], [1, 2, 4])
            self.assertTrue(model["steps"][1]["success"])
            self.assertFalse(model["steps"][2]["success"])
            self.assertEqual(model["steps"][0]["embedding_shape"], [1, 4])
            self.assertIn("capacity exceeded", model["steps"][2]["error"]["message"])

            summary = (output_directory / "encoding-capacity-summary.csv").read_text(encoding="utf-8")
            self.assertIn("fake/encoder", summary)
            self.assertIn("largest_successful_sample_count", summary)

    def test_benchmarks_huggingface_parquet_comments_with_filters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_directory = root / "hf-comments"
            python_dir = input_directory / "the-stack" / "Python"
            js_dir = input_directory / "the-stack" / "JavaScript"
            python_dir.mkdir(parents=True)
            js_dir.mkdir(parents=True)
            schema = pa.schema(
                [
                    ("dataset", pa.string()),
                    ("language", pa.string()),
                    ("opening_comment", pa.string()),
                ]
            )
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {"dataset": "the-stack", "language": "Python", "opening_comment": "one"},
                        {"dataset": "the-stack", "language": "Python", "opening_comment": "two"},
                    ],
                    schema=schema,
                ),
                python_dir / "part-00000.parquet",
            )
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {"dataset": "the-stack", "language": "JavaScript", "opening_comment": "skip"},
                    ],
                    schema=schema,
                ),
                js_dir / "part-00000.parquet",
            )

            encoder = FakeEncoder()

            def _loader(spec, device, cache_folder, trust_remote_code):
                return encoder

            stats = run_encoding_capacity_benchmark(
                input_directory,
                [EncodingModelSpec("fake/encoder", parameter_count=22_000_000)],
                output_directory=root / "benchmark",
                dataset_names=["the-stack"],
                languages=["Python"],
                sample_counts=[1, 2],
                encoder_loader=_loader,
            )

            self.assertEqual(stats.input_format, "parquet")
            self.assertEqual(stats.records_seen, 2)
            self.assertEqual(stats.samples_loaded, 2)
            self.assertEqual(stats.shards_read, 2)
            self.assertEqual(encoder.calls, [1, 2])


if __name__ == "__main__":
    unittest.main()
