from __future__ import annotations

import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from commentminer import license_scanner
from commentminer.license_scanner import (
    build_license_score_histogram,
    prewarm_huggingface_license_detection_cache,
    scan_huggingface_comment_licenses,
)


def _write_hf_dataset(root: Path) -> Path:
    input_directory = root / "comment-dataset"
    input_directory.mkdir(parents=True, exist_ok=True)
    (input_directory / "manifest.json").write_text(
        json.dumps({"records_written": 3, "groups": []}, indent=2),
        encoding="utf-8",
    )
    schema = pa.schema(
        [
            ("dataset", pa.string()),
            ("record_id", pa.string()),
            ("opening_comment", pa.string()),
            ("language", pa.string()),
            ("path", pa.string()),
            ("repo", pa.string()),
            ("extracted_at", pa.string()),
            ("metadata", pa.string()),
        ]
    )
    python_dir = input_directory / "the-stack" / "Python"
    python_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "dataset": "the-stack",
                    "record_id": "1",
                    "opening_comment": "Licensed under the Apache License, Version 2.0",
                    "language": "Python",
                    "path": "a.py",
                    "repo": "repo-a",
                    "extracted_at": "2026-04-07T00:00:00+00:00",
                    "metadata": "{}",
                },
                {
                    "dataset": "the-stack",
                    "record_id": "2",
                    "opening_comment": "just a normal header",
                    "language": "Python",
                    "path": "b.py",
                    "repo": "repo-b",
                    "extracted_at": "2026-04-07T00:00:00+00:00",
                    "metadata": "{}",
                },
            ],
            schema=schema,
        ),
        python_dir / "part-00000.parquet",
    )
    js_dir = input_directory / "the-stack" / "JavaScript"
    js_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "dataset": "the-stack",
                    "record_id": "3",
                    "opening_comment": "SPDX-License-Identifier: MIT",
                    "language": "JavaScript",
                    "path": "c.js",
                    "repo": "repo-c",
                    "extracted_at": "2026-04-07T00:00:00+00:00",
                    "metadata": "{}",
                },
            ],
            schema=schema,
        ),
        js_dir / "part-00000.parquet",
    )
    return input_directory


def _fake_api_mit_canary_result() -> dict[str, object]:
    return {
        "detected_license_expression": "mit",
        "detected_license_expression_spdx": "MIT",
        "percentage_of_license_text": 100,
        "license_detections": [
            {
                "license_expression": "mit",
                "license_expression_spdx": "MIT",
                "matches": [{"score": 100.0, "match_coverage": 100.0}],
            }
        ],
        "scan_errors": [],
    }


def _fake_scancode_runner(**kwargs):
    inputs_dir = Path(kwargs["inputs_dir"])
    resources = []
    for path in sorted(inputs_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        if "Apache License" in text:
            resources.append(
                {
                    "path": path.name,
                    "detected_license_expression": "apache-2.0",
                    "detected_license_expression_spdx": "Apache-2.0",
                    "percentage_of_license_text": 100,
                    "license_detections": [
                        {
                            "license_expression": "apache-2.0",
                            "license_expression_spdx": "Apache-2.0",
                            "matches": [{"score": 97.0, "match_coverage": 99.0}],
                        }
                    ],
                    "scan_errors": [],
                }
            )
        elif "SPDX-License-Identifier: MIT" in text:
            resources.append(
                {
                    "path": path.name,
                    "detected_license_expression": "mit",
                    "detected_license_expression_spdx": "MIT",
                    "percentage_of_license_text": 100,
                    "license_detections": [
                        {
                            "license_expression": "mit",
                            "license_expression_spdx": "MIT",
                            "matches": [{"score": 100.0, "match_coverage": 100.0}],
                        }
                    ],
                    "scan_errors": [],
                }
            )
        else:
            resources.append(
                {
                    "path": path.name,
                    "detected_license_expression": "mit",
                    "detected_license_expression_spdx": "MIT",
                    "percentage_of_license_text": 50,
                    "license_detections": [
                        {
                            "license_expression": "mit",
                            "license_expression_spdx": "MIT",
                            "matches": [{"score": 80.0, "match_coverage": 90.0}],
                        }
                    ],
                    "scan_errors": [],
                }
            )
    return {"headers": [{"tool_name": "scancode-toolkit", "tool_version": "test"}], "files": resources}


class LicenseScannerTests(unittest.TestCase):
    def test_scan_huggingface_comment_licenses_writes_enriched_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_directory = _write_hf_dataset(root)

            stats = scan_huggingface_comment_licenses(
                input_directory,
                runner=_fake_scancode_runner,
                batch_size=2,
            )

            self.assertEqual(stats.records_scanned, 3)
            self.assertEqual(stats.records_with_detected_license, 2)
            self.assertEqual(stats.records_without_detected_license, 1)
            self.assertEqual(stats.shards_processed, 2)

            output_directory = input_directory.parent / f"{input_directory.name}-license-scan"
            output_shard = output_directory / "the-stack" / "Python" / "part-00000.parquet"
            rows = pq.read_table(output_shard).to_pylist()
            self.assertIn("comment_license_detection", rows[0])
            self.assertIn("comment_license_score", rows[0])
            self.assertEqual(rows[0]["comment_license_score"], 97.0)
            detection = json.loads(rows[0]["comment_license_detection"])
            self.assertTrue(detection["contains_license_notice"])
            self.assertEqual(detection["best_license_score"], 97.0)
            self.assertEqual(detection["detected_license_expression"], "apache-2.0")
            normal_detection = json.loads(rows[1]["comment_license_detection"])
            self.assertFalse(normal_detection["contains_license_notice"])
            self.assertEqual(rows[1]["comment_license_score"], 80.0)

            manifest = json.loads((output_directory / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["records_scanned"], 3)
            self.assertEqual(manifest["records_with_detected_license"], 2)
            self.assertEqual(manifest["source_manifest"]["records_written"], 3)
            self.assertEqual(len(manifest["output_shards"]), 2)

    def test_build_license_score_histogram_reads_parquet_scan_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_directory = _write_hf_dataset(root)

            scan_huggingface_comment_licenses(
                input_directory,
                runner=_fake_scancode_runner,
                batch_size=2,
            )

            output_directory = input_directory.parent / f"{input_directory.name}-license-scan"
            histogram = build_license_score_histogram(
                output_directory,
                bins=10,
                languages=["Python"],
            )
            self.assertEqual(histogram.shard_format, "parquet")
            self.assertEqual(histogram.shards_read, 1)
            self.assertEqual(histogram.records_seen, 2)
            self.assertEqual(histogram.scores_seen, 2)
            self.assertEqual(histogram.bin_counts[8], 1)
            self.assertEqual(histogram.bin_counts[9], 1)

            rendered = license_scanner.format_license_score_histogram(histogram, width=12)
            self.assertIn("ScanCode score histogram", rendered)
            self.assertIn("Scores read: 2", rendered)

    def test_scan_huggingface_comment_licenses_api_backend_uses_scancode_api(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_directory = _write_hf_dataset(root)
            calls: list[tuple[object, float, str]] = []

            def _fake_get_licenses(location, min_score=0, **kwargs):
                query_string = str(kwargs.get("query_string", ""))
                calls.append((location, float(min_score), query_string))
                if query_string == "MIT License":
                    return _fake_api_mit_canary_result()
                if "Apache License" in query_string:
                    return {
                        "detected_license_expression": "apache-2.0",
                        "detected_license_expression_spdx": "Apache-2.0",
                        "percentage_of_license_text": 100,
                        "license_detections": [
                            {
                                "license_expression": "apache-2.0",
                                "license_expression_spdx": "Apache-2.0",
                                "matches": [{"score": 97.0, "match_coverage": 99.0}],
                            }
                        ],
                    }
                return {
                    "detected_license_expression": None,
                    "detected_license_expression_spdx": None,
                    "percentage_of_license_text": 0,
                    "license_detections": [],
                }

            with mock.patch("scancode.api.get_licenses", side_effect=_fake_get_licenses):
                stats = scan_huggingface_comment_licenses(
                    input_directory,
                    scanner_backend="api",
                    batch_size=10,
                    languages=["Python"],
                )

            self.assertEqual(stats.records_scanned, 2)
            self.assertEqual(stats.records_with_detected_license, 1)
            self.assertEqual(len(calls), 3)
            self.assertEqual({call[1] for call in calls}, {0.0})
            self.assertTrue(all(call[0] is None for call in calls))

            output_directory = input_directory.parent / f"{input_directory.name}-license-scan"
            rows = pq.read_table(output_directory / "the-stack" / "Python" / "part-00000.parquet").to_pylist()
            detections = [json.loads(row["comment_license_detection"]) for row in rows]
            self.assertEqual(
                [detection["contains_license_notice"] for detection in detections],
                [True, False],
            )
            self.assertEqual([row["comment_license_score"] for row in rows], [97.0, 0.0])

    def test_scan_huggingface_comment_licenses_fails_before_writing_when_api_is_broken(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_directory = _write_hf_dataset(root)

            with (
                mock.patch(
                    "scancode.api.get_licenses",
                    side_effect=RuntimeError("license index unavailable"),
                ),
                self.assertRaisesRegex(RuntimeError, "ScanCode Python API failed"),
            ):
                scan_huggingface_comment_licenses(
                    input_directory,
                    scanner_backend="api",
                    batch_size=10,
                )

            output_directory = input_directory.parent / f"{input_directory.name}-license-scan"
            self.assertFalse(output_directory.exists())

    def test_scan_huggingface_comment_licenses_requires_opening_comment_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_directory = _write_hf_dataset(root)
            shard = input_directory / "the-stack" / "Python" / "part-00000.parquet"
            pq.write_table(pq.read_table(shard).drop_columns(["opening_comment"]), shard)

            with self.assertRaisesRegex(ValueError, "opening_comment"):
                scan_huggingface_comment_licenses(
                    input_directory,
                    runner=_fake_scancode_runner,
                    batch_size=10,
                    languages=["Python"],
                )

    def test_scan_huggingface_comment_licenses_truncates_oversized_api_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_directory = _write_hf_dataset(root)
            python_shard = input_directory / "the-stack" / "Python" / "part-00000.parquet"
            table = pq.read_table(python_shard)
            rows = table.to_pylist()
            rows[0]["opening_comment"] = ("x" * 20) + " Apache License"
            pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), python_shard)
            query_strings: list[str] = []

            def _fake_get_licenses(location, min_score=0, **kwargs):
                query_strings.append(str(kwargs.get("query_string", "")))
                if query_strings[-1] == "MIT License":
                    return _fake_api_mit_canary_result()
                return {
                    "detected_license_expression": None,
                    "detected_license_expression_spdx": None,
                    "percentage_of_license_text": 0,
                    "license_detections": [],
                }

            with (
                mock.patch.object(license_scanner, "_MAX_SCANCODE_API_QUERY_CHARS", 12),
                mock.patch("scancode.api.get_licenses", side_effect=_fake_get_licenses),
            ):
                stats = scan_huggingface_comment_licenses(
                    input_directory,
                    scanner_backend="api",
                    batch_size=10,
                    languages=["Python"],
                )

            self.assertEqual(stats.records_scanned, 2)
            self.assertEqual(query_strings[0], "MIT License")
            self.assertEqual(query_strings[1], "x" * 12)

            output_directory = input_directory.parent / f"{input_directory.name}-license-scan"
            output_rows = pq.read_table(
                output_directory / "the-stack" / "Python" / "part-00000.parquet"
            ).to_pylist()
            detection = json.loads(output_rows[0]["comment_license_detection"])
            self.assertFalse(detection["contains_license_notice"])
            self.assertIn(
                "Opening comment truncated from 35 to 12 characters",
                detection["scan_errors"][0],
            )
            self.assertEqual(output_rows[0]["comment_license_score"], 0.0)

    def test_scan_huggingface_comment_licenses_treats_null_comment_as_empty_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_directory = _write_hf_dataset(root)
            python_shard = input_directory / "the-stack" / "Python" / "part-00000.parquet"
            table = pq.read_table(python_shard)
            rows = table.to_pylist()
            rows[0]["opening_comment"] = None
            pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), python_shard)
            query_strings: list[str] = []

            def _fake_get_licenses(location, min_score=0, **kwargs):
                query_strings.append(str(kwargs.get("query_string", "")))
                if query_strings[-1] == "MIT License":
                    return _fake_api_mit_canary_result()
                return {
                    "detected_license_expression": None,
                    "detected_license_expression_spdx": None,
                    "percentage_of_license_text": 0,
                    "license_detections": [],
                    "license_clues": [],
                }

            with mock.patch("scancode.api.get_licenses", side_effect=_fake_get_licenses):
                stats = scan_huggingface_comment_licenses(
                    input_directory,
                    scanner_backend="api",
                    batch_size=10,
                    languages=["Python"],
                )

            self.assertEqual(stats.records_scanned, 2)
            self.assertIn("", query_strings)
            self.assertNotIn("None", query_strings)

            output_directory = input_directory.parent / f"{input_directory.name}-license-scan"
            output_rows = pq.read_table(
                output_directory / "the-stack" / "Python" / "part-00000.parquet"
            ).to_pylist()
            self.assertEqual([row["comment_license_score"] for row in output_rows], [0.0, 0.0])

    def test_scan_huggingface_comment_licenses_retries_broken_worker_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_directory = _write_hf_dataset(root)
            submitted_worker_counts: list[int] = []

            class _FakeFuture:
                def __init__(self, result=None, exception=None):
                    self._result = result
                    self._exception = exception

                def result(self):
                    if self._exception is not None:
                        raise self._exception
                    return self._result

            class _FakeExecutor:
                pool_count = 0

                def __init__(self, **kwargs):
                    type(self).pool_count += 1
                    self.pool_number = type(self).pool_count
                    submitted_worker_counts.append(kwargs["max_workers"])

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, traceback):
                    return False

                def submit(self, fn, input_shard, input_directory_arg, output_directory, *args, **kwargs):
                    if self.pool_number == 1:
                        return _FakeFuture(
                            exception=license_scanner.BrokenProcessPool(
                                "simulated worker failure"
                            )
                        )
                    result = fn(
                        input_shard,
                        input_directory_arg,
                        output_directory,
                        *args,
                        **kwargs,
                    )
                    return _FakeFuture(result=result)

            with (
                mock.patch.object(license_scanner, "ProcessPoolExecutor", _FakeExecutor),
                mock.patch.object(
                    license_scanner,
                    "as_completed",
                    side_effect=lambda futures: list(futures),
                ),
                mock.patch.object(license_scanner, "_warm_scancode_api", return_value=None),
            ):
                stats = scan_huggingface_comment_licenses(
                    input_directory,
                    scanner_backend="api",
                    batch_size=10,
                    workers=2,
                )

            self.assertEqual(_FakeExecutor.pool_count, 2)
            self.assertEqual(submitted_worker_counts, [2, 1])
            self.assertEqual(stats.shards_processed, 2)

            output_directory = input_directory.parent / f"{input_directory.name}-license-scan"
            self.assertTrue(
                (output_directory / "the-stack" / "Python" / "part-00000.parquet").exists()
            )
            self.assertTrue(
                (output_directory / "the-stack" / "JavaScript" / "part-00000.parquet").exists()
            )

    def test_scan_huggingface_comment_licenses_resumes_completed_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_directory = _write_hf_dataset(root)
            calls: list[list[str]] = []

            def _counting_runner(**kwargs):
                inputs_dir = Path(kwargs["inputs_dir"])
                calls.append(sorted(path.name for path in inputs_dir.glob("*.txt")))
                return _fake_scancode_runner(**kwargs)

            first_stats = scan_huggingface_comment_licenses(
                input_directory,
                runner=_counting_runner,
                batch_size=10,
            )
            second_stats = scan_huggingface_comment_licenses(
                input_directory,
                runner=_counting_runner,
                batch_size=10,
            )

            self.assertEqual(first_stats.shards_processed, 2)
            self.assertEqual(second_stats.shards_processed, 0)
            self.assertEqual(second_stats.shards_skipped, 2)
            self.assertEqual(len(calls), 2)

    def test_scan_huggingface_comment_licenses_rescans_shard_missing_score_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_directory = _write_hf_dataset(root)

            scan_huggingface_comment_licenses(
                input_directory,
                runner=_fake_scancode_runner,
                batch_size=10,
            )

            output_directory = input_directory.parent / f"{input_directory.name}-license-scan"
            python_shard = output_directory / "the-stack" / "Python" / "part-00000.parquet"
            old_table = pq.read_table(python_shard).drop_columns(["comment_license_score"])
            pq.write_table(old_table, python_shard)

            calls: list[list[str]] = []

            def _counting_runner(**kwargs):
                inputs_dir = Path(kwargs["inputs_dir"])
                calls.append(sorted(path.name for path in inputs_dir.glob("*.txt")))
                return _fake_scancode_runner(**kwargs)

            stats = scan_huggingface_comment_licenses(
                input_directory,
                runner=_counting_runner,
                batch_size=10,
            )

            self.assertEqual(stats.shards_processed, 1)
            self.assertEqual(stats.shards_skipped, 1)
            schema = pq.read_schema(python_shard)
            self.assertIn("comment_license_score", schema.names)

    def test_scan_huggingface_comment_licenses_rescans_when_thresholds_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_directory = _write_hf_dataset(root)

            scan_huggingface_comment_licenses(
                input_directory,
                runner=_fake_scancode_runner,
                batch_size=10,
            )
            rescanned = scan_huggingface_comment_licenses(
                input_directory,
                runner=_fake_scancode_runner,
                batch_size=10,
                min_license_score=70,
                min_match_coverage=80,
            )

            self.assertEqual(rescanned.shards_processed, 2)
            self.assertEqual(rescanned.shards_skipped, 0)
            self.assertEqual(rescanned.records_with_detected_license, 3)
            output_directory = input_directory.parent / f"{input_directory.name}-license-scan"
            manifest = json.loads((output_directory / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["scanner_backend"], "api")
            self.assertEqual(manifest["scan_configuration"]["min_match_coverage"], 80.0)

    def test_scan_huggingface_comment_licenses_reuses_duplicate_comment_scans(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_directory = _write_hf_dataset(root)
            schema = pa.schema(
                [
                    ("dataset", pa.string()),
                    ("record_id", pa.string()),
                    ("opening_comment", pa.string()),
                    ("language", pa.string()),
                    ("path", pa.string()),
                    ("repo", pa.string()),
                    ("extracted_at", pa.string()),
                    ("metadata", pa.string()),
                ]
            )
            duplicate_comment = "Licensed under the Apache License, Version 2.0"
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "dataset": "the-stack",
                            "record_id": "1",
                            "opening_comment": duplicate_comment,
                            "language": "Python",
                            "path": "a.py",
                            "repo": "repo-a",
                            "extracted_at": "2026-04-07T00:00:00+00:00",
                            "metadata": "{}",
                        },
                        {
                            "dataset": "the-stack",
                            "record_id": "2",
                            "opening_comment": duplicate_comment,
                            "language": "Python",
                            "path": "b.py",
                            "repo": "repo-b",
                            "extracted_at": "2026-04-07T00:00:00+00:00",
                            "metadata": "{}",
                        },
                        {
                            "dataset": "the-stack",
                            "record_id": "3",
                            "opening_comment": duplicate_comment,
                            "language": "Python",
                            "path": "c.py",
                            "repo": "repo-c",
                            "extracted_at": "2026-04-07T00:00:00+00:00",
                            "metadata": "{}",
                        },
                        {
                            "dataset": "the-stack",
                            "record_id": "4",
                            "opening_comment": "just a normal header",
                            "language": "Python",
                            "path": "d.py",
                            "repo": "repo-d",
                            "extracted_at": "2026-04-07T00:00:00+00:00",
                            "metadata": "{}",
                        },
                    ],
                    schema=schema,
                ),
                input_directory / "the-stack" / "Python" / "part-00000.parquet",
            )
            calls: list[list[str]] = []

            def _counting_runner(**kwargs):
                inputs_dir = Path(kwargs["inputs_dir"])
                calls.append(sorted(path.name for path in inputs_dir.glob("*.txt")))
                return _fake_scancode_runner(**kwargs)

            stats = scan_huggingface_comment_licenses(
                input_directory,
                runner=_counting_runner,
                batch_size=10,
                languages=["Python"],
            )

            self.assertEqual(stats.records_scanned, 4)
            self.assertEqual(stats.records_with_detected_license, 3)
            self.assertEqual(len(calls), 1)
            self.assertEqual(len(calls[0]), 2)

            output_directory = input_directory.parent / f"{input_directory.name}-license-scan"
            rows = pq.read_table(output_directory / "the-stack" / "Python" / "part-00000.parquet").to_pylist()
            detections = [json.loads(row["comment_license_detection"]) for row in rows]
            self.assertEqual(
                [detection["contains_license_notice"] for detection in detections],
                [True, True, True, False],
            )
            self.assertEqual(
                [row["comment_license_score"] for row in rows],
                [97.0, 97.0, 97.0, 80.0],
            )

    def test_scan_huggingface_comment_licenses_reuses_cache_across_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_directory = _write_hf_dataset(root)
            duplicate_comment = "Licensed under the Apache License, Version 2.0"
            js_shard = input_directory / "the-stack" / "JavaScript" / "part-00000.parquet"
            rows = pq.read_table(js_shard).to_pylist()
            rows[0]["opening_comment"] = duplicate_comment
            pq.write_table(pa.Table.from_pylist(rows, schema=pq.read_schema(js_shard)), js_shard)
            calls: list[list[str]] = []

            def _counting_runner(**kwargs):
                inputs_dir = Path(kwargs["inputs_dir"])
                calls.append(sorted(path.name for path in inputs_dir.glob("*.txt")))
                return _fake_scancode_runner(**kwargs)

            stats = scan_huggingface_comment_licenses(
                input_directory,
                runner=_counting_runner,
                batch_size=10,
            )

            self.assertEqual(stats.records_scanned, 3)
            self.assertEqual(stats.records_with_detected_license, 2)
            self.assertEqual(sum(len(call) for call in calls), 2)

            output_directory = input_directory.parent / f"{input_directory.name}-license-scan"
            cache_path = output_directory / "license-detection-cache.sqlite"
            self.assertTrue(cache_path.exists())

    def test_prewarm_huggingface_license_detection_cache_reuses_scan_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_directory = _write_hf_dataset(root)
            cache_path = root / "license-score-cache.sqlite"
            calls: list[list[str]] = []

            def _counting_runner(**kwargs):
                inputs_dir = Path(kwargs["inputs_dir"])
                calls.append(sorted(path.name for path in inputs_dir.glob("*.txt")))
                return _fake_scancode_runner(**kwargs)

            first_stats = prewarm_huggingface_license_detection_cache(
                input_directory,
                detection_cache_path=cache_path,
                runner=_counting_runner,
                batch_size=10,
                languages=["Python"],
            )
            second_stats = prewarm_huggingface_license_detection_cache(
                input_directory,
                detection_cache_path=cache_path,
                runner=_counting_runner,
                batch_size=10,
                languages=["Python"],
            )

            self.assertEqual(first_stats.records_seen, 2)
            self.assertEqual(first_stats.unique_comments_seen, 2)
            self.assertEqual(first_stats.cached_comments, 0)
            self.assertEqual(first_stats.comments_scanned, 2)
            self.assertEqual(first_stats.unique_comments_with_detected_license, 1)
            self.assertEqual(second_stats.records_seen, 2)
            self.assertEqual(second_stats.unique_comments_seen, 2)
            self.assertEqual(second_stats.cached_comments, 2)
            self.assertEqual(second_stats.comments_scanned, 0)
            self.assertEqual(len(calls), 1)
            self.assertEqual(len(calls[0]), 2)
            self.assertTrue(cache_path.exists())
            self.assertFalse((input_directory.parent / f"{input_directory.name}-license-scan").exists())

            def _failing_runner(**kwargs):
                raise AssertionError("scan should reuse prewarmed cache")

            scan_stats = scan_huggingface_comment_licenses(
                input_directory,
                detection_cache_path=cache_path,
                runner=_failing_runner,
                batch_size=10,
                languages=["Python"],
            )

            self.assertEqual(scan_stats.records_scanned, 2)
            output_directory = input_directory.parent / f"{input_directory.name}-license-scan"
            rows = pq.read_table(
                output_directory / "the-stack" / "Python" / "part-00000.parquet"
            ).to_pylist()
            self.assertEqual([row["comment_license_score"] for row in rows], [97.0, 80.0])

    def test_scan_huggingface_comment_licenses_filters_and_limits_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_directory = _write_hf_dataset(root)

            stats = scan_huggingface_comment_licenses(
                input_directory,
                runner=_fake_scancode_runner,
                batch_size=10,
                languages=["Python"],
                max_shards=1,
            )

            self.assertEqual(stats.records_scanned, 2)
            self.assertEqual(stats.shards_processed, 1)
            output_directory = input_directory.parent / f"{input_directory.name}-license-scan"
            self.assertTrue((output_directory / "the-stack" / "Python" / "part-00000.parquet").exists())
            self.assertFalse((output_directory / "the-stack" / "JavaScript" / "part-00000.parquet").exists())


if __name__ == "__main__":
    unittest.main()
