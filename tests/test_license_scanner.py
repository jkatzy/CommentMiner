from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from commentminer.license_scanner import scan_comment_licenses


def _write_input_run(root: Path) -> Path:
    input_directory = root / "input-run"
    input_directory.mkdir(parents=True, exist_ok=True)
    (input_directory / "manifest.json").write_text(
        json.dumps({"dataset": "the-stack", "run_id": "run-1"}, indent=2),
        encoding="utf-8",
    )
    shard_one = input_directory / "part-00000.jsonl"
    shard_one.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "dataset": "the-stack",
                        "record_id": "1",
                        "opening_comment": "Licensed under the Apache License, Version 2.0",
                        "language": "python",
                        "path": "a.py",
                        "repo": "repo-a",
                        "extracted_at": "2026-04-07T00:00:00+00:00",
                        "metadata": {},
                    }
                ),
                json.dumps(
                    {
                        "dataset": "the-stack",
                        "record_id": "2",
                        "opening_comment": "just a normal header",
                        "language": "python",
                        "path": "b.py",
                        "repo": "repo-b",
                        "extracted_at": "2026-04-07T00:00:00+00:00",
                        "metadata": {},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    shard_two = input_directory / "part-00001.jsonl"
    shard_two.write_text(
        json.dumps(
            {
                "dataset": "the-stack",
                "record_id": "3",
                "opening_comment": "SPDX-License-Identifier: MIT",
                "language": "python",
                "path": "c.py",
                "repo": "repo-c",
                "extracted_at": "2026-04-07T00:00:00+00:00",
                "metadata": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return input_directory


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
    def test_scan_comment_licenses_rejects_in_place_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_directory = _write_input_run(root)

            with self.assertRaisesRegex(ValueError, "Output directory must differ"):
                scan_comment_licenses(
                    input_directory,
                    output_directory=input_directory,
                    runner=_fake_scancode_runner,
                )

    def test_scan_comment_licenses_writes_enriched_output_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_directory = _write_input_run(root)

            stats = scan_comment_licenses(
                input_directory,
                runner=_fake_scancode_runner,
                batch_size=2,
            )

            self.assertEqual(stats.records_scanned, 3)
            self.assertEqual(stats.records_with_detected_license, 2)
            self.assertEqual(stats.records_without_detected_license, 1)
            self.assertEqual(stats.shards_processed, 2)
            self.assertEqual(stats.shards_skipped, 0)

            output_directory = input_directory.parent / f"{input_directory.name}-license-scan"
            shard_one_payloads = [
                json.loads(line)
                for line in (output_directory / "part-00000.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                shard_one_payloads[0]["comment_license_detection"]["detected_license_expression"],
                "apache-2.0",
            )
            self.assertTrue(
                shard_one_payloads[0]["comment_license_detection"]["contains_license_notice"]
            )
            self.assertIsNone(
                shard_one_payloads[1]["comment_license_detection"]["detected_license_expression"]
            )
            self.assertFalse(
                shard_one_payloads[1]["comment_license_detection"]["contains_license_notice"]
            )
            shard_two_payload = json.loads(
                (output_directory / "part-00001.jsonl").read_text(encoding="utf-8").strip()
            )
            self.assertEqual(
                shard_two_payload["comment_license_detection"]["detected_license_expression_spdx"],
                "MIT",
            )

            manifest = json.loads((output_directory / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["records_scanned"], 3)
            self.assertEqual(manifest["records_with_detected_license"], 2)
            self.assertEqual(manifest["min_license_score"], 95.0)
            self.assertEqual(manifest["min_match_coverage"], 95.0)
            self.assertEqual(manifest["source_manifest"]["dataset"], "the-stack")

    def test_scan_comment_licenses_resumes_completed_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_directory = _write_input_run(root)
            calls: list[list[str]] = []

            def _counting_runner(**kwargs):
                inputs_dir = Path(kwargs["inputs_dir"])
                calls.append(sorted(path.name for path in inputs_dir.glob("*.txt")))
                return _fake_scancode_runner(**kwargs)

            first_stats = scan_comment_licenses(
                input_directory,
                runner=_counting_runner,
                batch_size=10,
            )
            second_stats = scan_comment_licenses(
                input_directory,
                runner=_counting_runner,
                batch_size=10,
            )

            self.assertEqual(first_stats.shards_processed, 2)
            self.assertEqual(second_stats.shards_processed, 0)
            self.assertEqual(second_stats.shards_skipped, 2)
            self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
