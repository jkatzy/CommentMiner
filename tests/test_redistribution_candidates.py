from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from commentminer.redistribution_candidates import (
    DEFAULT_DATASET,
    DEFAULT_FUZZY_THRESHOLD,
    DEFAULT_LANGUAGE,
    FuzzySeedMatcher,
    SeedPhrase,
    _scan_candidates,
    _max_min_language_allocation,
    _seed_inventory,
    _source_identity,
    _systematic_sample_ranks,
    build_redistribution_candidates,
    verify_redistribution_candidates,
)
from commentminer.redistribution_judge import (
    JUDGMENT_PROFILE_NON_LICENSE_LIMITATIONS,
    LABEL_AMBIGUOUS,
    LABEL_CODE_REDISTRIBUTION_INTENT,
    LABEL_LICENSE_ONLY,
    LABEL_NON_LICENSE_LIMITATION,
    LABEL_NON_LICENSE_LIMITATION_WITH_LICENSE,
    LABEL_OTHER,
)


REMOTE_PATH = "the-stack-v2-dedup/Java/part-00000.parquet"


def _source_row(
    source_row_index: int,
    comment_index: int,
    comment: str,
    *,
    path: str | None = None,
    license_score: float = 0.1,
    license_detection: object = "unknown",
) -> dict[str, object]:
    record_id = (
        f"{REMOTE_PATH}::row::{source_row_index}::comment::{comment_index}"
    )
    return {
        "dataset": DEFAULT_DATASET,
        "record_id": record_id,
        "opening_comment": comment,
        "language": DEFAULT_LANGUAGE,
        "path": path or f"src/File{source_row_index}.java",
        "repo": "example/project",
        "metadata": json.dumps(
            {
                "remote_path": REMOTE_PATH,
                "row_index": source_row_index,
                "comment_index": comment_index,
            }
        ),
        "comment_license_score": license_score,
        "comment_license_detection": (
            license_detection
            if isinstance(license_detection, str)
            else json.dumps(license_detection, sort_keys=True)
        ),
    }


def _write_shard(root: Path, part: int, rows: list[dict[str, object]]) -> Path:
    path = (
        root
        / DEFAULT_DATASET
        / DEFAULT_LANGUAGE
        / f"part-{part:05d}.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def _write_language_shard(
    root: Path,
    language: str,
    rows: list[tuple[str, float]],
) -> Path:
    remote_path = f"data/{language}/train-00000-of-00001.parquet"
    payload = []
    for index, (comment, score) in enumerate(rows):
        payload.append(
            {
                "dataset": DEFAULT_DATASET,
                "record_id": f"{remote_path}::row::{index}::comment::0",
                "opening_comment": comment,
                "language": language,
                "path": f"src/{language}{index}.txt",
                "repo": "example/project",
                "metadata": json.dumps(
                    {
                        "remote_path": remote_path,
                        "row_index": index,
                        "comment_index": 0,
                    }
                ),
                "comment_license_score": score,
                "comment_license_detection": json.dumps(
                    {
                        "contains_license_notice": False,
                        "detected_license_expression": None,
                        "license_matches": [],
                        "scan_errors": [],
                    }
                ),
            }
        )
    path = root / DEFAULT_DATASET / language / "part-00000.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(payload), path)
    return path


def _write_sparse_prefix_fixture(root: Path) -> tuple[str, str]:
    repeated = "/**\r\n * Do\r\n * not\r\n * distribute.\r\n */"
    second = "// Private and confidential."
    _write_shard(
        root,
        0,
        [
            _source_row(0, 0, repeated),
            _source_row(
                0,
                1,
                "// Distribute values evenly across worker threads.",
            ),
            _source_row(50_000, 0, repeated),
        ],
    )
    _write_shard(
        root,
        1,
        [
            _source_row(99_999, 0, second),
            _source_row(99_999, 1, repeated),
            _source_row(100_000, 0, "// Do not distribute this source code."),
        ],
    )
    return repeated, second


def _scan(root: Path, *, source_files_limit: int = 100_000):
    return _scan_candidates(
        root,
        dataset=DEFAULT_DATASET,
        language=DEFAULT_LANGUAGE,
        source_files_limit=source_files_limit,
        fuzzy_threshold=DEFAULT_FUZZY_THRESHOLD,
        include_government_seeds=False,
        include_provenance_seeds=False,
        include_funding_seeds=False,
        include_export_control_seeds=False,
        include_unpublished_work_seeds=False,
        revision="main",
        hf_token=None,
        hf_cache_directory=None,
        batch_size=2,
    )


def _prompt_candidates(prompt: str) -> list[dict[str, object]]:
    marker = "Untrusted candidate data:\n"
    return json.loads(prompt.split(marker, 1)[1])


class FuzzySeedMatcherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.seed = SeedPhrase(
            group="sharing",
            phrase="do not distribute",
            tokens=("do", "not", "distribute"),
            ordinal=0,
        )

    def test_matches_across_line_endings_and_comment_decoration(self) -> None:
        matcher = FuzzySeedMatcher([self.seed], threshold=0.82)

        for newline in ("\n", "\r\n"):
            with self.subTest(newline=repr(newline)):
                comment = newline.join(
                    ["/**", " * Do", " * not", " * distribute.", " */"]
                )
                matches = matcher.match(comment)

                self.assertEqual(len(matches), 1)
                self.assertEqual(matches[0].phrase, "do not distribute")
                self.assertEqual(matches[0].score, 1.0)
                self.assertIn("* not", matches[0].excerpt)

    def test_default_threshold_accepts_one_near_insertion(self) -> None:
        self.assertEqual(DEFAULT_FUZZY_THRESHOLD, 0.82)
        comment = "Do not externally distribute this source."

        accepted = FuzzySeedMatcher(
            [self.seed], threshold=DEFAULT_FUZZY_THRESHOLD
        ).match(comment)
        rejected = FuzzySeedMatcher([self.seed], threshold=0.88).match(comment)

        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0].score, 0.875)
        self.assertEqual(accepted[0].excerpt, "Do not externally distribute")
        self.assertEqual(rejected, [])

    def test_max_min_language_allocation_is_exact_and_capacity_aware(self) -> None:
        allocation = _max_min_language_allocation(
            {"LargeB": 10, "Tiny": 1, "LargeA": 10},
            8,
        )

        self.assertEqual(allocation, {"LargeA": 4, "LargeB": 3, "Tiny": 1})
        self.assertEqual(sum(allocation.values()), 8)

    def test_systematic_ranks_cover_full_capacity_and_spread_subsamples(self) -> None:
        self.assertEqual(_systematic_sample_ranks(4, 4), (0, 1, 2, 3))
        self.assertEqual(_systematic_sample_ranks(10, 3), (1, 5, 8))

    def test_does_not_match_technical_distribution_language(self) -> None:
        matcher = FuzzySeedMatcher([self.seed], threshold=0.82)

        self.assertEqual(
            matcher.match("Distribute values evenly across worker threads."),
            [],
        )
        self.assertEqual(
            matcher.match("Copy values into a shared memory buffer."),
            [],
        )

    def test_expanded_seed_families_are_opt_in_and_deduplicated(self) -> None:
        baseline = _seed_inventory(include_government_seeds=False)
        expanded = _seed_inventory(
            include_government_seeds=True,
            include_provenance_seeds=True,
            include_funding_seeds=True,
            include_export_control_seeds=True,
            include_unpublished_work_seeds=True,
        )

        self.assertEqual(len(baseline), 126)
        self.assertGreater(len(expanded), len(baseline))
        self.assertEqual(len({seed.tokens for seed in expanded}), len(expanded))
        self.assertTrue(
            {
                "proprietary_provenance",
                "funding_dissemination",
                "export_controls",
                "unpublished_work",
                "government_restrictions",
            }.issubset({seed.group for seed in expanded})
        )


class SourceIdentityAndScanTests(unittest.TestCase):
    def test_recovers_source_identity_from_record_id_or_metadata(self) -> None:
        self.assertEqual(
            _source_identity(
                f"{REMOTE_PATH}::row::99999::comment::7",
                None,
            ),
            (REMOTE_PATH, 99_999, 7),
        )
        self.assertEqual(
            _source_identity(
                "legacy-record-id",
                json.dumps(
                    {
                        "remote_path": REMOTE_PATH,
                        "row_index": 42,
                        "comment_index": 3,
                    }
                ),
            ),
            (REMOTE_PATH, 42, 3),
        )

    def test_sparse_source_prefix_keeps_multiple_comments_and_all_occurrences(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            repeated, second = _write_sparse_prefix_fixture(root)

            result = _scan(root)

        self.assertEqual(result.source_remote_path, REMOTE_PATH)
        self.assertEqual(result.comment_bearing_files_seen, 3)
        self.assertEqual(result.comment_rows_seen, 5)
        self.assertEqual(result.shards_scanned, 2)
        self.assertEqual(len(result.source_shards), 2)
        self.assertEqual(len(result.occurrences), 4)
        self.assertEqual(len(result.candidates), 2)

        self.assertEqual(
            [row["source_file_row_index"] for row in result.occurrences],
            [0, 50_000, 99_999, 99_999],
        )
        self.assertEqual(
            [row["source_comment_index"] for row in result.occurrences],
            [0, 0, 0, 1],
        )
        self.assertTrue(
            all(
                row["source_file_row_index"] < 100_000
                for row in result.occurrences
            )
        )
        self.assertFalse(
            any("::row::100000::" in row["record_id"] for row in result.occurrences)
        )

        candidates_by_comment = {
            row["opening_comment"]: row for row in result.candidates
        }
        self.assertEqual(candidates_by_comment[repeated]["occurrence_count"], 3)
        self.assertEqual(candidates_by_comment[second]["occurrence_count"], 1)
        self.assertEqual(
            set(candidates_by_comment[repeated]["occurrence_ids"]),
            {
                row["occurrence_id"]
                for row in result.occurrences
                if row["opening_comment"] == repeated
            },
        )

    def test_rejects_reordered_source_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _write_shard(
                root,
                0,
                [
                    _source_row(5, 0, "// Private and confidential."),
                    _source_row(4, 0, "// Do not distribute."),
                    _source_row(10, 0, "// Boundary."),
                ],
            )

            with self.assertRaisesRegex(ValueError, "monotonically ordered"):
                _scan(root, source_files_limit=10)

    def test_opt_in_seed_families_retrieve_provenance_funding_and_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _write_shard(
                root,
                0,
                [
                    _source_row(0, 0, "// Decompiled by Procyon."),
                    _source_row(
                        1,
                        0,
                        "// Funded by Acme under a grant agreement. "
                        "Publication requires approval.",
                    ),
                    _source_row(
                        2,
                        0,
                        "// Subject to export control laws; export authorization "
                        "required before providing access to foreign persons.",
                    ),
                    _source_row(3, 0, "// Export values to a CSV file."),
                    _source_row(4, 0, "// Boundary row."),
                ],
            )

            result = _scan_candidates(
                root,
                dataset=DEFAULT_DATASET,
                language=DEFAULT_LANGUAGE,
                source_files_limit=4,
                fuzzy_threshold=DEFAULT_FUZZY_THRESHOLD,
                include_government_seeds=False,
                include_provenance_seeds=True,
                include_funding_seeds=True,
                include_export_control_seeds=True,
                include_unpublished_work_seeds=False,
                revision="main",
                hf_token=None,
                hf_cache_directory=None,
                batch_size=8,
            )

        self.assertEqual(len(result.occurrences), 3)
        self.assertEqual(len(result.candidates), 3)
        groups = {
            group
            for row in result.candidates
            for group in row["matched_seed_groups"]
        }
        self.assertTrue(
            {
                "proprietary_provenance",
                "funding_dissemination",
                "export_controls",
            }.issubset(groups)
        )

    def test_unpublished_work_family_retrieves_code_not_publication_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _write_shard(
                root,
                0,
                [
                    _source_row(
                        0,
                        0,
                        "// The source code for this program is not published "
                        "or otherwise divested of its trade secrets.",
                    ),
                    _source_row(
                        1,
                        0,
                        "// The copyright notice does not evidence any actual "
                        "or intended publication of this source code.",
                    ),
                    _source_row(
                        2,
                        0,
                        "// Implements the algorithm from an unpublished paper.",
                    ),
                    _source_row(3, 0, "// Publication date: 2026-07-21."),
                    _source_row(4, 0, "// Boundary row."),
                ],
            )

            result = _scan_candidates(
                root,
                dataset=DEFAULT_DATASET,
                language=DEFAULT_LANGUAGE,
                source_files_limit=4,
                fuzzy_threshold=DEFAULT_FUZZY_THRESHOLD,
                include_government_seeds=False,
                include_provenance_seeds=False,
                include_funding_seeds=False,
                include_export_control_seeds=False,
                include_unpublished_work_seeds=True,
                revision="main",
                hf_token=None,
                hf_cache_directory=None,
                batch_size=8,
            )

        self.assertEqual(len(result.occurrences), 2)
        self.assertEqual(len(result.candidates), 2)
        self.assertTrue(
            all(
                "unpublished_work" in row["matched_seed_groups"]
                for row in result.candidates
            )
        )


class RedistributionCandidateBuildTests(unittest.TestCase):
    def test_all_language_scan_is_balanced_score_filtered_and_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_directory = root / "input"
            output_directory = root / "output"
            repeated = "// Confidential source. Do not distribute."
            _write_language_shard(
                input_directory,
                "Alpha",
                [
                    (repeated, 0.0),
                    ("// Decompiled by Procyon.", 10.0),
                    ("// Copy values into a worker queue.", 89.999),
                    ("// Do not distribute this high-score notice.", 90.0),
                ],
            )
            _write_language_shard(
                input_directory,
                "Beta",
                [
                    (repeated, 20.0),
                    ("// This is unpublished proprietary source code.", 1.0),
                    ("// Do not distribute this excluded notice.", 100.0),
                ],
            )

            stats = build_redistribution_candidates(
                output_directory,
                input_source=input_directory,
                dataset=DEFAULT_DATASET,
                all_languages=True,
                comment_rows_limit=4,
                scancode_score_threshold=0.9,
                scan_workers=2,
                include_provenance_seeds=True,
                include_unpublished_work_seeds=True,
                scan_only=True,
                batch_size=2,
            )

            manifest_path = output_directory / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            candidates = pq.read_table(
                output_directory / "candidates.parquet"
            ).to_pylist()
            occurrences = pq.read_table(
                output_directory / "occurrences.parquet"
            ).to_pylist()

            self.assertEqual(stats.selection_mode, "stratified_comment_rows")
            self.assertEqual(stats.comment_rows_in_scope, 4)
            self.assertEqual(stats.languages_in_scope, 2)
            self.assertEqual(len(occurrences), 3)
            self.assertEqual(len(candidates), 2)
            self.assertEqual(
                sorted(row["comment_license_score"] for row in occurrences),
                [0.0, 1.0, 20.0],
            )
            self.assertEqual(
                manifest["source"]["language_allocations"],
                [
                    {"eligible_rows": 3, "language": "Alpha", "selected_rows": 2},
                    {"eligible_rows": 2, "language": "Beta", "selected_rows": 2},
                ],
            )
            self.assertEqual(
                manifest["parameters"]["scancode_score_threshold_percent"],
                90.0,
            )
            self.assertEqual(manifest["results"]["comment_rows_in_scope"], 4)
            self.assertTrue(
                verify_redistribution_candidates(output_directory).valid
            )

            manifest["source"]["language_allocations"][0]["selected_rows"] = 1
            manifest_path.write_text(json.dumps(manifest))
            tampered = verify_redistribution_candidates(output_directory)
            self.assertFalse(tampered.valid)
            self.assertIn(
                "language allocations do not sum to comment row limit",
                tampered.errors,
            )

    def test_non_license_profile_writes_limitation_and_missed_license_subsets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_directory = root / "input"
            output_directory = root / "judged"
            not_detected = {
                "contains_license_notice": False,
                "detected_license_expression": None,
                "detected_license_expression_spdx": None,
                "license_matches": [],
                "scan_errors": [],
            }
            detected_mit = {
                "contains_license_notice": True,
                "detected_license_expression": "mit",
                "detected_license_expression_spdx": "MIT",
                "license_matches": [{"license_expression": "mit"}],
                "scan_errors": [],
            }
            _write_shard(
                input_directory,
                0,
                [
                    _source_row(
                        0,
                        0,
                        "// Confidential source code. Do not distribute outside Acme.",
                        path="Restricted.java",
                        license_detection=not_detected,
                    ),
                    _source_row(
                        1,
                        0,
                        "// Licensed to you under the Apache License, Version 2.0. Redistribution and use in source and binary forms are permitted.",
                        path="License.java",
                        license_score=94.0,
                        license_detection=not_detected,
                    ),
                    _source_row(
                        2,
                        0,
                        "// Licensed under MIT. Confidential; do not distribute externally.",
                        path="Mixed.java",
                        license_score=100.0,
                        license_detection=detected_mit,
                    ),
                    _source_row(
                        3,
                        0,
                        "// Do not distribute tasks until every worker is ready.",
                        path="Technical.java",
                        license_detection=not_detected,
                    ),
                    _source_row(
                        4,
                        0,
                        "// Boundary row outside the requested prefix.",
                        path="Outside.java",
                        license_detection=not_detected,
                    ),
                ],
            )

            def fake_runner(prompt: str) -> tuple[str, dict[str, int]]:
                decisions = []
                for candidate in _prompt_candidates(prompt):
                    comment = str(candidate["comment"])
                    common = {
                        "candidate_id": candidate["candidate_id"],
                        "confidence": 0.98,
                        "rationale": "Fixture decision.",
                    }
                    if "outside Acme" in comment:
                        decision = {
                            **common,
                            "label": LABEL_NON_LICENSE_LIMITATION,
                            "is_non_license_redistribution_limitation": True,
                            "is_license_notice": False,
                            "is_known_license": False,
                            "known_license": None,
                            "restriction_evidence": "Do not distribute outside Acme",
                            "license_evidence": None,
                            "evidence": "Do not distribute outside Acme",
                        }
                    elif "Apache License" in comment:
                        decision = {
                            **common,
                            "label": LABEL_LICENSE_ONLY,
                            "is_non_license_redistribution_limitation": False,
                            "is_license_notice": True,
                            "is_known_license": True,
                            "known_license": "Apache-2.0",
                            "restriction_evidence": None,
                            "license_evidence": "Apache License, Version 2.0",
                            "evidence": "Apache License, Version 2.0",
                        }
                    elif "Licensed under MIT" in comment:
                        decision = {
                            **common,
                            "label": LABEL_NON_LICENSE_LIMITATION_WITH_LICENSE,
                            "is_non_license_redistribution_limitation": True,
                            "is_license_notice": True,
                            "is_known_license": True,
                            "known_license": "MIT",
                            "restriction_evidence": "do not distribute externally",
                            "license_evidence": "Licensed under MIT",
                            "evidence": "Confidential; do not distribute externally",
                        }
                    else:
                        decision = {
                            **common,
                            "label": LABEL_OTHER,
                            "is_non_license_redistribution_limitation": False,
                            "is_license_notice": False,
                            "is_known_license": False,
                            "known_license": None,
                            "restriction_evidence": None,
                            "license_evidence": None,
                            "evidence": "Do not distribute tasks",
                        }
                    decisions.append(decision)
                return json.dumps({"decisions": decisions}), {}

            stats = build_redistribution_candidates(
                output_directory,
                input_source=input_directory,
                source_files_limit=4,
                judgment_profile=JUDGMENT_PROFILE_NON_LICENSE_LIMITATIONS,
                judge_batch_size=1,
                judge_workers=2,
                judge_max_attempts=1,
                judge_cache_path=root / "cache.sqlite",
                judge_runner=fake_runner,
            )
            report = verify_redistribution_candidates(output_directory)
            limitations = pq.read_table(
                output_directory / "non-license-limitations.parquet"
            ).to_pylist()
            missed = pq.read_table(
                output_directory / "scancode-missed-licenses.parquet"
            ).to_pylist()

            self.assertTrue(report.valid, report.errors)
            self.assertEqual(stats.judged_count, 4)
            self.assertEqual(
                {row["path"] for row in limitations},
                {"Restricted.java", "Mixed.java"},
            )
            self.assertEqual([row["path"] for row in missed], ["License.java"])
            self.assertEqual(missed[0]["known_license"], "Apache-2.0")
            self.assertFalse(missed[0]["scancode_contains_license_notice"])
            self.assertTrue(missed[0]["is_scancode_missed_license"])

    def test_scan_only_build_writes_and_verifies_bounded_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_directory = root / "input"
            output_directory = root / "scan-only"
            _write_sparse_prefix_fixture(input_directory)

            stats = build_redistribution_candidates(
                output_directory,
                input_source=input_directory,
                source_files_limit=100_000,
                fuzzy_threshold=0.82,
                scan_only=True,
                batch_size=2,
            )
            report = verify_redistribution_candidates(output_directory)
            manifest = json.loads(
                (output_directory / "manifest.json").read_text(encoding="utf-8")
            )

            self.assertTrue(report.valid, report.errors)
            self.assertTrue(stats.scan_only)
            self.assertEqual(stats.source_files_in_scope, 100_000)
            self.assertEqual(stats.comment_rows_seen, 5)
            self.assertEqual(stats.matched_occurrences, 4)
            self.assertEqual(stats.candidate_count, 2)
            self.assertEqual(report.matched_occurrences, 4)
            self.assertEqual(report.candidate_count, 2)
            self.assertTrue((output_directory / "occurrences.parquet").is_file())
            self.assertTrue((output_directory / "candidates.parquet").is_file())
            self.assertTrue((output_directory / "verification.json").is_file())
            self.assertFalse((output_directory / "dataset.parquet").exists())
            self.assertFalse(
                (output_directory / "labeled-occurrences.parquet").exists()
            )
            self.assertEqual(
                manifest["parameters"]["source_prefix_end_row_exclusive"],
                100_000,
            )
            self.assertFalse(manifest["parameters"]["include_provenance_seeds"])
            self.assertFalse(manifest["parameters"]["include_funding_seeds"])
            self.assertFalse(
                manifest["parameters"]["include_export_control_seeds"]
            )
            self.assertFalse(
                manifest["parameters"]["include_unpublished_work_seeds"]
            )
            self.assertEqual(manifest["results"]["judged_count"], 0)

    def test_judged_build_uses_fake_runner_and_preserves_exact_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_directory = root / "input"
            output_directory = root / "judged"
            _write_shard(
                input_directory,
                0,
                [
                    _source_row(
                        0,
                        0,
                        "// Private and confidential source code. Do not distribute.",
                        path="src/Restricted.java",
                    ),
                    _source_row(
                        1,
                        0,
                        "// Do not distribute tasks until every worker is ready.",
                        path="src/Scheduler.java",
                    ),
                    _source_row(
                        2,
                        0,
                        "// Redistribution prohibited.",
                        path="src/Unclear.java",
                    ),
                    _source_row(
                        3,
                        0,
                        "// Do not distribute this source code.",
                        path="src/OutsidePrefix.java",
                    ),
                ],
            )
            prompt_lock = threading.Lock()
            prompts: list[str] = []

            def fake_runner(prompt: str) -> tuple[str, dict[str, int]]:
                with prompt_lock:
                    prompts.append(prompt)
                decisions = []
                for candidate in _prompt_candidates(prompt):
                    comment = str(candidate["comment"])
                    if "Private and confidential" in comment:
                        label = LABEL_CODE_REDISTRIBUTION_INTENT
                        evidence = "Private and confidential source code"
                    elif "tasks until" in comment:
                        label = LABEL_OTHER
                        evidence = "Do not distribute tasks"
                    else:
                        label = LABEL_AMBIGUOUS
                        evidence = "Redistribution prohibited"
                    decisions.append(
                        {
                            "candidate_id": candidate["candidate_id"],
                            "label": label,
                            "confidence": 0.9,
                            "evidence": evidence,
                            "rationale": "Fixture decision.",
                        }
                    )
                return json.dumps({"decisions": decisions}), {
                    "input_tokens": 10,
                    "cached_input_tokens": 0,
                    "output_tokens": 5,
                }

            stats = build_redistribution_candidates(
                output_directory,
                input_source=input_directory,
                source_files_limit=3,
                fuzzy_threshold=0.82,
                scan_only=False,
                batch_size=2,
                judge_batch_size=1,
                judge_workers=2,
                judge_max_attempts=1,
                judge_cache_path=root / "judge-cache.sqlite",
                reasoning_effort="low",
                judge_runner=fake_runner,
            )
            report = verify_redistribution_candidates(output_directory)
            judged = pq.read_table(output_directory / "dataset.parquet").to_pylist()
            labeled = pq.read_table(
                output_directory / "labeled-occurrences.parquet"
            ).to_pylist()
            manifest = json.loads(
                (output_directory / "manifest.json").read_text(encoding="utf-8")
            )

            self.assertTrue(report.valid, report.errors)
            self.assertEqual(stats.judged_count, 3)
            self.assertEqual(len(prompts), 3)
            self.assertEqual(len(judged), 3)
            self.assertEqual(len(labeled), 3)
            self.assertEqual(manifest["judge"]["model"], "gpt-5.6-luna")
            self.assertEqual(manifest["judge"]["reasoning_effort"], "low")

            by_path = {row["path"]: row for row in judged}
            self.assertEqual(
                by_path["src/Restricted.java"]["judge_label"],
                LABEL_CODE_REDISTRIBUTION_INTENT,
            )
            self.assertIs(
                by_path["src/Restricted.java"]["is_code_redistribution_intent"],
                True,
            )
            self.assertEqual(
                by_path["src/Scheduler.java"]["judge_label"], LABEL_OTHER
            )
            self.assertIs(
                by_path["src/Scheduler.java"]["is_code_redistribution_intent"],
                False,
            )
            self.assertEqual(
                by_path["src/Unclear.java"]["judge_label"], LABEL_AMBIGUOUS
            )
            self.assertIsNone(
                by_path["src/Unclear.java"]["is_code_redistribution_intent"]
            )

            labeled_path = output_directory / "labeled-occurrences.parquet"
            labeled_table = pq.read_table(labeled_path)
            tampered_rows = labeled_table.to_pylist()
            tampered_rows[0]["path"] = "src/Wrong.java"
            pq.write_table(
                pa.Table.from_pylist(tampered_rows, schema=labeled_table.schema),
                labeled_path,
                compression="zstd",
            )
            manifest["artifacts"]["labeled-occurrences.parquet"] = {
                "size": labeled_path.stat().st_size,
                "sha256": hashlib.sha256(labeled_path.read_bytes()).hexdigest(),
            }
            (output_directory / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            tampered_report = verify_redistribution_candidates(output_directory)

            self.assertFalse(tampered_report.valid)
            self.assertIn(
                "labeled occurrence differs from source occurrence",
                tampered_report.errors,
            )


if __name__ == "__main__":
    unittest.main()
