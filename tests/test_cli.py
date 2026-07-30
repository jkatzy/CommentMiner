from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import unittest
from unittest.mock import Mock, patch

from commentminer.cli import build_parser, main


class CliTests(unittest.TestCase):
    def test_exposes_only_parquet_data_workflow_commands(self) -> None:
        parser = build_parser()
        subparser_action = next(
            action for action in parser._actions if action.dest == "command"  # noqa: SLF001
        )
        commands = set(subparser_action.choices)

        self.assertIn("mine-dataset", commands)
        self.assertIn("export-hf-dataset", commands)
        self.assertIn("scan-hf-comment-licenses", commands)
        self.assertIn("build-redistribution-candidate-dataset", commands)
        self.assertIn("verify-redistribution-candidate-dataset", commands)
        self.assertNotIn("aggregate-comment-runs", commands)
        self.assertNotIn("deduplicate-comment-run", commands)
        self.assertNotIn("scan-comment-licenses", commands)

    def test_export_has_no_alternate_data_format_option(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["export-hf-dataset", "config.json", "output"]
        )
        self.assertFalse(hasattr(args, "format"))

        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "export-hf-dataset",
                    "config.json",
                    "output",
                    "--format",
                    "anything",
                ]
            )

    def test_redistribution_candidate_parser_has_reproducible_defaults(self) -> None:
        parser = build_parser()

        args = parser.parse_args(
            ["build-redistribution-candidate-dataset", "output"]
        )

        self.assertEqual(args.input_source, "Jkatzy/code-comments")
        self.assertEqual(args.dataset, "the-stack-v2-dedup")
        self.assertEqual(args.language, "Java")
        self.assertEqual(args.source_files_limit, 100_000)
        self.assertFalse(args.all_languages)
        self.assertIsNone(args.comment_rows_limit)
        self.assertIsNone(args.scancode_score_threshold)
        self.assertEqual(args.scan_workers, 1)
        self.assertEqual(args.fuzzy_threshold, 0.82)
        self.assertFalse(args.include_government_seeds)
        self.assertFalse(args.include_provenance_seeds)
        self.assertFalse(args.include_funding_seeds)
        self.assertFalse(args.include_export_control_seeds)
        self.assertFalse(args.include_unpublished_work_seeds)
        self.assertFalse(args.scan_only)
        self.assertEqual(args.batch_size, 8192)
        self.assertEqual(args.judge_batch_size, 64)
        self.assertEqual(args.judge_workers, 4)
        self.assertEqual(args.judge_max_batch_chars, 160_000)
        self.assertEqual(args.judge_max_comment_chars, 12_000)
        self.assertEqual(args.judge_max_attempts, 3)
        self.assertEqual(args.judge_timeout_seconds, 900.0)
        self.assertEqual(args.codex_model, "gpt-5.6-luna")
        self.assertEqual(args.codex_reasoning_effort, "max")
        self.assertEqual(args.judgment_profile, "redistribution_intent")

        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "build-redistribution-candidate-dataset",
                    "output",
                    "--codex-reasoning-effort",
                    "high",
                ]
            )

    def test_redistribution_candidate_cli_dispatches_parallel_scan_only_build(self) -> None:
        stats = Mock(
            output_directory="output",
            input_format="huggingface",
            source_files_in_scope=100_000,
            comment_bearing_files_seen=35_895,
            comment_rows_seen=42_613,
            shards_scanned=1,
            matched_occurrences=1_177,
            candidate_count=679,
            judged_count=0,
            judge_batches=0,
            judge_attempts=0,
            judge_cache_hits=0,
            scan_only=True,
            occurrences_path="output/occurrences.parquet",
            candidates_path="output/candidates.parquet",
            dataset_path=None,
            labeled_occurrences_path=None,
            manifest_path="output/manifest.json",
            verification_path="output/verification.json",
        )
        with patch(
            "commentminer.cli.build_redistribution_candidates",
            return_value=stats,
        ) as build_candidates, redirect_stdout(StringIO()):
            exit_code = main(
                [
                    "build-redistribution-candidate-dataset",
                    "output",
                    "--scan-only",
                    "--all-languages",
                    "--comment-rows-limit",
                    "5000000",
                    "--scancode-score-below",
                    "0.9",
                    "--scan-workers",
                    "32",
                    "--judge-workers",
                    "4",
                    "--include-government-seeds",
                    "--include-provenance-seeds",
                    "--include-funding-seeds",
                    "--include-export-control-seeds",
                    "--include-unpublished-work-seeds",
                    "--judgment-profile",
                    "non_license_limitations",
                    "--codex-reasoning-effort",
                    "low",
                ]
            )

        self.assertEqual(exit_code, 0)
        kwargs = build_candidates.call_args.kwargs
        self.assertEqual(kwargs["input_source"], "Jkatzy/code-comments")
        self.assertEqual(kwargs["dataset"], "the-stack-v2-dedup")
        self.assertEqual(kwargs["language"], "Java")
        self.assertEqual(kwargs["source_files_limit"], 100_000)
        self.assertTrue(kwargs["all_languages"])
        self.assertEqual(kwargs["comment_rows_limit"], 5_000_000)
        self.assertEqual(kwargs["scancode_score_threshold"], 0.9)
        self.assertEqual(kwargs["scan_workers"], 32)
        self.assertEqual(kwargs["fuzzy_threshold"], 0.82)
        self.assertTrue(kwargs["include_government_seeds"])
        self.assertTrue(kwargs["include_provenance_seeds"])
        self.assertTrue(kwargs["include_funding_seeds"])
        self.assertTrue(kwargs["include_export_control_seeds"])
        self.assertTrue(kwargs["include_unpublished_work_seeds"])
        self.assertTrue(kwargs["scan_only"])
        self.assertEqual(kwargs["batch_size"], 8192)
        self.assertEqual(kwargs["judge_batch_size"], 64)
        self.assertEqual(kwargs["judge_workers"], 4)
        self.assertEqual(kwargs["judge_max_batch_chars"], 160_000)
        self.assertEqual(kwargs["judge_max_comment_chars"], 12_000)
        self.assertEqual(kwargs["codex_model"], "gpt-5.6-luna")
        self.assertEqual(kwargs["reasoning_effort"], "low")
        self.assertEqual(kwargs["judgment_profile"], "non_license_limitations")

    def test_redistribution_candidate_verify_cli_reports_failure(self) -> None:
        report = Mock(
            valid=False,
            candidate_count=4,
            matched_occurrences=6,
            judged_count=4,
            errors=("artifact hash mismatch",),
        )
        with patch(
            "commentminer.cli.verify_redistribution_candidates",
            return_value=report,
        ), redirect_stdout(StringIO()) as stdout:
            exit_code = main(
                ["verify-redistribution-candidate-dataset", "output"]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("artifact hash mismatch", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
