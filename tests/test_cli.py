from __future__ import annotations

from contextlib import redirect_stderr
from io import StringIO
import unittest

from commentminer.cli import build_parser


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


if __name__ == "__main__":
    unittest.main()
