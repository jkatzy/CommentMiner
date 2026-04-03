from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from commentminer.cli import _mine_dataset
from commentminer.pipeline import PipelineRunStats


class CliTests(unittest.TestCase):
    def test_mine_dataset_prints_warning_when_max_records_is_used(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            with config_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "storage": {
                            "working_directory": f"{tmp_dir}/work",
                            "output_directory": f"{tmp_dir}/output",
                            "checkpoint_directory": f"{tmp_dir}/checkpoints",
                            "download_directory": f"{tmp_dir}/downloads",
                            "huggingface_cache_directory": f"{tmp_dir}/hf-cache",
                        },
                        "datasets": [],
                    },
                    handle,
                )

            stderr = io.StringIO()
            stdout = io.StringIO()
            fake_stats = PipelineRunStats(
                dataset="the-stack",
                run_id="run",
                records_seen=3,
                comments_written=1,
                skipped_without_comment=2,
                shards_written=1,
                failed_shards=0,
            )

            with (
                patch("commentminer.cli._build_source", return_value=(type("Dataset", (), {"name": "the-stack"})(), object())),
                patch("commentminer.cli.run_dataset", return_value=fake_stats),
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                exit_code = _mine_dataset(
                    config_path=config_path,
                    dataset_name="the-stack",
                    language="python",
                    show_progress=False,
                    token_env=None,
                    max_records=3,
                    max_comment_start_row=3,
                    progress_every=1000,
                    workers=4,
                )

            self.assertEqual(exit_code, 0)
            self.assertIn("WARNING: --max-records=3 uses sequential record-level mining", stderr.getvalue())
            self.assertIn("ignores --workers", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
