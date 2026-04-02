from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from commentminer.config import PipelineConfig


class PipelineConfigTests(unittest.TestCase):
    def test_from_path_resolves_relative_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_path = root / "pipeline.json"
            config_path.write_text(
                json.dumps(
                    {
                        "storage": {
                            "working_directory": "var/work",
                            "output_directory": "var/output",
                            "checkpoint_directory": "var/checkpoints",
                            "download_directory": "var/downloads",
                            "huggingface_cache_directory": "var/hf-cache",
                        },
                        "datasets": [
                            {
                                "name": "toy",
                                "input_uri": "memory://toy",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            config = PipelineConfig.from_path(config_path)

            self.assertEqual(config.storage.working_directory, (root / "var/work").resolve())
            self.assertEqual(config.storage.output_directory, (root / "var/output").resolve())
            self.assertEqual(config.storage.checkpoint_directory, (root / "var/checkpoints").resolve())
            self.assertEqual(config.storage.download_directory, (root / "var/downloads").resolve())
            self.assertEqual(config.storage.huggingface_cache_directory, (root / "var/hf-cache").resolve())
            self.assertEqual(config.datasets[0].name, "toy")

    def test_dataset_language_patterns_are_resolved(self) -> None:
        config = PipelineConfig.from_dict(
            {
                "storage": {
                    "working_directory": "var/work",
                    "output_directory": "var/output",
                    "checkpoint_directory": "var/checkpoints",
                },
                "datasets": [
                    {
                        "name": "stack",
                        "source": "huggingface_hub",
                        "repo_id": "bigcode/the-stack-v2",
                        "allow_patterns": ["data/{language}/**"],
                        "languages": ["python", "java"],
                    }
                ],
            },
            base_dir=Path("/tmp/project"),
        )

        dataset = config.require_dataset("stack")

        self.assertTrue(dataset.supports_language_selection())
        self.assertEqual(dataset.available_languages(), ["java", "python"])
        allow_patterns, ignore_patterns = dataset.resolve_patterns("python")
        self.assertEqual(allow_patterns, ["data/python/**"])
        self.assertEqual(ignore_patterns, [])


if __name__ == "__main__":
    unittest.main()
