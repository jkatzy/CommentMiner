from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from commentminer.config import DatasetSpec, PipelineConfig, StorageConfig
from commentminer.downloader import HuggingFaceDownloader, RedPajamaManifestDownloader


@dataclass
class FakeRepoFile:
    path: str
    size: int


class FakeApi:
    def __init__(self, files: list[FakeRepoFile]) -> None:
        self._files = files

    def list_repo_tree(self, **_: object):
        return list(self._files)


def fake_download_file(**kwargs: object) -> str:
    local_dir = Path(str(kwargs["local_dir"]))
    filename = Path(str(kwargs["filename"]))
    target_path = local_dir / filename
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text("payload", encoding="utf-8")
    return str(target_path)


def fake_manifest_download_factory(manifest_path: Path):
    def _download(**_: object) -> str:
        return str(manifest_path)

    return _download


def fake_remote_download(url: str, target_path: Path) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(f"downloaded:{url}", encoding="utf-8")
    return target_path


class HuggingFaceDownloaderTests(unittest.TestCase):
    def test_list_languages_discovers_languages_from_remote_paths(self) -> None:
        dataset = DatasetSpec(
            name="the-stack",
            source="huggingface_hub",
            repo_id="bigcode/the-stack",
            repo_type="dataset",
            revision="main",
            allow_patterns=["data/{language}/**"],
        )
        downloader = HuggingFaceDownloader(
            api=FakeApi(
                [
                    FakeRepoFile("data/python/train-00000-of-00001.parquet", 10),
                    FakeRepoFile("data/java/train-00000-of-00001.parquet", 12),
                    FakeRepoFile("data/python/train-00001-of-00001.parquet", 14),
                ]
            )
        )

        languages = downloader.list_languages(dataset)

        self.assertEqual(languages, ["java", "python"])

    def test_plan_download_uses_language_filter_and_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = PipelineConfig(
                storage=StorageConfig(
                    working_directory=root / "work",
                    output_directory=root / "output",
                    checkpoint_directory=root / "checkpoints",
                    download_directory=root / "downloads",
                    huggingface_cache_directory=root / "hf-cache",
                    max_records_per_shard=10,
                    max_bytes_per_shard=1024,
                ),
                datasets=[],
                checkpoint_interval_records=1,
            )
            config.datasets.append(
                DatasetSpec(
                    name="the-stack-v2",
                    source="huggingface_hub",
                    repo_id="bigcode/the-stack-v2",
                    repo_type="dataset",
                    revision="main",
                    allow_patterns=["data/{language}/**"],
                    languages=["python", "java"],
                )
            )

            checkpoint_root = config.storage.checkpoint_directory / "downloads"
            checkpoint_root.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_root / "the-stack-v2-python.json"
            checkpoint_path.write_text(
                json.dumps(
                    {
                        "dataset": "the-stack-v2",
                        "repo_id": "bigcode/the-stack-v2",
                        "revision": "main",
                        "language": "python",
                        "completed_files": ["data/python/part-00000.parquet"],
                    }
                ),
                encoding="utf-8",
            )

            downloader = HuggingFaceDownloader(
                api=FakeApi(
                    [
                        FakeRepoFile("data/python/part-00000.parquet", 10),
                        FakeRepoFile("data/python/part-00001.parquet", 12),
                        FakeRepoFile("data/java/part-00000.parquet", 14),
                    ]
                )
            )

            plan = downloader.plan_download(
                config,
                config.require_dataset("the-stack-v2"),
                language="python",
            )

            self.assertEqual(plan.matched_count, 2)
            self.assertEqual(plan.completed_count, 1)
            self.assertEqual(plan.pending_count, 1)
            self.assertEqual(plan.pending_files[0].path, "data/python/part-00001.parquet")

    def test_download_writes_files_and_updates_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = PipelineConfig(
                storage=StorageConfig(
                    working_directory=root / "work",
                    output_directory=root / "output",
                    checkpoint_directory=root / "checkpoints",
                    download_directory=root / "downloads",
                    huggingface_cache_directory=root / "hf-cache",
                    max_records_per_shard=10,
                    max_bytes_per_shard=1024,
                ),
                datasets=[],
                checkpoint_interval_records=1,
            )
            config.datasets.append(
                DatasetSpec(
                    name="the-stack",
                    source="huggingface_hub",
                    repo_id="bigcode/the-stack",
                    repo_type="dataset",
                    revision="main",
                    allow_patterns=["data/**"],
                )
            )

            downloader = HuggingFaceDownloader(
                api=FakeApi(
                    [
                        FakeRepoFile("data/a.parquet", 10),
                        FakeRepoFile("data/b.parquet", 12),
                    ]
                ),
                download_file=fake_download_file,
            )

            summary = downloader.download(config, config.require_dataset("the-stack"))

            self.assertEqual(summary.matched_count, 2)
            self.assertEqual(summary.already_downloaded_count, 0)
            self.assertEqual(summary.downloaded_count, 2)
            self.assertTrue((summary.download_root / "data/a.parquet").exists())
            self.assertTrue((summary.download_root / "data/b.parquet").exists())

            checkpoint = json.loads(summary.checkpoint_path.read_text(encoding="utf-8"))
            self.assertEqual(
                checkpoint["completed_files"],
                ["data/a.parquet", "data/b.parquet"],
            )

            second_summary = downloader.download(config, config.require_dataset("the-stack"))
            self.assertEqual(second_summary.already_downloaded_count, 2)
            self.assertEqual(second_summary.downloaded_count, 0)

    def test_redpajama_manifest_download_uses_url_manifest_and_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = root / "github.txt"
            manifest_path.write_text(
                "\n".join(
                    [
                        "https://data.together.xyz/redpajama-data-1T/v1.0.0/github/shard-000.jsonl",
                        "https://data.together.xyz/redpajama-data-1T/v1.0.0/github/shard-001.jsonl",
                    ]
                ),
                encoding="utf-8",
            )
            config = PipelineConfig(
                storage=StorageConfig(
                    working_directory=root / "work",
                    output_directory=root / "output",
                    checkpoint_directory=root / "checkpoints",
                    download_directory=root / "downloads",
                    huggingface_cache_directory=root / "hf-cache",
                    max_records_per_shard=10,
                    max_bytes_per_shard=1024,
                ),
                datasets=[
                    DatasetSpec(
                        name="redpajama-github",
                        source="redpajama_manifest",
                        repo_id="togethercomputer/RedPajama-Data-1T",
                        repo_type="dataset",
                        revision="main",
                        streaming=True,
                        extra={"manifest_path": "urls/github.txt"},
                    )
                ],
                checkpoint_interval_records=1,
            )
            downloader = RedPajamaManifestDownloader(
                manifest_download_file=fake_manifest_download_factory(manifest_path),
                remote_download_file=fake_remote_download,
            )

            plan = downloader.plan_download(config, config.require_dataset("redpajama-github"))

            self.assertEqual(plan.matched_count, 2)
            self.assertEqual(plan.pending_count, 2)
            self.assertEqual(plan.completed_count, 0)

            summary = downloader.download(config, config.require_dataset("redpajama-github"))

            self.assertEqual(summary.matched_count, 2)
            self.assertEqual(summary.downloaded_count, 2)
            self.assertTrue(
                (
                    summary.download_root
                    / "redpajama-data-1T"
                    / "v1.0.0"
                    / "github"
                    / "shard-000.jsonl"
                ).exists()
            )
            checkpoint = json.loads(summary.checkpoint_path.read_text(encoding="utf-8"))
            self.assertEqual(
                checkpoint["completed_files"],
                [
                    "https://data.together.xyz/redpajama-data-1T/v1.0.0/github/shard-000.jsonl",
                    "https://data.together.xyz/redpajama-data-1T/v1.0.0/github/shard-001.jsonl",
                ],
            )

            second_summary = downloader.download(config, config.require_dataset("redpajama-github"))
            self.assertEqual(second_summary.already_downloaded_count, 2)
            self.assertEqual(second_summary.downloaded_count, 0)


if __name__ == "__main__":
    unittest.main()
