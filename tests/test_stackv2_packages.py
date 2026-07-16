from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from commentminer.config import DatasetSpec, PipelineConfig, StorageConfig
from commentminer.downloader import HuggingFaceDownloader, RemoteFile
from commentminer.stackv2_packages import (
    StackV2IdPackage,
    StackV2IdSegment,
    StackV2SWHContentPackageSource,
    _package_worker_max_tasks_per_child_option,
    _run_packages,
    mine_stack_v2_id_packages,
    plan_stack_v2_id_packages,
)


@dataclass
class FakeRepoFile:
    path: str
    size: int


class FakeApi:
    def __init__(self, files: list[FakeRepoFile]) -> None:
        self._files = files

    def list_repo_tree(self, **_: object):
        return list(self._files)


class FakeContentFetcher:
    url_template = "fake://{blob_id}"

    def __init__(self, content_by_blob: dict[str, str]) -> None:
        self.content_by_blob = content_by_blob
        self.calls: list[str] = []

    def fetch(self, blob_id: str, src_encoding: str | None) -> str:
        self.calls.append(blob_id)
        return self.content_by_blob[blob_id]


class FirstLineExtractor:
    def extract_opening_comment(self, record) -> str | None:
        if not record.content:
            return None
        return record.content.splitlines()[0]

    def supports_language_value(self, value: str) -> bool:
        return True


def _build_config(root: Path) -> PipelineConfig:
    return PipelineConfig(
        storage=StorageConfig(
            working_directory=root / "work",
            output_directory=root / "output",
            checkpoint_directory=root / "checkpoints",
            download_directory=root / "downloads",
            huggingface_cache_directory=root / "hf-cache",
            max_records_per_shard=10,
            max_bytes_per_shard=1024 * 1024,
        ),
        datasets=[],
        checkpoint_interval_records=1,
    )


def _build_dataset() -> DatasetSpec:
    return DatasetSpec(
        name="the-stack-v2",
        source="huggingface_hub",
        repo_id="bigcode/the-stack-v2",
        repo_type="dataset",
        allow_patterns=["data/{language}/**"],
        languages=["Java", "Python"],
        streaming=True,
        batch_size=2,
        extra={
            "content_backend": "softwareheritage_s3",
            "language_columns": ["language", "gha_language", "extension"],
            "path_columns": ["path"],
            "repo_columns": ["repo_name"],
            "metadata_columns": [
                "blob_id",
                "src_encoding",
                "language",
                "gha_language",
                "extension",
                "path",
                "repo_name",
            ],
        },
    )


def _rows(language: str, count: int, *, prefix: str) -> list[dict[str, object]]:
    return [
        {
            "blob_id": f"{prefix}-{index}",
            "path": f"/src/{prefix}-{index}.txt",
            "repo_name": f"org/{prefix}",
            "src_encoding": "utf-8",
            "language": language,
            "gha_language": None,
            "extension": "txt",
        }
        for index in range(count)
    ]


def _write_stack_v2_fixture(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _copy_by_remote(fixtures: dict[str, Path]):
    def download_file(**kwargs: object) -> str:
        remote_path = str(kwargs["filename"])
        local_dir = Path(str(kwargs["local_dir"]))
        target_path = local_dir / remote_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(fixtures[remote_path], target_path)
        return str(target_path)

    return download_file


class StackV2PackageTests(unittest.TestCase):
    def test_package_worker_max_tasks_option_defaults_and_validates(self) -> None:
        self.assertEqual(
            _package_worker_max_tasks_per_child_option(None, default=1),
            1,
        )
        self.assertEqual(
            _package_worker_max_tasks_per_child_option("3", default=1),
            3,
        )
        self.assertIsNone(
            _package_worker_max_tasks_per_child_option(0, default=1),
        )
        self.assertIsNone(
            _package_worker_max_tasks_per_child_option(None, default=None),
        )
        with self.assertRaises(ValueError):
            _package_worker_max_tasks_per_child_option(-1, default=1)

    def test_process_package_workers_pass_max_tasks_per_child_to_executor(self) -> None:
        calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        class RecordingProcessPoolExecutor:
            def __init__(self, *args: object, **kwargs: object) -> None:
                calls.append((args, kwargs))

            def __enter__(self):
                return self

            def __exit__(self, *args: object) -> bool:
                return False

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = _build_config(root)
            dataset = _build_dataset()
            downloader = HuggingFaceDownloader(api=FakeApi([]))

            with patch(
                "commentminer.stackv2_packages.ProcessPoolExecutor",
                RecordingProcessPoolExecutor,
            ):
                _run_packages(
                    config,
                    dataset,
                    [],
                    downloader=downloader,
                    token=None,
                    package_workers=1,
                    package_worker_backend="process",
                    package_worker_max_tasks_per_child=7,
                    content_download_workers=1,
                    content_prefetch_records=1,
                    content_fetcher=None,
                    content_executor=None,
                    content_language_filter=None,
                    extractor_factory=FirstLineExtractor,
                    extraction_workers=1,
                    extraction_buffer=None,
                    cache_source_files=False,
                    show_progress=False,
                    progress_every=0,
                    skip_errors=False,
                )

        self.assertEqual(calls, [((), {"max_workers": 1, "max_tasks_per_child": 7})])

    def test_plan_splits_id_packages_across_language_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = _build_config(root)
            dataset = _build_dataset()
            java_remote = "data/Java/train-00000.parquet"
            python_remote = "data/Python/train-00000.parquet"
            java_fixture = root / "fixtures" / "java.parquet"
            python_fixture = root / "fixtures" / "python.parquet"
            _write_stack_v2_fixture(java_fixture, _rows("Java", 2, prefix="java"))
            _write_stack_v2_fixture(python_fixture, _rows("Python", 3, prefix="python"))
            downloader = HuggingFaceDownloader(
                api=FakeApi(
                    [
                        FakeRepoFile(java_remote, java_fixture.stat().st_size),
                        FakeRepoFile(python_remote, python_fixture.stat().st_size),
                    ]
                ),
                download_file=_copy_by_remote(
                    {
                        java_remote: java_fixture,
                        python_remote: python_fixture,
                    }
                ),
            )

            plan = plan_stack_v2_id_packages(
                config,
                dataset,
                downloader=downloader,
                languages=["Java", "Python"],
                package_size=3,
                metadata_download_workers=2,
            )

            self.assertEqual(plan.id_count, 5)
            self.assertEqual(len(plan.packages), 2)
            self.assertEqual(plan.packages[0].id_count, 3)
            self.assertEqual(
                [
                    (segment.language, segment.remote.path, segment.start_row, segment.end_row)
                    for segment in plan.packages[0].segments
                ],
                [
                    ("Java", java_remote, 0, 2),
                    ("Python", python_remote, 0, 1),
                ],
            )
            self.assertEqual(
                [
                    (segment.language, segment.remote.path, segment.start_row, segment.end_row)
                    for segment in plan.packages[1].segments
                ],
                [("Python", python_remote, 1, 3)],
            )
            self.assertTrue((root / "downloads" / "the-stack-v2" / java_remote).exists())
            self.assertTrue((root / "downloads" / "the-stack-v2" / python_remote).exists())

    def test_package_source_processes_only_requested_row_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = _build_config(root)
            dataset = _build_dataset()
            remote_path = "data/Python/train-00000.parquet"
            fixture = root / "fixtures" / "python.parquet"
            rows = _rows("Python", 5, prefix="python")
            _write_stack_v2_fixture(fixture, rows)
            downloader = HuggingFaceDownloader(
                api=FakeApi([FakeRepoFile(remote_path, fixture.stat().st_size)]),
                download_file=_copy_by_remote({remote_path: fixture}),
            )
            package = StackV2IdPackage(
                index=7,
                segments=(
                    StackV2IdSegment(
                        language="Python",
                        remote=RemoteFile(remote_path, fixture.stat().st_size),
                        start_row=1,
                        end_row=4,
                    ),
                ),
            )
            content_fetcher = FakeContentFetcher(
                {row["blob_id"]: f"# {row['blob_id']}\nbody\n" for row in rows}
            )
            source = StackV2SWHContentPackageSource(
                config,
                dataset,
                package,
                show_progress=False,
                downloader=downloader,
                content_fetcher=content_fetcher,
                content_download_workers=2,
                content_prefetch_records=2,
            )

            records = list(source.iter_records())

            self.assertEqual(
                [record.record_id for record in records],
                [
                    f"{remote_path}::row::1",
                    f"{remote_path}::row::2",
                    f"{remote_path}::row::3",
                ],
            )
            self.assertEqual(content_fetcher.calls, ["python-1", "python-2", "python-3"])
            self.assertEqual(
                [record.metadata["stack_v2_id_package_index"] for record in records],
                [7, 7, 7],
            )
            self.assertEqual(
                [record.metadata["selected_language"] for record in records],
                ["Python", "Python", "Python"],
            )

    def test_mine_stack_v2_id_packages_runs_and_skips_completed_packages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = _build_config(root)
            dataset = _build_dataset()
            remote_path = "data/Python/train-00000.parquet"
            fixture = root / "fixtures" / "python.parquet"
            rows = _rows("Python", 5, prefix="python")
            _write_stack_v2_fixture(fixture, rows)
            downloader = HuggingFaceDownloader(
                api=FakeApi([FakeRepoFile(remote_path, fixture.stat().st_size)]),
                download_file=_copy_by_remote({remote_path: fixture}),
            )
            content_fetcher = FakeContentFetcher(
                {row["blob_id"]: f"# {row['blob_id']}\nbody\n" for row in rows}
            )

            summary = mine_stack_v2_id_packages(
                config,
                dataset,
                downloader=downloader,
                languages=["Python"],
                package_size=2,
                metadata_download_workers=1,
                package_workers=2,
                content_download_workers=4,
                content_prefetch_records=4,
                extraction_workers=1,
                show_progress=False,
                progress_every=0,
                content_fetcher=content_fetcher,
                extractor_factory=FirstLineExtractor,
            )

            self.assertEqual(summary.packages_planned, 3)
            self.assertEqual(summary.packages_completed, 3)
            self.assertEqual(summary.records_seen, 5)
            self.assertEqual(summary.comments_written, 5)
            output_records = []
            for path in config.storage.output_directory.rglob("part-*.jsonl"):
                output_records.extend(
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line
                )
            self.assertEqual(len(output_records), 5)
            self.assertEqual({record["dataset"] for record in output_records}, {"the-stack-v2"})
            self.assertEqual(
                {record["metadata"]["stack_v2_id_package_index"] for record in output_records},
                {0, 1, 2},
            )

            second_summary = mine_stack_v2_id_packages(
                config,
                dataset,
                downloader=downloader,
                languages=["Python"],
                package_size=2,
                metadata_download_workers=1,
                package_workers=2,
                content_download_workers=4,
                content_prefetch_records=4,
                extraction_workers=1,
                show_progress=False,
                progress_every=0,
                content_fetcher=content_fetcher,
                extractor_factory=FirstLineExtractor,
            )

            self.assertEqual(second_summary.packages_planned, 3)
            self.assertEqual(second_summary.packages_skipped, 3)
            self.assertEqual(second_summary.packages_completed, 0)

    def test_mine_stack_v2_id_packages_supports_process_package_workers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = _build_config(root)
            dataset = _build_dataset()
            content_root = root / "content"
            content_root.mkdir()
            dataset.extra = dict(dataset.extra)
            dataset.extra["swh_content_url_template"] = f"file://{content_root}/{{blob_id}}"
            dataset.extra["swh_content_compression"] = "none"
            remote_path = "data/Python/train-00000.parquet"
            fixture = root / "fixtures" / "python.parquet"
            rows = _rows("Python", 4, prefix="python")
            _write_stack_v2_fixture(fixture, rows)
            for row in rows:
                (content_root / str(row["blob_id"])).write_text(
                    f"# {row['blob_id']}\nbody\n",
                    encoding="utf-8",
                )
            downloader = HuggingFaceDownloader(
                api=FakeApi([FakeRepoFile(remote_path, fixture.stat().st_size)]),
                download_file=_copy_by_remote({remote_path: fixture}),
            )

            summary = mine_stack_v2_id_packages(
                config,
                dataset,
                downloader=downloader,
                languages=["Python"],
                package_size=2,
                metadata_download_workers=1,
                package_workers=2,
                package_worker_backend="process",
                content_download_workers=4,
                content_prefetch_records=4,
                extraction_workers=1,
                show_progress=False,
                progress_every=0,
                extractor_factory=FirstLineExtractor,
            )

            self.assertEqual(summary.packages_planned, 2)
            self.assertEqual(summary.packages_completed, 2)
            self.assertEqual(summary.records_seen, 4)
            self.assertEqual(summary.comments_written, 4)


if __name__ == "__main__":
    unittest.main()
