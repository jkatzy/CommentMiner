from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch

import httpx
from huggingface_hub.errors import HfHubHTTPError
import pyarrow as pa
import pyarrow.parquet as pq

from commentminer.cli import build_parser
from commentminer.config import DatasetSpec, PipelineConfig
from commentminer.stackv3_bucket import (
    StackV3BucketPlan,
    StackV3MemorySafeguardError,
    StackV3ShardDeferred,
    StackV3BucketShard,
    StackV3BucketShardSource,
    _wait_for_memory_headroom,
    _completion_path,
    _is_transient_stack_v3_error,
    _mine_stack_v3_shard_process,
    _required_launch_memory_bytes,
    _stack_v3_content,
    mine_stack_v3_bucket_shards,
    plan_stack_v3_bucket_shards,
)
from commentminer.pipeline import PipelineRunStats


def _defer_once_then_succeed(
    config: PipelineConfig,
    dataset: DatasetSpec,
    shard: StackV3BucketShard,
    **_: object,
) -> PipelineRunStats:
    attempts = config.storage.working_directory / "scheduler-test-attempts.txt"
    attempts.parent.mkdir(parents=True, exist_ok=True)
    previous = attempts.read_text(encoding="utf-8") if attempts.exists() else ""
    attempts.write_text(f"{previous}{os.getpid()}\n", encoding="utf-8")
    if not previous:
        raise StackV3ShardDeferred("forced memory pressure")
    return PipelineRunStats(dataset=dataset.name, run_id=f"test-{shard.index}")


def _transient_once_then_succeed(
    config: PipelineConfig,
    dataset: DatasetSpec,
    shard: StackV3BucketShard,
    **_: object,
) -> PipelineRunStats:
    attempts = config.storage.working_directory / "transient-test-attempts.txt"
    attempts.parent.mkdir(parents=True, exist_ok=True)
    previous = attempts.read_text(encoding="utf-8") if attempts.exists() else ""
    attempts.write_text(f"{previous}{os.getpid()}\n", encoding="utf-8")
    if not previous:
        response = httpx.Response(
            504,
            request=httpx.Request("GET", "https://huggingface.co/api/buckets/example"),
        )
        raise HfHubHTTPError("Gateway Timeout", response=response)
    return PipelineRunStats(dataset=dataset.name, run_id=f"test-{shard.index}")


class _ListingFileSystem:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def ls(self, path: str, detail: bool = True) -> list[dict[str, object]]:
        self.calls.append(path)
        if path.endswith("/contents"):
            return [
                {"name": f"{path}/language=Python", "type": "directory"},
                {"name": f"{path}/language=Rust", "type": "directory"},
            ]
        language = path.rsplit("=", 1)[1]
        return [
            {
                "name": f"{path}/{language.lower()}-000.parquet",
                "type": "file",
                "size": 100 if language == "Python" else 200,
            }
        ]


class StackV3BucketTests(unittest.TestCase):
    def _config(self, root: Path) -> tuple[PipelineConfig, DatasetSpec]:
        config = PipelineConfig.from_dict(
            {
                "storage": {
                    "working_directory": str(root / "work"),
                    "output_directory": str(root / "output"),
                    "checkpoint_directory": str(root / "checkpoints"),
                    "download_directory": str(root / "downloads"),
                    "huggingface_cache_directory": str(root / "hf-cache"),
                    "max_records_per_shard": 100,
                    "max_bytes_per_shard": 1_000_000,
                },
                "datasets": [
                    {
                        "name": "the-stack-v3-full",
                        "source": "huggingface_bucket",
                        "repo_id": "HuggingFaceCode/stack-v3-full",
                        "repo_type": "bucket",
                        "batch_size": 2,
                    }
                ],
            },
            base_dir=root,
        )
        return config, config.require_dataset("the-stack-v3-full")

    def test_plan_caches_complete_inventory_before_filtering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config, dataset = self._config(Path(tmp_dir))
            filesystem = _ListingFileSystem()

            python_plan = plan_stack_v3_bucket_shards(
                config,
                dataset,
                filesystem=filesystem,
                languages=["Python"],
                listing_workers=2,
            )
            rust_plan = plan_stack_v3_bucket_shards(
                config,
                dataset,
                filesystem=_ListingFileSystem(),
                languages=["Rust"],
                listing_workers=2,
            )
            no_rust_plan = plan_stack_v3_bucket_shards(
                config,
                dataset,
                filesystem=_ListingFileSystem(),
                exclude_languages=["Rust"],
                listing_workers=2,
            )

            self.assertEqual(python_plan.languages, ("Python",))
            self.assertEqual(rust_plan.languages, ("Rust",))
            self.assertEqual(no_rust_plan.languages, ("Python",))
            self.assertEqual(python_plan.shards[0].index, 0)
            self.assertEqual(rust_plan.shards[0].index, 1)
            self.assertTrue(any(call.endswith("language=Python") for call in filesystem.calls))
            self.assertTrue(any(call.endswith("language=Rust") for call in filesystem.calls))

    def test_source_extracts_notebook_code_and_resumes_by_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config, dataset = self._config(root)
            local_path = root / "notebooks.parquet"
            notebook = json.dumps(
                {
                    "cells": [
                        {"cell_type": "markdown", "source": ["# title"]},
                        {"cell_type": "code", "source": ["# license\n", "print(1)\n"]},
                    ],
                    "metadata": {"language_info": {"name": "python"}},
                }
            )
            pq.write_table(
                pa.table(
                    {
                        "content_id": ["a", "b"],
                        "content": [notebook, notebook],
                        "repo_ids": [[1, 2], [3]],
                    }
                ),
                local_path,
            )
            shard = StackV3BucketShard(
                index=3,
                path="buckets/example/contents/language=Jupyter%20Notebook/0.parquet",
                language="Jupyter Notebook",
                size=local_path.stat().st_size,
            )
            source = StackV3BucketShardSource(dataset, shard, local_path)
            first = next(source.iter_records())
            resumed = list(source.iter_records(start_after=first.record_id))

            self.assertEqual(first.language, "python")
            self.assertEqual(first.content, "# license\nprint(1)\n")
            self.assertEqual(first.metadata["container_language"], "Jupyter Notebook")
            self.assertEqual([record.metadata["row_index"] for record in resumed], [1])

    def test_stack_v3_content_caps_sources_but_exempts_notebooks(self) -> None:
        source = "".join(f"line {index}\n" for index in range(300))
        truncated, language, was_truncated = _stack_v3_content("Python", source)

        self.assertEqual(language, "Python")
        self.assertTrue(was_truncated)
        self.assertEqual(len(truncated.splitlines()), 250)
        self.assertTrue(truncated.endswith("line 249\n"))

        notebook = json.dumps(
            {
                "cells": [{"cell_type": "code", "source": source}],
                "metadata": {"language_info": {"name": "python"}},
            }
        )
        notebook_source, notebook_language, notebook_truncated = _stack_v3_content(
            "Jupyter Notebook",
            notebook,
        )
        self.assertEqual(notebook_language, "python")
        self.assertFalse(notebook_truncated)
        self.assertEqual(len(notebook_source.splitlines()), 300)

    def test_stack_v3_content_sanitizes_surrogates_only_for_unity_assets(self) -> None:
        source = "//\udd12 comment \U0001f680\nvalue: 1\n"

        unity_content, unity_language, unity_truncated = _stack_v3_content(
            "Unity3D Asset",
            source,
        )
        python_content, _, _ = _stack_v3_content("Python", source)

        self.assertEqual(unity_language, "Unity3D Asset")
        self.assertFalse(unity_truncated)
        self.assertEqual(unity_content, "//\ufffd comment \U0001f680\nvalue: 1\n")
        self.assertEqual(unity_content.encode("utf-8").decode("utf-8"), unity_content)
        self.assertEqual(python_content, source)

    def test_single_shard_worker_downloads_extracts_marks_and_removes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config, dataset = self._config(root)
            remote_path = root / "remote.parquet"
            pq.write_table(
                pa.table(
                    {
                        "content_id": ["a", "b"],
                        "content": ["# first\nprint(1)\n", "print(2)\n"],
                        "repo_ids": [[1], [2]],
                    }
                ),
                remote_path,
            )
            shard = StackV3BucketShard(
                index=0,
                path="buckets/example/contents/language=Python/0.parquet",
                language="Python",
                size=remote_path.stat().st_size,
            )

            class _DownloadFileSystem:
                def __init__(self, token: object = None) -> None:
                    pass

                def get(self, remote: str, local: str) -> None:
                    self.assert_remote(remote)
                    shutil.copyfile(remote_path, local)

                @staticmethod
                def assert_remote(remote: str) -> None:
                    if remote != shard.path:
                        raise AssertionError(remote)

            with patch("commentminer.stackv3_bucket.HfFileSystem", _DownloadFileSystem):
                stats = _mine_stack_v3_shard_process(
                    config,
                    dataset,
                    shard,
                    token=None,
                    max_comment_start_row=10,
                    progress_every=100,
                )

            self.assertEqual(stats.records_seen, 2)
            self.assertEqual(stats.comments_written, 1)
            self.assertTrue(_completion_path(config, dataset, shard).exists())
            self.assertTrue(
                any(
                    (config.storage.output_directory / dataset.name).rglob("*.parquet")
                )
            )
            self.assertFalse(
                (config.storage.download_directory / dataset.name / "bucket-shards" / f"{shard.digest}.parquet").exists()
            )

    def test_single_shard_worker_removes_download_when_extraction_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config, dataset = self._config(root)
            remote_path = root / "remote.parquet"
            remote_path.write_bytes(b"not parquet")
            shard = StackV3BucketShard(
                index=1,
                path="buckets/example/contents/language=Python/broken.parquet",
                language="Python",
                size=remote_path.stat().st_size,
            )

            class _DownloadFileSystem:
                def __init__(self, token: object = None) -> None:
                    pass

                def get(self, remote: str, local: str) -> None:
                    shutil.copyfile(remote_path, local)

            with patch("commentminer.stackv3_bucket.HfFileSystem", _DownloadFileSystem):
                with self.assertRaises(Exception):
                    _mine_stack_v3_shard_process(
                        config,
                        dataset,
                        shard,
                        token=None,
                        max_comment_start_row=10,
                        progress_every=100,
                    )

            scratch = config.storage.download_directory / dataset.name / "bucket-shards"
            self.assertFalse((scratch / f"{shard.digest}.parquet").exists())
            self.assertFalse((scratch / f"{shard.digest}.partial").exists())

    def test_single_shard_worker_converts_memory_error_and_removes_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config, dataset = self._config(root)
            remote_path = root / "remote.parquet"
            remote_path.write_bytes(b"downloaded shard")
            shard = StackV3BucketShard(
                index=2,
                path="buckets/example/contents/language=Python/memory.parquet",
                language="Python",
                size=remote_path.stat().st_size,
            )

            class _DownloadFileSystem:
                def __init__(self, token: object = None) -> None:
                    pass

                def get(self, remote: str, local: str) -> None:
                    shutil.copyfile(remote_path, local)

            with (
                patch("commentminer.stackv3_bucket.HfFileSystem", _DownloadFileSystem),
                patch("commentminer.stackv3_bucket.run_dataset", side_effect=MemoryError),
            ):
                with self.assertRaises(StackV3MemorySafeguardError):
                    _mine_stack_v3_shard_process(
                        config,
                        dataset,
                        shard,
                        token=None,
                        max_comment_start_row=10,
                        progress_every=100,
                    )

            scratch = config.storage.download_directory / dataset.name / "bucket-shards"
            self.assertFalse((scratch / f"{shard.digest}.parquet").exists())

    def test_worker_checkpoints_removes_and_defers_shard_under_memory_pressure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config, dataset = self._config(root)
            shard = StackV3BucketShard(
                index=4,
                path="buckets/example/contents/language=Python/deferred.parquet",
                language="Python",
                size=0,
            )
            scratch = config.storage.download_directory / dataset.name / "bucket-shards"
            scratch.mkdir(parents=True)
            local_path = scratch / f"{shard.digest}.parquet"
            pq.write_table(
                pa.table({"content_id": ["a"], "content": ["# comment\ncode\n"]}),
                local_path,
            )
            shard = StackV3BucketShard(
                index=shard.index,
                path=shard.path,
                language=shard.language,
                size=local_path.stat().st_size,
            )

            class _NoDownloadFileSystem:
                def __init__(self, token: object = None) -> None:
                    pass

                def get(self, remote: str, local: str) -> None:
                    raise AssertionError("existing shard should not be downloaded")

            available = [32 * 1024**3, 32 * 1024**3, 8 * 1024**3, 8 * 1024**3]
            with (
                patch("commentminer.stackv3_bucket.HfFileSystem", _NoDownloadFileSystem),
                patch(
                    "commentminer.stackv3_bucket._available_memory_bytes",
                    side_effect=available,
                ),
            ):
                with self.assertRaises(StackV3ShardDeferred):
                    _mine_stack_v3_shard_process(
                        config,
                        dataset,
                        shard,
                        token=None,
                        max_comment_start_row=10,
                        progress_every=100,
                        min_available_memory_gb=16,
                    )

            self.assertFalse(local_path.exists())
            checkpoint = (
                config.storage.checkpoint_directory
                / f"the-stack-v3-full-stack-v3-shard-00000004-{shard.digest}.json"
            )
            self.assertTrue(checkpoint.exists())

    def test_memory_headroom_waits_until_memory_recovers(self) -> None:
        with (
            patch(
                "commentminer.stackv3_bucket._available_memory_bytes",
                side_effect=[8 * 1024**3, 20 * 1024**3],
            ),
            patch("commentminer.stackv3_bucket.time.sleep") as sleep,
        ):
            _wait_for_memory_headroom(
                16,
                shard_path="example.parquet",
                poll_seconds=0.01,
            )

        sleep.assert_called_once_with(0.01)

    def test_launch_memory_reserves_largest_worker_plus_twenty_five_percent(self) -> None:
        gib = 1024**3
        self.assertEqual(
            _required_launch_memory_bytes(16, 2 * gib),
            int(18.5 * gib),
        )

    def test_huggingface_504_is_transient_but_404_is_not(self) -> None:
        request = httpx.Request("GET", "https://huggingface.co/api/buckets/example")
        self.assertTrue(
            _is_transient_stack_v3_error(
                HfHubHTTPError("Gateway Timeout", response=httpx.Response(504, request=request))
            )
        )
        self.assertFalse(
            _is_transient_stack_v3_error(
                HfHubHTTPError("Not Found", response=httpx.Response(404, request=request))
            )
        )

    def test_scheduler_requeues_deferred_shard_in_a_fresh_process(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config, dataset = self._config(Path(tmp_dir))
            shard = StackV3BucketShard(
                index=5,
                path="buckets/example/contents/language=Python/requeue.parquet",
                language="Python",
                size=1,
            )
            plan = StackV3BucketPlan(
                dataset=dataset.name,
                bucket_id="example",
                languages=("Python",),
                shards=(shard,),
            )

            with (
                patch(
                    "commentminer.stackv3_bucket.plan_stack_v3_bucket_shards",
                    return_value=plan,
                ),
                patch(
                    "commentminer.stackv3_bucket._mine_stack_v3_shard_process",
                    side_effect=_defer_once_then_succeed,
                ),
                patch(
                    "commentminer.stackv3_bucket._available_memory_bytes",
                    return_value=128 * 1024**3,
                ),
            ):
                started_at = time.monotonic()
                summary = mine_stack_v3_bucket_shards(
                    config,
                    dataset,
                    shard_workers=1,
                    max_extraction_workers=1,
                    min_free_gb=0,
                    min_available_memory_gb=1,
                    memory_recovery_seconds=0.1,
                    shard_launch_interval_seconds=0,
                )
                elapsed = time.monotonic() - started_at

            attempt_pids = (
                config.storage.working_directory / "scheduler-test-attempts.txt"
            ).read_text(encoding="utf-8").splitlines()
            self.assertEqual(summary.shards_completed, 1)
            self.assertEqual(len(attempt_pids), 2)
            self.assertNotEqual(attempt_pids[0], attempt_pids[1])
            self.assertGreaterEqual(elapsed, 0.1)

    def test_scheduler_retries_transient_download_without_stopping_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config, dataset = self._config(Path(tmp_dir))
            shard = StackV3BucketShard(
                index=6,
                path="buckets/example/contents/language=Python/transient.parquet",
                language="Python",
                size=1,
            )
            plan = StackV3BucketPlan(
                dataset=dataset.name,
                bucket_id="example",
                languages=("Python",),
                shards=(shard,),
            )
            with (
                patch(
                    "commentminer.stackv3_bucket.plan_stack_v3_bucket_shards",
                    return_value=plan,
                ),
                patch(
                    "commentminer.stackv3_bucket._mine_stack_v3_shard_process",
                    side_effect=_transient_once_then_succeed,
                ),
            ):
                summary = mine_stack_v3_bucket_shards(
                    config,
                    dataset,
                    shard_workers=1,
                    max_extraction_workers=1,
                    min_free_gb=0,
                    min_available_memory_gb=0,
                    transient_retry_initial_seconds=0,
                    transient_retry_max_seconds=0,
                )

            attempt_pids = (
                config.storage.working_directory / "transient-test-attempts.txt"
            ).read_text(encoding="utf-8").splitlines()
            self.assertEqual(summary.shards_completed, 1)
            self.assertEqual(len(attempt_pids), 2)
            self.assertNotEqual(attempt_pids[0], attempt_pids[1])

    def test_cli_defaults_to_guarded_128_shard_workers(self) -> None:
        args = build_parser().parse_args(
            ["mine-stack-v3-shards", "config.json", "the-stack-v3-full"]
        )
        self.assertEqual(args.listing_workers, 64)
        self.assertEqual(args.shard_workers, 128)
        self.assertEqual(args.max_extraction_workers, 80)
        self.assertEqual(args.min_free_gb, 20)
        self.assertEqual(args.min_available_memory_gb, 16)
        self.assertIsNone(args.exclude_languages)
        self.assertFalse(args.shuffle_shards)
        self.assertEqual(args.shuffle_seed, 0)


if __name__ == "__main__":
    unittest.main()
