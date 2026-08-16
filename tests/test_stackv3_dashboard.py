from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from commentminer.stackv3_dashboard import (
    DashboardPaths,
    StackV3DashboardCollector,
    _argument_int,
    _bar,
    _key_value_fields,
    _parse_diskstats,
    _percentile,
    _window_rate,
    format_bytes,
    format_duration,
    format_active_languages,
    format_rate_short,
)


class StackV3DashboardTests(unittest.TestCase):
    def test_formatters(self) -> None:
        self.assertEqual(format_bytes(1536), "1.5 KiB")
        self.assertEqual(format_duration(90), "2m")
        self.assertEqual(format_duration(3 * 86400), "3.0d")
        self.assertEqual(_bar(1, 4, 8), "[##------]")
        self.assertEqual(_bar(1, 1000, 10), "[#---------]")
        self.assertEqual(format_rate_short(1024**2), "1.0MiB/s")
        self.assertEqual(
            format_active_languages(
                (("Python", 4, 12, 30), ("Rust", 2, 5, 10)),
                max_characters=50,
            ),
            "Python (4, 12/30), Rust (2, 5/10)",
        )
        self.assertEqual(_percentile([10.0, 20.0, 30.0], 0.5), 20.0)

    def test_field_and_argument_parsers(self) -> None:
        fields = _key_value_fields(
            "INFO Mining progress dataset=example run_id=abc records_seen=12 comments_written=20"
        )
        self.assertEqual(fields["dataset"], "example")
        self.assertEqual(fields["records_seen"], "12")
        self.assertEqual(_argument_int(("cmd", "--shard-workers", "64"), "--shard-workers"), 64)

    def test_window_rate_uses_runner_age(self) -> None:
        rate, interval = _window_rate(
            [850.0, 950.0, 990.0],
            now=1000.0,
            seconds=3600.0,
            runner_started_at=800.0,
        )
        self.assertEqual(interval, 200.0)
        self.assertAlmostEqual(rate, 54.0)

    def test_diskstats_parser_selects_major_minor(self) -> None:
        stats = _parse_diskstats(
            [
                "8 0 sda 1 0 10 0 2 0 20 0 0 30 0",
                "259 3 md125p1 4 0 100 0 5 0 200 0 0 300 0",
            ],
            "259:3",
        )

        self.assertEqual(stats, (100, 200, 300))

    def test_collector_reads_inventory_markers_scratch_and_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            paths = DashboardPaths.from_root(root, "the-stack-v3-full")
            paths.listing.parent.mkdir(parents=True)
            paths.completed.mkdir(parents=True)
            paths.scratch.mkdir(parents=True)
            paths.output.mkdir(parents=True)
            paths.logs.mkdir(parents=True)
            shard_path = "buckets/example/contents/language=Python/part.parquet"
            digest = hashlib.sha1(shard_path.encode()).hexdigest()[:12]
            paths.listing.write_text(
                json.dumps({"shards": [{"path": shard_path, "size": 1024}]}),
                encoding="utf-8",
            )
            (paths.completed / f"{digest}.json").write_text("{}", encoding="utf-8")
            (paths.scratch / "download.partial").write_bytes(b"1234")
            (paths.output / "part-00000.parquet").write_bytes(b"output")
            (paths.logs / "stack-v3-test.log").write_text(
                "2026-01-01 00:00:00,000 INFO Downloading Stack v3 shard "
                f"language=Python path={shard_path} size=1024\n"
                "2026-01-01 00:00:02,000 INFO Starting mining run "
                f"dataset=the-stack-v3-full__stack-v3-shard-00000000-{digest} run_id=run "
                "resume_from=None\n"
                "2026-01-01 00:00:05,000 INFO Mining progress "
                f"dataset=the-stack-v3-full__stack-v3-shard-00000000-{digest} run_id=run "
                "records_seen=10 comments_written=15\n"
                "2026-01-01 00:00:10,000 INFO Finished mining run "
                f"dataset=the-stack-v3-full__stack-v3-shard-00000000-{digest} run_id=run "
                "records_seen=20 comments_written=30\n",
                encoding="utf-8",
            )

            collector = StackV3DashboardCollector(paths, output_scan_interval=0)
            processes = [
                (100, 1, 0, 1, "S", ("commentminer", "mine-stack-v3-shards", "--shard-workers", "1")),
                (101, 100, 0, 1, "R", ("commentminer", "mine-stack-v3-shards", "--shard-workers", "1")),
            ]
            with (
                patch(
                    "commentminer.stackv3_dashboard._stack_v3_processes",
                    return_value=processes,
                ),
                patch("commentminer.stackv3_dashboard._host_cpu_totals", return_value=(100, 80)),
                patch("commentminer.stackv3_dashboard._network_rx_bytes", return_value=100),
            ):
                snapshot = collector.sample()

            self.assertEqual(snapshot.total_shards, 1)
            self.assertEqual(snapshot.completed_shards, 1)
            self.assertEqual(snapshot.completed_bytes, 1024)
            self.assertEqual(snapshot.scratch_files, 1)
            self.assertEqual(snapshot.partial_files, 1)
            self.assertEqual(snapshot.output_files, 1)
            self.assertEqual(snapshot.records_seen, 20)
            self.assertEqual(snapshot.comments_written, 30)
            self.assertEqual(snapshot.shard_duration_samples, 1)
            self.assertEqual(snapshot.shard_duration_average, 10.0)
            self.assertEqual(snapshot.recent_shard_durations, (10.0,))

    def test_collector_builds_active_record_progress_from_footer_and_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            paths = DashboardPaths.from_root(root, "the-stack-v3-full")
            paths.listing.parent.mkdir(parents=True)
            paths.completed.mkdir(parents=True)
            paths.scratch.mkdir(parents=True)
            paths.output.mkdir(parents=True)
            paths.checkpoints.mkdir(parents=True)
            paths.logs.mkdir(parents=True)
            shard_path = "buckets/example/contents/language=Python/part.parquet"
            digest = hashlib.sha1(shard_path.encode()).hexdigest()[:12]
            paths.listing.write_text(
                json.dumps(
                    {
                        "shards": [
                            {
                                "index": 7,
                                "path": shard_path,
                                "language": "Python",
                                "size": 1024,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            pq.write_table(
                pa.table({"content": ["a", "b", "c", "d"]}),
                paths.scratch / f"{digest}.parquet",
            )
            (paths.checkpoints / f"the-stack-v3-full-stack-v3-shard-00000007-{digest}.json").write_text(
                json.dumps(
                    {
                        "last_record_id": f"{shard_path}::row::1",
                        "records_seen": 2,
                    }
                ),
                encoding="utf-8",
            )

            collector = StackV3DashboardCollector(paths, output_scan_interval=0)
            processes = [
                (100, 1, 0, 1, "S", ("commentminer", "mine-stack-v3-shards", "--shard-workers", "1")),
                (101, 100, 0, 1, "R", ("commentminer", "mine-stack-v3-shards", "--shard-workers", "1")),
            ]
            with (
                patch(
                    "commentminer.stackv3_dashboard._stack_v3_processes",
                    return_value=processes,
                ),
                patch("commentminer.stackv3_dashboard._host_cpu_totals", return_value=(100, 80)),
                patch("commentminer.stackv3_dashboard._network_rx_bytes", return_value=100),
            ):
                snapshot = collector.sample()

            self.assertEqual(snapshot.active_record_position, 2)
            self.assertEqual(snapshot.active_record_total, 4)
            self.assertEqual(snapshot.active_record_shards, 1)
            self.assertEqual(snapshot.active_languages, (("Python", 1, 0, 1),))
            self.assertEqual(snapshot.corpus_record_position, 2)
            self.assertEqual(snapshot.corpus_record_total, 15_600_000_000)

    def test_active_shards_follow_live_worker_registry_not_stale_scratch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            paths = DashboardPaths.from_root(root, "the-stack-v3-full")
            paths.listing.parent.mkdir(parents=True)
            paths.completed.mkdir(parents=True)
            paths.scratch.mkdir(parents=True)
            paths.output.mkdir(parents=True)
            paths.checkpoints.mkdir(parents=True)
            paths.logs.mkdir(parents=True)
            shards = [
                {
                    "index": 0,
                    "path": "buckets/example/contents/language=Python/active.parquet",
                    "language": "Python",
                    "size": 100,
                },
                {
                    "index": 1,
                    "path": "buckets/example/contents/language=Rust/stale.parquet",
                    "language": "Rust",
                    "size": 100,
                },
            ]
            paths.listing.write_text(json.dumps({"shards": shards}), encoding="utf-8")
            digests = [
                hashlib.sha1(str(shard["path"]).encode()).hexdigest()[:12]
                for shard in shards
            ]
            for digest in digests:
                pq.write_table(
                    pa.table({"content": ["a", "b"]}),
                    paths.scratch / f"{digest}.parquet",
                )
            paths.active_workers.write_text(
                json.dumps(
                    {
                        "parent_pid": 200,
                        "workers": [
                            {"pid": 201, "digest": digests[0]},
                            {"pid": 999, "digest": digests[1]},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            processes = [
                (200, 1, 0, 1, "S", ("commentminer", "mine-stack-v3-shards", "--shard-workers", "1")),
                (201, 200, 0, 1, "R", ("commentminer", "mine-stack-v3-shards", "--shard-workers", "1")),
            ]
            collector = StackV3DashboardCollector(paths, output_scan_interval=0)
            with (
                patch(
                    "commentminer.stackv3_dashboard._stack_v3_processes",
                    return_value=processes,
                ),
                patch("commentminer.stackv3_dashboard._host_cpu_totals", return_value=(100, 80)),
                patch("commentminer.stackv3_dashboard._network_rx_bytes", return_value=100),
            ):
                snapshot = collector.sample()

            self.assertEqual(snapshot.active_workers, 1)
            self.assertEqual(snapshot.scratch_files, 2)
            self.assertEqual(snapshot.active_record_shards, 1)
            self.assertEqual(snapshot.active_languages, (("Python", 1, 0, 1),))


if __name__ == "__main__":
    unittest.main()
