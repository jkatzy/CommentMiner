from __future__ import annotations

import argparse
from collections import deque
import curses
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import time
from typing import Any, Iterable

import pyarrow.parquet as pq


_GIB = 1024**3
_CLOCK_TICKS = os.sysconf("SC_CLK_TCK")
_CPU_COUNT = os.cpu_count() or 1
_DEFAULT_TOTAL_RECORDS = 15_600_000_000
_DEFAULT_MAX_DISK_READ_MIB_S = 2_000.0
_DEFAULT_MAX_DISK_WRITE_MIB_S = 1_000.0


@dataclass(frozen=True, slots=True)
class DashboardPaths:
    root: Path
    dataset: str
    listing: Path
    completed: Path
    active_workers: Path
    scratch: Path
    output: Path
    checkpoints: Path
    logs: Path

    @classmethod
    def from_root(cls, root: Path, dataset: str) -> DashboardPaths:
        root = root.resolve()
        return cls(
            root=root,
            dataset=dataset,
            listing=root / "var/work/stack-v3-bucket" / f"{dataset}-listing.json",
            completed=root / "var/work/stack-v3-bucket" / dataset / "completed",
            active_workers=(
                root / "var/work/stack-v3-bucket" / dataset / "active-workers.json"
            ),
            scratch=root / "var/downloads" / dataset / "bucket-shards",
            output=root / "var/output" / dataset,
            checkpoints=root / "var/checkpoints",
            logs=root / "var/logs",
        )


@dataclass(slots=True)
class DashboardSnapshot:
    timestamp: float
    status: str
    main_pid: int | None
    configured_workers: int
    active_workers: int
    total_shards: int
    completed_shards: int
    remaining_shards: int
    total_bytes: int
    completed_bytes: int
    shard_rate_1h: float
    shard_rate_6h: float
    byte_rate_6h: float
    eta_seconds: float | None
    scratch_files: int
    partial_files: int
    scratch_bytes: int
    download_bytes_per_second: float
    downloads_per_second: float
    output_files: int
    output_bytes: int
    output_files_per_second: float
    host_rx_bytes_per_second: float
    host_cpu_percent: float
    pipeline_cpu_cores: float
    memory_total: int
    memory_available: int
    swap_total: int
    swap_free: int
    disk_total: int
    disk_free: int
    disk_device: str | None
    disk_read_bytes_per_second: float
    disk_write_bytes_per_second: float
    disk_read_max_bytes_per_second: float
    disk_write_max_bytes_per_second: float
    disk_read_percent: float
    disk_write_percent: float
    disk_busy_percent: float
    records_seen: int
    comments_written: int
    corpus_record_position: int
    corpus_record_total: int
    active_record_position: int
    active_record_total: int
    active_record_shards: int
    records_per_second: float
    log_path: str | None
    guard_events: int
    shard_duration_samples: int
    shard_duration_average: float | None
    shard_duration_stddev: float | None
    shard_duration_median: float | None
    shard_duration_p10: float | None
    shard_duration_p90: float | None
    recent_shard_durations: tuple[float, ...]
    active_languages: tuple[tuple[str, int, int, int], ...]
    recent_events: tuple[str, ...]


class StackV3DashboardCollector:
    def __init__(
        self,
        paths: DashboardPaths,
        *,
        output_scan_interval: float = 20.0,
        checkpoint_scan_interval: float = 5.0,
        total_records: int = _DEFAULT_TOTAL_RECORDS,
        max_disk_read_mib_s: float = _DEFAULT_MAX_DISK_READ_MIB_S,
        max_disk_write_mib_s: float = _DEFAULT_MAX_DISK_WRITE_MIB_S,
    ) -> None:
        self.paths = paths
        self.output_scan_interval = output_scan_interval
        self.checkpoint_scan_interval = checkpoint_scan_interval
        self.total_records = total_records
        self.max_disk_read_bytes_per_second = max_disk_read_mib_s * 1024**2
        self.max_disk_write_bytes_per_second = max_disk_write_mib_s * 1024**2
        self.total_shards = 0
        self.total_bytes = 0
        self.size_by_digest: dict[str, int] = {}
        self.index_by_digest: dict[str, int] = {}
        self.language_by_digest: dict[str, str] = {}
        self.total_by_language: dict[str, int] = {}
        self.source_by_remote_path: dict[str, str] = {}
        self._load_inventory()

        self._last_sample_at: float | None = None
        self._scratch_sizes: dict[str, int] = {}
        self._download_rate = 0.0
        self._download_file_rate = 0.0
        self._record_rate = 0.0
        self._record_positions: dict[str, int] = {}
        self._row_totals: dict[str, int] = {}
        self._checkpoint_cache: dict[str, tuple[int, int]] = {}
        self._active_languages: tuple[tuple[str, int, int, int], ...] = ()
        self._completed_by_language: dict[str, int] = {}
        self._corpus_checkpoint_cache: dict[str, tuple[int, int]] = {}
        self._corpus_records = 0
        self._next_checkpoint_scan_at = 0.0
        self._last_net_rx: int | None = None
        self._net_rate = 0.0
        self._last_host_cpu: tuple[int, int] | None = None
        self._last_pipeline_ticks: int | None = None
        self._last_pipeline_at: float | None = None
        self._worker_pids: set[int] = set()
        self._disk_device = _mount_block_device(self.paths.root)
        self._last_diskstats: tuple[int, int, int] | None = None
        self._disk_read_rate = 0.0
        self._disk_write_rate = 0.0
        self._disk_busy_percent = 0.0

        self._output_files = 0
        self._output_bytes = 0
        self._output_rate = 0.0
        self._last_output_scan_at: float | None = None
        self._next_output_scan_at = 0.0

        self._log_path: Path | None = None
        self._log_offset = 0
        self._run_progress: dict[tuple[str, str], tuple[int, int]] = {}
        self._guard_events = 0
        self._recent_events: deque[str] = deque(maxlen=5)
        self._download_started: dict[tuple[str, str], float] = {}
        self._run_started: dict[tuple[str, str, str], float] = {}
        self._completed_duration_keys: set[tuple[str, str, str]] = set()
        self._shard_durations: deque[tuple[float, float]] = deque(maxlen=10_000)
        self._load_duration_history()

    def _load_inventory(self) -> None:
        try:
            payload = json.loads(self.paths.listing.read_text(encoding="utf-8"))
            shards = payload.get("shards", [])
        except (OSError, ValueError, TypeError):
            shards = []
        self.total_shards = len(shards)
        self.total_bytes = 0
        self.size_by_digest.clear()
        self.index_by_digest.clear()
        self.language_by_digest.clear()
        self.total_by_language.clear()
        self.source_by_remote_path.clear()
        for fallback_index, shard in enumerate(shards):
            path = str(shard.get("path", ""))
            size = int(shard.get("size") or 0)
            index = int(shard.get("index", fallback_index))
            digest = hashlib.sha1(path.encode("utf-8")).hexdigest()[:12]
            self.size_by_digest[digest] = size
            self.index_by_digest[digest] = index
            language = str(shard.get("language") or "unknown")
            self.language_by_digest[digest] = language
            self.total_by_language[language] = self.total_by_language.get(language, 0) + 1
            self.source_by_remote_path[path] = (
                f"{self.paths.dataset}__stack-v3-shard-{index:08d}-{digest}"
            )
            self.total_bytes += size

    def sample(self) -> DashboardSnapshot:
        now = time.time()
        monotonic_now = time.monotonic()
        elapsed = (
            max(0.001, monotonic_now - self._last_sample_at)
            if self._last_sample_at is not None
            else None
        )
        self._last_sample_at = monotonic_now

        process = self._process_metrics(monotonic_now)
        completed = self._completion_metrics(now, process[3])
        scratch = self._scratch_metrics(
            elapsed,
            main_pid=process[0],
            active_worker_count=process[2],
        )
        corpus_records = self._corpus_record_metrics(monotonic_now)
        output = self._output_metrics(monotonic_now)
        host_cpu = self._host_cpu_percent()
        host_rx = self._host_network_rate(elapsed)
        memory = _memory_metrics()
        disk = shutil.disk_usage(self.paths.output if self.paths.output.exists() else self.paths.root)
        disk_io = self._disk_io_metrics(elapsed)
        self._read_log_updates()

        records_seen = sum(value[0] for value in self._run_progress.values())
        comments_written = sum(value[1] for value in self._run_progress.values())
        duration_stats = self._duration_stats()
        remaining_bytes = max(0, self.total_bytes - completed[1])
        eta_seconds = remaining_bytes / completed[4] if completed[4] > 0 else None
        return DashboardSnapshot(
            timestamp=now,
            status="RUNNING" if process[0] is not None else "STOPPED",
            main_pid=process[0],
            configured_workers=process[1],
            active_workers=process[2],
            total_shards=self.total_shards,
            completed_shards=completed[0],
            remaining_shards=max(0, self.total_shards - completed[0]),
            total_bytes=self.total_bytes,
            completed_bytes=completed[1],
            shard_rate_1h=completed[2],
            shard_rate_6h=completed[3],
            byte_rate_6h=completed[4],
            eta_seconds=eta_seconds,
            scratch_files=scratch[0],
            partial_files=scratch[1],
            scratch_bytes=scratch[2],
            download_bytes_per_second=self._download_rate,
            downloads_per_second=self._download_file_rate,
            output_files=output[0],
            output_bytes=output[1],
            output_files_per_second=self._output_rate,
            host_rx_bytes_per_second=host_rx,
            host_cpu_percent=host_cpu,
            pipeline_cpu_cores=process[4],
            memory_total=memory[0],
            memory_available=memory[1],
            swap_total=memory[2],
            swap_free=memory[3],
            disk_total=disk.total,
            disk_free=disk.free,
            disk_device=self._disk_device[1] if self._disk_device else None,
            disk_read_bytes_per_second=disk_io[0],
            disk_write_bytes_per_second=disk_io[1],
            disk_read_max_bytes_per_second=self.max_disk_read_bytes_per_second,
            disk_write_max_bytes_per_second=self.max_disk_write_bytes_per_second,
            disk_read_percent=(
                100.0 * disk_io[0] / self.max_disk_read_bytes_per_second
                if self.max_disk_read_bytes_per_second > 0
                else 0.0
            ),
            disk_write_percent=(
                100.0 * disk_io[1] / self.max_disk_write_bytes_per_second
                if self.max_disk_write_bytes_per_second > 0
                else 0.0
            ),
            disk_busy_percent=disk_io[2],
            records_seen=records_seen,
            comments_written=comments_written,
            corpus_record_position=corpus_records,
            corpus_record_total=self.total_records,
            active_record_position=scratch[3],
            active_record_total=scratch[4],
            active_record_shards=scratch[5],
            records_per_second=self._record_rate,
            log_path=str(self._log_path) if self._log_path else None,
            guard_events=self._guard_events,
            shard_duration_samples=duration_stats[0],
            shard_duration_average=duration_stats[1],
            shard_duration_stddev=duration_stats[2],
            shard_duration_median=duration_stats[3],
            shard_duration_p10=duration_stats[4],
            shard_duration_p90=duration_stats[5],
            recent_shard_durations=duration_stats[6],
            active_languages=self._active_languages,
            recent_events=tuple(self._recent_events),
        )

    def _completion_metrics(
        self,
        now: float,
        runner_started_at: float | None,
    ) -> tuple[int, int, float, float, float]:
        completed = 0
        completed_bytes = 0
        mtimes: list[float] = []
        try:
            entries = os.scandir(self.paths.completed)
        except OSError:
            self._completed_by_language = {}
            return 0, 0, 0.0, 0.0, 0.0
        completed_by_language: dict[str, int] = {}
        with entries:
            for entry in entries:
                if not entry.is_file() or not entry.name.endswith(".json"):
                    continue
                completed += 1
                digest = entry.name.removesuffix(".json")
                completed_bytes += self.size_by_digest.get(digest, 0)
                language = self.language_by_digest.get(digest)
                if language is not None:
                    completed_by_language[language] = (
                        completed_by_language.get(language, 0) + 1
                    )
                try:
                    mtimes.append(entry.stat().st_mtime)
                except OSError:
                    pass
        self._completed_by_language = completed_by_language

        one_hour = _window_rate(
            mtimes,
            now=now,
            seconds=3600.0,
            runner_started_at=runner_started_at,
        )
        six_hour = _window_rate(
            mtimes,
            now=now,
            seconds=6 * 3600.0,
            runner_started_at=runner_started_at,
        )
        six_start = now - six_hour[1]
        recent_bytes = sum(
            self.size_by_digest.get(path.name.removesuffix(".json"), 0)
            for path in _completion_paths_since(self.paths.completed, six_start)
        )
        byte_rate = recent_bytes / six_hour[1] if six_hour[1] > 0 else 0.0
        return completed, completed_bytes, one_hour[0], six_hour[0], byte_rate

    def _scratch_metrics(
        self,
        elapsed: float | None,
        *,
        main_pid: int | None,
        active_worker_count: int,
    ) -> tuple[int, int, int, int, int, int]:
        current: dict[str, int] = {}
        activity_mtimes: dict[str, int] = {}
        parquet_paths: dict[str, Path] = {}
        files = 0
        partials = 0
        total_bytes = 0
        try:
            entries = os.scandir(self.paths.scratch)
        except OSError:
            entries = None
        if entries is not None:
            with entries:
                for entry in entries:
                    if not entry.is_file():
                        continue
                    try:
                        stat_result = entry.stat()
                        size = stat_result.st_size
                    except OSError:
                        continue
                    files += 1
                    partials += int(entry.name.endswith(".partial"))
                    total_bytes += size
                    digest = Path(entry.name).stem
                    current[digest] = max(size, current.get(digest, 0))
                    activity_mtimes[digest] = max(
                        stat_result.st_mtime_ns,
                        activity_mtimes.get(digest, 0),
                    )
                    if entry.name.endswith(".parquet"):
                        parquet_paths[digest] = Path(entry.path)

        if elapsed is not None:
            downloaded = sum(
                max(0, size - self._scratch_sizes.get(digest, 0))
                for digest, size in current.items()
            )
            new_files = sum(digest not in self._scratch_sizes for digest in current)
            self._download_rate = _smooth(self._download_rate, downloaded / elapsed)
            self._download_file_rate = _smooth(self._download_file_rate, new_files / elapsed)
        self._scratch_sizes = current
        active_digests = self._registered_active_digests(main_pid)
        if active_digests is None:
            ranked = sorted(
                current,
                key=lambda digest: (
                    self._checkpoint_modified_ns(digest),
                    activity_mtimes.get(digest, 0),
                ),
                reverse=True,
            )
            active_digests = set(ranked[:active_worker_count])

        language_counts: dict[str, int] = {}
        for digest in active_digests:
            language = self.language_by_digest.get(digest, "unknown")
            language_counts[language] = language_counts.get(language, 0) + 1
        self._active_languages = tuple(
            (
                language,
                active_count,
                self._completed_by_language.get(language, 0),
                self.total_by_language.get(language, 0),
            )
            for language, active_count in sorted(
                language_counts.items(),
                key=lambda item: (-item[1], item[0].casefold()),
            )
        )

        positions: dict[str, int] = {}
        active_total = 0
        for digest, path in parquet_paths.items():
            if digest not in active_digests:
                continue
            row_total = self._row_totals.get(digest)
            if row_total is None:
                try:
                    row_total = pq.read_metadata(path).num_rows
                except Exception:
                    continue
                self._row_totals[digest] = row_total
            position = min(row_total, self._checkpoint_record_position(digest))
            positions[digest] = position
            active_total += row_total
        if elapsed is not None:
            advanced = sum(
                max(0, position - self._record_positions.get(digest, position))
                for digest, position in positions.items()
            )
            self._record_rate = _smooth(self._record_rate, advanced / elapsed)
        self._record_positions = positions
        current_digests = set(parquet_paths)
        self._row_totals = {
            digest: rows for digest, rows in self._row_totals.items() if digest in current_digests
        }
        self._checkpoint_cache = {
            digest: value for digest, value in self._checkpoint_cache.items() if digest in current_digests
        }
        return (
            files,
            partials,
            total_bytes,
            sum(positions.values()),
            active_total,
            len(positions),
        )

    def _registered_active_digests(self, main_pid: int | None) -> set[str] | None:
        if main_pid is None:
            return set()
        try:
            payload = json.loads(self.paths.active_workers.read_text(encoding="utf-8"))
            if int(payload.get("parent_pid") or 0) != main_pid:
                return None
            workers = payload.get("workers") or []
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None
        digests: set[str] = set()
        for worker in workers:
            if not isinstance(worker, dict) or not worker.get("digest"):
                continue
            try:
                pid = int(worker.get("pid") or 0)
            except (ValueError, TypeError):
                continue
            if pid in self._worker_pids:
                digests.add(str(worker["digest"]))
        return digests

    def _checkpoint_modified_ns(self, digest: str) -> int:
        index = self.index_by_digest.get(digest)
        if index is None:
            return 0
        path = self.paths.checkpoints / (
            f"{self.paths.dataset}-stack-v3-shard-{index:08d}-{digest}.json"
        )
        try:
            return path.stat().st_mtime_ns
        except OSError:
            return 0

    def _corpus_record_metrics(self, monotonic_now: float) -> int:
        if monotonic_now < self._next_checkpoint_scan_at:
            return self._corpus_records
        prefix = f"{self.paths.dataset}-stack-v3-shard-"
        current_names: set[str] = set()
        try:
            entries = os.scandir(self.paths.checkpoints)
        except OSError:
            return self._corpus_records
        with entries:
            for entry in entries:
                if (
                    not entry.is_file()
                    or not entry.name.startswith(prefix)
                    or not entry.name.endswith(".json")
                ):
                    continue
                current_names.add(entry.name)
                try:
                    modified_ns = entry.stat().st_mtime_ns
                except OSError:
                    continue
                cached = self._corpus_checkpoint_cache.get(entry.name)
                if cached is not None and cached[0] == modified_ns:
                    continue
                try:
                    payload = json.loads(Path(entry.path).read_text(encoding="utf-8"))
                    records_seen = int(payload.get("records_seen") or 0)
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    continue
                self._corpus_checkpoint_cache[entry.name] = modified_ns, records_seen
        self._corpus_checkpoint_cache = {
            name: value
            for name, value in self._corpus_checkpoint_cache.items()
            if name in current_names
        }
        self._corpus_records = sum(value[1] for value in self._corpus_checkpoint_cache.values())
        self._next_checkpoint_scan_at = monotonic_now + self.checkpoint_scan_interval
        return self._corpus_records

    def _checkpoint_record_position(self, digest: str) -> int:
        index = self.index_by_digest.get(digest)
        if index is None:
            return 0
        path = self.paths.checkpoints / (
            f"{self.paths.dataset}-stack-v3-shard-{index:08d}-{digest}.json"
        )
        try:
            modified_ns = path.stat().st_mtime_ns
        except OSError:
            return 0
        cached = self._checkpoint_cache.get(digest)
        if cached is not None and cached[0] == modified_ns:
            return cached[1]
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            last_record_id = str(payload.get("last_record_id") or "")
            position = int(last_record_id.rsplit("::row::", 1)[1]) + 1
        except (OSError, ValueError, IndexError, TypeError, json.JSONDecodeError):
            position = 0
        self._checkpoint_cache[digest] = modified_ns, position
        return position

    def _output_metrics(self, monotonic_now: float) -> tuple[int, int]:
        if monotonic_now < self._next_output_scan_at:
            return self._output_files, self._output_bytes
        files = 0
        total_bytes = 0
        for directory, _, names in os.walk(self.paths.output):
            for name in names:
                if not name.endswith(".parquet"):
                    continue
                files += 1
                try:
                    total_bytes += (Path(directory) / name).stat().st_size
                except OSError:
                    pass
        if self._last_output_scan_at is not None:
            elapsed = max(0.001, monotonic_now - self._last_output_scan_at)
            created = max(0, files - self._output_files)
            self._output_rate = _smooth(self._output_rate, created / elapsed)
        self._output_files = files
        self._output_bytes = total_bytes
        self._last_output_scan_at = monotonic_now
        self._next_output_scan_at = monotonic_now + self.output_scan_interval
        return files, total_bytes

    def _process_metrics(
        self,
        monotonic_now: float,
    ) -> tuple[int | None, int, int, float | None, float]:
        processes = _stack_v3_processes()
        processes = [process for process in processes if process[4] != "Z"]
        if not processes:
            self._last_pipeline_ticks = None
            self._last_pipeline_at = None
            self._worker_pids.clear()
            return None, 0, 0, None, 0.0

        matching_pids = {process[0] for process in processes}
        child_counts: dict[int, int] = {}
        for _, ppid, _, _, _, _ in processes:
            if ppid in matching_pids:
                child_counts[ppid] = child_counts.get(ppid, 0) + 1
        main_pid = max(matching_pids, key=lambda pid: child_counts.get(pid, 0))
        self._worker_pids = {
            pid for pid, ppid, _, _, _, _ in processes if ppid == main_pid
        }
        main = next(process for process in processes if process[0] == main_pid)
        configured = _argument_int(main[5], "--shard-workers") or max(
            0, child_counts.get(main_pid, 0) - 1
        )
        active = min(configured, child_counts.get(main_pid, 0))
        total_ticks = sum(process[2] for process in processes)
        cpu_cores = 0.0
        if self._last_pipeline_ticks is not None and self._last_pipeline_at is not None:
            elapsed = max(0.001, monotonic_now - self._last_pipeline_at)
            cpu_cores = max(0.0, total_ticks - self._last_pipeline_ticks) / _CLOCK_TICKS / elapsed
        self._last_pipeline_ticks = total_ticks
        self._last_pipeline_at = monotonic_now
        runner_started_at = _process_start_wall(main[3])
        return main_pid, configured, active, runner_started_at, cpu_cores

    def _host_cpu_percent(self) -> float:
        current = _host_cpu_totals()
        percent = 0.0
        if current is not None and self._last_host_cpu is not None:
            total_delta = current[0] - self._last_host_cpu[0]
            idle_delta = current[1] - self._last_host_cpu[1]
            if total_delta > 0:
                percent = 100.0 * max(0, total_delta - idle_delta) / total_delta
        self._last_host_cpu = current
        return percent

    def _host_network_rate(self, elapsed: float | None) -> float:
        current = _network_rx_bytes()
        if current is not None and self._last_net_rx is not None and elapsed is not None:
            self._net_rate = _smooth(self._net_rate, max(0, current - self._last_net_rx) / elapsed)
        self._last_net_rx = current
        return self._net_rate

    def _disk_io_metrics(self, elapsed: float | None) -> tuple[float, float, float]:
        if self._disk_device is None:
            return 0.0, 0.0, 0.0
        current = _block_device_stats(self._disk_device[0])
        if current is not None and self._last_diskstats is not None and elapsed is not None:
            read_bytes = max(0, current[0] - self._last_diskstats[0]) * 512
            write_bytes = max(0, current[1] - self._last_diskstats[1]) * 512
            busy_ms = max(0, current[2] - self._last_diskstats[2])
            self._disk_read_rate = _smooth(self._disk_read_rate, read_bytes / elapsed)
            self._disk_write_rate = _smooth(self._disk_write_rate, write_bytes / elapsed)
            self._disk_busy_percent = _smooth(
                self._disk_busy_percent,
                min(100.0, 100.0 * busy_ms / (elapsed * 1000.0)),
            )
        self._last_diskstats = current
        return self._disk_read_rate, self._disk_write_rate, self._disk_busy_percent

    def _read_log_updates(self) -> None:
        newest = _newest_stack_v3_log(self.paths.logs)
        if newest != self._log_path:
            self._log_path = newest
            self._log_offset = 0
            self._run_progress.clear()
            self._guard_events = 0
            self._recent_events.clear()
        if newest is None:
            return
        try:
            with newest.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(self._log_offset)
                for line in handle:
                    self._parse_log_line(line.rstrip(), log_key=str(newest))
                self._log_offset = handle.tell()
        except OSError:
            return

    def _parse_log_line(self, line: str, *, log_key: str) -> None:
        self._parse_duration_line(line, log_key=log_key)
        if "Mining progress dataset=" in line or "Finished mining run dataset=" in line:
            fields = _key_value_fields(line)
            dataset = fields.get("dataset")
            run_id = fields.get("run_id")
            if dataset and run_id:
                try:
                    records = int(fields.get("records_seen", 0))
                    comments = int(fields.get("comments_written", 0))
                except ValueError:
                    records = comments = 0
                key = (dataset, run_id)
                previous = self._run_progress.get(key, (0, 0))
                if records >= previous[0]:
                    self._run_progress[key] = records, comments
        is_guard = any(
            marker in line
            for marker in (
                "Pausing Stack v3 extraction",
                "Requeued Stack v3 shard after memory-pressure checkpoint",
                "Requeued Stack v3 shard after transient download failure",
                "Stack v3 scheduler waiting for memory",
                "Stack v3 memory launch gate is closed",
                "Stack v3 memory is above the guarded launch requirement",
                "Stack v3 memory recovery was sustained",
                "Memory safeguard triggered",
                "Memory allocation failed",
                "worker exited unexpectedly",
                "attempt exited with status",
            )
        )
        if is_guard:
            self._guard_events += 1
        if is_guard or "Finished mining run dataset=" in line or "Starting Stack v3 with" in line:
            self._recent_events.append(_compact_event(line))

    def _load_duration_history(self) -> None:
        try:
            logs = sorted(
                self.paths.logs.glob("stack-v3*.log"),
                key=lambda path: path.stat().st_mtime,
            )[-5:]
        except OSError:
            return
        for path in logs:
            try:
                with path.open("r", encoding="utf-8", errors="replace") as handle:
                    for line in handle:
                        self._parse_duration_line(line.rstrip(), log_key=str(path))
            except OSError:
                continue

    def _parse_duration_line(self, line: str, *, log_key: str) -> None:
        timestamp = _log_timestamp(line)
        if timestamp is None:
            return
        if "Downloading Stack v3 shard" in line:
            remote_path = _key_value_fields(line).get("path")
            source = self.source_by_remote_path.get(remote_path or "")
            if source:
                self._download_started[(log_key, source)] = timestamp
            return
        if "Starting mining run dataset=" in line:
            fields = _key_value_fields(line)
            dataset = fields.get("dataset")
            run_id = fields.get("run_id")
            if dataset and run_id:
                self._run_started[(log_key, dataset, run_id)] = self._download_started.pop(
                    (log_key, dataset),
                    timestamp,
                )
            return
        if "Finished mining run dataset=" not in line:
            return
        fields = _key_value_fields(line)
        dataset = fields.get("dataset")
        run_id = fields.get("run_id")
        if not dataset or not run_id:
            return
        key = (log_key, dataset, run_id)
        if key in self._completed_duration_keys:
            return
        started_at = self._run_started.pop(key, None)
        if started_at is None or timestamp < started_at:
            return
        self._completed_duration_keys.add(key)
        self._shard_durations.append((timestamp, timestamp - started_at))

    def _duration_stats(
        self,
    ) -> tuple[
        int,
        float | None,
        float | None,
        float | None,
        float | None,
        float | None,
        tuple[float, ...],
    ]:
        recent_pairs = sorted(self._shard_durations, key=lambda value: value[0])[-100:]
        durations = [value[1] for value in recent_pairs]
        latest = tuple(value[1] for value in recent_pairs[-5:])
        if not durations:
            return 0, None, None, None, None, None, latest
        ordered = sorted(durations)
        return (
            len(durations),
            statistics.fmean(durations),
            statistics.pstdev(durations),
            statistics.median(durations),
            _percentile(ordered, 0.10),
            _percentile(ordered, 0.90),
            latest,
        )


def _window_rate(
    mtimes: Iterable[float],
    *,
    now: float,
    seconds: float,
    runner_started_at: float | None,
) -> tuple[float, float]:
    interval = seconds
    if runner_started_at is not None:
        interval = min(seconds, max(1.0, now - runner_started_at))
    start = now - interval
    count = sum(mtime >= start for mtime in mtimes)
    return count * 3600.0 / interval, interval


def _completion_paths_since(directory: Path, started_at: float) -> list[Path]:
    paths: list[Path] = []
    try:
        entries = os.scandir(directory)
    except OSError:
        return paths
    with entries:
        for entry in entries:
            if not entry.is_file() or not entry.name.endswith(".json"):
                continue
            try:
                if entry.stat().st_mtime >= started_at:
                    paths.append(Path(entry.path))
            except OSError:
                pass
    return paths


def _stack_v3_processes() -> list[tuple[int, int, int, int, str, tuple[str, ...]]]:
    processes: list[tuple[int, int, int, int, str, tuple[str, ...]]] = []
    try:
        proc_entries = os.scandir("/proc")
    except OSError:
        return processes
    with proc_entries:
        for entry in proc_entries:
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            try:
                cmdline = tuple(
                    value.decode("utf-8", errors="replace")
                    for value in (Path(entry.path) / "cmdline").read_bytes().split(b"\0")
                    if value
                )
                if not any("mine-stack-v3-shards" in value for value in cmdline):
                    continue
                stat = (Path(entry.path) / "stat").read_text(encoding="utf-8")
                close = stat.rfind(")")
                values = stat[close + 2 :].split()
                ppid = int(values[1])
                ticks = int(values[11]) + int(values[12])
                start_ticks = int(values[19])
                state = values[0]
            except (OSError, ValueError, IndexError):
                continue
            processes.append((pid, ppid, ticks, start_ticks, state, cmdline))
    return processes


def _process_start_wall(start_ticks: int) -> float | None:
    try:
        uptime = float(Path("/proc/uptime").read_text(encoding="utf-8").split()[0])
    except (OSError, ValueError, IndexError):
        return None
    return time.time() - uptime + start_ticks / _CLOCK_TICKS


def _argument_int(arguments: tuple[str, ...], name: str) -> int | None:
    try:
        return int(arguments[arguments.index(name) + 1])
    except (ValueError, IndexError):
        return None


def _host_cpu_totals() -> tuple[int, int] | None:
    try:
        fields = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()[1:]
        values = [int(value) for value in fields]
    except (OSError, ValueError, IndexError):
        return None
    total = sum(values)
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return total, idle


def _network_rx_bytes() -> int | None:
    try:
        lines = Path("/proc/net/dev").read_text(encoding="utf-8").splitlines()[2:]
    except OSError:
        return None
    total = 0
    for line in lines:
        try:
            interface, values = line.split(":", 1)
            if interface.strip() == "lo":
                continue
            total += int(values.split()[0])
        except (ValueError, IndexError):
            continue
    return total


def _mount_block_device(root: Path) -> tuple[str, str] | None:
    target = str(root.resolve())
    best: tuple[int, str, str] | None = None
    try:
        lines = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        fields = line.split()
        if len(fields) < 10 or "-" not in fields:
            continue
        mountpoint = _unescape_mount_path(fields[4])
        if target != mountpoint and not target.startswith(mountpoint.rstrip("/") + "/"):
            continue
        separator = fields.index("-")
        source = fields[separator + 2] if len(fields) > separator + 2 else fields[2]
        candidate = (len(mountpoint), fields[2], Path(source).name)
        if best is None or candidate[0] > best[0]:
            best = candidate
    return (best[1], best[2]) if best else None


def _unescape_mount_path(value: str) -> str:
    return (
        value.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
    )


def _block_device_stats(major_minor: str) -> tuple[int, int, int] | None:
    try:
        lines = Path("/proc/diskstats").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    return _parse_diskstats(lines, major_minor)


def _parse_diskstats(lines: Iterable[str], major_minor: str) -> tuple[int, int, int] | None:
    try:
        expected_major, expected_minor = (int(value) for value in major_minor.split(":", 1))
    except ValueError:
        return None
    for line in lines:
        fields = line.split()
        if len(fields) < 14:
            continue
        try:
            if int(fields[0]) != expected_major or int(fields[1]) != expected_minor:
                continue
            return int(fields[5]), int(fields[9]), int(fields[12])
        except ValueError:
            continue
    return None


def _memory_metrics() -> tuple[int, int, int, int]:
    values: dict[str, int] = {}
    try:
        lines = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        return 0, 0, 0, 0
    for line in lines:
        key, _, raw = line.partition(":")
        if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            try:
                values[key] = int(raw.split()[0]) * 1024
            except (ValueError, IndexError):
                pass
    return (
        values.get("MemTotal", 0),
        values.get("MemAvailable", 0),
        values.get("SwapTotal", 0),
        values.get("SwapFree", 0),
    )


def _newest_stack_v3_log(directory: Path) -> Path | None:
    try:
        candidates = list(directory.glob("stack-v3*.log"))
    except OSError:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime, default=None)


def _key_value_fields(line: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in line.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key] = value
    return fields


def _compact_event(line: str) -> str:
    if len(line) >= 24 and line[4] == "-" and line[13] == ":":
        line = line[11:19] + " " + line[24:]
    return line


def _log_timestamp(line: str) -> float | None:
    if len(line) < 23:
        return None
    try:
        parsed = datetime.strptime(line[:23], "%Y-%m-%d %H:%M:%S,%f").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None
    return parsed.timestamp()


def _percentile(ordered: list[float], fraction: float) -> float:
    if not ordered:
        raise ValueError("ordered values must not be empty")
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _smooth(previous: float, current: float, alpha: float = 0.35) -> float:
    return current if previous == 0 else alpha * current + (1 - alpha) * previous


def format_bytes(value: float) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    value = float(max(0, value))
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if value < 100 else f"{value:.0f} {unit}"
        value /= 1024
    return f"{value:.1f} PiB"


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "unknown"
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.0f}m"
    hours = minutes / 60
    if hours < 48:
        return f"{hours:.1f}h"
    return f"{hours / 24:.1f}d"


def format_rate_short(value: float) -> str:
    return format_bytes(value).replace(" ", "") + "/s"


def format_active_languages(
    languages: tuple[tuple[str, int, int, int], ...],
    *,
    max_characters: int,
) -> str:
    if not languages:
        return "none"
    parts: list[str] = []
    for index, (language, active, completed, total) in enumerate(languages):
        part = f"{language} ({active}, {completed}/{total})"
        remaining = len(languages) - index - 1
        suffix = f", +{remaining} more" if remaining else ""
        candidate = ", ".join((*parts, part)) + suffix
        if parts and len(candidate) > max_characters:
            return ", ".join(parts) + f", +{len(languages) - len(parts)} more"
        parts.append(part)
    return ", ".join(parts)


def _percentage(part: int, total: int) -> float:
    return 100.0 * part / total if total else 0.0


def _bar(part: int, total: int, width: int) -> str:
    width = max(1, width)
    filled = round(width * part / total) if total else 0
    if part > 0 and total > 0 and filled == 0:
        filled = 1
    filled = min(width, max(0, filled))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _render(stdscr: Any, snapshot: DashboardSnapshot) -> None:
    stdscr.erase()
    height, width = stdscr.getmaxyx()
    normal = curses.color_pair(1)
    heading = curses.color_pair(2) | curses.A_BOLD
    good = curses.color_pair(3) | curses.A_BOLD
    warning = curses.color_pair(4) | curses.A_BOLD
    dim = curses.color_pair(5)

    def put(row: int, text: str, attr: int = normal) -> None:
        if row < 0 or row >= height or width < 2:
            return
        try:
            stdscr.addnstr(row, 0, text.ljust(max(0, width - 1)), max(0, width - 1), attr)
        except curses.error:
            pass

    status_attr = good if snapshot.status == "RUNNING" else warning
    clock = datetime.fromtimestamp(snapshot.timestamp, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    title = f" CommentMiner / Stack v3   {clock}   {snapshot.status} "
    put(0, title, status_attr)
    if height < 24 or width < 72:
        put(2, f"Shards  {snapshot.completed_shards:,}/{snapshot.total_shards:,}  ({_percentage(snapshot.completed_shards, snapshot.total_shards):.3f}%)")
        put(3, f"Workers {snapshot.active_workers}/{snapshot.configured_workers}   CPU {snapshot.pipeline_cpu_cores:.1f} cores")
        put(4, f"Download {format_bytes(snapshot.download_bytes_per_second)}/s   Host RX {format_bytes(snapshot.host_rx_bytes_per_second)}/s")
        put(5, f"RAM available {format_bytes(snapshot.memory_available)}   Disk free {format_bytes(snapshot.disk_free)}")
        put(6, f"Active records {snapshot.active_record_position:,}/{snapshot.active_record_total:,}   {snapshot.records_per_second:,.0f}/s")
        put(7, f"Shard time avg {format_duration(snapshot.shard_duration_average)} +/- {format_duration(snapshot.shard_duration_stddev)}")
        put(8, f"Languages {format_active_languages(snapshot.active_languages, max_characters=max(10, width - 12))}")
        put(10, "Terminal is too small for the full dashboard (minimum 72x24).", warning)
        put(height - 1, " q quit   r reset/reload ", dim)
        stdscr.refresh()
        return

    shard_bar_width = max(10, min(50, width - 42))
    record_corpus_bar_width = max(10, min(50, width - 53))
    progress = _bar(snapshot.completed_shards, snapshot.total_shards, shard_bar_width)
    put(2, "CORPUS", heading)
    put(
        3,
        f"Shards       {snapshot.completed_shards:>7,} / {snapshot.total_shards:<7,} "
        f"{progress}  {_percentage(snapshot.completed_shards, snapshot.total_shards):6.3f}%",
    )
    put(
        4,
        f"Records*     {snapshot.corpus_record_position:>10,} / {snapshot.corpus_record_total:<14,} "
        f"{_bar(snapshot.corpus_record_position, snapshot.corpus_record_total, record_corpus_bar_width)}  "
        f"{_percentage(snapshot.corpus_record_position, snapshot.corpus_record_total):6.3f}%",
    )
    put(
        5,
        f"Compressed   {format_bytes(snapshot.completed_bytes):>10} / {format_bytes(snapshot.total_bytes):<10}"
        f"   Remaining {snapshot.remaining_shards:,} shards",
    )
    put(
        6,
        f"Throughput   {snapshot.shard_rate_1h:7.1f} shards/h (1h)   "
        f"{snapshot.shard_rate_6h:7.1f} shards/h (6h)   "
        f"{format_bytes(snapshot.byte_rate_6h)}/s   ETA {format_duration(snapshot.eta_seconds)}",
    )
    put(
        7,
        f"Shard time   last {snapshot.shard_duration_samples:>3}  "
        f"avg {format_duration(snapshot.shard_duration_average)} +/- {format_duration(snapshot.shard_duration_stddev)}   "
        f"median {format_duration(snapshot.shard_duration_median)}   "
        f"p10-p90 {format_duration(snapshot.shard_duration_p10)}-{format_duration(snapshot.shard_duration_p90)}",
    )

    put(8, "LIVE PIPELINE", heading)
    put(
        9,
        f"Workers      {snapshot.active_workers:>3}/{snapshot.configured_workers:<3} active   "
        f"Pipeline CPU {snapshot.pipeline_cpu_cores:6.1f} cores   Host CPU {snapshot.host_cpu_percent:5.1f}%   PID {snapshot.main_pid or '-'}",
    )
    put(
        10,
        f"Downloads    {snapshot.scratch_files:>4} files ({snapshot.partial_files} partial)   "
        f"Scratch {format_bytes(snapshot.scratch_bytes):>10}   "
        f"Write {format_bytes(snapshot.download_bytes_per_second)}/s   {snapshot.downloads_per_second:.2f} files/s",
    )
    put(
        11,
        f"Host network RX {format_bytes(snapshot.host_rx_bytes_per_second)}/s   "
        f"Output {snapshot.output_files:,} parquet files / {format_bytes(snapshot.output_bytes)}   "
        f"{snapshot.output_files_per_second:.2f} files/s",
    )
    put(
        12,
        f"Current log  {snapshot.records_seen:,} records   {snapshot.comments_written:,} comments   "
        f"Memory/supervisor events {snapshot.guard_events}",
    )
    record_rate = f"{snapshot.records_per_second:,.0f}/s"
    record_bar_width = max(8, min(40, width - 62 - len(record_rate)))
    record_suffix = (
        f"  {snapshot.active_record_shards} ready shards" if width >= 105 else ""
    )
    put(
        13,
        f"Active rows  {snapshot.active_record_position:>10,} / {snapshot.active_record_total:<10,} "
        f"{_bar(snapshot.active_record_position, snapshot.active_record_total, record_bar_width)} "
        f"{_percentage(snapshot.active_record_position, snapshot.active_record_total):5.1f}%  "
        f"{record_rate}{record_suffix}",
    )
    put(
        14,
        "Languages    "
        + format_active_languages(
            snapshot.active_languages,
            max_characters=max(10, width - 14),
        ),
    )

    put(15, "SYSTEM", heading)
    memory_used = max(0, snapshot.memory_total - snapshot.memory_available)
    swap_used = max(0, snapshot.swap_total - snapshot.swap_free)
    disk_used = max(0, snapshot.disk_total - snapshot.disk_free)
    put(
        16,
        f"Memory       {format_bytes(memory_used):>10} used   {format_bytes(snapshot.memory_available):>10} available   "
        f"{_percentage(memory_used, snapshot.memory_total):5.1f}% pressure",
        warning if snapshot.memory_available < 16 * _GIB else normal,
    )
    put(
        17,
        f"Swap         {format_bytes(swap_used):>10} used / {format_bytes(snapshot.swap_total):<10}   "
        f"Disk {format_bytes(disk_used)} used / {format_bytes(snapshot.disk_total)}   {format_bytes(snapshot.disk_free)} free",
    )
    put(
        18,
        f"Disk I/O     {snapshot.disk_device or '-'}   "
        f"R {format_rate_short(snapshot.disk_read_bytes_per_second)} "
        f"{snapshot.disk_read_percent:5.1f}%/{format_rate_short(snapshot.disk_read_max_bytes_per_second)}   "
        f"W {format_rate_short(snapshot.disk_write_bytes_per_second)} "
        f"{snapshot.disk_write_percent:5.1f}%/{format_rate_short(snapshot.disk_write_max_bytes_per_second)}   "
        f"busy {snapshot.disk_busy_percent:4.1f}%",
    )

    event_row = 20
    put(event_row, "RECENT SHARD TIMES / ACTIVITY", heading)
    latest_durations = ", ".join(
        format_duration(value) for value in snapshot.recent_shard_durations
    )
    put(event_row + 1, f"  Latest: {latest_durations or 'waiting for completed shards...'}", dim)
    events = snapshot.recent_events[-max(0, height - event_row - 3) :]
    if events:
        for offset, event in enumerate(events, 2):
            put(event_row + offset, "  " + event, dim)
    footer = " q quit   r reload   * corpus records approximate (43.9B - 28.3B stubs) "
    put(height - 1, footer, dim)
    stdscr.refresh()


def _run_tui(stdscr: Any, collector: StackV3DashboardCollector, refresh: float) -> None:
    curses.curs_set(0)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_WHITE, -1)
    curses.init_pair(2, curses.COLOR_CYAN, -1)
    curses.init_pair(3, curses.COLOR_GREEN, -1)
    curses.init_pair(4, curses.COLOR_YELLOW, -1)
    curses.init_pair(5, curses.COLOR_BLUE, -1)
    stdscr.nodelay(True)
    stdscr.keypad(True)
    while True:
        started = time.monotonic()
        snapshot = collector.sample()
        _render(stdscr, snapshot)
        while time.monotonic() - started < refresh:
            try:
                key = stdscr.getch()
            except curses.error:
                key = -1
            if key in (ord("q"), ord("Q")):
                return
            if key in (ord("r"), ord("R")):
                collector._load_inventory()
                break
            if key == curses.KEY_RESIZE:
                _render(stdscr, snapshot)
            time.sleep(0.05)


def _plain_snapshot(snapshot: DashboardSnapshot) -> str:
    return "\n".join(
        (
            f"Status: {snapshot.status} (PID {snapshot.main_pid or '-'})",
            f"Shards: {snapshot.completed_shards:,}/{snapshot.total_shards:,} "
            f"({_percentage(snapshot.completed_shards, snapshot.total_shards):.3f}%), "
            f"{snapshot.remaining_shards:,} remaining",
            f"Corpus records*: {snapshot.corpus_record_position:,}/{snapshot.corpus_record_total:,} "
            f"({_percentage(snapshot.corpus_record_position, snapshot.corpus_record_total):.3f}%)",
            f"Throughput: {snapshot.shard_rate_1h:.1f} shards/h (1h), "
            f"{snapshot.shard_rate_6h:.1f} shards/h (6h), ETA {format_duration(snapshot.eta_seconds)}",
            f"Shard time (last {snapshot.shard_duration_samples}): average "
            f"{format_duration(snapshot.shard_duration_average)} +/- {format_duration(snapshot.shard_duration_stddev)}, "
            f"median {format_duration(snapshot.shard_duration_median)}, "
            f"p10-p90 {format_duration(snapshot.shard_duration_p10)}-{format_duration(snapshot.shard_duration_p90)}",
            "Recent shard times: "
            + (", ".join(format_duration(value) for value in snapshot.recent_shard_durations) or "none"),
            f"Workers: {snapshot.active_workers}/{snapshot.configured_workers}; "
            f"pipeline CPU {snapshot.pipeline_cpu_cores:.1f} cores; host CPU {snapshot.host_cpu_percent:.1f}%",
            f"Downloads: {snapshot.scratch_files} files ({snapshot.partial_files} partial), "
            f"{format_bytes(snapshot.scratch_bytes)} scratch, "
            f"{format_bytes(snapshot.download_bytes_per_second)}/s, {snapshot.downloads_per_second:.2f} files/s",
            f"Host network RX: {format_bytes(snapshot.host_rx_bytes_per_second)}/s",
            f"Output: {snapshot.output_files:,} parquet files, {format_bytes(snapshot.output_bytes)}, "
            f"{snapshot.output_files_per_second:.2f} files/s",
            f"Current log: {snapshot.records_seen:,} records, {snapshot.comments_written:,} comments",
            f"Active records: {snapshot.active_record_position:,}/{snapshot.active_record_total:,} "
            f"({_percentage(snapshot.active_record_position, snapshot.active_record_total):.1f}%), "
            f"{snapshot.records_per_second:,.0f} records/s across {snapshot.active_record_shards} ready shards",
            "Active languages: "
            + format_active_languages(snapshot.active_languages, max_characters=10_000),
            f"Memory: {format_bytes(snapshot.memory_available)} available / {format_bytes(snapshot.memory_total)}; "
            f"swap {format_bytes(snapshot.swap_total - snapshot.swap_free)} used",
            f"Disk: {format_bytes(snapshot.disk_free)} free / {format_bytes(snapshot.disk_total)}",
            f"Disk I/O ({snapshot.disk_device or '-'}): read "
            f"{format_bytes(snapshot.disk_read_bytes_per_second)}/s "
            f"({snapshot.disk_read_percent:.1f}% of {format_bytes(snapshot.disk_read_max_bytes_per_second)}/s), "
            f"write {format_bytes(snapshot.disk_write_bytes_per_second)}/s "
            f"({snapshot.disk_write_percent:.1f}% of {format_bytes(snapshot.disk_write_max_bytes_per_second)}/s), "
            f"busy {snapshot.disk_busy_percent:.1f}%",
            f"Memory/supervisor events: {snapshot.guard_events}",
            f"Log: {snapshot.log_path or '-'}",
            "* Corpus record denominator is approximate: 43.9B metadata entries minus 28.3B stubs.",
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live TUI dashboard for the Stack v3 shard runner.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="CommentMiner repository root.")
    parser.add_argument("--dataset", default="the-stack-v3-full")
    parser.add_argument(
        "--total-records",
        type=int,
        default=_DEFAULT_TOTAL_RECORDS,
        help="Approximate non-stub corpus records; defaults to 43.9B metadata entries minus 28.3B stubs.",
    )
    parser.add_argument(
        "--max-disk-read-mib-s",
        type=float,
        default=_DEFAULT_MAX_DISK_READ_MIB_S,
        help="Read-speed ceiling used for the disk percentage (default: 2000 MiB/s).",
    )
    parser.add_argument(
        "--max-disk-write-mib-s",
        type=float,
        default=_DEFAULT_MAX_DISK_WRITE_MIB_S,
        help="Write-speed ceiling used for the disk percentage (default: 1000 MiB/s).",
    )
    parser.add_argument("--refresh", type=float, default=2.0, help="TUI refresh interval in seconds.")
    parser.add_argument("--once", action="store_true", help="Print one non-interactive snapshot and exit.")
    parser.add_argument("--json", action="store_true", help="With --once, emit JSON.")
    parser.add_argument(
        "--sample-seconds",
        type=float,
        default=1.0,
        help="With --once, wait this long between samples to measure instantaneous rates.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.refresh <= 0:
        raise SystemExit("--refresh must be greater than zero")
    if args.sample_seconds < 0:
        raise SystemExit("--sample-seconds must be at least zero")
    if args.total_records < 1:
        raise SystemExit("--total-records must be at least one")
    if args.max_disk_read_mib_s <= 0 or args.max_disk_write_mib_s <= 0:
        raise SystemExit("disk speed ceilings must be greater than zero")
    paths = DashboardPaths.from_root(args.root, args.dataset)
    collector = StackV3DashboardCollector(
        paths,
        total_records=args.total_records,
        max_disk_read_mib_s=args.max_disk_read_mib_s,
        max_disk_write_mib_s=args.max_disk_write_mib_s,
    )
    if args.once:
        collector.sample()
        if args.sample_seconds:
            time.sleep(args.sample_seconds)
        snapshot = collector.sample()
        if args.json:
            print(json.dumps(asdict(snapshot), indent=2))
        else:
            print(_plain_snapshot(snapshot))
        return 0
    if not os.isatty(0) or not os.isatty(1):
        raise SystemExit("The interactive dashboard requires a TTY; use --once for plain output.")
    curses.wrapper(_run_tui, collector, args.refresh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
