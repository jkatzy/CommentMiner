from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from typing import Any, Callable, Iterable


_LOGGER = logging.getLogger(__name__)
_STACK_V2_MIN_LICENSE_SCORE = 95.0
_STACK_V2_MIN_MATCH_COVERAGE = 95.0


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


@dataclass(slots=True)
class LicenseScanCheckpoint:
    source_directory: str
    completed_shards: list[str] = field(default_factory=list)
    updated_at: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["completed_shards"] = sorted(self.completed_shards)
        return payload

    @classmethod
    def from_path(cls, path: Path, *, source_directory: Path) -> "LicenseScanCheckpoint":
        if not path.exists():
            return cls(source_directory=str(source_directory))
        raw = json.loads(path.read_text(encoding="utf-8"))
        checkpoint = cls(
            source_directory=str(raw.get("source_directory", source_directory)),
            completed_shards=[str(item) for item in raw.get("completed_shards", [])],
            updated_at=str(raw["updated_at"]) if raw.get("updated_at") is not None else None,
        )
        if checkpoint.source_directory != str(source_directory):
            return cls(source_directory=str(source_directory))
        return checkpoint

    def save(self, path: Path) -> Path:
        self.updated_at = _utc_now()
        temp_path = path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        temp_path.replace(path)
        return path


@dataclass(slots=True)
class LicenseScanStats:
    input_directory: Path
    output_directory: Path
    records_scanned: int = 0
    records_with_detected_license: int = 0
    records_without_detected_license: int = 0
    shards_processed: int = 0
    shards_skipped: int = 0
    batches_run: int = 0


@dataclass(slots=True)
class _ScannedBatch:
    headers: list[dict[str, Any]]
    records_scanned: int
    detections: int


def scan_comment_licenses(
    input_directory: Path,
    *,
    output_directory: Path | None = None,
    scancode_command: str = "scancode",
    batch_size: int = 500,
    min_license_score: float = _STACK_V2_MIN_LICENSE_SCORE,
    min_match_coverage: float = _STACK_V2_MIN_MATCH_COVERAGE,
    progress_every: int = 1000,
    runner: Callable[..., dict[str, Any]] | None = None,
) -> LicenseScanStats:
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if progress_every < 1:
        raise ValueError(f"progress_every must be >= 1, got {progress_every}")

    input_directory = input_directory.resolve()
    if not input_directory.exists() or not input_directory.is_dir():
        raise ValueError(f"Input run directory does not exist: {input_directory}")

    input_shards = sorted(input_directory.glob("part-*.jsonl"))
    if not input_shards:
        raise ValueError(f"No input shard files found in: {input_directory}")

    output_directory = (output_directory or input_directory.parent / f"{input_directory.name}-license-scan").resolve()
    if output_directory == input_directory:
        raise ValueError("Output directory must differ from the input run directory")
    output_directory.mkdir(parents=True, exist_ok=True)
    temp_root = output_directory / ".tmp"
    temp_root.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_directory / "license-scan-checkpoint.json"
    checkpoint = LicenseScanCheckpoint.from_path(checkpoint_path, source_directory=input_directory)
    completed_shards = set(checkpoint.completed_shards)
    stats = LicenseScanStats(input_directory=input_directory, output_directory=output_directory)
    scancode_headers: list[dict[str, Any]] = []
    next_progress_update = progress_every

    _LOGGER.info(
        "Starting comment license scan input_directory=%s output_directory=%s batch_size=%s scancode_command=%s",
        input_directory,
        output_directory,
        batch_size,
        scancode_command,
    )

    try:
        for input_shard in input_shards:
            shard_name = input_shard.name
            output_shard = output_directory / shard_name
            if shard_name in completed_shards:
                stats.shards_skipped += 1
                _LOGGER.info("Skipping already scanned shard=%s", shard_name)
                continue

            _LOGGER.info("Scanning shard=%s", input_shard)
            records_in_shard = 0
            detections_in_shard = 0
            with input_shard.open("r", encoding="utf-8") as source_handle, output_shard.open(
                "w", encoding="utf-8"
            ) as output_handle:
                batch: list[dict[str, Any]] = []
                for line in source_handle:
                    if not line.strip():
                        continue
                    batch.append(json.loads(line))
                    if len(batch) >= batch_size:
                        batch_result = _scan_batch(
                            batch,
                            output_handle,
                            temp_root,
                            scancode_command=scancode_command,
                            min_license_score=min_license_score,
                            min_match_coverage=min_match_coverage,
                            runner=runner,
                        )
                        if batch_result.headers and not scancode_headers:
                            scancode_headers = batch_result.headers
                        records_in_shard += batch_result.records_scanned
                        detections_in_shard += batch_result.detections
                        next_progress_update = _apply_scanned_batch(
                            stats,
                            batch_result,
                            shard_name=shard_name,
                            progress_every=progress_every,
                            next_progress_update=next_progress_update,
                        )
                        batch = []
                if batch:
                    batch_result = _scan_batch(
                        batch,
                        output_handle,
                        temp_root,
                        scancode_command=scancode_command,
                        min_license_score=min_license_score,
                        min_match_coverage=min_match_coverage,
                        runner=runner,
                    )
                    if batch_result.headers and not scancode_headers:
                        scancode_headers = batch_result.headers
                    records_in_shard += batch_result.records_scanned
                    detections_in_shard += batch_result.detections
                    next_progress_update = _apply_scanned_batch(
                        stats,
                        batch_result,
                        shard_name=shard_name,
                        progress_every=progress_every,
                        next_progress_update=next_progress_update,
                    )

            completed_shards.add(shard_name)
            checkpoint.completed_shards = sorted(completed_shards)
            checkpoint.save(checkpoint_path)
            stats.shards_processed += 1
            _LOGGER.info(
                "Finished scanning shard=%s records=%s detected_licenses=%s",
                shard_name,
                records_in_shard,
                detections_in_shard,
            )
    finally:
        if temp_root.exists():
            shutil.rmtree(temp_root)

    _write_license_scan_manifest(
        input_directory=input_directory,
        output_directory=output_directory,
        checkpoint_path=checkpoint_path,
        stats=stats,
        batch_size=batch_size,
        min_license_score=min_license_score,
        min_match_coverage=min_match_coverage,
        scancode_command=scancode_command,
        scancode_headers=scancode_headers,
    )
    _LOGGER.info(
        "Finished comment license scan records_scanned=%s records_with_detected_license=%s shards_processed=%s shards_skipped=%s",
        stats.records_scanned,
        stats.records_with_detected_license,
        stats.shards_processed,
        stats.shards_skipped,
    )
    return stats


def _scan_batch(
    batch: list[dict[str, Any]],
    output_handle,
    temp_root: Path,
    *,
    scancode_command: str,
    min_license_score: float,
    min_match_coverage: float,
    runner: Callable[..., dict[str, Any]] | None,
) -> _ScannedBatch:
    with TemporaryDirectory(dir=temp_root) as temp_dir:
        batch_root = Path(temp_dir)
        inputs_dir = batch_root / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        mapping: dict[str, dict[str, Any]] = {}

        for index, payload in enumerate(batch):
            file_name = f"comment-{index:06d}.txt"
            mapping[file_name] = payload
            comment = str(payload.get("opening_comment", ""))
            (inputs_dir / file_name).write_text(comment, encoding="utf-8")

        scan_result = _run_scancode(
            inputs_dir,
            batch_root / "scancode-result.json",
            scancode_command=scancode_command,
            runner=runner,
        )
        resource_results = _resource_results_by_filename(scan_result)

        for file_name, payload in mapping.items():
            payload["comment_license_detection"] = _extract_license_detection(
                resource_results.get(file_name),
                min_license_score=min_license_score,
                min_match_coverage=min_match_coverage,
            )
            output_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

        detections = sum(
            1
            for payload in batch
            if payload["comment_license_detection"]["contains_license_notice"]
        )
        headers = scan_result.get("headers", [])
        return _ScannedBatch(
            headers=headers if isinstance(headers, list) else [],
            records_scanned=len(batch),
            detections=detections,
        )


def _apply_scanned_batch(
    stats: LicenseScanStats,
    batch_result: _ScannedBatch,
    *,
    shard_name: str,
    progress_every: int,
    next_progress_update: int,
) -> int:
    stats.batches_run += 1
    stats.records_scanned += batch_result.records_scanned
    stats.records_with_detected_license += batch_result.detections
    stats.records_without_detected_license += batch_result.records_scanned - batch_result.detections

    while stats.records_scanned >= next_progress_update:
        _LOGGER.info(
            "License scan progress records_scanned=%s records_with_detected_license=%s shards_processed=%s current_shard=%s",
            stats.records_scanned,
            stats.records_with_detected_license,
            stats.shards_processed,
            shard_name,
        )
        next_progress_update += progress_every

    return next_progress_update


def _run_scancode(
    inputs_dir: Path,
    output_path: Path,
    *,
    scancode_command: str,
    runner: Callable[..., dict[str, Any]] | None,
) -> dict[str, Any]:
    if runner is not None:
        return runner(
            inputs_dir=inputs_dir,
            output_path=output_path,
            scancode_command=scancode_command,
        )

    command = [scancode_command, "--quiet", "--license", "--json-pp", str(output_path)]
    command.append(str(inputs_dir))

    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ScanCode CLI was not found. Install 'scancode-toolkit' or pass --scancode with the executable path."
        ) from exc

    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        detail = stderr or stdout or "no output"
        raise RuntimeError(f"ScanCode failed with exit code {completed.returncode}: {detail}")

    return json.loads(output_path.read_text(encoding="utf-8"))


def _resource_results_by_filename(scan_result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    resources = scan_result.get("files")
    if resources is None:
        resources = scan_result.get("resources", [])
    result: dict[str, dict[str, Any]] = {}
    for resource in resources or []:
        if not isinstance(resource, dict):
            continue
        path = resource.get("path")
        if path is None:
            continue
        result[Path(str(path)).name] = resource
    return result


def _extract_license_detection(
    resource: dict[str, Any] | None,
    *,
    min_license_score: float,
    min_match_coverage: float,
) -> dict[str, Any]:
    if resource is None:
        return {
            "contains_license_notice": False,
            "detected_license_expression": None,
            "detected_license_expression_spdx": None,
            "percentage_of_license_text": 0,
            "license_matches": [],
            "scan_errors": [],
        }

    matching_licenses = _filtered_license_matches(
        resource,
        min_license_score=min_license_score,
        min_match_coverage=min_match_coverage,
    )
    contains_license_notice = bool(matching_licenses)
    detected_expression = _resource_field(
        resource,
        "detected_license_expression",
        "detected_license_expressions",
    )
    detected_expression_spdx = _resource_field(
        resource,
        "detected_license_expression_spdx",
        "detected_license_expressions_spdx",
    )

    return {
        "contains_license_notice": contains_license_notice,
        "detected_license_expression": detected_expression if contains_license_notice else None,
        "detected_license_expression_spdx": detected_expression_spdx if contains_license_notice else None,
        "percentage_of_license_text": resource.get("percentage_of_license_text", 0) if contains_license_notice else 0,
        "license_matches": matching_licenses,
        "scan_errors": resource.get("scan_errors", []),
    }


def _filtered_license_matches(
    resource: dict[str, Any],
    *,
    min_license_score: float,
    min_match_coverage: float,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for match in _iter_scancode_matches(resource):
        score = float(match.get("score", 0.0) or 0.0)
        coverage = float(match.get("match_coverage", 0.0) or 0.0)
        if score < min_license_score or coverage < min_match_coverage:
            continue
        matches.append(match)
    return matches


def _iter_scancode_matches(resource: dict[str, Any]) -> Iterable[dict[str, Any]]:
    legacy_licenses = resource.get("licenses") or []
    for lic in legacy_licenses:
        if not isinstance(lic, dict):
            continue
        matched_rule = lic.get("matched_rule", {}) or {}
        yield {
            "license_expression": lic.get("spdx_license_key") or lic.get("key"),
            "score": float(lic.get("score", 0.0) or 0.0),
            "match_coverage": float(
                lic.get("match_coverage", 0.0) or matched_rule.get("coverage", 0.0) or 0.0
            ),
            "rule_identifier": matched_rule.get("identifier"),
            "matcher": lic.get("matcher"),
            "start_line": lic.get("start_line"),
            "end_line": lic.get("end_line"),
        }

    for detection in resource.get("license_detections", []) or []:
        if not isinstance(detection, dict):
            continue
        detection_expression = detection.get("license_expression")
        detection_spdx = detection.get("license_expression_spdx")
        for match in detection.get("matches", []) or []:
            if not isinstance(match, dict):
                continue
            yield {
                "license_expression": match.get("license_expression") or detection_expression,
                "license_expression_spdx": match.get("license_expression_spdx") or detection_spdx,
                "score": float(match.get("score", 0.0) or 0.0),
                "match_coverage": float(match.get("match_coverage", 0.0) or 0.0),
                "rule_identifier": match.get("rule_identifier"),
                "matcher": match.get("matcher"),
                "start_line": match.get("start_line"),
                "end_line": match.get("end_line"),
            }


def _resource_field(resource: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = resource.get(key)
        if value is not None:
            return value
    return None


def _write_license_scan_manifest(
    *,
    input_directory: Path,
    output_directory: Path,
    checkpoint_path: Path,
    stats: LicenseScanStats,
    batch_size: int,
    min_license_score: float,
    min_match_coverage: float,
    scancode_command: str,
    scancode_headers: list[dict[str, Any]],
) -> None:
    source_manifest_path = input_directory / "manifest.json"
    source_manifest = None
    if source_manifest_path.exists():
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))

    manifest = {
        "source_directory": str(input_directory),
        "created_at": _utc_now(),
        "records_scanned": stats.records_scanned,
        "records_with_detected_license": stats.records_with_detected_license,
        "records_without_detected_license": stats.records_without_detected_license,
        "shards_processed": stats.shards_processed,
        "shards_skipped": stats.shards_skipped,
        "batches_run": stats.batches_run,
        "batch_size": batch_size,
        "min_license_score": min_license_score,
        "min_match_coverage": min_match_coverage,
        "scancode_command": scancode_command,
        "scancode_headers": scancode_headers,
        "checkpoint_path": str(checkpoint_path),
        "input_shards": sorted(path.name for path in input_directory.glob("part-*.jsonl")),
        "output_shards": sorted(path.name for path in output_directory.glob("part-*.jsonl")),
        "source_manifest": source_manifest,
    }
    (output_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
