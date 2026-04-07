from __future__ import annotations

from dataclasses import dataclass, replace
import json
import logging
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

import pyarrow.parquet as pq
from tqdm.auto import tqdm

from .config import DatasetSpec, PipelineConfig
from .downloader import HuggingFaceDownloader, RedPajamaManifestDownloader, RemoteFile
from .extractors import _normalize_language_token
from .models import InputRecord


_ROW_ID_SEPARATOR = "::row::"
_PROCESSED_CHECKPOINT_NAMESPACE = "processed-shards"
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ShardRowCursor:
    remote_path: str
    row_index: int

    def to_record_id(self) -> str:
        return f"{self.remote_path}{_ROW_ID_SEPARATOR}{self.row_index}"

    @classmethod
    def parse(cls, value: str) -> "ShardRowCursor":
        remote_path, separator, row_index = value.rpartition(_ROW_ID_SEPARATOR)
        if not separator:
            raise ValueError(f"Invalid shard row record id: {value}")
        return cls(remote_path=remote_path, row_index=int(row_index))


class TheStackParquetSource:
    def __init__(
        self,
        config: PipelineConfig,
        dataset: DatasetSpec,
        *,
        language: str | None = None,
        show_progress: bool = True,
        token: str | bool | None = None,
        downloader: HuggingFaceDownloader | None = None,
    ) -> None:
        self.name = dataset.name
        self.config = config
        self.dataset = dataset
        self.language = language
        self.show_progress = show_progress
        self.token = token
        self.downloader = downloader or HuggingFaceDownloader()
        self.failed_shards: list[str] = []

    def pending_shards(self) -> list[RemoteFile]:
        return list(self.iter_pending_shards())

    def iter_pending_shards(self) -> Iterator[RemoteFile]:
        self.config.ensure_directories()
        self.failed_shards = []
        yield from self.downloader.iter_pending_files(
            self.config,
            self.dataset,
            language=self.language,
            token=self.token,
            checkpoint_namespace=_PROCESSED_CHECKPOINT_NAMESPACE,
        )

    def iter_shard_records(
        self,
        remote: RemoteFile,
        *,
        show_progress: bool | None = None,
    ) -> Iterator[InputRecord]:
        local_path = self.downloader.download_remote_file(
            self.config,
            self.dataset,
            remote.path,
            language=self.language,
            token=self.token,
        )
        try:
            yield from self._iter_file_records(
                remote,
                local_path,
                None,
                show_progress=show_progress,
            )
        finally:
            if self.dataset.streaming:
                self.downloader.remove_local_file(
                    self.config,
                    self.dataset,
                    remote.path,
                    language=self.language,
                )

    def mark_shard_completed(self, remote: RemoteFile) -> Path:
        return self.downloader.mark_file_completed(
            self.config,
            self.dataset,
            remote.path,
            language=self.language,
            checkpoint_namespace=_PROCESSED_CHECKPOINT_NAMESPACE,
        )

    def note_shard_failure(self, remote: RemoteFile, exc: BaseException) -> None:
        self.failed_shards.append(remote.path)
        _LOGGER.error(
            "Shard processing failed dataset=%s language=%s remote_path=%s error=%s",
            self.dataset.name,
            self.language or "all",
            remote.path,
            exc,
        )

    def iter_records(self, start_after: str | None = None) -> Iterator[InputRecord]:
        self.config.ensure_directories()
        resume_cursor = ShardRowCursor.parse(start_after) if start_after else None
        _LOGGER.info(
            "Preparing source iteration dataset=%s language=%s resume_from=%s",
            self.dataset.name,
            self.language or "all",
            start_after,
        )
        for remote in self.iter_pending_shards():
            _LOGGER.info(
                "Starting shard processing dataset=%s language=%s remote_path=%s",
                self.dataset.name,
                self.language or "all",
                remote.path,
            )
            local_path = self.downloader.download_remote_file(
                self.config,
                self.dataset,
                remote.path,
                language=self.language,
                token=self.token,
            )
            fully_processed = False
            try:
                yield from self._iter_file_records(remote, local_path, resume_cursor)
                fully_processed = True
            finally:
                if fully_processed:
                    self.downloader.mark_file_completed(
                        self.config,
                        self.dataset,
                        remote.path,
                        language=self.language,
                        checkpoint_namespace=_PROCESSED_CHECKPOINT_NAMESPACE,
                    )
                    resume_cursor = None
                    _LOGGER.info(
                        "Finished shard processing dataset=%s language=%s remote_path=%s",
                        self.dataset.name,
                        self.language or "all",
                        remote.path,
                    )
                if self.dataset.streaming:
                    self.downloader.remove_local_file(
                        self.config,
                        self.dataset,
                        remote.path,
                        language=self.language,
                    )

    def _iter_file_records(
        self,
        remote: RemoteFile,
        local_path: Any,
        resume_cursor: ShardRowCursor | None,
        *,
        show_progress: bool | None = None,
    ) -> Iterator[InputRecord]:
        start_row = 0
        if resume_cursor and resume_cursor.remote_path == remote.path:
            start_row = resume_cursor.row_index + 1

        parquet_file = pq.ParquetFile(local_path)
        _LOGGER.info(
            "Reading parquet shard dataset=%s language=%s remote_path=%s total_rows=%s start_row=%s",
            self.dataset.name,
            self.language or "all",
            remote.path,
            parquet_file.metadata.num_rows if parquet_file.metadata is not None else "unknown",
            start_row,
        )
        progress = tqdm(
            total=parquet_file.metadata.num_rows if parquet_file.metadata is not None else None,
            initial=start_row,
            desc=_progress_description(self.dataset.name, remote.path),
            unit="rows",
            dynamic_ncols=True,
            leave=False,
            disable=not (self.show_progress if show_progress is None else show_progress),
        )
        row_index = 0
        try:
            for batch in parquet_file.iter_batches(batch_size=self.dataset.batch_size):
                for row in _iter_batch_rows(batch):
                    if row_index < start_row:
                        row_index += 1
                        continue
                    progress.update(1)
                    yield self._row_to_input_record(remote.path, row_index, row)
                    row_index += 1
        finally:
            progress.close()

    def _row_to_input_record(
        self,
        remote_path: str,
        row_index: int,
        row: dict[str, Any],
    ) -> InputRecord:
        repo = _first_non_null(
            row,
            "max_stars_repo_name",
            "max_issues_repo_name",
            "max_forks_repo_name",
        )
        path = _first_non_null(
            row,
            "max_stars_repo_path",
            "max_issues_repo_path",
            "max_forks_repo_path",
        )
        language = row.get("lang")
        metadata = {key: value for key, value in row.items() if key != "content"}
        metadata["remote_path"] = remote_path
        metadata["row_index"] = row_index
        if self.language is not None:
            metadata["selected_language"] = self.language

        return InputRecord(
            dataset=self.dataset.name,
            record_id=ShardRowCursor(remote_path, row_index).to_record_id(),
            content=str(row.get("content", "")),
            language=str(language) if language is not None else self.language,
            path=str(path) if path is not None else None,
            repo=str(repo) if repo is not None else None,
            metadata=metadata,
        )


class RedPajamaGithubSource:
    def __init__(
        self,
        config: PipelineConfig,
        dataset: DatasetSpec,
        *,
        language: str | None = None,
        show_progress: bool = True,
        token: str | bool | None = None,
        downloader: RedPajamaManifestDownloader | None = None,
    ) -> None:
        self.name = dataset.name
        self.config = config
        self.dataset = dataset
        self.language = language
        self.show_progress = show_progress
        self.token = token
        self.downloader = downloader or RedPajamaManifestDownloader()
        self.failed_shards: list[str] = []

    def pending_shards(self) -> list[RemoteFile]:
        return list(self.iter_pending_shards())

    def iter_pending_shards(self) -> Iterator[RemoteFile]:
        self.config.ensure_directories()
        self.failed_shards = []
        yield from self.downloader.iter_pending_files(
            self.config,
            self.dataset,
            language=self.language,
            token=self.token,
            checkpoint_namespace=_PROCESSED_CHECKPOINT_NAMESPACE,
        )

    def iter_shard_records(
        self,
        remote: RemoteFile,
        *,
        show_progress: bool | None = None,
    ) -> Iterator[InputRecord]:
        local_path = self.downloader.download_remote_file(
            self.config,
            self.dataset,
            remote.path,
            language=self.language,
            token=self.token,
        )
        try:
            yield from self._iter_file_records(
                remote,
                local_path,
                None,
                show_progress=show_progress,
            )
        finally:
            if self.dataset.streaming:
                self.downloader.remove_local_file(
                    self.config,
                    self.dataset,
                    remote.path,
                    language=self.language,
                )

    def mark_shard_completed(self, remote: RemoteFile) -> Path:
        return self.downloader.mark_file_completed(
            self.config,
            self.dataset,
            remote.path,
            language=self.language,
            checkpoint_namespace=_PROCESSED_CHECKPOINT_NAMESPACE,
        )

    def note_shard_failure(self, remote: RemoteFile, exc: BaseException) -> None:
        self.failed_shards.append(remote.path)
        _LOGGER.error(
            "Shard processing failed dataset=%s language=%s remote_path=%s error=%s",
            self.dataset.name,
            self.language or "all",
            remote.path,
            exc,
        )

    def iter_records(self, start_after: str | None = None) -> Iterator[InputRecord]:
        self.config.ensure_directories()
        resume_cursor = ShardRowCursor.parse(start_after) if start_after else None
        _LOGGER.info(
            "Preparing source iteration dataset=%s language=%s resume_from=%s",
            self.dataset.name,
            self.language or "all",
            start_after,
        )
        for remote in self.iter_pending_shards():
            local_path = self.downloader.download_remote_file(
                self.config,
                self.dataset,
                remote.path,
                language=self.language,
                token=self.token,
            )
            fully_processed = False
            try:
                yield from self._iter_file_records(remote, local_path, resume_cursor)
                fully_processed = True
            finally:
                if fully_processed:
                    self.downloader.mark_file_completed(
                        self.config,
                        self.dataset,
                        remote.path,
                        language=self.language,
                        checkpoint_namespace=_PROCESSED_CHECKPOINT_NAMESPACE,
                    )
                    resume_cursor = None
                if self.dataset.streaming:
                    self.downloader.remove_local_file(
                        self.config,
                        self.dataset,
                        remote.path,
                        language=self.language,
                    )

    def _iter_file_records(
        self,
        remote: RemoteFile,
        local_path: Any,
        resume_cursor: ShardRowCursor | None,
        *,
        show_progress: bool | None = None,
    ) -> Iterator[InputRecord]:
        start_row = 0
        if resume_cursor and resume_cursor.remote_path == remote.path:
            start_row = resume_cursor.row_index + 1

        local_path = Path(local_path)
        total_bytes = local_path.stat().st_size if local_path.exists() else None
        progress = tqdm(
            total=total_bytes,
            desc=_progress_description(self.dataset.name, remote.path),
            unit="B",
            unit_scale=True,
            dynamic_ncols=True,
            leave=False,
            disable=not (self.show_progress if show_progress is None else show_progress),
        )
        try:
            with local_path.open("rb") as handle:
                for row_index, raw_line in enumerate(handle):
                    progress.update(len(raw_line))
                    if row_index < start_row:
                        continue
                    if not raw_line.strip():
                        continue
                    row = json.loads(raw_line.decode("utf-8"))
                    record = self._row_to_input_record(remote.path, row_index, row)
                    if self.language is not None and not _record_matches_language(record, self.language):
                        continue
                    if self.language is not None:
                        metadata = dict(record.metadata)
                        metadata["selected_language"] = self.language
                        record = replace(record, metadata=metadata)
                    yield record
        finally:
            progress.close()

    def _row_to_input_record(
        self,
        remote_path: str,
        row_index: int,
        row: dict[str, Any],
    ) -> InputRecord:
        metadata = _redpajama_metadata(row)
        metadata["remote_path"] = remote_path
        metadata["row_index"] = row_index

        path = _first_non_null(metadata, "path", "repo_path", "file_path", "filepath", "filename")
        repo = _first_non_null(metadata, "repo_name", "repo", "repository", "repo_id")
        if repo is None:
            repo = _github_repo_from_url(metadata.get("url"))

        ext = _extension_token_from_path(str(path)) if path is not None else None
        if ext and "ext" not in metadata:
            metadata["ext"] = ext

        language = _first_non_null(metadata, "lang", "language", "programming_language")
        if language is None and ext is not None:
            normalized = _normalize_language_token(ext)
            language = normalized[0] if normalized else ext

        return InputRecord(
            dataset=self.dataset.name,
            record_id=ShardRowCursor(remote_path, row_index).to_record_id(),
            content=str(row.get("text", "")),
            language=str(language) if language is not None else None,
            path=str(path) if path is not None else None,
            repo=str(repo) if repo is not None else None,
            metadata=metadata,
        )


def _first_non_null(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return value
    return None


def _redpajama_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any]
    raw_meta = row.get("meta")
    if isinstance(raw_meta, str):
        try:
            parsed = json.loads(raw_meta)
        except json.JSONDecodeError:
            parsed = {"meta": raw_meta}
        metadata = parsed if isinstance(parsed, dict) else {"meta": parsed}
    elif isinstance(raw_meta, dict):
        metadata = dict(raw_meta)
    else:
        metadata = {key: value for key, value in row.items() if key != "text"}

    subset = row.get("red_pajama_subset")
    if subset is not None:
        metadata.setdefault("red_pajama_subset", subset)
    return metadata


def _github_repo_from_url(url: Any) -> str | None:
    if url is None:
        return None
    parsed = urlparse(str(url))
    if parsed.netloc not in {"github.com", "www.github.com", "raw.githubusercontent.com"}:
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        return None
    return "/".join(parts[:2])


def _extension_token_from_path(path: str) -> str | None:
    path_obj = Path(path)
    suffix = path_obj.suffix.lower().lstrip(".")
    if suffix:
        return suffix

    basename = path_obj.name.lower()
    special_names = {
        "dockerfile": "dockerfile",
        "makefile": "makefile",
        "cmakelists.txt": "cmake",
    }
    return special_names.get(basename)


def _record_matches_language(record: InputRecord, language: str) -> bool:
    target_candidates = set(_normalize_language_token(language))
    if not target_candidates:
        return False

    raw_candidates: list[str] = []
    for candidate in (
        record.language,
        record.metadata.get("lang"),
        record.metadata.get("language"),
        record.metadata.get("programming_language"),
        record.metadata.get("ext"),
    ):
        if candidate is None:
            continue
        raw_candidates.append(str(candidate))

    if record.path is not None:
        ext = _extension_token_from_path(record.path)
        if ext is not None:
            raw_candidates.append(ext)

    resolved_candidates: set[str] = set()
    for candidate in raw_candidates:
        resolved_candidates.update(_normalize_language_token(candidate))
    return bool(target_candidates.intersection(resolved_candidates))


def _iter_batch_rows(batch) -> Iterator[dict[str, Any]]:
    column_names = batch.schema.names
    columns = [batch.column(index) for index in range(batch.num_columns)]
    for row_index in range(batch.num_rows):
        row: dict[str, Any] = {}
        for name, column in zip(column_names, columns, strict=True):
            row[name] = column[row_index].as_py()
        yield row


def _progress_description(dataset_name: str, remote_path: str) -> str:
    parts = remote_path.split("/")
    shard_label = "/".join(parts[-2:]) if len(parts) >= 2 else remote_path
    return f"{dataset_name}:{shard_label}"
