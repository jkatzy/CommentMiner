from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import pyarrow.parquet as pq

from .config import DatasetSpec, PipelineConfig
from .downloader import HuggingFaceDownloader, RemoteFile
from .models import InputRecord


_ROW_ID_SEPARATOR = "::row::"
_PROCESSED_CHECKPOINT_NAMESPACE = "processed-shards"


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
        token: str | bool | None = None,
        downloader: HuggingFaceDownloader | None = None,
    ) -> None:
        self.name = dataset.name
        self.config = config
        self.dataset = dataset
        self.language = language
        self.token = token
        self.downloader = downloader or HuggingFaceDownloader()

    def iter_records(self, start_after: str | None = None) -> Iterator[InputRecord]:
        self.config.ensure_directories()
        resume_cursor = ShardRowCursor.parse(start_after) if start_after else None
        plan = self.downloader.plan_download(
            self.config,
            self.dataset,
            language=self.language,
            token=self.token,
            checkpoint_namespace=_PROCESSED_CHECKPOINT_NAMESPACE,
        )

        for remote in plan.pending_files:
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
    ) -> Iterator[InputRecord]:
        start_row = 0
        if resume_cursor and resume_cursor.remote_path == remote.path:
            start_row = resume_cursor.row_index + 1

        parquet_file = pq.ParquetFile(local_path)
        row_index = 0
        for batch in parquet_file.iter_batches(batch_size=self.dataset.batch_size):
            for row in batch.to_pylist():
                if row_index < start_row:
                    row_index += 1
                    continue
                yield self._row_to_input_record(remote.path, row_index, row)
                row_index += 1

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


def _first_non_null(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return value
    return None
