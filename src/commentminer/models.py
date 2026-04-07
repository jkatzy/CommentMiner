from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class InputRecord:
    dataset: str
    record_id: str
    content: str
    language: str | None = None
    path: str | None = None
    repo: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CommentRecord:
    dataset: str
    record_id: str
    opening_comment: str
    language: str | None = None
    path: str | None = None
    repo: str | None = None
    extracted_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "record_id": self.record_id,
            "opening_comment": self.opening_comment,
            "language": self.language,
            "path": self.path,
            "repo": self.repo,
            "extracted_at": self.extracted_at,
            "metadata": self.metadata,
        }


class DatasetSource(Protocol):
    name: str

    def iter_records(self, start_after: str | None = None) -> Iterable[InputRecord]:
        """Yield records, optionally resuming after a record identifier."""


@runtime_checkable
class ShardedDatasetSource(DatasetSource, Protocol):
    dataset: Any
    downloader: Any
    language: str | None
    token: str | bool | None
    show_progress: bool

    def pending_shards(self) -> list[Any]:
        """Return shard descriptors that still need to be processed."""

    def iter_pending_shards(self) -> Iterator[Any]:
        """Stream shard descriptors that still need to be processed."""

    def iter_shard_records(
        self,
        remote: Any,
        *,
        show_progress: bool | None = None,
    ) -> Iterable[InputRecord]:
        """Yield records from one shard."""

    def mark_shard_completed(self, remote: Any) -> Path:
        """Persist shard completion state and return the checkpoint path."""

    def note_shard_failure(self, remote: Any, exc: BaseException) -> None:
        """Record a shard failure without aborting the entire run."""


class CommentExtractor(Protocol):
    def extract_opening_comment(self, record: InputRecord) -> str | None:
        """Return the opening comment for a record, or None if not found."""
