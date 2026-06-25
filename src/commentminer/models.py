from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Iterable, Protocol


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
        return _json_safe(
            {
                "dataset": self.dataset,
                "record_id": self.record_id,
                "opening_comment": self.opening_comment,
                "language": self.language,
                "path": self.path,
                "repo": self.repo,
                "extracted_at": self.extracted_at,
                "metadata": self.metadata,
            }
        )


@dataclass(frozen=True, slots=True)
class ExtractedComment:
    text: str
    start_line: int | None = None
    index: int | None = None


class DatasetSource(Protocol):
    name: str

    def iter_records(self, start_after: str | None = None) -> Iterable[InputRecord]:
        """Yield records, optionally resuming after a record identifier."""


class CommentExtractor(Protocol):
    def extract_opening_comment(self, record: InputRecord) -> str | None:
        """Return the opening comment for a record, or None if not found."""


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except (TypeError, ValueError):
            pass
    return str(value)
