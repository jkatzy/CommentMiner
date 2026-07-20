from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from ml4setk import OpeningCommentQuery, get_supported_comment_languages
from ml4setk.Parsing.Comments.CommentQuery import (
    CommentQuery,
    _comment_start_ignored_ranges,
)

from .models import ExtractedComment, InputRecord


_LANGUAGE_ALIASES = {
    "bash": "shell",
    "c#": "csharp",
    "c-sharp": "c#",
    "c++": "cpp",
    "cc": "c++",
    "cs": "c#",
    "cxx": "c++",
    "f-sharp": "f#",
    "cplusplus": "c++",
    "common-lisp": "lisp",
    "cpp": "c++",
    "f#": "fsharp",
    "fs": "f#",
    "fsi": "f#",
    "fsx": "f#",
    "hh": "c++",
    "hpp": "c++",
    "hxx": "c++",
    "jl": "julia",
    "js": "javascript",
    "jsx": "javascript",
    "kt": "kotlin",
    "kts": "kotlin",
    "mjs": "javascript",
    "objective-c": "objectivec",
    "objectivec": "objective-c",
    "pl": "perl",
    "pm": "perl",
    "ps1": "powershell",
    "py": "python",
    "pyw": "python",
    "rb": "ruby",
    "rst": "restructuredtext",
    "rs": "rust",
    "sh": "shell",
    "ts": "typescript",
    "visual-basic-net": "vbnet",
    "visual-basic-.net": "vbnet",
    "zsh": "shell",
}

_MAX_LINE_COMMENT_BLANK_LINES = 5


def _normalize_language_token(value: str) -> list[str]:
    token = value.strip().lower()
    if not token:
        return []

    candidates: list[str] = []
    separators_normalized = (
        token,
        token.replace(" ", "-"),
        token.replace(" ", "_"),
        token.replace(" ", ""),
    )
    for item in separators_normalized:
        candidates.extend(
            [
                item,
                item.replace("_", "-"),
                item.replace("-", "_"),
                item.replace("-", ""),
                item.replace("_", ""),
            ]
        )

    alias = _LANGUAGE_ALIASES.get(token)
    if alias is not None:
        candidates.append(alias)

    deduped: list[str] = []
    seen = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            deduped.append(candidate)
            seen.add(candidate)
    return deduped


def _comments_are_contiguous(
    text: str,
    previous_start: int,
    previous_end: int,
    next_start: int,
    next_end: int,
) -> bool:
    """Return whether two comment ranges belong to the same logical block.

    Preserve the existing adjacent-comment behavior, but allow up to five
    blank lines between line comments that use the same delimiter family.
    """

    between = text[previous_end:next_start]
    if between.strip():
        return False

    line_breaks = between.count("\n")
    if line_breaks <= 1:
        return True
    if line_breaks > _MAX_LINE_COMMENT_BLANK_LINES + 1:
        return False

    previous_key = CommentQuery._line_comment_group_key(
        text[previous_start:previous_end]
    )
    next_key = CommentQuery._line_comment_group_key(text[next_start:next_end])
    return previous_key is not None and previous_key == next_key


def _opening_start_cutoff(text: str, max_start_row: int) -> int:
    if max_start_row < 1:
        return 0

    index = -1
    for _ in range(max_start_row):
        index = text.find("\n", index + 1)
        if index == -1:
            return len(text)
    return index + 1


def _starts_inside_ranges(start: int, ranges: list[tuple[int, int]]) -> bool:
    for range_start, range_end in ranges:
        if start < range_start:
            return False
        if range_start <= start < range_end:
            return True
    return False


def _regex_comment_ranges_starting_before(
    query: Any,
    text: str,
    cutoff: int,
    ignored_ranges: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    line_comments = query.line_comments
    for pattern in line_comments.regexes:
        seen_starts: set[int] = set()
        for match in pattern.finditer(text, overlapped=True):
            start = match.start()
            if start >= cutoff:
                break
            if start in seen_starts:
                continue
            seen_starts.add(start)
            if not _starts_inside_ranges(start, ignored_ranges):
                ranges.append((start, match.end()))
    return line_comments._dedupe_match_ranges(ranges)


def _parse_nested_ranges_starting_before(
    open_delim: str,
    close_delim: str,
    text: str,
    cutoff: int,
) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    stack_depth = 0
    block_start: int | None = None
    open_len = len(open_delim)
    close_len = len(close_delim)
    search_from = 0

    while True:
        open_index = text.find(open_delim, search_from)
        close_index = text.find(close_delim, search_from)
        if open_index == -1 and close_index == -1:
            break
        if stack_depth == 0 and (open_index == -1 or open_index >= cutoff):
            break

        if open_index != -1 and (close_index == -1 or open_index <= close_index):
            if stack_depth == 0:
                block_start = open_index
            stack_depth += 1
            search_from = open_index + open_len
            continue

        if stack_depth:
            stack_depth -= 1
            if stack_depth == 0 and block_start is not None:
                result.append((block_start, close_index + close_len))
                block_start = None
        search_from = close_index + close_len

    return result


def _nested_comment_ranges_starting_before(
    query: Any,
    text: str,
    cutoff: int,
    ignored_ranges: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for open_delim, close_delim in query.nested_comments.delimiters:
        ranges.extend(
            item
            for item in _parse_nested_ranges_starting_before(
                open_delim,
                close_delim,
                text,
                cutoff,
            )
            if not _starts_inside_ranges(item[0], ignored_ranges)
        )
    return sorted(ranges)


def _opening_comment_ranges_starting_before(
    query: Any,
    text: str,
    start_anchor: int,
    cutoff: int,
) -> list[tuple[int, int]]:
    ignored_ranges = _comment_start_ignored_ranges(query.language, text[:cutoff])
    ranges = []
    ranges.extend(
        _regex_comment_ranges_starting_before(
            query,
            text,
            cutoff,
            ignored_ranges,
        )
    )
    ranges.extend(
        _nested_comment_ranges_starting_before(
            query,
            text,
            cutoff,
            ignored_ranges,
        )
    )
    return sorted(
        (start, end)
        for start, end in query.line_comments._dedupe_match_ranges(ranges)
        if end > start_anchor
    )


@dataclass(slots=True)
class ML4SEOpeningCommentExtractor:
    max_start_row: int = 10
    _supported_languages: set[str] = field(
        default_factory=lambda: set(get_supported_comment_languages())
    )
    _query_languages: dict[str, str | None] = field(default_factory=dict)
    _queries: dict[str, Any] = field(default_factory=dict)
    _lock: Any = field(default_factory=Lock, repr=False, compare=False)

    def extract_opening_comment(self, record: InputRecord) -> str | None:
        comments = self.extract_opening_comments(record)
        return comments[0].text if comments else None

    def extract_opening_comments(self, record: InputRecord) -> list[ExtractedComment]:
        if not record.content:
            return []

        language = self._resolve_language(record)
        if language is None:
            return []

        query = self._query_for_language(language)
        start_anchor = query._hashbang_end(record.content) if query.skip_hashbang else 0
        cutoff = _opening_start_cutoff(record.content, self.max_start_row)
        comment_ranges: list[tuple[int, int]] = []
        previous_range: tuple[int, int] | None = None
        for start, end in _opening_comment_ranges_starting_before(
            query,
            record.content,
            start_anchor,
            cutoff,
        ):
            if previous_range is not None and _comments_are_contiguous(
                record.content,
                previous_range[0],
                previous_range[1],
                start,
                end,
            ):
                previous_start, _ = comment_ranges[-1]
                comment_ranges[-1] = (previous_start, end)
            else:
                comment_ranges.append((start, end))
            previous_range = (start, end)

        comments: list[ExtractedComment] = []
        for start, end in comment_ranges:
            start_line = query._row_number(record.content, start)
            text = record.content[start:end].strip()
            if text:
                comments.append(
                    ExtractedComment(
                        text=text,
                        start_line=start_line,
                        index=len(comments),
                    )
                )
        return comments

    def supports_language_value(self, value: str) -> bool:
        with self._lock:
            return any(
                candidate in self._supported_languages
                for candidate in _normalize_language_token(value)
            )

    def _resolve_language(self, record: InputRecord) -> str | None:
        with self._lock:
            for raw_language in self._language_candidates(record):
                if raw_language in self._query_languages:
                    cached = self._query_languages[raw_language]
                    if cached is not None:
                        return cached
                    continue

                resolved = next(
                    (
                        candidate
                        for candidate in _normalize_language_token(raw_language)
                        if candidate in self._supported_languages
                    ),
                    None,
                )
                self._query_languages[raw_language] = resolved
                if resolved is not None:
                    return resolved

        return None

    def _query_for_language(self, language: str) -> Any:
        with self._lock:
            query = self._queries.get(language)
            if query is None:
                query = OpeningCommentQuery(language, max_start_row=self.max_start_row)
                self._queries[language] = query
            return query

    @staticmethod
    def _language_candidates(record: InputRecord) -> list[str]:
        metadata = record.metadata
        candidates = [
            metadata.get("selected_language"),
            record.language,
            metadata.get("path_language"),
            metadata.get("ext"),
            metadata.get("lang"),
        ]
        result: list[str] = []
        seen = set()
        for candidate in candidates:
            if candidate is None:
                continue
            value = str(candidate)
            if value not in seen:
                result.append(value)
                seen.add(value)
        return result
