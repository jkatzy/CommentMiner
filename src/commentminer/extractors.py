from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ml4setk import OpeningCommentQuery, get_supported_comment_languages

from .models import InputRecord


_LANGUAGE_ALIASES = {
    "c-sharp": "csharp",
    "common-lisp": "lisp",
    "f-sharp": "fsharp",
    "objective-c": "objectivec",
    "visual-basic-net": "vbnet",
    "visual-basic-.net": "vbnet",
}


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
@dataclass(slots=True)
class ML4SEOpeningCommentExtractor:
    max_start_row: int = 3
    max_input_characters: int | None = None
    _supported_languages: set[str] = field(
        default_factory=lambda: set(get_supported_comment_languages())
    )
    _query_languages: dict[str, str | None] = field(default_factory=dict)
    _queries: dict[str, Any] = field(default_factory=dict)

    def extract_opening_comment(self, record: InputRecord) -> str | None:
        if not record.content:
            return None

        language = self._resolve_language(record)
        if language is None:
            return None

        query = self._query_for_language(language)
        matches = query.parse(_truncate_text(record.content, self.max_input_characters))
        if not matches:
            return None
        return matches[0].match

    def _resolve_language(self, record: InputRecord) -> str | None:
        for raw_language in self._language_candidates(record):
            cached = self._query_languages.get(raw_language)
            if cached is not None or raw_language in self._query_languages:
                return cached

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
            metadata.get("ext"),
            record.language,
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


def _truncate_text(text: str, max_characters: int | None) -> str:
    if max_characters is None:
        return text
    if max_characters < 1:
        raise ValueError(f"max_input_characters must be >= 1, got {max_characters}")
    if len(text) <= max_characters:
        return text
    return text[:max_characters]
