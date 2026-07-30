from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from difflib import SequenceMatcher
import fcntl
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Any, Iterable, Iterator, Mapping, Sequence
import unicodedata
from uuid import uuid4

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .redistribution_judge import (
    JUDGE_LABELS,
    JUDGMENT_PROFILE_NON_LICENSE_LIMITATIONS,
    JUDGMENT_PROFILE_REDISTRIBUTION_INTENT,
    JUDGMENT_PROFILES,
    LABEL_AMBIGUOUS,
    LABEL_CODE_REDISTRIBUTION_INTENT,
    LABEL_LICENSE_ONLY,
    LABEL_NON_LICENSE_LIMITATION,
    LABEL_NON_LICENSE_LIMITATION_WITH_LICENSE,
    LIMITATION_JUDGE_LABELS,
    LABEL_OTHER,
    SUPPORTED_REASONING_EFFORTS,
    _evidence_is_source_text,
    judge_redistribution_candidates,
)
from .topic_modelling import (
    EXPORT_CONTROL_SEEDS,
    FUNDING_DISSEMINATION_SEEDS,
    GOVERNMENT_RESTRICTION_SEEDS,
    PROPRIETARY_PROVENANCE_SEEDS,
    SEED_TOPICS,
    UNPUBLISHED_WORK_SEEDS,
)


FORMAT_VERSION = 2
SUPPORTED_FORMAT_VERSIONS = (1, FORMAT_VERSION)
MANIFEST_KIND = "commentminer-redistribution-candidate-dataset"
DEFAULT_INPUT_SOURCE = "Jkatzy/code-comments"
DEFAULT_DATASET = "the-stack-v2-dedup"
DEFAULT_LANGUAGE = "Java"
DEFAULT_SOURCE_FILES_LIMIT = 100_000
DEFAULT_FUZZY_THRESHOLD = 0.82
DEFAULT_CODEX_MODEL = "gpt-5.6-luna"
DEFAULT_REASONING_EFFORT = "max"
DEFAULT_REVISION = "0d4c83fac76705d2e2388186b628543a4916dab8"
SELECTION_SOURCE_FILE_PREFIX = "source_file_prefix"
SELECTION_STRATIFIED_COMMENTS = "stratified_comment_rows"

_RECORD_ID_RE = re.compile(
    r"^(?P<remote_path>.+)::row::(?P<row_index>\d+)"
    r"(?:::(?:comment)::(?P<comment_index>\d+))?$"
)
_TOKEN_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)
_ANCHOR_STOPWORDS = frozenset(
    {
        "a",
        "all",
        "and",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "no",
        "not",
        "of",
        "on",
        "only",
        "or",
        "the",
        "this",
        "to",
        "under",
        "use",
        "with",
        "without",
    }
)


_OCCURRENCE_SCHEMA = pa.schema(
    [
        ("occurrence_id", pa.string()),
        ("candidate_id", pa.string()),
        ("comment_hash", pa.string()),
        ("source_file_id", pa.string()),
        ("source_remote_path", pa.string()),
        ("source_file_row_index", pa.int64()),
        ("source_comment_index", pa.int64()),
        ("source_parquet_path", pa.string()),
        ("source_parquet_row_index", pa.int64()),
        ("dataset", pa.string()),
        ("record_id", pa.string()),
        ("language", pa.string()),
        ("path", pa.string()),
        ("repo", pa.string()),
        ("opening_comment", pa.string()),
        ("comment_license_score", pa.float64()),
        ("comment_license_detection", pa.string()),
        ("matched_seed_groups", pa.list_(pa.string())),
        ("matched_seed_phrases", pa.list_(pa.string())),
        ("match_scores", pa.list_(pa.float64())),
        ("match_excerpts", pa.list_(pa.string())),
        ("best_match_score", pa.float64()),
        ("best_match_excerpt", pa.string()),
    ]
)

_CANDIDATE_SCHEMA = pa.schema(
    [
        *_OCCURRENCE_SCHEMA,
        ("occurrence_count", pa.int64()),
        ("occurrence_ids", pa.list_(pa.string())),
    ]
)

_JUDGED_SCHEMA = pa.schema(
    [
        *_CANDIDATE_SCHEMA,
        ("judge_label", pa.string()),
        ("is_code_redistribution_intent", pa.bool_()),
        ("judge_confidence", pa.float64()),
        ("judge_evidence", pa.string()),
        ("judge_rationale", pa.string()),
    ]
)

_LABELED_OCCURRENCE_SCHEMA = pa.schema(
    [
        *_OCCURRENCE_SCHEMA,
        ("judge_label", pa.string()),
        ("is_code_redistribution_intent", pa.bool_()),
        ("judge_confidence", pa.float64()),
        ("judge_evidence", pa.string()),
        ("judge_rationale", pa.string()),
    ]
)

_LIMITATION_JUDGMENT_FIELDS = [
    ("judge_label", pa.string()),
    ("is_non_license_redistribution_limitation", pa.bool_()),
    ("is_license_notice", pa.bool_()),
    ("is_known_license", pa.bool_()),
    ("known_license", pa.string()),
    ("restriction_evidence", pa.string()),
    ("license_evidence", pa.string()),
    ("scancode_contains_license_notice", pa.bool_()),
    ("scancode_detected_license_expression", pa.string()),
    ("is_scancode_missed_license", pa.bool_()),
    ("judge_confidence", pa.float64()),
    ("judge_evidence", pa.string()),
    ("judge_rationale", pa.string()),
]

_LIMITATION_JUDGED_SCHEMA = pa.schema(
    [*_CANDIDATE_SCHEMA, *_LIMITATION_JUDGMENT_FIELDS]
)

_LIMITATION_LABELED_OCCURRENCE_SCHEMA = pa.schema(
    [*_OCCURRENCE_SCHEMA, *_LIMITATION_JUDGMENT_FIELDS]
)


@dataclass(frozen=True, slots=True)
class SeedPhrase:
    group: str
    phrase: str
    tokens: tuple[str, ...]
    ordinal: int


@dataclass(frozen=True, slots=True)
class SeedMatch:
    group: str
    phrase: str
    score: float
    excerpt: str
    start: int
    end: int
    ordinal: int


@dataclass(slots=True)
class RedistributionBuildStats:
    output_directory: Path
    input_directory: Path | None = None
    input_format: str = "local"
    source_files_in_scope: int = 0
    comment_rows_in_scope: int = 0
    languages_in_scope: int = 1
    selection_mode: str = SELECTION_SOURCE_FILE_PREFIX
    comment_bearing_files_seen: int = 0
    comment_rows_seen: int = 0
    shards_scanned: int = 0
    matched_occurrences: int = 0
    candidate_count: int = 0
    judged_count: int = 0
    judge_batches: int = 0
    judge_attempts: int = 0
    judge_cache_hits: int = 0
    scan_only: bool = False
    occurrences_path: Path | None = None
    candidates_path: Path | None = None
    dataset_path: Path | None = None
    labeled_occurrences_path: Path | None = None
    non_license_limitations_path: Path | None = None
    scancode_missed_licenses_path: Path | None = None
    manifest_path: Path | None = None
    verification_path: Path | None = None


@dataclass(slots=True)
class RedistributionVerificationReport:
    valid: bool
    candidate_count: int = 0
    matched_occurrences: int = 0
    judged_count: int = 0
    errors: tuple[str, ...] = ()


@dataclass(slots=True)
class _ScanResult:
    occurrences: list[dict[str, Any]]
    candidates: list[dict[str, Any]]
    source_remote_path: str
    comment_bearing_files_seen: int
    comment_rows_seen: int
    shards_scanned: int
    source_shards: list[dict[str, Any]]
    input_directory: Path | None
    input_format: str
    selection_mode: str = SELECTION_SOURCE_FILE_PREFIX
    comment_rows_examined: int = 0
    score_filtered_rows: int = 0
    eligible_comment_rows_available: int | None = None
    language_allocations: tuple[dict[str, Any], ...] = ()
    source_inventory_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class _ShardEligibility:
    language: str
    path: str
    relative_path: str
    size: int
    total_rows: int
    eligible_rows: int


@dataclass(frozen=True, slots=True)
class _ShardScanTask:
    inventory: _ShardEligibility
    selected_eligible_ranks: tuple[int, ...]


@dataclass(slots=True)
class _ShardScanResult:
    language: str
    path: str
    selected_rows: int
    occurrences: list[dict[str, Any]]
    source_file_runs: int
    first_source_file_id: str | None
    last_source_file_id: str | None
    source_remote_paths: tuple[str, ...]


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_comment(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


def _tokenize_with_spans(text: str) -> list[tuple[str, int, int]]:
    tokens: list[tuple[str, int, int]] = []
    for match in _TOKEN_RE.finditer(text):
        token = unicodedata.normalize("NFKC", match.group(0)).casefold()
        if token:
            tokens.append((token, match.start(), match.end()))
    return tokens


def _seed_inventory(
    *,
    include_government_seeds: bool,
    include_provenance_seeds: bool = False,
    include_funding_seeds: bool = False,
    include_export_control_seeds: bool = False,
    include_unpublished_work_seeds: bool = False,
) -> tuple[SeedPhrase, ...]:
    inventory: list[SeedPhrase] = []
    groups: list[tuple[str, Iterable[str]]] = []
    # Let the dedicated family own its exact overlaps when it is enabled. The
    # same phrases remain in their original groups when this flag is off.
    if include_unpublished_work_seeds:
        groups.append(("unpublished_work", UNPUBLISHED_WORK_SEEDS))
    groups.extend(SEED_TOPICS.items())
    # Put the specialized opt-in groups before the broader government group so
    # a phrase shared by both retains the more informative retrieval group.
    if include_provenance_seeds:
        groups.append(("proprietary_provenance", PROPRIETARY_PROVENANCE_SEEDS))
    if include_funding_seeds:
        groups.append(("funding_dissemination", FUNDING_DISSEMINATION_SEEDS))
    if include_export_control_seeds:
        groups.append(("export_controls", EXPORT_CONTROL_SEEDS))
    if include_government_seeds:
        groups.append(("government_restrictions", GOVERNMENT_RESTRICTION_SEEDS))
    seen_token_phrases: set[tuple[str, ...]] = set()
    for group, phrases in groups:
        for phrase in phrases:
            tokens = tuple(item[0] for item in _tokenize_with_spans(phrase))
            if not tokens or tokens in seen_token_phrases:
                continue
            seen_token_phrases.add(tokens)
            inventory.append(
                SeedPhrase(
                    group=group,
                    phrase=phrase,
                    tokens=tokens,
                    ordinal=len(inventory),
                )
            )
    return tuple(inventory)


@lru_cache(maxsize=65_536)
def _token_substitution_cost(left: str, right: str) -> float:
    if left == right:
        return 0.0
    similarity = SequenceMatcher(None, left, right, autojunk=False).ratio()
    return 1.0 - similarity if similarity >= 0.8 else 1.0


def _weighted_token_edit_distance(
    seed_tokens: Sequence[str], window_tokens: Sequence[str]
) -> float:
    insertion_cost = 0.5
    deletion_cost = 1.0
    previous = [index * insertion_cost for index in range(len(window_tokens) + 1)]
    for seed_index, seed_token in enumerate(seed_tokens, start=1):
        current = [seed_index * deletion_cost]
        for window_index, window_token in enumerate(window_tokens, start=1):
            current.append(
                min(
                    previous[window_index] + deletion_cost,
                    current[window_index - 1] + insertion_cost,
                    previous[window_index - 1]
                    + _token_substitution_cost(seed_token, window_token),
                )
            )
        previous = current
    return previous[-1]


class FuzzySeedMatcher:
    """Formatting-insensitive, bounded token edit matcher for seed phrases."""

    def __init__(
        self,
        seeds: Sequence[SeedPhrase],
        *,
        threshold: float = DEFAULT_FUZZY_THRESHOLD,
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("fuzzy_threshold must be in (0, 1]")
        if not seeds:
            raise ValueError("At least one seed phrase is required")
        self.seeds = tuple(seeds)
        self.threshold = float(threshold)
        index: dict[str, set[int]] = defaultdict(set)
        anchor_positions: list[dict[str, tuple[int, ...]]] = []
        for seed_index, seed in enumerate(self.seeds):
            anchors = {
                token
                for token in seed.tokens
                if token not in _ANCHOR_STOPWORDS and len(token) >= 3
            }
            if not anchors:
                anchors = set(seed.tokens)
            anchor_positions.append(
                {
                    anchor: tuple(
                        position
                        for position, token in enumerate(seed.tokens)
                        if token == anchor
                    )
                    for anchor in anchors
                }
            )
            for anchor in anchors:
                index[anchor].add(seed_index)
        self._anchor_index = {key: frozenset(value) for key, value in index.items()}
        self._anchor_positions = tuple(anchor_positions)

    def match(self, text: str) -> list[SeedMatch]:
        source_tokens = _tokenize_with_spans(text)
        if not source_tokens:
            return []
        normalized_tokens = [token for token, _, _ in source_tokens]
        anchor_hits: dict[int, list[tuple[int, str]]] = defaultdict(list)
        for token_position, token in enumerate(normalized_tokens):
            for seed_index in self._anchor_index.get(token, ()):
                anchor_hits[seed_index].append((token_position, token))

        matches: list[SeedMatch] = []
        for seed_index in sorted(anchor_hits):
            seed = self.seeds[seed_index]
            seed_length = len(seed.tokens)
            best: tuple[float, int, int] | None = None
            min_window = max(1, seed_length - 1)
            max_window = min(len(source_tokens), seed_length + 2)
            candidate_starts: set[int] = set()
            for source_position, anchor in anchor_hits[seed_index]:
                for seed_position in self._anchor_positions[seed_index][anchor]:
                    expected_start = source_position - seed_position
                    for delta in range(-2, 3):
                        candidate_starts.add(expected_start + delta)
            for window_length in range(min_window, max_window + 1):
                last_start = len(source_tokens) - window_length
                for start in sorted(candidate_starts):
                    if start < 0 or start > last_start:
                        continue
                    window = normalized_tokens[start : start + window_length]
                    distance = _weighted_token_edit_distance(seed.tokens, window)
                    score = max(
                        0.0,
                        1.0 - distance / max(seed_length, window_length),
                    )
                    candidate = (score, -window_length, -start)
                    if best is None or candidate > (best[0], -best[2], -best[1]):
                        best = (score, start, window_length)
            if best is None or best[0] + 1e-12 < self.threshold:
                continue
            _, start, window_length = best
            source_start = source_tokens[start][1]
            source_end = source_tokens[start + window_length - 1][2]
            matches.append(
                SeedMatch(
                    group=seed.group,
                    phrase=seed.phrase,
                    score=round(best[0], 6),
                    excerpt=text[source_start:source_end],
                    start=source_start,
                    end=source_end,
                    ordinal=seed.ordinal,
                )
            )
        return sorted(matches, key=lambda item: (-item.score, item.ordinal))


def _source_identity(record_id: str, metadata: str | None) -> tuple[str, int, int | None]:
    match = _RECORD_ID_RE.fullmatch(record_id)
    if match is not None:
        comment_index = match.group("comment_index")
        return (
            match.group("remote_path"),
            int(match.group("row_index")),
            int(comment_index) if comment_index is not None else None,
        )
    if metadata:
        try:
            payload = json.loads(metadata)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, Mapping):
            remote_path = payload.get("remote_path")
            row_index = payload.get("row_index")
            comment_index = payload.get("comment_index")
            if isinstance(remote_path, str) and isinstance(row_index, int):
                return (
                    remote_path,
                    row_index,
                    comment_index if isinstance(comment_index, int) else None,
                )
    raise ValueError(f"Cannot recover Stack v2 source identity from record_id {record_id!r}")


def _local_shards(source: Path, *, dataset: str, language: str) -> list[Path]:
    candidates = (
        source / dataset / language,
        source / language,
        source,
    )
    for directory in candidates:
        if directory.is_dir():
            shards = sorted(directory.glob("part-*.parquet"))
            if shards:
                return shards
    raise ValueError(
        f"No Parquet shards found for {dataset}/{language} under {source.resolve()}"
    )


def _remote_shards(
    source: str,
    *,
    dataset: str,
    language: str,
    revision: str,
    hf_token: str | bool | None,
    hf_cache_directory: Path | None,
) -> Iterator[Path]:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface-hub is required for remote dataset input") from exc
    prefix = f"{dataset}/{language}/"
    try:
        paths = sorted(
            path
            for path in HfApi().list_repo_files(
                source,
                repo_type="dataset",
                revision=revision,
                token=hf_token,
            )
            if path.startswith(prefix)
            and Path(path).name.startswith("part-")
            and path.endswith(".parquet")
        )
    except Exception as exc:
        raise RuntimeError(f"Unable to list Hugging Face dataset {source!r}: {exc}") from exc
    if not paths:
        raise ValueError(f"No remote Parquet shards found for {dataset}/{language}")
    for remote_path in paths:
        try:
            local = hf_hub_download(
                repo_id=source,
                filename=remote_path,
                repo_type="dataset",
                revision=revision,
                token=hf_token,
                cache_dir=str(hf_cache_directory) if hf_cache_directory else None,
            )
        except Exception as exc:
            raise RuntimeError(f"Unable to cache {remote_path}: {exc}") from exc
        yield Path(local).resolve()


def _source_shards(
    input_source: Path | str,
    *,
    dataset: str,
    language: str,
    revision: str,
    hf_token: str | bool | None,
    hf_cache_directory: Path | None,
) -> tuple[Iterable[Path], Path | None, str]:
    local = Path(input_source).expanduser()
    if local.exists():
        if not local.is_dir():
            raise ValueError(f"Input source must be a directory: {local}")
        resolved = local.resolve()
        return _local_shards(resolved, dataset=dataset, language=language), resolved, "local"
    if isinstance(input_source, Path) or "/" not in str(input_source):
        raise ValueError(f"Input source directory does not exist: {local.resolve()}")
    return (
        _remote_shards(
            str(input_source),
            dataset=dataset,
            language=language,
            revision=revision,
            hf_token=hf_token,
            hf_cache_directory=hf_cache_directory,
        ),
        None,
        "huggingface",
    )


def _row_value(columns: Mapping[str, list[Any]], name: str, index: int) -> Any:
    values = columns.get(name)
    return values[index] if values is not None else None


def _normalized_scancode_threshold_percent(value: float) -> float:
    if isinstance(value, bool):
        raise ValueError("ScanCode score threshold must be numeric")
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("ScanCode score threshold must be numeric") from exc
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("ScanCode score threshold must be finite and positive")
    if threshold <= 1.0:
        threshold *= 100.0
    if threshold > 100.0:
        raise ValueError("ScanCode score threshold cannot exceed 100")
    return threshold


def _discover_all_language_shards(
    input_source: Path | str,
    *,
    dataset: str,
) -> tuple[Path, dict[str, tuple[Path, ...]]]:
    source = Path(input_source).expanduser()
    if not source.exists() or not source.is_dir():
        raise ValueError(
            "All-language stratified scans require a local input directory"
        )
    source = source.resolve()
    dataset_root = source / dataset
    if not dataset_root.is_dir():
        raise ValueError(
            f"Dataset directory does not exist for all-language scan: {dataset_root}"
        )
    language_shards: dict[str, tuple[Path, ...]] = {}
    for directory in sorted(
        (path for path in dataset_root.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    ):
        shards = tuple(sorted(directory.glob("part-*.parquet")))
        if shards:
            language_shards[directory.name] = shards
    if not language_shards:
        raise ValueError(f"No language Parquet partitions found under {dataset_root}")
    return source, language_shards


def _count_shard_eligibility(
    task: tuple[str, str, str, float, int],
) -> _ShardEligibility:
    language, path_text, relative_path, threshold_percent, batch_size = task
    path = Path(path_text)
    parquet = pq.ParquetFile(path)
    if "comment_license_score" not in parquet.schema_arrow.names:
        raise ValueError(f"Input shard {path} has no comment_license_score column")
    total_rows = parquet.metadata.num_rows
    eligible_rows = 0
    rows_seen = 0
    for batch in parquet.iter_batches(
        batch_size=batch_size,
        columns=["comment_license_score"],
    ):
        scores = batch.column(0)
        if scores.null_count:
            raise ValueError(f"Input shard {path} has a null ScanCode score")
        finite = pc.all(pc.is_finite(scores)).as_py()
        if finite is not True:
            raise ValueError(f"Input shard {path} has a non-finite ScanCode score")
        bounds = pc.min_max(scores).as_py()
        if bounds["min"] < 0 or bounds["max"] > 100:
            raise ValueError(f"Input shard {path} has a ScanCode score outside 0..100")
        eligible_rows += int(pc.sum(pc.less(scores, threshold_percent)).as_py())
        rows_seen += batch.num_rows
    if rows_seen != total_rows:
        raise RuntimeError(f"Parquet row count changed while reading {path}")
    return _ShardEligibility(
        language=language,
        path=str(path),
        relative_path=relative_path,
        size=path.stat().st_size,
        total_rows=total_rows,
        eligible_rows=eligible_rows,
    )


def _max_min_language_allocation(
    capacities: Mapping[str, int],
    total: int,
) -> dict[str, int]:
    if total < 1:
        raise ValueError("comment_rows_limit must be >= 1")
    normalized: dict[str, int] = {}
    for language, capacity in capacities.items():
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 0:
            raise ValueError(f"Invalid eligible capacity for {language!r}")
        normalized[str(language)] = capacity
    positive = sorted(language for language, count in normalized.items() if count > 0)
    if not positive:
        raise ValueError("No comments satisfy the ScanCode score threshold")
    if total < len(positive):
        raise ValueError(
            "comment_rows_limit is too small to include every eligible language"
        )
    available = sum(normalized.values())
    if total > available:
        raise ValueError(
            f"Requested {total:,} comments but only {available:,} are eligible"
        )

    allocation = {language: 0 for language in sorted(normalized)}
    active = positive
    remaining = total
    while active:
        share = remaining // len(active)
        exhausted = [
            language for language in active if normalized[language] <= share
        ]
        if exhausted:
            for language in exhausted:
                allocation[language] = normalized[language]
                remaining -= normalized[language]
            exhausted_set = set(exhausted)
            active = [
                language for language in active if language not in exhausted_set
            ]
            continue
        for language in active:
            allocation[language] = share
        remainder = remaining - share * len(active)
        for language in active[:remainder]:
            allocation[language] += 1
        remaining = 0
        break
    if remaining != 0 or sum(allocation.values()) != total:
        raise RuntimeError("Unable to construct exact max-min language allocation")
    return allocation


def _systematic_sample_ranks(available: int, selected: int) -> tuple[int, ...]:
    if not 0 <= selected <= available:
        raise ValueError("Systematic sample size must lie within capacity")
    if selected == 0:
        return ()
    ranks = tuple(
        ((2 * index + 1) * available) // (2 * selected)
        for index in range(selected)
    )
    if len(set(ranks)) != selected or ranks[0] < 0 or ranks[-1] >= available:
        raise RuntimeError("Systematic sampling produced invalid eligible ranks")
    return ranks


_STRATIFIED_MATCHER: FuzzySeedMatcher | None = None
_STRATIFIED_DATASET = ""
_STRATIFIED_THRESHOLD_PERCENT = 0.0
_STRATIFIED_BATCH_SIZE = 0


def _init_stratified_scan_worker(
    seeds: tuple[SeedPhrase, ...],
    fuzzy_threshold: float,
    dataset: str,
    score_threshold_percent: float,
    batch_size: int,
) -> None:
    global _STRATIFIED_MATCHER
    global _STRATIFIED_DATASET
    global _STRATIFIED_THRESHOLD_PERCENT
    global _STRATIFIED_BATCH_SIZE
    _STRATIFIED_MATCHER = FuzzySeedMatcher(seeds, threshold=fuzzy_threshold)
    _STRATIFIED_DATASET = dataset
    _STRATIFIED_THRESHOLD_PERCENT = score_threshold_percent
    _STRATIFIED_BATCH_SIZE = batch_size


def _scan_stratified_shard(task: _ShardScanTask) -> _ShardScanResult:
    matcher = _STRATIFIED_MATCHER
    if matcher is None or _STRATIFIED_BATCH_SIZE < 1:
        raise RuntimeError("Stratified scan worker was not initialized")
    inventory = task.inventory
    target_ranks = task.selected_eligible_ranks
    if not target_ranks:
        return _ShardScanResult(
            language=inventory.language,
            path=inventory.path,
            selected_rows=0,
            occurrences=[],
            source_file_runs=0,
            first_source_file_id=None,
            last_source_file_id=None,
            source_remote_paths=(),
        )
    if len(set(target_ranks)) != len(target_ranks):
        raise RuntimeError(f"Repeated systematic rank for {inventory.path}")

    path = Path(inventory.path)
    parquet = pq.ParquetFile(path)
    required = {
        "dataset",
        "record_id",
        "opening_comment",
        "language",
        "comment_license_score",
    }
    available_columns = set(parquet.schema_arrow.names)
    missing = sorted(required - available_columns)
    if missing:
        raise ValueError(f"Input shard {path} is missing columns: {', '.join(missing)}")
    requested = [
        "dataset",
        "record_id",
        "opening_comment",
        "language",
        "path",
        "repo",
        "metadata",
        "comment_license_score",
        "comment_license_detection",
    ]
    columns = [name for name in requested if name in available_columns]
    eligible_rank = 0
    physical_row_index = 0
    selected_rows = 0
    occurrences: list[dict[str, Any]] = []
    source_file_runs = 0
    first_source_file_id: str | None = None
    last_source_file_id: str | None = None
    previous_source_file_id: str | None = None
    last_source_index_by_remote: dict[str, int] = {}
    source_remote_paths: set[str] = set()
    target_index = 0
    score_column_index = columns.index("comment_license_score")

    for batch in parquet.iter_batches(
        batch_size=_STRATIFIED_BATCH_SIZE,
        columns=columns,
    ):
        scores = batch.column(score_column_index)
        if scores.null_count or pc.all(pc.is_finite(scores)).as_py() is not True:
            raise ValueError(f"Input shard {path} has an invalid ScanCode score")
        bounds = pc.min_max(scores).as_py()
        if bounds["min"] < 0 or bounds["max"] > 100:
            raise ValueError(f"Input shard {path} has a ScanCode score outside 0..100")
        eligible_indices = pc.indices_nonzero(
            pc.less(scores, _STRATIFIED_THRESHOLD_PERCENT)
        ).to_pylist()
        batch_eligible_end = eligible_rank + len(eligible_indices)
        selected_physical_indices: list[int] = []
        while (
            target_index < len(target_ranks)
            and target_ranks[target_index] < batch_eligible_end
        ):
            target_rank = target_ranks[target_index]
            if target_rank < eligible_rank:
                raise RuntimeError(f"Systematic ranks are not ordered for {path}")
            selected_physical_indices.append(
                int(eligible_indices[target_rank - eligible_rank])
            )
            target_index += 1

        if selected_physical_indices:
            selected_batch = batch.take(
                pa.array(selected_physical_indices, type=pa.int64())
            )
            values = selected_batch.to_pydict()
            for index, batch_physical_index in enumerate(selected_physical_indices):
                score = float(_row_value(values, "comment_license_score", index))
                row_dataset = str(_row_value(values, "dataset", index) or "")
                row_language = str(_row_value(values, "language", index) or "")
                if row_dataset != _STRATIFIED_DATASET:
                    raise ValueError(
                        f"Input shard {path} declares dataset {row_dataset!r}"
                    )
                if row_language != inventory.language:
                    raise ValueError(
                        f"Input shard {path} declares language {row_language!r}, "
                        f"expected {inventory.language!r}"
                    )
                record_id = str(_row_value(values, "record_id", index) or "")
                metadata = _row_value(values, "metadata", index)
                remote_path, source_row_index, comment_index = _source_identity(
                    record_id,
                    str(metadata) if metadata is not None else None,
                )
                if source_row_index < 0:
                    raise ValueError(f"Negative source row index in {record_id!r}")
                previous_index = last_source_index_by_remote.get(remote_path)
                if previous_index is not None and source_row_index < previous_index:
                    raise ValueError(
                        f"Source rows are not ordered within {remote_path!r}"
                    )
                last_source_index_by_remote[remote_path] = source_row_index
                source_remote_paths.add(remote_path)
                source_file_id = f"{remote_path}::row::{source_row_index}"
                if source_file_id != previous_source_file_id:
                    source_file_runs += 1
                    previous_source_file_id = source_file_id
                if first_source_file_id is None:
                    first_source_file_id = source_file_id
                last_source_file_id = source_file_id
                selected_rows += 1

                comment = str(_row_value(values, "opening_comment", index) or "")
                matches = matcher.match(comment)
                if matches:
                    normalized = _normalized_comment(comment)
                    comment_hash = _sha256_text(normalized)
                    candidate_id = f"redistribution-{comment_hash[:24]}"
                    occurrence_id = _sha256_text(
                        f"{_STRATIFIED_DATASET}\0{record_id}\0{comment_hash}"
                    )
                    occurrences.append(
                        {
                            "occurrence_id": occurrence_id,
                            "candidate_id": candidate_id,
                            "comment_hash": comment_hash,
                            "source_file_id": source_file_id,
                            "source_remote_path": remote_path,
                            "source_file_row_index": source_row_index,
                            "source_comment_index": comment_index,
                            "source_parquet_path": str(path),
                            "source_parquet_row_index": (
                                physical_row_index + batch_physical_index
                            ),
                            "dataset": row_dataset,
                            "record_id": record_id,
                            "language": row_language,
                            "path": str(_row_value(values, "path", index) or ""),
                            "repo": str(_row_value(values, "repo", index) or ""),
                            "opening_comment": comment,
                            "comment_license_score": score,
                            "comment_license_detection": str(
                                _row_value(
                                    values,
                                    "comment_license_detection",
                                    index,
                                )
                                or ""
                            ),
                            "matched_seed_groups": [
                                match.group for match in matches
                            ],
                            "matched_seed_phrases": [
                                match.phrase for match in matches
                            ],
                            "match_scores": [match.score for match in matches],
                            "match_excerpts": [match.excerpt for match in matches],
                            "best_match_score": matches[0].score,
                            "best_match_excerpt": matches[0].excerpt,
                        }
                    )
        eligible_rank = batch_eligible_end
        physical_row_index += batch.num_rows
    if physical_row_index != inventory.total_rows:
        raise RuntimeError(f"Parquet row count changed while scanning {path}")
    if eligible_rank != inventory.eligible_rows:
        raise RuntimeError(f"Eligible row count changed while scanning {path}")
    if selected_rows != len(target_ranks):
        raise RuntimeError(f"Systematic sample underfilled in {path}")
    if target_index != len(target_ranks):
        raise RuntimeError(f"Systematic sample ranks were not consumed in {path}")
    return _ShardScanResult(
        language=inventory.language,
        path=inventory.path,
        selected_rows=selected_rows,
        occurrences=occurrences,
        source_file_runs=source_file_runs,
        first_source_file_id=first_source_file_id,
        last_source_file_id=last_source_file_id,
        source_remote_paths=tuple(sorted(source_remote_paths)),
    )


def _scan_candidates_all_languages(
    input_source: Path | str,
    *,
    dataset: str,
    comment_rows_limit: int,
    scancode_score_threshold: float,
    fuzzy_threshold: float,
    include_government_seeds: bool,
    include_provenance_seeds: bool,
    include_funding_seeds: bool,
    include_export_control_seeds: bool,
    include_unpublished_work_seeds: bool,
    batch_size: int,
    scan_workers: int,
) -> _ScanResult:
    if comment_rows_limit < 1:
        raise ValueError("comment_rows_limit must be >= 1")
    if batch_size < 1 or scan_workers < 1:
        raise ValueError("batch_size and scan_workers must be >= 1")
    threshold_percent = _normalized_scancode_threshold_percent(
        scancode_score_threshold
    )
    input_directory, language_shards = _discover_all_language_shards(
        input_source,
        dataset=dataset,
    )
    inventory_tasks = [
        (
            language,
            str(path),
            path.relative_to(input_directory).as_posix(),
            threshold_percent,
            batch_size,
        )
        for language, shards in language_shards.items()
        for path in shards
    ]
    with ProcessPoolExecutor(max_workers=scan_workers) as executor:
        inventory = list(executor.map(_count_shard_eligibility, inventory_tasks))
    inventory.sort(key=lambda item: (item.language, item.relative_path))

    capacities = Counter()
    for item in inventory:
        capacities[item.language] += item.eligible_rows
    allocation = _max_min_language_allocation(capacities, comment_rows_limit)
    inventory_by_language: dict[str, list[_ShardEligibility]] = defaultdict(list)
    for item in inventory:
        inventory_by_language[item.language].append(item)

    scan_tasks: list[_ShardScanTask] = []
    selected_by_path: dict[str, int] = {}
    for language in sorted(inventory_by_language):
        selected_ranks = _systematic_sample_ranks(
            capacities[language], allocation[language]
        )
        rank_index = 0
        eligible_offset = 0
        for item in inventory_by_language[language]:
            local_ranks: list[int] = []
            limit = eligible_offset + item.eligible_rows
            while (
                rank_index < len(selected_ranks)
                and selected_ranks[rank_index] < limit
            ):
                local_ranks.append(selected_ranks[rank_index] - eligible_offset)
                rank_index += 1
            eligible_offset = limit
            selected_by_path[item.path] = len(local_ranks)
            if local_ranks:
                scan_tasks.append(
                    _ShardScanTask(
                        inventory=item,
                        selected_eligible_ranks=tuple(local_ranks),
                    )
                )
        if rank_index != len(selected_ranks) or eligible_offset != capacities[language]:
            raise RuntimeError(f"Unable to map systematic sample for {language}")

    seeds = _seed_inventory(
        include_government_seeds=include_government_seeds,
        include_provenance_seeds=include_provenance_seeds,
        include_funding_seeds=include_funding_seeds,
        include_export_control_seeds=include_export_control_seeds,
        include_unpublished_work_seeds=include_unpublished_work_seeds,
    )
    with ProcessPoolExecutor(
        max_workers=scan_workers,
        initializer=_init_stratified_scan_worker,
        initargs=(
            seeds,
            fuzzy_threshold,
            dataset,
            threshold_percent,
            batch_size,
        ),
    ) as executor:
        shard_results = list(executor.map(_scan_stratified_shard, scan_tasks))

    occurrences: list[dict[str, Any]] = []
    by_candidate: dict[str, dict[str, Any]] = {}
    candidate_occurrence_ids: dict[str, list[str]] = defaultdict(list)
    comment_bearing_files_seen = 0
    previous_last_source_file: str | None = None
    source_remote_paths: set[str] = set()
    for result in shard_results:
        comment_bearing_files_seen += result.source_file_runs
        if (
            previous_last_source_file is not None
            and result.first_source_file_id == previous_last_source_file
        ):
            comment_bearing_files_seen -= 1
        if result.last_source_file_id is not None:
            previous_last_source_file = result.last_source_file_id
        source_remote_paths.update(result.source_remote_paths)
        for occurrence in result.occurrences:
            occurrences.append(occurrence)
            candidate_id = str(occurrence["candidate_id"])
            occurrence_id = str(occurrence["occurrence_id"])
            candidate_occurrence_ids[candidate_id].append(occurrence_id)
            by_candidate.setdefault(candidate_id, occurrence)

    if sum(result.selected_rows for result in shard_results) != comment_rows_limit:
        raise RuntimeError("Stratified scan did not select exactly the requested rows")
    candidates: list[dict[str, Any]] = []
    for candidate_id, representative in sorted(by_candidate.items()):
        candidate = dict(representative)
        occurrence_ids = candidate_occurrence_ids[candidate_id]
        candidate["occurrence_count"] = len(occurrence_ids)
        candidate["occurrence_ids"] = occurrence_ids
        candidates.append(candidate)

    inventory_payload = [
        {
            "language": item.language,
            "path": item.relative_path,
            "size": item.size,
            "total_rows": item.total_rows,
            "eligible_rows": item.eligible_rows,
            "selected_rows": selected_by_path.get(item.path, 0),
        }
        for item in inventory
    ]
    inventory_sha256 = _sha256_text(
        json.dumps(
            inventory_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    )
    language_allocations = tuple(
        {
            "language": language,
            "eligible_rows": capacities[language],
            "selected_rows": allocation[language],
        }
        for language in sorted(capacities)
    )
    return _ScanResult(
        occurrences=occurrences,
        candidates=candidates,
        source_remote_path="multiple",
        comment_bearing_files_seen=comment_bearing_files_seen,
        comment_rows_seen=comment_rows_limit,
        shards_scanned=len(scan_tasks),
        source_shards=inventory_payload,
        input_directory=input_directory,
        input_format="local",
        selection_mode=SELECTION_STRATIFIED_COMMENTS,
        comment_rows_examined=sum(item.total_rows for item in inventory),
        score_filtered_rows=sum(
            item.total_rows - item.eligible_rows for item in inventory
        ),
        eligible_comment_rows_available=sum(capacities.values()),
        language_allocations=language_allocations,
        source_inventory_sha256=inventory_sha256,
    )


def _scan_candidates(
    input_source: Path | str,
    *,
    dataset: str,
    language: str,
    source_files_limit: int,
    fuzzy_threshold: float,
    include_government_seeds: bool,
    include_provenance_seeds: bool,
    include_funding_seeds: bool,
    include_export_control_seeds: bool,
    include_unpublished_work_seeds: bool,
    revision: str,
    hf_token: str | bool | None,
    hf_cache_directory: Path | None,
    batch_size: int,
) -> _ScanResult:
    if source_files_limit < 1:
        raise ValueError("source_files_limit must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    matcher = FuzzySeedMatcher(
        _seed_inventory(
            include_government_seeds=include_government_seeds,
            include_provenance_seeds=include_provenance_seeds,
            include_funding_seeds=include_funding_seeds,
            include_export_control_seeds=include_export_control_seeds,
            include_unpublished_work_seeds=include_unpublished_work_seeds,
        ),
        threshold=fuzzy_threshold,
    )
    shards, input_directory, input_format = _source_shards(
        input_source,
        dataset=dataset,
        language=language,
        revision=revision,
        hf_token=hf_token,
        hf_cache_directory=hf_cache_directory,
    )
    required = {"dataset", "record_id", "opening_comment", "language"}
    requested = [
        "dataset",
        "record_id",
        "opening_comment",
        "language",
        "path",
        "repo",
        "metadata",
        "comment_license_score",
        "comment_license_detection",
    ]
    occurrences: list[dict[str, Any]] = []
    by_candidate: dict[str, dict[str, Any]] = {}
    candidate_occurrence_ids: dict[str, list[str]] = defaultdict(list)
    comment_bearing_file_ids: set[str] = set()
    source_remote_path: str | None = None
    comment_rows_seen = 0
    shards_scanned = 0
    source_shards: list[dict[str, Any]] = []
    reached_boundary = False
    last_source_row_index: int | None = None

    for shard in shards:
        parquet = pq.ParquetFile(shard)
        available = set(parquet.schema_arrow.names)
        missing = sorted(required - available)
        if missing:
            raise ValueError(f"Input shard {shard} is missing columns: {', '.join(missing)}")
        columns = [name for name in requested if name in available]
        shard_row_index = 0
        shard_used = False
        for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
            values = batch.to_pydict()
            for index in range(batch.num_rows):
                record_id = str(_row_value(values, "record_id", index) or "")
                metadata = _row_value(values, "metadata", index)
                remote_path, source_row_index, comment_index = _source_identity(
                    record_id,
                    str(metadata) if metadata is not None else None,
                )
                if source_remote_path is None:
                    source_remote_path = remote_path
                if remote_path != source_remote_path:
                    raise ValueError(
                        "Input changed source remote_path before an explicit "
                        "source-file prefix boundary was observed"
                    )
                if source_row_index < 0:
                    raise ValueError(f"Negative source row index in {record_id!r}")
                if (
                    last_source_row_index is not None
                    and source_row_index < last_source_row_index
                ):
                    raise ValueError(
                        "Input source row indices are not monotonically ordered"
                    )
                if source_row_index >= source_files_limit:
                    reached_boundary = True
                    break
                last_source_row_index = source_row_index
                shard_used = True
                comment_rows_seen += 1
                source_file_id = f"{remote_path}::row::{source_row_index}"
                comment_bearing_file_ids.add(source_file_id)
                comment = str(_row_value(values, "opening_comment", index) or "")
                matches = matcher.match(comment)
                if not matches:
                    shard_row_index += 1
                    continue
                normalized = _normalized_comment(comment)
                comment_hash = _sha256_text(normalized)
                candidate_id = f"redistribution-{comment_hash[:24]}"
                occurrence_id = _sha256_text(
                    f"{dataset}\0{record_id}\0{comment_hash}"
                )
                occurrence = {
                    "occurrence_id": occurrence_id,
                    "candidate_id": candidate_id,
                    "comment_hash": comment_hash,
                    "source_file_id": source_file_id,
                    "source_remote_path": remote_path,
                    "source_file_row_index": source_row_index,
                    "source_comment_index": comment_index,
                    "source_parquet_path": str(shard),
                    "source_parquet_row_index": shard_row_index,
                    "dataset": str(_row_value(values, "dataset", index) or ""),
                    "record_id": record_id,
                    "language": str(_row_value(values, "language", index) or ""),
                    "path": str(_row_value(values, "path", index) or ""),
                    "repo": str(_row_value(values, "repo", index) or ""),
                    "opening_comment": comment,
                    "comment_license_score": _row_value(
                        values, "comment_license_score", index
                    ),
                    "comment_license_detection": str(
                        _row_value(values, "comment_license_detection", index) or ""
                    ),
                    "matched_seed_groups": [match.group for match in matches],
                    "matched_seed_phrases": [match.phrase for match in matches],
                    "match_scores": [match.score for match in matches],
                    "match_excerpts": [match.excerpt for match in matches],
                    "best_match_score": matches[0].score,
                    "best_match_excerpt": matches[0].excerpt,
                }
                occurrences.append(occurrence)
                candidate_occurrence_ids[candidate_id].append(occurrence_id)
                by_candidate.setdefault(candidate_id, occurrence)
                shard_row_index += 1
            if reached_boundary:
                break
        if shard_used:
            shards_scanned += 1
            source_shards.append(
                {
                    "path": str(shard),
                    "size": shard.stat().st_size,
                    "sha256": _sha256_file(shard),
                }
            )
        if reached_boundary:
            break
    if source_remote_path is None:
        raise ValueError("Selected input contains no comment rows")
    if not reached_boundary:
        raise ValueError(
            "Input ended before the requested original source-file prefix boundary"
        )

    candidates: list[dict[str, Any]] = []
    for candidate_id, representative in sorted(by_candidate.items()):
        candidate = dict(representative)
        occurrence_ids = candidate_occurrence_ids[candidate_id]
        candidate["occurrence_count"] = len(occurrence_ids)
        candidate["occurrence_ids"] = occurrence_ids
        candidates.append(candidate)
    return _ScanResult(
        occurrences=occurrences,
        candidates=candidates,
        source_remote_path=source_remote_path,
        comment_bearing_files_seen=len(comment_bearing_file_ids),
        comment_rows_seen=comment_rows_seen,
        shards_scanned=shards_scanned,
        source_shards=source_shards,
        input_directory=input_directory,
        input_format=input_format,
        comment_rows_examined=comment_rows_seen,
    )


def _write_parquet(path: Path, rows: Sequence[Mapping[str, Any]], schema: pa.Schema) -> None:
    table = pa.Table.from_pylist([dict(row) for row in rows], schema=schema)
    pq.write_table(table, path, compression="zstd")


def _scancode_judgment_fields(
    row: Mapping[str, Any], decision: Mapping[str, Any]
) -> dict[str, Any]:
    raw_detection = row.get("comment_license_detection")
    detection: Mapping[str, Any] | None = None
    if isinstance(raw_detection, Mapping):
        detection = raw_detection
    elif isinstance(raw_detection, str) and raw_detection.strip():
        try:
            parsed = json.loads(raw_detection)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, Mapping):
            detection = parsed

    raw_contains = detection.get("contains_license_notice") if detection else None
    contains_notice = raw_contains if isinstance(raw_contains, bool) else None
    expression = None
    if detection:
        expression = detection.get("detected_license_expression_spdx")
        if expression is None:
            expression = detection.get("detected_license_expression")
    if expression is not None:
        expression = str(expression).strip() or None

    is_license_notice = decision.get("is_license_notice")
    if is_license_notice is True:
        missed = None if contains_notice is None else not contains_notice
    elif is_license_notice is False:
        missed = False
    else:
        missed = None
    return {
        "scancode_contains_license_notice": contains_notice,
        "scancode_detected_license_expression": expression,
        "is_scancode_missed_license": missed,
    }


def _judged_schema_for_profile(profile: str) -> pa.Schema:
    return (
        _LIMITATION_JUDGED_SCHEMA
        if profile == JUDGMENT_PROFILE_NON_LICENSE_LIMITATIONS
        else _JUDGED_SCHEMA
    )


def _labeled_occurrence_schema_for_profile(profile: str) -> pa.Schema:
    return (
        _LIMITATION_LABELED_OCCURRENCE_SCHEMA
        if profile == JUDGMENT_PROFILE_NON_LICENSE_LIMITATIONS
        else _LABELED_OCCURRENCE_SCHEMA
    )


def _expected_limitation_label(
    restriction: bool | None, license_notice: bool | None
) -> str:
    if not isinstance(restriction, bool) or not isinstance(license_notice, bool):
        return LABEL_AMBIGUOUS
    if restriction and license_notice:
        return LABEL_NON_LICENSE_LIMITATION_WITH_LICENSE
    if restriction:
        return LABEL_NON_LICENSE_LIMITATION
    if license_notice:
        return LABEL_LICENSE_ONLY
    return LABEL_OTHER


def _artifact_metadata(path: Path) -> dict[str, Any]:
    return {"size": path.stat().st_size, "sha256": _sha256_file(path)}


def _dataset_card(manifest: Mapping[str, Any]) -> str:
    results = manifest["results"]
    judge = manifest.get("judge")
    profile = manifest["parameters"].get(
        "judgment_profile", JUDGMENT_PROFILE_REDISTRIBUTION_INTENT
    )
    judge_text = (
        "This is a scan-only retrieval artifact; no model labels are present."
        if manifest["parameters"]["scan_only"]
        else (
            "Every unique comment was judged by `gpt-5.6-luna` at "
            f"`{judge['reasoning_effort']}` "
            "reasoning, and every occurrence maps back to that decision."
        )
    )
    labels = ""
    if isinstance(judge, Mapping):
        labels = "\nLabel counts: " + json.dumps(
            results.get("label_counts", {}), sort_keys=True
        ) + ".\n"
    if profile == JUDGMENT_PROFILE_NON_LICENSE_LIMITATIONS:
        title = "Non-License Redistribution-Limitation Candidate Comments"
        purpose = (
            "The judge records non-license redistribution limitations and genuine "
            "license text as independent facts. `non-license-limitations.parquet` "
            "contains every affirmative non-license limitation, including mixed "
            "headers; `scancode-missed-licenses.parquet` contains judged license "
            "comments for which ScanCode `contains_license_notice` was false."
        )
    else:
        title = "Redistribution-Intent Candidate Comments"
        purpose = (
            "Use `dataset.parquet` for unique judged candidates and "
            "`labeled-occurrences.parquet` when source-level multiplicity matters."
        )
    if manifest["parameters"].get(
        "selection_mode", SELECTION_SOURCE_FILE_PREFIX
    ) == SELECTION_STRATIFIED_COMMENTS:
        scope_text = (
            f"It contains a deterministic, max-min language-balanced sample of "
            f"{manifest['parameters']['comment_rows_limit']:,} comments from "
            f"{results['language_count']:,} language partitions. Eligible comments "
            f"have raw ScanCode score strictly below "
            f"{manifest['parameters']['scancode_score_threshold_percent']}."
        )
        language_text = "all locally available languages"
    else:
        scope_text = (
            f"It covers the first {manifest['parameters']['source_files_limit']:,} "
            "original Stack source-file rows; files without extracted comments "
            "remain part of that bound."
        )
        language_text = f"`{manifest['source']['language']}`"
    return f"""---
license: other
task_categories:
- text-classification
language:
- en
---

# {title}

This bounded audit dataset was retrieved from `{manifest['source']['dataset_id']}`
configuration `{manifest['source']['dataset']}`, language
{language_text}. {scope_text}

The formatting-tolerant token matcher used
{results['seed_phrase_count']} phrase seeds and retained
{results['matched_occurrences']:,} source occurrences representing
{results['candidate_count']:,} normalized unique comments. Retrieval is not a
label. {judge_text}
{labels}
{purpose}
Use `dataset.parquet` for all unique judged candidates and
`labeled-occurrences.parquet` when source-level multiplicity matters.
`candidates.parquet` and `occurrences.parquet` preserve pre-judge retrieval
evidence. See `manifest.json`, `verification.json`, and `judge-rubric.md` for
the auditable configuration and semantics.
"""


def _recognizable_output(path: Path) -> bool:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return manifest.get("kind") == MANIFEST_KIND


def _publish_staging(staging: Path, output: Path, *, overwrite: bool) -> None:
    if not output.exists():
        os.replace(staging, output)
        return
    if not overwrite:
        raise ValueError(f"Output directory already exists: {output}")
    if not output.is_dir() or not _recognizable_output(output):
        raise ValueError(
            f"Refusing to replace unrecognized redistribution output: {output}"
        )
    backup = output.with_name(f".{output.name}.backup-{uuid4().hex}")
    os.replace(output, backup)
    try:
        os.replace(staging, output)
    except BaseException:
        os.replace(backup, output)
        raise
    shutil.rmtree(backup)


def build_redistribution_candidates(
    output_directory: Path,
    *,
    input_source: Path | str = DEFAULT_INPUT_SOURCE,
    dataset: str = DEFAULT_DATASET,
    language: str = DEFAULT_LANGUAGE,
    source_files_limit: int = DEFAULT_SOURCE_FILES_LIMIT,
    all_languages: bool = False,
    comment_rows_limit: int | None = None,
    scancode_score_threshold: float | None = None,
    scan_workers: int = 1,
    fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
    include_government_seeds: bool = False,
    include_provenance_seeds: bool = False,
    include_funding_seeds: bool = False,
    include_export_control_seeds: bool = False,
    include_unpublished_work_seeds: bool = False,
    scan_only: bool = False,
    batch_size: int = 8192,
    judge_batch_size: int = 64,
    judge_workers: int = 4,
    judge_max_attempts: int = 3,
    judge_timeout_seconds: float = 900.0,
    judge_max_batch_chars: int = 160_000,
    judge_max_comment_chars: int = 12_000,
    judge_cache_path: Path | None = None,
    revision: str = DEFAULT_REVISION,
    hf_token: str | bool | None = None,
    hf_cache_directory: Path | None = None,
    codex_command: str = "codex",
    codex_model: str = DEFAULT_CODEX_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    judgment_profile: str = JUDGMENT_PROFILE_REDISTRIBUTION_INTENT,
    overwrite: bool = False,
    judge_runner: Any | None = None,
) -> RedistributionBuildStats:
    output = output_directory.expanduser().resolve()
    if not dataset.strip() or not language.strip():
        raise ValueError("dataset and language must be non-empty")
    if all_languages:
        if comment_rows_limit is None:
            raise ValueError("all_languages requires comment_rows_limit")
        if scancode_score_threshold is None:
            raise ValueError("all_languages requires scancode_score_threshold")
    elif comment_rows_limit is not None or scancode_score_threshold is not None:
        raise ValueError(
            "comment_rows_limit and scancode_score_threshold require all_languages"
        )
    if scan_workers < 1:
        raise ValueError("scan_workers must be >= 1")
    if codex_model != DEFAULT_CODEX_MODEL:
        raise ValueError(f"codex_model must be {DEFAULT_CODEX_MODEL!r}")
    if reasoning_effort not in SUPPORTED_REASONING_EFFORTS:
        raise ValueError(
            "reasoning_effort must be one of "
            + ", ".join(SUPPORTED_REASONING_EFFORTS)
        )
    if judgment_profile not in JUDGMENT_PROFILES:
        raise ValueError(
            f"judgment_profile must be one of {', '.join(JUDGMENT_PROFILES)}"
        )
    if output.exists() and not overwrite:
        raise ValueError(f"Output directory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.with_name(f".{output.name}.lock")
    lock_stream = lock_path.open("a+")
    try:
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another build holds the output lock: {lock_path}") from exc
        staging = output.with_name(f".{output.name}.staging-{uuid4().hex}")
        staging.mkdir()
        try:
            seeds = _seed_inventory(
                include_government_seeds=include_government_seeds,
                include_provenance_seeds=include_provenance_seeds,
                include_funding_seeds=include_funding_seeds,
                include_export_control_seeds=include_export_control_seeds,
                include_unpublished_work_seeds=include_unpublished_work_seeds,
            )
            if all_languages:
                assert comment_rows_limit is not None
                assert scancode_score_threshold is not None
                scan = _scan_candidates_all_languages(
                    input_source,
                    dataset=dataset,
                    comment_rows_limit=comment_rows_limit,
                    scancode_score_threshold=scancode_score_threshold,
                    fuzzy_threshold=fuzzy_threshold,
                    include_government_seeds=include_government_seeds,
                    include_provenance_seeds=include_provenance_seeds,
                    include_funding_seeds=include_funding_seeds,
                    include_export_control_seeds=include_export_control_seeds,
                    include_unpublished_work_seeds=(
                        include_unpublished_work_seeds
                    ),
                    batch_size=batch_size,
                    scan_workers=scan_workers,
                )
            else:
                scan = _scan_candidates(
                    input_source,
                    dataset=dataset,
                    language=language,
                    source_files_limit=source_files_limit,
                    fuzzy_threshold=fuzzy_threshold,
                    include_government_seeds=include_government_seeds,
                    include_provenance_seeds=include_provenance_seeds,
                    include_funding_seeds=include_funding_seeds,
                    include_export_control_seeds=include_export_control_seeds,
                    include_unpublished_work_seeds=(
                        include_unpublished_work_seeds
                    ),
                    revision=revision,
                    hf_token=hf_token,
                    hf_cache_directory=hf_cache_directory,
                    batch_size=batch_size,
                )
            occurrences_path = staging / "occurrences.parquet"
            candidates_path = staging / "candidates.parquet"
            _write_parquet(occurrences_path, scan.occurrences, _OCCURRENCE_SCHEMA)
            _write_parquet(candidates_path, scan.candidates, _CANDIDATE_SCHEMA)

            judge_stats = None
            judge_configuration: dict[str, Any] | None = None
            judged_rows: list[dict[str, Any]] = []
            labeled_occurrences: list[dict[str, Any]] = []
            dataset_path: Path | None = None
            labeled_occurrences_path: Path | None = None
            if not scan_only:
                cache_path = (
                    judge_cache_path.expanduser().resolve()
                    if judge_cache_path is not None
                    else output.with_name(f"{output.name}-judge-cache.sqlite")
                )
                decisions, judge_stats, judge_configuration = (
                    judge_redistribution_candidates(
                        scan.candidates,
                        output_directory=staging,
                        cache_path=cache_path,
                        codex_command=codex_command,
                        codex_model=codex_model,
                        reasoning_effort=reasoning_effort,
                        batch_size=judge_batch_size,
                        max_batch_chars=judge_max_batch_chars,
                        workers=judge_workers,
                        max_attempts=judge_max_attempts,
                        timeout_seconds=judge_timeout_seconds,
                        max_comment_chars=judge_max_comment_chars,
                        judgment_profile=judgment_profile,
                        runner=judge_runner,
                    )
                )
                for candidate in scan.candidates:
                    decision = decisions[candidate["candidate_id"]]
                    judged_rows.append(
                        {
                            **candidate,
                            **decision,
                            **(
                                _scancode_judgment_fields(candidate, decision)
                                if judgment_profile
                                == JUDGMENT_PROFILE_NON_LICENSE_LIMITATIONS
                                else {}
                            ),
                        }
                    )
                for occurrence in scan.occurrences:
                    decision = decisions[occurrence["candidate_id"]]
                    labeled_occurrences.append(
                        {
                            **occurrence,
                            **decision,
                            **(
                                _scancode_judgment_fields(occurrence, decision)
                                if judgment_profile
                                == JUDGMENT_PROFILE_NON_LICENSE_LIMITATIONS
                                else {}
                            ),
                        }
                    )
                dataset_path = staging / "dataset.parquet"
                labeled_occurrences_path = staging / "labeled-occurrences.parquet"
                judged_schema = _judged_schema_for_profile(judgment_profile)
                labeled_schema = _labeled_occurrence_schema_for_profile(
                    judgment_profile
                )
                _write_parquet(dataset_path, judged_rows, judged_schema)
                _write_parquet(
                    labeled_occurrences_path,
                    labeled_occurrences,
                    labeled_schema,
                )
                if judgment_profile == JUDGMENT_PROFILE_NON_LICENSE_LIMITATIONS:
                    _write_parquet(
                        staging / "non-license-limitations.parquet",
                        [
                            row
                            for row in judged_rows
                            if row["is_non_license_redistribution_limitation"] is True
                        ],
                        judged_schema,
                    )
                    _write_parquet(
                        staging / "scancode-missed-licenses.parquet",
                        [
                            row
                            for row in judged_rows
                            if row["is_scancode_missed_license"] is True
                        ],
                        judged_schema,
                    )

            label_counts = Counter(
                str(row["judge_label"]) for row in judged_rows
            )
            manifest: dict[str, Any] = {
                "kind": MANIFEST_KIND,
                "format_version": FORMAT_VERSION,
                "created_at": _utc_now(),
                "source": {
                    "dataset_id": DEFAULT_INPUT_SOURCE,
                    "input_source": str(input_source),
                    "input_format": scan.input_format,
                    "input_directory": (
                        str(scan.input_directory) if scan.input_directory else None
                    ),
                    "revision": revision,
                    "dataset": dataset,
                    "language": "all" if all_languages else language,
                    "languages": (
                        [row["language"] for row in scan.language_allocations]
                        if all_languages
                        else [language]
                    ),
                    "language_allocations": list(scan.language_allocations),
                    "source_remote_path": scan.source_remote_path,
                    "source_shards": scan.source_shards,
                    "source_inventory_sha256": scan.source_inventory_sha256,
                },
                "parameters": {
                    "selection_mode": scan.selection_mode,
                    "source_files_limit": (
                        None if all_languages else source_files_limit
                    ),
                    "source_prefix_start_row": None if all_languages else 0,
                    "source_prefix_end_row_exclusive": (
                        None if all_languages else source_files_limit
                    ),
                    "all_languages": all_languages,
                    "comment_rows_limit": comment_rows_limit,
                    "scancode_score_threshold_requested": (
                        scancode_score_threshold
                    ),
                    "scancode_score_threshold_percent": (
                        _normalized_scancode_threshold_percent(
                            scancode_score_threshold
                        )
                        if scancode_score_threshold is not None
                        else None
                    ),
                    "scancode_score_comparison": (
                        "strictly_less_than" if all_languages else None
                    ),
                    "language_allocation": (
                        "max_min_water_fill" if all_languages else None
                    ),
                    "within_language_sampling": (
                        "systematic_midpoint_eligible_ranks"
                        if all_languages
                        else None
                    ),
                    "scan_workers": scan_workers,
                    "fuzzy_threshold": fuzzy_threshold,
                    "include_government_seeds": include_government_seeds,
                    "include_provenance_seeds": include_provenance_seeds,
                    "include_funding_seeds": include_funding_seeds,
                    "include_export_control_seeds": include_export_control_seeds,
                    "include_unpublished_work_seeds": (
                        include_unpublished_work_seeds
                    ),
                    "scan_only": scan_only,
                    "batch_size": batch_size,
                    "judgment_profile": judgment_profile,
                },
                "matching": {
                    "algorithm": "bounded-weighted-token-edit-distance",
                    "normalization": (
                        "Unicode NFKC casefolded word tokens; punctuation, "
                        "comment decoration, whitespace, and line endings are separators"
                    ),
                    "insertion_cost": 0.5,
                    "deletion_cost": 1.0,
                    "substitution": "character similarity when >=0.8, else cost 1",
                    "window_token_delta": [-1, 2],
                    "seeds": [asdict(seed) for seed in seeds],
                },
                "results": {
                    "source_files_in_scope": (
                        0 if all_languages else source_files_limit
                    ),
                    "comment_rows_in_scope": scan.comment_rows_seen,
                    "comment_rows_examined": scan.comment_rows_examined,
                    "score_filtered_rows": scan.score_filtered_rows,
                    "eligible_comment_rows_available": (
                        scan.eligible_comment_rows_available
                    ),
                    "language_count": (
                        len(scan.language_allocations) if all_languages else 1
                    ),
                    "comment_bearing_files_seen": scan.comment_bearing_files_seen,
                    "comment_rows_seen": scan.comment_rows_seen,
                    "shards_scanned": scan.shards_scanned,
                    "seed_phrase_count": len(seeds),
                    "matched_occurrences": len(scan.occurrences),
                    "candidate_count": len(scan.candidates),
                    "judged_count": len(judged_rows),
                    "label_counts": dict(sorted(label_counts.items())),
                    "non_license_limitation_count": sum(
                        row.get("is_non_license_redistribution_limitation") is True
                        for row in judged_rows
                    ),
                    "scancode_missed_license_count": sum(
                        row.get("is_scancode_missed_license") is True
                        for row in judged_rows
                    ),
                },
                "judge": judge_configuration,
                "artifacts": {},
            }
            readme_path = staging / "README.md"
            readme_path.write_text(_dataset_card(manifest), encoding="utf-8")
            artifact_names = ["occurrences.parquet", "candidates.parquet", "README.md"]
            if not scan_only:
                artifact_names.extend(
                    [
                        "dataset.parquet",
                        "labeled-occurrences.parquet",
                        "judge-rubric.md",
                        "judge-output.schema.json",
                        "judge-responses.jsonl",
                        "judge-errors.jsonl",
                        "judge-decisions.jsonl",
                    ]
                )
                if judgment_profile == JUDGMENT_PROFILE_NON_LICENSE_LIMITATIONS:
                    artifact_names.extend(
                        [
                            "non-license-limitations.parquet",
                            "scancode-missed-licenses.parquet",
                        ]
                    )
            manifest["artifacts"] = {
                name: _artifact_metadata(staging / name) for name in artifact_names
            }
            manifest_path = staging / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )
            report = verify_redistribution_candidates(staging)
            if not report.valid:
                raise RuntimeError(
                    "Staged redistribution dataset failed verification: "
                    + "; ".join(report.errors)
                )
            verification_path = staging / "verification.json"
            verification_path.write_text(
                json.dumps(asdict(report), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _publish_staging(staging, output, overwrite=overwrite)
        except BaseException:
            if staging.exists():
                shutil.rmtree(staging)
            raise
    finally:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
        lock_stream.close()

    stats = RedistributionBuildStats(
        output_directory=output,
        input_directory=scan.input_directory,
        input_format=scan.input_format,
        source_files_in_scope=0 if all_languages else source_files_limit,
        comment_rows_in_scope=scan.comment_rows_seen,
        languages_in_scope=(
            len(scan.language_allocations) if all_languages else 1
        ),
        selection_mode=scan.selection_mode,
        comment_bearing_files_seen=scan.comment_bearing_files_seen,
        comment_rows_seen=scan.comment_rows_seen,
        shards_scanned=scan.shards_scanned,
        matched_occurrences=len(scan.occurrences),
        candidate_count=len(scan.candidates),
        judged_count=len(judged_rows),
        judge_batches=judge_stats.batches if judge_stats else 0,
        judge_attempts=judge_stats.calls if judge_stats else 0,
        judge_cache_hits=judge_stats.cache_hits if judge_stats else 0,
        scan_only=scan_only,
        occurrences_path=output / "occurrences.parquet",
        candidates_path=output / "candidates.parquet",
        dataset_path=(output / "dataset.parquet") if not scan_only else None,
        labeled_occurrences_path=(
            output / "labeled-occurrences.parquet" if not scan_only else None
        ),
        non_license_limitations_path=(
            output / "non-license-limitations.parquet"
            if not scan_only
            and judgment_profile == JUDGMENT_PROFILE_NON_LICENSE_LIMITATIONS
            else None
        ),
        scancode_missed_licenses_path=(
            output / "scancode-missed-licenses.parquet"
            if not scan_only
            and judgment_profile == JUDGMENT_PROFILE_NON_LICENSE_LIMITATIONS
            else None
        ),
        manifest_path=output / "manifest.json",
        verification_path=output / "verification.json",
    )
    return stats


def _schema_matches(path: Path, expected: pa.Schema) -> bool:
    try:
        actual = pq.ParquetFile(path).schema_arrow
    except (OSError, pa.ArrowException):
        return False
    return actual.equals(expected, check_metadata=False)


def _rows_match_on_fields(
    left: Mapping[str, Any], right: Mapping[str, Any], fields: Iterable[str]
) -> bool:
    for field in fields:
        left_value = left.get(field)
        right_value = right.get(field)
        if (
            isinstance(left_value, float)
            and isinstance(right_value, float)
            and math.isnan(left_value)
            and math.isnan(right_value)
        ):
            continue
        if left_value != right_value:
            return False
    return True


def verify_redistribution_candidates(
    output_directory: Path,
) -> RedistributionVerificationReport:
    output = output_directory.expanduser().resolve()
    errors: list[str] = []
    manifest_path = output / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return RedistributionVerificationReport(
            valid=False, errors=(f"Cannot read manifest: {exc}",)
        )
    if manifest.get("kind") != MANIFEST_KIND:
        errors.append("manifest kind is not a redistribution candidate dataset")
    if manifest.get("format_version") not in SUPPORTED_FORMAT_VERSIONS:
        errors.append("unsupported manifest format version")
    parameters = manifest.get("parameters")
    results = manifest.get("results")
    if not isinstance(parameters, Mapping) or not isinstance(results, Mapping):
        return RedistributionVerificationReport(
            valid=False, errors=tuple(errors + ["manifest parameters/results are missing"])
        )
    for parameter_name in (
        "include_government_seeds",
        "include_provenance_seeds",
        "include_funding_seeds",
        "include_export_control_seeds",
        "include_unpublished_work_seeds",
    ):
        parameter_value = parameters.get(parameter_name, False)
        if not isinstance(parameter_value, bool):
            errors.append(f"manifest {parameter_name} parameter is not boolean")
    scan_only = parameters.get("scan_only") is True
    judgment_profile = str(
        parameters.get(
            "judgment_profile", JUDGMENT_PROFILE_REDISTRIBUTION_INTENT
        )
    )
    if judgment_profile not in JUDGMENT_PROFILES:
        errors.append("manifest has an invalid judgment profile")
    limitation_profile = (
        judgment_profile == JUDGMENT_PROFILE_NON_LICENSE_LIMITATIONS
    )
    candidate_count = int(results.get("candidate_count", -1))
    occurrence_count = int(results.get("matched_occurrences", -1))
    judged_count = int(results.get("judged_count", -1))
    selection_mode = str(
        parameters.get("selection_mode", SELECTION_SOURCE_FILE_PREFIX)
    )
    if selection_mode not in (
        SELECTION_SOURCE_FILE_PREFIX,
        SELECTION_STRATIFIED_COMMENTS,
    ):
        errors.append("manifest has an invalid selection mode")
    stratified = selection_mode == SELECTION_STRATIFIED_COMMENTS
    if stratified and manifest.get("format_version") != FORMAT_VERSION:
        errors.append("stratified selection requires manifest format version 2")
    source = manifest.get("source")
    if not isinstance(source, Mapping):
        source = {}
        errors.append("manifest source configuration is missing")

    language_set: set[str] = set()
    score_threshold_percent: float | None = None
    if stratified:
        try:
            comment_rows_limit = int(parameters.get("comment_rows_limit", -1))
            requested_threshold = float(
                parameters.get("scancode_score_threshold_requested")
            )
            score_threshold_percent = float(
                parameters.get("scancode_score_threshold_percent")
            )
        except (TypeError, ValueError):
            comment_rows_limit = -1
            requested_threshold = math.nan
            score_threshold_percent = None
            errors.append("manifest stratified thresholds are invalid")
        if comment_rows_limit < 1:
            errors.append("manifest comment row limit is invalid")
        if (
            score_threshold_percent is None
            or not math.isfinite(score_threshold_percent)
            or score_threshold_percent <= 0
            or score_threshold_percent > 100
        ):
            errors.append("manifest normalized ScanCode threshold is invalid")
        else:
            try:
                expected_threshold = _normalized_scancode_threshold_percent(
                    requested_threshold
                )
            except ValueError:
                expected_threshold = math.nan
            if expected_threshold != score_threshold_percent:
                errors.append("manifest ScanCode threshold normalization changed")
        if parameters.get("scancode_score_comparison") != "strictly_less_than":
            errors.append("manifest ScanCode comparison is not strict")
        if parameters.get("language_allocation") != "max_min_water_fill":
            errors.append("manifest language allocation algorithm changed")
        if (
            parameters.get("within_language_sampling")
            != "systematic_midpoint_eligible_ranks"
        ):
            errors.append("manifest within-language sampling algorithm changed")
        if int(results.get("source_files_in_scope", -1)) != 0:
            errors.append("stratified manifest reports a source-file prefix")
        if int(results.get("comment_rows_in_scope", -1)) != comment_rows_limit:
            errors.append("comment row scope count is inconsistent")
        if int(results.get("comment_rows_seen", -1)) != comment_rows_limit:
            errors.append("selected comment row count is inconsistent")

        allocations = source.get("language_allocations")
        if not isinstance(allocations, list) or not allocations:
            errors.append("stratified manifest has no language allocations")
            allocations = []
        capacities: dict[str, int] = {}
        declared_allocations: dict[str, int] = {}
        for row in allocations:
            if not isinstance(row, Mapping):
                errors.append("manifest has an invalid language allocation")
                continue
            language = row.get("language")
            eligible = row.get("eligible_rows")
            selected = row.get("selected_rows")
            if (
                not isinstance(language, str)
                or not language
                or isinstance(eligible, bool)
                or not isinstance(eligible, int)
                or eligible < 1
                or isinstance(selected, bool)
                or not isinstance(selected, int)
                or not 1 <= selected <= eligible
                or language in capacities
            ):
                errors.append("manifest has an invalid language allocation")
                continue
            capacities[language] = eligible
            declared_allocations[language] = selected
        language_set = set(capacities)
        if list(capacities) != sorted(capacities):
            errors.append("manifest language allocations are not sorted")
        if sum(declared_allocations.values()) != comment_rows_limit:
            errors.append("language allocations do not sum to comment row limit")
        if capacities:
            try:
                expected_allocation = _max_min_language_allocation(
                    capacities, comment_rows_limit
                )
            except ValueError:
                expected_allocation = {}
            if expected_allocation != declared_allocations:
                errors.append("language allocations are not max-min balanced")
        if int(results.get("language_count", -1)) != len(language_set):
            errors.append("language count does not match allocations")
        if source.get("languages") != sorted(language_set):
            errors.append("source language list does not match allocations")
        if source.get("language") != "all":
            errors.append("stratified source language is not all")

        shard_rows = source.get("source_shards")
        if not isinstance(shard_rows, list) or not shard_rows:
            errors.append("stratified manifest has no source shard inventory")
            shard_rows = []
        inventory_payload: list[dict[str, Any]] = []
        total_rows = 0
        eligible_rows = 0
        selected_rows = 0
        selected_shards = 0
        for row in shard_rows:
            if not isinstance(row, Mapping):
                errors.append("manifest has an invalid source shard entry")
                continue
            try:
                normalized_row = {
                    "language": str(row["language"]),
                    "path": str(row["path"]),
                    "size": int(row["size"]),
                    "total_rows": int(row["total_rows"]),
                    "eligible_rows": int(row["eligible_rows"]),
                    "selected_rows": int(row["selected_rows"]),
                }
            except (KeyError, TypeError, ValueError):
                errors.append("manifest has an invalid source shard entry")
                continue
            if (
                normalized_row["language"] not in language_set
                or normalized_row["size"] < 0
                or normalized_row["total_rows"] < 0
                or not 0
                <= normalized_row["eligible_rows"]
                <= normalized_row["total_rows"]
                or not 0
                <= normalized_row["selected_rows"]
                <= normalized_row["eligible_rows"]
            ):
                errors.append("manifest has an invalid source shard entry")
            inventory_payload.append(normalized_row)
            total_rows += normalized_row["total_rows"]
            eligible_rows += normalized_row["eligible_rows"]
            selected_rows += normalized_row["selected_rows"]
            selected_shards += normalized_row["selected_rows"] > 0
        inventory_hash = _sha256_text(
            json.dumps(
                inventory_payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
        )
        if source.get("source_inventory_sha256") != inventory_hash:
            errors.append("source shard inventory hash mismatch")
        if total_rows != int(results.get("comment_rows_examined", -1)):
            errors.append("examined comment count does not match shard inventory")
        if eligible_rows != int(
            results.get("eligible_comment_rows_available", -1)
        ):
            errors.append("eligible comment count does not match shard inventory")
        if total_rows - eligible_rows != int(results.get("score_filtered_rows", -1)):
            errors.append("score-filtered count does not match shard inventory")
        if selected_rows != comment_rows_limit:
            errors.append("selected shard rows do not sum to comment row limit")
        if selected_shards != int(results.get("shards_scanned", -1)):
            errors.append("selected shard count does not match results")
    else:
        source_limit = int(parameters.get("source_files_limit", -1))
        if int(results.get("source_files_in_scope", -2)) != source_limit:
            errors.append("source-file scope count is inconsistent")

    matching = manifest.get("matching")
    declared_seed_pairs: set[tuple[str, str]] = set()
    if not isinstance(matching, Mapping) or not isinstance(
        matching.get("seeds"), list
    ):
        errors.append("manifest matching seed inventory is missing")
    else:
        seed_rows = matching["seeds"]
        for ordinal, seed in enumerate(seed_rows):
            if not isinstance(seed, Mapping):
                errors.append("manifest contains an invalid seed entry")
                continue
            group = seed.get("group")
            phrase = seed.get("phrase")
            tokens = seed.get("tokens")
            if (
                not isinstance(group, str)
                or not group
                or not isinstance(phrase, str)
                or not phrase
                or not isinstance(tokens, list)
                or not tokens
                or seed.get("ordinal") != ordinal
            ):
                errors.append("manifest contains an invalid seed entry")
                continue
            pair = (group, phrase)
            if pair in declared_seed_pairs:
                errors.append("manifest contains a duplicate seed group/phrase")
            declared_seed_pairs.add(pair)
        if len(seed_rows) != int(results.get("seed_phrase_count", -1)):
            errors.append("manifest seed count does not match seed inventory")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        errors.append("manifest artifacts are missing")
        artifacts = {}
    required_artifacts = {
        "occurrences.parquet",
        "candidates.parquet",
        "README.md",
    }
    if not scan_only:
        required_artifacts.update(
            {
                "dataset.parquet",
                "labeled-occurrences.parquet",
                "judge-rubric.md",
                "judge-output.schema.json",
                "judge-responses.jsonl",
                "judge-errors.jsonl",
                "judge-decisions.jsonl",
            }
        )
        if limitation_profile:
            required_artifacts.update(
                {
                    "non-license-limitations.parquet",
                    "scancode-missed-licenses.parquet",
                }
            )
    for name in sorted(required_artifacts - set(artifacts)):
        errors.append(f"manifest is missing required artifact {name}")
    for name, metadata in artifacts.items():
        path = output / str(name)
        if not path.is_file():
            errors.append(f"missing artifact {name}")
            continue
        if not isinstance(metadata, Mapping):
            errors.append(f"invalid artifact metadata for {name}")
            continue
        if path.stat().st_size != metadata.get("size"):
            errors.append(f"artifact size mismatch for {name}")
        if _sha256_file(path) != metadata.get("sha256"):
            errors.append(f"artifact hash mismatch for {name}")

    occurrences_path = output / "occurrences.parquet"
    candidates_path = output / "candidates.parquet"
    if not _schema_matches(occurrences_path, _OCCURRENCE_SCHEMA):
        errors.append("occurrences.parquet schema mismatch")
        occurrences: list[dict[str, Any]] = []
    else:
        occurrences = pq.read_table(occurrences_path).to_pylist()
    if not _schema_matches(candidates_path, _CANDIDATE_SCHEMA):
        errors.append("candidates.parquet schema mismatch")
        candidates: list[dict[str, Any]] = []
    else:
        candidates = pq.read_table(candidates_path).to_pylist()
    if len(occurrences) != occurrence_count:
        errors.append("occurrence row count does not match manifest")
    if len(candidates) != candidate_count:
        errors.append("candidate row count does not match manifest")

    threshold = float(parameters.get("fuzzy_threshold", -1.0))
    source_limit = (
        None
        if stratified
        else int(parameters.get("source_files_limit", -1))
    )
    candidate_ids: set[str] = set()
    candidates_by_id: dict[str, dict[str, Any]] = {}
    normalized_comments: set[str] = set()
    counts_by_candidate = Counter()
    occurrence_ids: set[str] = set()
    for row in occurrences:
        candidate_id = row.get("candidate_id")
        counts_by_candidate[candidate_id] += 1
        occurrence_id = row.get("occurrence_id")
        if occurrence_id in occurrence_ids:
            errors.append("duplicate occurrence_id")
        occurrence_ids.add(occurrence_id)
        source_index = row.get("source_file_row_index")
        if not isinstance(source_index, int) or source_index < 0:
            errors.append("occurrence has an invalid source row index")
        elif source_limit is not None and source_index >= source_limit:
            errors.append("occurrence lies outside source-file prefix")
        if stratified:
            if row.get("language") not in language_set:
                errors.append("occurrence language is outside the allocation")
            raw_score = row.get("comment_license_score")
            try:
                occurrence_score = float(raw_score)
            except (TypeError, ValueError):
                occurrence_score = math.nan
            if (
                score_threshold_percent is None
                or not math.isfinite(occurrence_score)
                or occurrence_score < 0
                or occurrence_score >= score_threshold_percent
            ):
                errors.append("occurrence violates the ScanCode score filter")
        phrases = row.get("matched_seed_phrases") or []
        groups = row.get("matched_seed_groups") or []
        scores = row.get("match_scores") or []
        excerpts = row.get("match_excerpts") or []
        if not phrases or not (len(phrases) == len(groups) == len(scores) == len(excerpts)):
            errors.append("occurrence has misaligned or empty seed matches")
        if any(
            (str(group), str(phrase)) not in declared_seed_pairs
            for group, phrase in zip(groups, phrases, strict=False)
        ):
            errors.append("occurrence references a seed absent from the manifest")
        if any(not isinstance(score, (int, float)) or score < threshold for score in scores):
            errors.append("occurrence has a seed score below the threshold")
    for row in candidates:
        candidate_id = row.get("candidate_id")
        if candidate_id in candidate_ids:
            errors.append("duplicate candidate_id")
        candidate_ids.add(candidate_id)
        candidates_by_id[candidate_id] = row
        normalized = _normalized_comment(str(row.get("opening_comment") or ""))
        if normalized in normalized_comments:
            errors.append("duplicate normalized candidate comment")
        normalized_comments.add(normalized)
        expected_hash = _sha256_text(normalized)
        if row.get("comment_hash") != expected_hash:
            errors.append("candidate comment hash mismatch")
        if candidate_id != f"redistribution-{expected_hash[:24]}":
            errors.append("candidate_id does not match comment hash")
        if row.get("occurrence_count") != counts_by_candidate[candidate_id]:
            errors.append("candidate occurrence_count mismatch")
        if sorted(row.get("occurrence_ids") or []) != sorted(
            occurrence["occurrence_id"]
            for occurrence in occurrences
            if occurrence["candidate_id"] == candidate_id
        ):
            errors.append("candidate occurrence_ids mismatch")
    if set(counts_by_candidate) != candidate_ids:
        errors.append("occurrence candidate IDs do not equal candidate table IDs")

    if scan_only:
        if judged_count != 0:
            errors.append("scan-only manifest reports judged candidates")
        if (output / "dataset.parquet").exists():
            errors.append("scan-only output unexpectedly contains dataset.parquet")
    else:
        judge = manifest.get("judge")
        if not isinstance(judge, Mapping):
            errors.append("judged output has no judge configuration")
        else:
            if judge.get("model") != DEFAULT_CODEX_MODEL:
                errors.append("judge model is not gpt-5.6-luna")
            if judge.get("reasoning_effort") not in SUPPORTED_REASONING_EFFORTS:
                errors.append("judge reasoning effort is unsupported")
            if str(
                judge.get(
                    "judgment_profile",
                    JUDGMENT_PROFILE_REDISTRIBUTION_INTENT,
                )
            ) != judgment_profile:
                errors.append("judge configuration profile does not match manifest")
        dataset_path = output / "dataset.parquet"
        labeled_path = output / "labeled-occurrences.parquet"
        judged_schema = _judged_schema_for_profile(judgment_profile)
        labeled_schema = _labeled_occurrence_schema_for_profile(judgment_profile)
        if not _schema_matches(dataset_path, judged_schema):
            errors.append("dataset.parquet schema mismatch")
            judged_rows: list[dict[str, Any]] = []
        else:
            judged_rows = pq.read_table(dataset_path).to_pylist()
        if not _schema_matches(labeled_path, labeled_schema):
            errors.append("labeled-occurrences.parquet schema mismatch")
            labeled_rows: list[dict[str, Any]] = []
        else:
            labeled_rows = pq.read_table(labeled_path).to_pylist()
        if len(judged_rows) != candidate_count or len(judged_rows) != judged_count:
            errors.append("judged dataset count mismatch")
        if len(labeled_rows) != occurrence_count:
            errors.append("labeled occurrence count mismatch")
        judged_by_id = {row.get("candidate_id"): row for row in judged_rows}
        if set(judged_by_id) != candidate_ids:
            errors.append("judged candidate IDs do not match candidates")
        for row in judged_rows:
            candidate_id = row.get("candidate_id")
            candidate = candidates_by_id.get(candidate_id)
            if candidate is None or not _rows_match_on_fields(
                row, candidate, _CANDIDATE_SCHEMA.names
            ):
                errors.append("judged candidate differs from pre-judge candidate")
            label = row.get("judge_label")
            if limitation_profile:
                restriction = row.get(
                    "is_non_license_redistribution_limitation"
                )
                license_notice = row.get("is_license_notice")
                if label not in LIMITATION_JUDGE_LABELS:
                    errors.append("invalid judge label")
                if label != _expected_limitation_label(
                    restriction, license_notice
                ):
                    errors.append("judge label/fact mismatch")
                known = row.get("is_known_license")
                known_license = row.get("known_license")
                if license_notice is None:
                    if known is not None or known_license is not None:
                        errors.append("uncertain license has an identity")
                elif license_notice is False:
                    if known is not False or known_license is not None:
                        errors.append("non-license has a license identity")
                elif not isinstance(known, bool):
                    errors.append("license row has no known-license fact")
                elif known and not str(known_license or "").strip():
                    errors.append("known license has no name")
                elif not known and known_license is not None:
                    errors.append("unknown license has a name")
                expected_scan = _scancode_judgment_fields(row, row)
                if any(
                    row.get(field) != expected_scan[field]
                    for field in expected_scan
                ):
                    errors.append("ScanCode comparison fields are inconsistent")
                for fact_field, evidence_field in (
                    (
                        "is_non_license_redistribution_limitation",
                        "restriction_evidence",
                    ),
                    ("is_license_notice", "license_evidence"),
                ):
                    axis_evidence = row.get(evidence_field)
                    if row.get(fact_field) is True:
                        if not str(axis_evidence or "").strip():
                            errors.append(f"empty {evidence_field}")
                        elif not _evidence_is_source_text(
                            str(axis_evidence),
                            str(row.get("opening_comment") or ""),
                        ):
                            errors.append(f"{evidence_field} is not grounded")
                    elif axis_evidence is not None:
                        errors.append(f"unexpected {evidence_field}")
            else:
                expected_boolean = (
                    True
                    if label == LABEL_CODE_REDISTRIBUTION_INTENT
                    else False if label == LABEL_OTHER else None
                )
                if label not in JUDGE_LABELS:
                    errors.append("invalid judge label")
                if row.get("is_code_redistribution_intent") is not expected_boolean:
                    errors.append("judge label/boolean mismatch")
            confidence = row.get("judge_confidence")
            if not isinstance(confidence, (int, float)) or not 0.0 <= confidence <= 1.0:
                errors.append("invalid judge confidence")
            evidence = str(row.get("judge_evidence") or "")
            if not evidence.strip():
                errors.append("empty judge evidence")
            elif not _evidence_is_source_text(
                evidence, str(row.get("opening_comment") or "")
            ):
                errors.append("judge evidence is not grounded in candidate comment")
            if not str(row.get("judge_rationale") or "").strip():
                errors.append("empty judge rationale")
        actual_label_counts = dict(
            sorted(Counter(str(row.get("judge_label")) for row in judged_rows).items())
        )
        if results.get("label_counts") != actual_label_counts:
            errors.append("judge label counts do not match manifest")
        occurrences_by_id = {
            row.get("occurrence_id"): row for row in occurrences
        }
        labeled_ids: set[Any] = set()
        for row in labeled_rows:
            occurrence_id = row.get("occurrence_id")
            if occurrence_id in labeled_ids:
                errors.append("duplicate labeled occurrence_id")
            labeled_ids.add(occurrence_id)
            occurrence = occurrences_by_id.get(occurrence_id)
            if occurrence is None or not _rows_match_on_fields(
                row, occurrence, _OCCURRENCE_SCHEMA.names
            ):
                errors.append("labeled occurrence differs from source occurrence")
            judged = judged_by_id.get(row.get("candidate_id"))
            judgment_fields = (
                [
                    field[0]
                    for field in _LIMITATION_JUDGMENT_FIELDS
                    if not field[0].startswith("scancode_")
                    and field[0] != "is_scancode_missed_license"
                ]
                if limitation_profile
                else [
                    "judge_label",
                    "is_code_redistribution_intent",
                    "judge_confidence",
                    "judge_evidence",
                    "judge_rationale",
                ]
            )
            if judged is None or any(
                row.get(field) != judged.get(field)
                for field in judgment_fields
            ):
                errors.append("labeled occurrence does not match its candidate judgment")
        if labeled_ids != set(occurrences_by_id):
            errors.append("labeled occurrence IDs do not match occurrences")
        if limitation_profile:
            expected_subsets = {
                "non-license-limitations.parquet": {
                    row["candidate_id"]
                    for row in judged_rows
                    if row.get("is_non_license_redistribution_limitation") is True
                },
                "scancode-missed-licenses.parquet": {
                    row["candidate_id"]
                    for row in judged_rows
                    if row.get("is_scancode_missed_license") is True
                },
            }
            result_keys = {
                "non-license-limitations.parquet": "non_license_limitation_count",
                "scancode-missed-licenses.parquet": "scancode_missed_license_count",
            }
            for name, expected_ids in expected_subsets.items():
                path = output / name
                if not _schema_matches(path, _LIMITATION_JUDGED_SCHEMA):
                    errors.append(f"{name} schema mismatch")
                    continue
                subset_rows = pq.read_table(path).to_pylist()
                subset_ids = [row.get("candidate_id") for row in subset_rows]
                if len(subset_ids) != len(set(subset_ids)):
                    errors.append(f"{name} contains duplicate candidates")
                if set(subset_ids) != expected_ids:
                    errors.append(f"{name} does not match its dataset predicate")
                if len(subset_ids) != int(results.get(result_keys[name], -1)):
                    errors.append(f"{name} count does not match manifest")
                for subset_row in subset_rows:
                    judged = judged_by_id.get(subset_row.get("candidate_id"))
                    if judged is None or not _rows_match_on_fields(
                        subset_row, judged, _LIMITATION_JUDGED_SCHEMA.names
                    ):
                        errors.append(f"{name} row differs from dataset.parquet")

    return RedistributionVerificationReport(
        valid=not errors,
        candidate_count=max(candidate_count, 0),
        matched_occurrences=max(occurrence_count, 0),
        judged_count=max(judged_count, 0),
        errors=tuple(dict.fromkeys(errors)),
    )
