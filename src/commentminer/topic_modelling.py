from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import logging
import math
from pathlib import Path
import re
import shutil
import shlex
import subprocess
from typing import Any, Callable, Iterable, Pattern, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from .models import _json_safe


_LOGGER = logging.getLogger(__name__)
_DEFAULT_SCORE_THRESHOLD = 0.95
_DEFAULT_BATCH_SIZE = 8192
_DEFAULT_TOPIC_SAMPLE_SIZE = 10
_DEFAULT_JUDGE_SAMPLE_SIZE = 8
_DEFAULT_JUDGE_MAX_COMMENT_CHARS = 1200

# Phrase-level seeds for guided BERTopic modelling. The mapping names are
# descriptive metadata only: BERTopic still discovers its own topic IDs.
SEED_TOPICS = {
    "proprietary_or_confidential": [
        "strictly confidential",
        "company confidential",
        "confidential information",
        "confidential and proprietary",
        "proprietary information",
        "proprietary material",
        "trade secret",
        "valuable trade secrets",
        "company internal",
        "internal use only",
        "for internal use",
        "not for public release",
        "not intended for publication",
        "unpublished work",
        "private and confidential",
        "remains the property of",
        "property of the company",
        "source code access",
        "employees and contractors",
        "current employees only",
    ],
    "non_license_sharing_restrictions": [
        "do not distribute",
        "not for distribution",
        "distribution prohibited",
        "redistribution prohibited",
        "do not disclose",
        "disclosure prohibited",
        "unauthorized disclosure",
        "do not disseminate",
        "dissemination prohibited",
        "do not share",
        "unauthorized sharing",
        "do not copy",
        "copying prohibited",
        "unauthorized copying",
        "reproduction prohibited",
        "unauthorized reproduction",
        "strictly forbidden",
        "prior written permission",
        "express written permission",
        "express written consent",
        "without written authorization",
        "authorized personnel only",
        "intended recipient only",
        "need to know",
        "outside the company",
        "no external distribution",
        "no external disclosure",
        "non-disclosure agreement",
        "nondisclosure agreement",
        "confidentiality agreement",
        "return or destroy all copies",
    ],
    "custom_or_unrecognized_license": [
        "license grant",
        "permission is granted",
        "subject to these terms",
        "subject to the terms and conditions",
        "provided that",
        "except as expressly permitted",
        "non-exclusive license",
        "non-transferable license",
        "non-sublicensable license",
        "revocable license",
        "limited license",
        "source available",
        "non-commercial use only",
        "no commercial use",
        "research purposes only",
        "academic use only",
        "educational use only",
        "evaluation purposes only",
        "personal use only",
        "field of use",
        "specific purpose only",
        "binary form only",
        "executable form only",
        "complete and unmodified",
        "unmodified form only",
        "no derivative works",
        "derivative works prohibited",
        "registered developer",
        "single user license",
        "named user license",
        "authorized users",
        "authorized site",
        "geographical location",
        "citation required",
        "attribution required",
        "commercial license required",
        "separate license required",
        "license fee",
        "royalty bearing",
        "termination of license",
    ],
    "customer_or_contract_specific": [
        "licensed to",
        "licensed for",
        "provided to",
        "delivered to",
        "furnished to",
        "supplied to",
        "prepared for",
        "developed for",
        "commissioned by",
        "on behalf of",
        "exclusive use of",
        "sole use of",
        "for customer use only",
        "for client use only",
        "customer confidential",
        "client confidential",
        "customer proprietary",
        "named customer",
        "registered customer",
        "authorized customer",
        "customer site",
        "licensed materials",
        "pursuant to the agreement",
        "pursuant to contract",
        "under contract",
        "contract number",
        "contract no",
        "purchase order",
        "statement of work",
        "source code escrow",
        "escrow beneficiary",
        "third-party licensor",
        "authorized reseller",
        "authorized distributor",
        "original equipment manufacturer",
    ],
}

# Kept separate so a run can test this more specialized cluster explicitly via
# bertopic_model_kwargs={"seed_topic_list": [*SEED_TOPICS.values(),
# GOVERNMENT_RESTRICTION_SEEDS]}.
GOVERNMENT_RESTRICTION_SEEDS = [
    "restricted rights notice",
    "limited rights notice",
    "government purpose rights",
    "restricted computer software",
    "commercial computer software",
    "developed at private expense",
    "use reproduction or disclosure",
    "subject to restrictions",
    "government end users",
    "contractor and subcontractor",
    "controlled unclassified information",
    "controlled technical information",
    "export controlled",
    "export control",
    "ITAR controlled",
    "EAR controlled",
    "US persons only",
    "classified information",
]

# Opt-in provenance indicators. These retrieve comments worth reviewing for
# copied, decompiled, reconstructed, or unexpectedly exposed proprietary code;
# they are signals for a downstream judge, not conclusions about authorization.
PROPRIETARY_PROVENANCE_SEEDS = [
    "decompiled by",
    "decompiled with",
    "source code recreated from a .class file",
    "powered by Fernflower",
    "reconstructed from bytecode",
    "extracted from APK",
    "extracted from JAR",
    "extracted from DEX",
    "proprietary and confidential",
    "proprietary/confidential",
    "confidential property",
    "proprietary source code",
    "unpublished proprietary source code",
    "stolen source code",
    "misappropriated code",
    "copied without permission",
    "accidentally committed",
    "should not be public",
    "remove before release",
    "copied from",
    "taken from",
    "borrowed from",
    "adapted from",
    "ported from",
]

# Opt-in unpublished-work indicators. Some phrases overlap the core
# proprietary/confidential and provenance inventories intentionally: when this
# family is enabled it gives those matches a specific retrieval group, while
# the default 126-seed inventory remains unchanged. These are review signals;
# a downstream judge must still distinguish a source-code nonpublication
# notice from references to unpublished papers, data, or ordinary metadata.
UNPUBLISHED_WORK_SEEDS = [
    "unpublished work",
    "unpublished work under U.S. copyright laws",
    "unpublished work pursuant to Title 17",
    "unpublished material",
    "unpublished copyrighted work",
    "unpublished source code",
    "unpublished proprietary source code",
    "unpublished proprietary material",
    "unpublished and confidential work",
    "unpublished copyright",
    "unpublished rights reserved",
    "unpublished all rights reserved",
    "source code for this program is not published",
    "source code is not published",
    "program is not published",
    "not published or otherwise divested",
    "does not evidence publication",
    "does not evidence any actual or intended publication",
    "does not evidence any actual or intended publication or disclosure",
    "copyright notice is precautionary only",
    "not intended for publication",
    "not for publication",
    "publication prohibited",
    "publication or disclosure prohibited",
    "may not be published",
    "not authorized for public release",
    "not publicly released",
    "no publication intended",
    "does not imply publication or disclosure",
    "not to be published",
]

# Opt-in project funding, sponsorship, and procurement phrases. A funding
# acknowledgement alone is not a dissemination limitation; the control and
# release phrases let a downstream judge determine whether the two are linked.
FUNDING_DISSEMINATION_SEEDS = [
    "funded by",
    "funding from",
    "under sponsorship of",
    "grant agreement",
    "cooperative agreement",
    "supported under Air Force contract",
    "developed pursuant to contract",
    "subject to sponsor approval",
    "approval prior to publication",
    "publication requires approval",
    "prepublication review",
    "publication embargo",
    "Distribution Statement B",
    "Distribution Statement C",
    "Distribution Statement D",
    "Distribution Statement E",
    "Distribution Statement F",
    "other requests shall be referred to",
    "DFARS 252.227",
    "SBIR data rights",
    "limited rights",
    "unlimited rights",
    "approved for public release",
    "distribution unlimited",
]

# Opt-in export-control vocabulary, including restrictive language,
# classifications/regimes, recipient controls, and explicit release/exception
# indicators. A few phrases intentionally overlap the older government list so
# this specialized family can be enabled independently.
EXPORT_CONTROL_SEEDS = [
    "export controlled",
    "export control",
    "ITAR controlled",
    "EAR controlled",
    "US persons only",
    "subject to export control laws",
    "export license required",
    "requires an export license",
    "export authorization required",
    "requires export authorization",
    "export or re-export",
    "deemed export",
    "foreign person",
    "foreign national",
    "ITAR",
    "International Traffic in Arms Regulations",
    "Export Administration Regulations",
    "EAR99",
    "ECCN",
    "Commerce Control List",
    "USML",
    "OFAC",
    "AECA",
    "Export Administration Act",
    "technical data",
    "defense article",
    "5D002",
    "5D992",
    "License Exception ENC",
    "EU dual-use controls",
    "Wassenaar Arrangement",
    "sanctioned countries",
    "embargoed countries",
    "denied parties",
    "NLR",
    "no license required",
    "not subject to export control",
]

# High-signal words and phrases for building a sharing-related candidate set
# before BERTopic runs. Deliberately omit noisy standalone inflections such as
# ``shared`` and ``distributed`` because they mostly describe programming
# concepts (for example, CUDA shared memory and distributed computation).
SHARING_PREFILTER_KEYWORDS = (
    "confidential",
    "confidentiality",
    "proprietary",
    "distribute",
    "distribution",
    "redistribute",
    "redistribution",
    "disclose",
    "disclosure",
    "disseminate",
    "dissemination",
    "share",
    "sharing",
    "reproduce",
    "reproduction",
    "permission",
    "consent",
    "authorization",
    "authorisation",
    "unauthorized",
    "unauthorised",
    "nondisclosure",
    "non-disclosure",
    # Phrase expansions capture common inflected restriction wording without
    # admitting noisy standalone words such as ``shared`` or ``copying``.
    "trade secret",
    "trade secrets",
    "internal use only",
    "for internal use",
    "not for public release",
    "not intended for publication",
    "do not distribute",
    "not be distributed",
    "not for distribution",
    "distribution prohibited",
    "do not disclose",
    "not be disclosed",
    "disclosure prohibited",
    "do not disseminate",
    "not be disseminated",
    "dissemination prohibited",
    "do not share",
    "not be shared",
    "sharing prohibited",
    "do not copy",
    "not be copied",
    "copying prohibited",
    "copying is prohibited",
    "do not reproduce",
    "not be reproduced",
    "reproduction prohibited",
    "strictly forbidden",
    "prior written permission",
    "express written permission",
    "express written consent",
    "without written authorization",
    "without written authorisation",
    "authorized personnel only",
    "authorised personnel only",
    "intended recipient only",
    "no external distribution",
    "no external disclosure",
    "return or destroy all copies",
)

_TOPIC_ASSIGNMENT_SCHEMA = pa.schema(
    [
        ("ordinal", pa.int64()),
        ("topic_id", pa.int64()),
        ("topic_label", pa.string()),
        ("topic_probability", pa.float64()),
        ("source_path", pa.string()),
        ("source_row_index", pa.int64()),
        ("dataset", pa.string()),
        ("record_id", pa.string()),
        ("language", pa.string()),
        ("path", pa.string()),
        ("repo", pa.string()),
        ("comment_license_score", pa.float64()),
        ("comment_license_score_percent", pa.float64()),
        ("opening_comment", pa.string()),
        ("source_record", pa.string()),
    ]
)


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


@dataclass(slots=True)
class TopicModellingStats:
    input_directory: Path
    output_directory: Path
    input_format: str
    score_threshold: float
    normalized_score_threshold: float
    records_seen: int = 0
    records_missing_score: int = 0
    records_without_comment: int = 0
    records_selected: int = 0
    records_modelled: int = 0
    topics_discovered: int = 0
    outlier_records: int = 0
    shards_read: int = 0
    assignments_path: Path | None = None
    topics_path: Path | None = None
    manifest_path: Path | None = None
    model_path: Path | None = None
    codex_judge_report_path: Path | None = None
    prefilter_keywords: tuple[str, ...] = ()
    records_before_keyword_prefilter: int = 0
    records_prefiltered_out: int = 0


@dataclass(slots=True)
class _SelectedComment:
    ordinal: int
    text: str
    source_path: str
    source_row_index: int
    payload: dict[str, Any]
    score: float
    score_percent: float


@dataclass(slots=True)
class _LoadedComments:
    comments: list[_SelectedComment]
    input_format: str
    records_seen: int = 0
    records_missing_score: int = 0
    records_without_comment: int = 0
    records_before_keyword_prefilter: int = 0
    records_prefiltered_out: int = 0
    shards_read: int = 0


def _resolve_topic_modelling_input(
    source: Path | str,
    *,
    dataset_names: Iterable[str] | None,
    languages: Iterable[str] | None,
) -> Path:
    """Resolve a local directory or a Hub dataset ID to a local cached snapshot."""

    candidate = Path(source).expanduser()
    if candidate.exists():
        if not candidate.is_dir():
            raise ValueError(f"Input ScanCode output is not a directory: {candidate}")
        return candidate.resolve()

    # Path objects retain the old, unambiguous local-directory behavior. A Hub
    # ID is accepted as a string in the conventional ``owner/dataset`` form.
    if isinstance(source, Path) or "/" not in source:
        raise ValueError(f"Input ScanCode output directory does not exist: {candidate.resolve()}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface-hub is required to use a Hugging Face dataset ID"
        ) from exc

    datasets = list(dataset_names or [])
    selected_languages = list(languages or [])
    if datasets and selected_languages:
        allow_patterns = [
            f"{dataset}/{language}/part-*.parquet"
            for dataset in datasets
            for language in selected_languages
        ]
    elif datasets:
        allow_patterns = [f"{dataset}/*/part-*.parquet" for dataset in datasets]
    elif selected_languages:
        allow_patterns = [f"*/{language}/part-*.parquet" for language in selected_languages]
    else:
        allow_patterns = ["*/*/part-*.parquet", "part-*.parquet"]

    try:
        snapshot_path = snapshot_download(
            repo_id=source,
            repo_type="dataset",
            allow_patterns=allow_patterns,
        )
    except Exception as exc:
        raise RuntimeError(f"Unable to cache Hugging Face dataset '{source}': {exc}") from exc
    return Path(snapshot_path).resolve()


def _normalize_prefilter_keywords(
    keywords: Iterable[str] | None,
) -> tuple[str, ...]:
    if keywords is None:
        return ()
    raw_keywords = [keywords] if isinstance(keywords, str) else keywords
    normalized: list[str] = []
    seen: set[str] = set()
    for keyword in raw_keywords:
        if not isinstance(keyword, str):
            raise ValueError(f"Prefilter keywords must be strings, got {keyword!r}")
        value = " ".join(keyword.split()).casefold()
        if not value:
            raise ValueError("Prefilter keywords must not be empty")
        if value not in seen:
            normalized.append(value)
            seen.add(value)
    return tuple(normalized)


def _compile_keyword_prefilter(keywords: Sequence[str]) -> Pattern[str] | None:
    if not keywords:
        return None
    alternatives = []
    for keyword in sorted(keywords, key=len, reverse=True):
        # Normalization collapses phrase whitespace to spaces. Match those
        # spaces flexibly so a phrase can span lines in a block comment.
        alternatives.append(re.escape(keyword).replace(r"\ ", r"\s+"))
    return re.compile(
        r"(?<!\w)(?:" + "|".join(alternatives) + r")(?!\w)",
        flags=re.IGNORECASE,
    )


def run_low_scancode_topic_modelling(
    input_directory: Path | str,
    *,
    output_directory: Path | None = None,
    score_threshold: float = _DEFAULT_SCORE_THRESHOLD,
    prefilter_keywords: Iterable[str] | None = None,
    dataset_names: Iterable[str] | None = None,
    languages: Iterable[str] | None = None,
    max_shards: int | None = None,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    min_topic_size: int = 10,
    calculate_probabilities: bool = False,
    bertopic_model: Any | None = None,
    bertopic_model_kwargs: dict[str, Any] | None = None,
    topic_sample_size: int = _DEFAULT_TOPIC_SAMPLE_SIZE,
    save_model: bool = False,
    judge_with_codex: bool = False,
    codex_command: str = "codex",
    codex_model: str | None = None,
    codex_timeout: int = 600,
    codex_runner: Callable[[str], str] | None = None,
    judge_sample_size: int = _DEFAULT_JUDGE_SAMPLE_SIZE,
    judge_max_topics: int | None = None,
    overwrite: bool = False,
) -> TopicModellingStats:
    """Run BERTopic over comments whose ScanCode score is below the threshold.

    ScanCode scores in this project are stored on a 0-100 scale, but this
    function accepts either 0.95-style ratio thresholds or 95-style percentage
    thresholds. Scores read from inputs are normalized the same way before
    filtering. When ``prefilter_keywords`` are supplied, only low-score
    comments containing at least one case-insensitive whole word or literal
    phrase are passed to BERTopic.
    """

    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if min_topic_size < 1:
        raise ValueError(f"min_topic_size must be >= 1, got {min_topic_size}")
    if topic_sample_size < 1:
        raise ValueError(f"topic_sample_size must be >= 1, got {topic_sample_size}")
    if judge_sample_size < 1:
        raise ValueError(f"judge_sample_size must be >= 1, got {judge_sample_size}")
    if judge_max_topics is not None and judge_max_topics < 1:
        raise ValueError(f"judge_max_topics must be >= 1, got {judge_max_topics}")
    if codex_timeout < 1:
        raise ValueError(f"codex_timeout must be >= 1, got {codex_timeout}")
    if max_shards is not None and max_shards < 1:
        raise ValueError(f"max_shards must be >= 1, got {max_shards}")

    normalized_prefilter_keywords = _normalize_prefilter_keywords(prefilter_keywords)
    keyword_prefilter = _compile_keyword_prefilter(normalized_prefilter_keywords)

    input_source = input_directory
    is_huggingface_source = (
        isinstance(input_source, str)
        and "/" in input_source
        and not Path(input_source).expanduser().exists()
    )
    input_directory = _resolve_topic_modelling_input(
        input_source,
        dataset_names=dataset_names,
        languages=languages,
    )

    normalized_threshold = _normalize_score(score_threshold)
    if output_directory is None and is_huggingface_source:
        output_directory = Path.cwd() / f"{input_source.rsplit('/', 1)[-1]}-topic-modelling"
    output_directory = (
        output_directory
        or input_directory.parent / f"{input_directory.name}-topic-modelling"
    ).resolve()
    if output_directory == input_directory:
        raise ValueError("Output directory must differ from the input directory")
    if overwrite and output_directory.exists():
        shutil.rmtree(output_directory)
    if output_directory.exists() and any(output_directory.iterdir()):
        raise ValueError(
            f"Output directory '{output_directory}' already exists and is not empty. "
            "Use overwrite=True or choose another output directory."
        )
    output_directory.mkdir(parents=True, exist_ok=True)

    loaded = _load_low_score_comments(
        input_directory,
        score_threshold=normalized_threshold,
        keyword_prefilter=keyword_prefilter,
        dataset_names=dataset_names,
        languages=languages,
        max_shards=max_shards,
        batch_size=batch_size,
    )
    stats = TopicModellingStats(
        input_directory=input_directory,
        output_directory=output_directory,
        input_format=loaded.input_format,
        score_threshold=score_threshold,
        normalized_score_threshold=normalized_threshold,
        records_seen=loaded.records_seen,
        records_missing_score=loaded.records_missing_score,
        records_without_comment=loaded.records_without_comment,
        records_before_keyword_prefilter=loaded.records_before_keyword_prefilter,
        records_prefiltered_out=loaded.records_prefiltered_out,
        records_selected=len(loaded.comments),
        shards_read=loaded.shards_read,
        prefilter_keywords=normalized_prefilter_keywords,
    )

    assignments_path = output_directory / "topic-assignments.parquet"
    topics_path = output_directory / "topics.json"
    stats.assignments_path = assignments_path
    stats.topics_path = topics_path
    stats.manifest_path = output_directory / "manifest.json"

    if not loaded.comments:
        _write_topic_assignments(assignments_path, [], [], [], topic_labels={})
        _write_topics_json(topics_path, [])
        _write_topic_modelling_manifest(
            stats,
            bertopic_config={
                "min_topic_size": min_topic_size,
                "calculate_probabilities": calculate_probabilities,
            },
            dataset_names=list(dataset_names or []),
            languages=list(languages or []),
            judge_with_codex=judge_with_codex,
        )
        return stats

    documents = [comment.text for comment in loaded.comments]
    model = bertopic_model or _build_bertopic_model(
        min_topic_size=min_topic_size,
        calculate_probabilities=calculate_probabilities,
        model_kwargs=bertopic_model_kwargs or {},
    )
    topics, probabilities = model.fit_transform(documents)
    topics = [int(topic) for topic in topics]
    if len(topics) != len(loaded.comments):
        raise RuntimeError(
            "BERTopic returned a topic assignment count that does not match the selected comments"
        )

    topic_summaries = _summarize_topics(
        loaded.comments,
        topics,
        probabilities,
        model,
        sample_size=topic_sample_size,
    )
    topic_labels = {
        int(summary["topic_id"]): str(summary["label"]) for summary in topic_summaries
    }
    _write_topic_assignments(
        assignments_path,
        loaded.comments,
        topics,
        probabilities,
        topic_labels=topic_labels,
    )
    _write_topics_json(topics_path, topic_summaries)

    stats.records_modelled = len(loaded.comments)
    stats.topics_discovered = len({topic for topic in topics if topic != -1})
    stats.outlier_records = sum(1 for topic in topics if topic == -1)

    if save_model:
        stats.model_path = output_directory / "bertopic-model"
        _save_bertopic_model(model, stats.model_path)

    if judge_with_codex:
        stats.codex_judge_report_path = validate_topic_clusters_with_codex(
            topic_summaries,
            output_directory,
            input_directory=input_directory,
            score_threshold=score_threshold,
            normalized_score_threshold=normalized_threshold,
            records_selected=len(loaded.comments),
            codex_command=codex_command,
            codex_model=codex_model,
            codex_timeout=codex_timeout,
            codex_runner=codex_runner,
            sample_size=judge_sample_size,
            max_topics=judge_max_topics,
        )

    _write_topic_modelling_manifest(
        stats,
        bertopic_config={
            "min_topic_size": min_topic_size,
            "calculate_probabilities": calculate_probabilities,
            "model_kwargs": _json_safe(bertopic_model_kwargs or {}),
        },
        dataset_names=list(dataset_names or []),
        languages=list(languages or []),
        judge_with_codex=judge_with_codex,
    )
    return stats


def validate_topic_clusters_with_codex(
    topic_summaries: Sequence[dict[str, Any]],
    output_directory: Path,
    *,
    input_directory: Path,
    score_threshold: float,
    normalized_score_threshold: float,
    records_selected: int,
    codex_command: str = "codex",
    codex_model: str | None = None,
    codex_timeout: int = 600,
    codex_runner: Callable[[str], str] | None = None,
    sample_size: int = _DEFAULT_JUDGE_SAMPLE_SIZE,
    max_topics: int | None = None,
) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    clusters = [
        _topic_summary_for_judge(summary, sample_size=sample_size)
        for summary in topic_summaries
        if int(summary.get("topic_id", -1)) != -1
    ]
    if max_topics is not None:
        clusters = clusters[:max_topics]

    prompt = _build_codex_judge_prompt(
        clusters,
        input_directory=input_directory,
        score_threshold=score_threshold,
        normalized_score_threshold=normalized_score_threshold,
        records_selected=records_selected,
    )
    prompt_path = output_directory / "codex-cluster-validation-prompt.md"
    response_path = output_directory / "codex-cluster-validation-response.md"
    report_path = output_directory / "codex-cluster-validation.json"
    prompt_path.write_text(prompt, encoding="utf-8")

    if codex_runner is None:
        response = _run_codex_exec(
            prompt,
            codex_command=codex_command,
            codex_model=codex_model,
            timeout=codex_timeout,
            cwd=Path.cwd(),
        )
    else:
        response = codex_runner(prompt)
    response_path.write_text(response, encoding="utf-8")

    parsed_response = _extract_json_object(response)
    report = {
        "created_at": _utc_now(),
        "input_directory": str(input_directory),
        "score_threshold": score_threshold,
        "normalized_score_threshold": normalized_score_threshold,
        "records_selected": records_selected,
        "clusters_submitted": len(clusters),
        "codex_command": codex_command,
        "codex_model": codex_model,
        "prompt_path": str(prompt_path),
        "response_path": str(response_path),
        "parsed_response": parsed_response,
    }
    report_path.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")
    return report_path


def _load_low_score_comments(
    input_directory: Path,
    *,
    score_threshold: float,
    keyword_prefilter: Pattern[str] | None,
    dataset_names: Iterable[str] | None,
    languages: Iterable[str] | None,
    max_shards: int | None,
    batch_size: int,
) -> _LoadedComments:
    dataset_filter = set(dataset_names or [])
    language_filter = set(languages or [])
    parquet_shards = _hf_parquet_shards(
        input_directory,
        dataset_filter=dataset_filter,
        language_filter=language_filter,
    )
    if max_shards is not None:
        parquet_shards = parquet_shards[:max_shards]
    if not parquet_shards:
        raise ValueError(f"No ScanCode-enriched Parquet shards found in: {input_directory}")

    loaded = _LoadedComments(
        comments=[],
        input_format="parquet",
    )
    for shard in parquet_shards:
        _load_parquet_low_score_comments(
            shard,
            input_directory,
            loaded,
            score_threshold=score_threshold,
            keyword_prefilter=keyword_prefilter,
            batch_size=batch_size,
            dataset_filter=dataset_filter,
            language_filter=language_filter,
        )
    return loaded


def _load_parquet_low_score_comments(
    shard: Path,
    input_directory: Path,
    loaded: _LoadedComments,
    *,
    score_threshold: float,
    keyword_prefilter: Pattern[str] | None,
    batch_size: int,
    dataset_filter: set[str],
    language_filter: set[str],
) -> None:
    relative_path = shard.relative_to(input_directory).as_posix()
    parquet_file = pq.ParquetFile(shard)
    row_offset = 0
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        for batch_index, payload in enumerate(batch.to_pylist()):
            _maybe_add_low_score_comment(
                payload,
                loaded,
                score_threshold=score_threshold,
                keyword_prefilter=keyword_prefilter,
                dataset_filter=dataset_filter,
                language_filter=language_filter,
                source_path=relative_path,
                source_row_index=row_offset + batch_index,
            )
        row_offset += batch.num_rows
    loaded.shards_read += 1


def _maybe_add_low_score_comment(
    payload: dict[str, Any],
    loaded: _LoadedComments,
    *,
    score_threshold: float,
    keyword_prefilter: Pattern[str] | None,
    dataset_filter: set[str],
    language_filter: set[str],
    source_path: str,
    source_row_index: int,
) -> None:
    if dataset_filter and str(payload.get("dataset") or "") not in dataset_filter:
        return
    if language_filter and str(payload.get("language") or "") not in language_filter:
        return

    loaded.records_seen += 1
    score = payload.get("comment_license_score")
    if score is None:
        loaded.records_missing_score += 1
        return
    try:
        score_percent = _normalize_score(score)
    except ValueError:
        loaded.records_missing_score += 1
        return
    if score_percent >= score_threshold:
        return

    text = payload.get("opening_comment")
    if text is None or not str(text).strip():
        loaded.records_without_comment += 1
        return

    text = str(text)
    loaded.records_before_keyword_prefilter += 1
    if keyword_prefilter is not None and keyword_prefilter.search(text) is None:
        loaded.records_prefiltered_out += 1
        return

    loaded.comments.append(
        _SelectedComment(
            ordinal=len(loaded.comments),
            text=text,
            source_path=source_path,
            source_row_index=source_row_index,
            payload=payload,
            score=float(score),
            score_percent=score_percent,
        )
    )


def _build_bertopic_model(
    *,
    min_topic_size: int,
    calculate_probabilities: bool,
    model_kwargs: dict[str, Any],
) -> Any:
    try:
        from bertopic import BERTopic
        from sklearn.feature_extraction.text import CountVectorizer
    except ImportError as exc:
        raise RuntimeError(
            "BERTopic is not installed. Run `uv sync` or `uv add bertopic` before topic modelling."
        ) from exc
    kwargs = {
        "language": "english",
        "min_topic_size": min_topic_size,
        "calculate_probabilities": calculate_probabilities,
        "seed_topic_list": [list(phrases) for phrases in SEED_TOPICS.values()],
        "vectorizer_model": CountVectorizer(
            lowercase=True,
            ngram_range=(1, 3),
            min_df=2,
            stop_words=None,
        ),
    }
    kwargs.update(model_kwargs)
    return BERTopic(**kwargs)


def _summarize_topics(
    comments: Sequence[_SelectedComment],
    topics: Sequence[int],
    probabilities: Any,
    model: Any,
    *,
    sample_size: int,
) -> list[dict[str, Any]]:
    grouped: dict[int, list[tuple[_SelectedComment, float | None]]] = {}
    for index, topic_id in enumerate(topics):
        grouped.setdefault(int(topic_id), []).append(
            (comments[index], _assignment_probability(probabilities, index))
        )

    summaries: list[dict[str, Any]] = []
    for topic_id in sorted(grouped):
        assignments = grouped[topic_id]
        keywords = _topic_keywords(model, topic_id)
        label = _topic_label(model, topic_id, keywords)
        summaries.append(
            {
                "topic_id": topic_id,
                "label": label,
                "count": len(assignments),
                "keywords": keywords,
                "representative_comments": [
                    _comment_example(comment, probability=probability, max_chars=500)
                    for comment, probability in assignments[:sample_size]
                ],
            }
        )
    return summaries


def _write_topic_assignments(
    path: Path,
    comments: Sequence[_SelectedComment],
    topics: Sequence[int],
    probabilities: Any,
    *,
    topic_labels: dict[int, str],
) -> None:
    rows = []
    for index, comment in enumerate(comments):
        topic_id = int(topics[index])
        rows.append(
            {
                "ordinal": comment.ordinal,
                "topic_id": topic_id,
                "topic_label": topic_labels.get(topic_id, f"Topic {topic_id}"),
                "topic_probability": _assignment_probability(probabilities, index),
                "source_path": comment.source_path,
                "source_row_index": comment.source_row_index,
                "dataset": comment.payload.get("dataset"),
                "record_id": comment.payload.get("record_id"),
                "language": comment.payload.get("language"),
                "path": comment.payload.get("path"),
                "repo": comment.payload.get("repo"),
                "comment_license_score": comment.score,
                "comment_license_score_percent": comment.score_percent,
                "opening_comment": comment.text,
                "source_record": json.dumps(
                    _json_safe(comment.payload), ensure_ascii=False, sort_keys=True
                ),
            }
        )
    table = pa.Table.from_pylist(rows, schema=_TOPIC_ASSIGNMENT_SCHEMA)
    pq.write_table(table, path, compression="zstd")


def _write_topics_json(path: Path, topic_summaries: Sequence[dict[str, Any]]) -> None:
    payload = {
        "created_at": _utc_now(),
        "topic_count": len(
            [summary for summary in topic_summaries if int(summary.get("topic_id", -1)) != -1]
        ),
        "outlier_count": sum(
            int(summary.get("count", 0))
            for summary in topic_summaries
            if int(summary.get("topic_id", -1)) == -1
        ),
        "topics": list(topic_summaries),
    }
    path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")


def _write_topic_modelling_manifest(
    stats: TopicModellingStats,
    *,
    bertopic_config: dict[str, Any],
    dataset_names: list[str],
    languages: list[str],
    judge_with_codex: bool,
) -> None:
    assert stats.manifest_path is not None
    payload = {
        "created_at": _utc_now(),
        "source_directory": str(stats.input_directory),
        "input_format": stats.input_format,
        "score_threshold": stats.score_threshold,
        "normalized_score_threshold": stats.normalized_score_threshold,
        "threshold_semantics": "select comments where normalized comment_license_score < normalized_score_threshold",
        "records_seen": stats.records_seen,
        "records_missing_score": stats.records_missing_score,
        "records_without_comment": stats.records_without_comment,
        "records_before_keyword_prefilter": stats.records_before_keyword_prefilter,
        "records_prefiltered_out": stats.records_prefiltered_out,
        "records_selected": stats.records_selected,
        "records_modelled": stats.records_modelled,
        "topics_discovered": stats.topics_discovered,
        "outlier_records": stats.outlier_records,
        "shards_read": stats.shards_read,
        "dataset_filter": sorted(dataset_names),
        "language_filter": sorted(languages),
        "keyword_prefilter": {
            "enabled": bool(stats.prefilter_keywords),
            "keywords": list(stats.prefilter_keywords),
            "match_mode": "any_case_insensitive_whole_word_or_phrase",
        },
        "bertopic": bertopic_config,
        "assignments_path": str(stats.assignments_path) if stats.assignments_path else None,
        "topics_path": str(stats.topics_path) if stats.topics_path else None,
        "model_path": str(stats.model_path) if stats.model_path else None,
        "judge_with_codex": judge_with_codex,
        "codex_judge_report_path": (
            str(stats.codex_judge_report_path) if stats.codex_judge_report_path else None
        ),
    }
    stats.manifest_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")


def _save_bertopic_model(model: Any, model_path: Path) -> None:
    save = getattr(model, "save", None)
    if not callable(save):
        raise RuntimeError("The BERTopic model object does not expose a callable save method")
    save(str(model_path))


def _hf_parquet_shards(
    input_directory: Path,
    *,
    dataset_filter: set[str],
    language_filter: set[str],
) -> list[Path]:
    shards: list[Path] = []
    shards.extend(sorted(input_directory.glob("part-*.parquet")))
    for path in sorted(input_directory.glob("*/*/part-*.parquet")):
        relative = path.relative_to(input_directory)
        if len(relative.parts) < 3 or relative.parts[0].startswith("."):
            continue
        dataset, language = relative.parts[0], relative.parts[1]
        if dataset_filter and dataset not in dataset_filter:
            continue
        if language_filter and language not in language_filter:
            continue
        shards.append(path)
    return shards


def _normalize_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"ScanCode score must be numeric, got {value!r}") from exc
    if not math.isfinite(score) or score < 0:
        raise ValueError(f"ScanCode score must be finite and non-negative, got {value!r}")
    if score <= 1.0:
        return score * 100.0
    return score


def _assignment_probability(probabilities: Any, index: int) -> float | None:
    if probabilities is None:
        return None
    try:
        row = probabilities[index]
    except (IndexError, KeyError, TypeError):
        return None
    try:
        if isinstance(row, (str, bytes)):
            return None
        iterator = iter(row)
    except TypeError:
        try:
            value = float(row)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    values: list[float] = []
    for item in iterator:
        try:
            value = float(item)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return max(values) if values else None


def _topic_keywords(model: Any, topic_id: int) -> list[dict[str, float | str]]:
    get_topic = getattr(model, "get_topic", None)
    if not callable(get_topic):
        return []
    try:
        raw_keywords = get_topic(topic_id) or []
    except Exception:
        return []
    keywords: list[dict[str, float | str]] = []
    for item in raw_keywords:
        if not isinstance(item, (list, tuple)) or not item:
            continue
        term = str(item[0])
        try:
            weight = float(item[1]) if len(item) > 1 else 0.0
        except (TypeError, ValueError):
            weight = 0.0
        keywords.append({"term": term, "weight": weight})
    return keywords


def _topic_label(
    model: Any,
    topic_id: int,
    keywords: Sequence[dict[str, float | str]],
) -> str:
    if topic_id == -1:
        return "Outliers"
    labels = getattr(model, "topic_labels_", None)
    if isinstance(labels, dict) and topic_id in labels:
        return str(labels[topic_id])
    label = _topic_label_from_info(model, topic_id)
    if label:
        return label
    terms = [str(item["term"]) for item in keywords[:4]]
    if terms:
        return f"{topic_id}_" + "_".join(terms)
    return f"Topic {topic_id}"


def _topic_label_from_info(model: Any, topic_id: int) -> str | None:
    get_topic_info = getattr(model, "get_topic_info", None)
    if not callable(get_topic_info):
        return None
    try:
        info = get_topic_info()
    except Exception:
        return None
    to_dict = getattr(info, "to_dict", None)
    if not callable(to_dict):
        return None
    try:
        rows = to_dict("records")
    except Exception:
        return None
    for row in rows:
        try:
            row_topic = int(row.get("Topic"))
        except (TypeError, ValueError):
            continue
        if row_topic == topic_id and row.get("Name"):
            return str(row["Name"])
    return None


def _comment_example(
    comment: _SelectedComment,
    *,
    probability: float | None,
    max_chars: int,
) -> dict[str, Any]:
    return {
        "ordinal": comment.ordinal,
        "dataset": comment.payload.get("dataset"),
        "record_id": comment.payload.get("record_id"),
        "language": comment.payload.get("language"),
        "path": comment.payload.get("path"),
        "repo": comment.payload.get("repo"),
        "comment_license_score": comment.score,
        "comment_license_score_percent": comment.score_percent,
        "topic_probability": probability,
        "opening_comment": _truncate_text(comment.text, max_chars=max_chars),
    }


def _topic_summary_for_judge(
    summary: dict[str, Any],
    *,
    sample_size: int,
) -> dict[str, Any]:
    examples = list(summary.get("representative_comments") or [])[:sample_size]
    return {
        "topic_id": summary.get("topic_id"),
        "label": summary.get("label"),
        "count": summary.get("count"),
        "keywords": summary.get("keywords"),
        "sample_comments": [
            {
                **example,
                "opening_comment": _truncate_text(
                    str(example.get("opening_comment") or ""),
                    max_chars=_DEFAULT_JUDGE_MAX_COMMENT_CHARS,
                ),
            }
            for example in examples
        ],
    }


def _build_codex_judge_prompt(
    clusters: Sequence[dict[str, Any]],
    *,
    input_directory: Path,
    score_threshold: float,
    normalized_score_threshold: float,
    records_selected: int,
) -> str:
    payload = {
        "input_directory": str(input_directory),
        "score_threshold": score_threshold,
        "normalized_score_threshold": normalized_score_threshold,
        "records_selected": records_selected,
        "clusters": clusters,
    }
    return "\n".join(
        [
            "You are validating BERTopic clusters of code opening comments.",
            "",
            "Context:",
            "- The comments were selected because their ScanCode comment_license_score is below the configured threshold.",
            "- Thresholds may be entered as 0.95 or 95; this run uses the normalized 0-100 threshold shown in the JSON.",
            "- Treat topic -1 as BERTopic outliers; outliers are not included below.",
            "",
            "Task:",
            "For each cluster, judge whether the sample comments form a coherent and useful topic.",
            "Use only the JSON data in this prompt. Do not inspect local files.",
            "",
            "Return only a JSON object with this shape:",
            "{",
            '  "overall_assessment": "short summary",',
            '  "clusters": [',
            "    {",
            '      "topic_id": 0,',
            '      "valid_cluster": true,',
            '      "coherence_score": 0.0,',
            '      "suggested_label": "short label",',
            '      "rationale": "why the examples do or do not belong together",',
            '      "weak_or_off_topic_ordinals": [1, 2]',
            "    }",
            "  ]",
            "}",
            "",
            "Cluster data:",
            json.dumps(_json_safe(payload), indent=2),
        ]
    )


def _run_codex_exec(
    prompt: str,
    *,
    codex_command: str,
    codex_model: str | None,
    timeout: int,
    cwd: Path,
) -> str:
    command = [
        *shlex.split(codex_command),
        "exec",
        "--ephemeral",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
    ]
    if codex_model:
        command.extend(["--model", codex_model])
    command.append("-")
    try:
        completed = subprocess.run(
            command,
            input=prompt,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Codex command not found: {codex_command}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Codex judge timed out after {timeout} seconds") from exc
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        detail = stderr or stdout or f"exit status {completed.returncode}"
        raise RuntimeError(f"Codex judge failed: {detail}")
    return completed.stdout.strip()


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    for candidate in (stripped, _substring_between(stripped, "{", "}")):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _substring_between(text: str, start: str, end: str) -> str | None:
    start_index = text.find(start)
    end_index = text.rfind(end)
    if start_index < 0 or end_index < start_index:
        return None
    return text[start_index : end_index + 1]


def _truncate_text(text: str, *, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."
