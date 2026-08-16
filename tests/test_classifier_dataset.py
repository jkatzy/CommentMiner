from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import tempfile
import unittest
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from commentminer.classifier_dataset import (
    _AFFIRMATIVE_SHARING_CONTEXT,
    LABEL_IRRELEVANT,
    LABEL_MISSED_LICENSE,
    LABEL_SHARING_RESTRICTION,
    _BoundedCandidatePool,
    _Candidate,
    _TemplateFamilyIndex,
    ClassifierDatasetStats,
    _allocate_global_candidates,
    _allocate_scarcity_aware_candidates,
    _canonicalize_decision_evidence,
    _decision_semantic_signature,
    _final_selection_key,
    _hard_irrelevant_features,
    _leakage_aware_split_assignments,
    _near_duplicate_template,
    _near_duplicate_whole_template,
    _normalized_comment_template,
    _max_min_fair_limits,
    _select_exact_hash_diverse_candidates,
    _select_diverse_rows,
    _select_global_diverse_rows,
    _scan_candidates,
    _template_family_markers,
    _template_shingles,
    _whole_template_word_trigrams,
    build_classifier_dataset,
    verify_classifier_dataset,
)


_INPUT_SCHEMA = pa.schema(
    [
        ("dataset", pa.string()),
        ("record_id", pa.string()),
        ("opening_comment", pa.string()),
        ("language", pa.string()),
        ("path", pa.string()),
        ("repo", pa.string()),
        ("extracted_at", pa.string()),
        ("metadata", pa.string()),
        ("comment_license_detection", pa.string()),
        ("comment_license_score", pa.float64()),
    ]
)


def _row(
    record_id: str,
    comment: str,
    *,
    dataset: str = "ds",
    language: str = "Python",
    repo: str = "example/repo",
    score: float = 0.0,
    contains_license_notice: bool = False,
) -> dict[str, object]:
    detection = {
        "best_license_score": score,
        "contains_license_notice": contains_license_notice,
        "detected_license_expression": "mit" if contains_license_notice else None,
        "detected_license_expression_spdx": (
            "MIT" if contains_license_notice else None
        ),
        "license_matches": [],
        "percentage_of_license_text": 100.0 if contains_license_notice else 0.0,
        "scan_errors": [],
    }
    return {
        "dataset": dataset,
        "record_id": record_id,
        "opening_comment": comment,
        "language": language,
        "path": f"src/{record_id}.py",
        "repo": repo,
        "extracted_at": "2026-07-16T00:00:00+00:00",
        "metadata": "{}",
        "comment_license_detection": json.dumps(detection, sort_keys=True),
        "comment_license_score": score,
    }


def _write_combination(
    root: Path,
    dataset: str,
    language: str,
    rows: Iterable[dict[str, object]],
    *,
    shard: int = 0,
) -> Path:
    path = root / dataset / language / f"part-{shard:05d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(list(rows), schema=_INPUT_SCHEMA), path)
    return path


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def _candidate_fixture(
    example_id: str,
    *,
    comment_hash: str,
    template_hash: str,
    priority: float,
    dataset: str = "ds",
    candidate_class: str = LABEL_IRRELEVANT,
) -> _Candidate:
    return _Candidate(
        example_id=example_id,
        comment_hash=comment_hash,
        template_hash=template_hash,
        candidate_class=candidate_class,
        heuristic_score=priority,
        matched_terms=["scancode_zero_random_negative"],
        selection_priority=priority,
        dataset=dataset,
        record_id=example_id,
        language="Python",
        path=f"{example_id}.py",
        repo=f"org/{example_id}",
        source_path="part-00000.parquet",
        source_row_index=0,
        opening_comment=f"Comment for {example_id}",
        comment_license_score=0.0,
        comment_license_contains_notice=False,
        comment_license_expression=None,
        comment_license_detection='{"contains_license_notice": false}',
    )


class _ScriptedJudge:
    """Return one structured decision for every candidate in a judge batch."""

    def __init__(
        self,
        votes: dict[str, str | list[str]],
        *,
        invalid_invariant_markers: Iterable[str] = (),
        semantic_overrides: dict[str, dict[str, object]] | None = None,
        coverage_error: str | None = None,
    ) -> None:
        self._votes = {
            marker: [labels] if isinstance(labels, str) else list(labels)
            for marker, labels in votes.items()
        }
        self._invalid_invariant_markers = set(invalid_invariant_markers)
        self._semantic_overrides = semantic_overrides or {}
        self._coverage_error = coverage_error
        self._calls_by_marker: dict[str, int] = defaultdict(int)
        self.calls: list[str] = []

    def __call__(self, *args: object, **kwargs: object) -> str:
        # Accept either a prompt string or a structured judge request. Candidate
        # IDs are deliberately read from the request rather than reconstructed
        # from provenance, so the response must target the builder's exact IDs.
        if len(args) == 1 and isinstance(args[0], str) and not kwargs:
            request = args[0]
        else:
            request = json.dumps(
                {"args": args, "kwargs": kwargs},
                default=str,
                ensure_ascii=False,
                sort_keys=True,
            )
        markers = [marker for marker in self._votes if marker in request]
        if not markers:
            raise AssertionError(
                f"judge batch contained no scripted candidates: {request[:500]!r}"
            )

        id_matches = list(
            re.finditer(
                r'''["']candidate_id["']\s*:\s*["']([^"']+)["']''',
                request,
            )
        )
        if not id_matches:
            raise AssertionError(
                f"judge prompt did not expose candidate_id values: {request[:500]!r}"
            )

        decisions: list[dict[str, object]] = []
        for marker in markers:
            marker_offset = request.index(marker)
            id_match = min(
                id_matches,
                key=lambda match: abs(match.start() - marker_offset),
            )
            candidate_id = id_match.group(1)
            vote_index = self._calls_by_marker[marker]
            scripted_votes = self._votes[marker]
            vote = (
                scripted_votes[-1]
                if vote_index >= len(scripted_votes)
                else scripted_votes[vote_index]
            )
            self._calls_by_marker[marker] += 1
            self.calls.append(marker)
            decisions.append(
                self._decision(
                    candidate_id,
                    marker,
                    vote,
                    invalidate=marker in self._invalid_invariant_markers,
                    semantic_override=self._semantic_overrides.get(marker, {}),
                )
            )

        if self._coverage_error == "omit_last":
            decisions.pop()
        elif self._coverage_error == "duplicate_first" and decisions:
            decisions.append(dict(decisions[0]))
        return json.dumps({"decisions": decisions})

    @staticmethod
    def _decision(
        candidate_id: str,
        marker: str,
        label: str,
        *,
        invalidate: bool,
        semantic_override: dict[str, object],
    ) -> dict[str, object]:
        if label == LABEL_SHARING_RESTRICTION:
            is_sharing_restriction = True
            is_license_notice = False
        elif label == LABEL_MISSED_LICENSE:
            is_sharing_restriction = False
            is_license_notice = True
        elif label == LABEL_IRRELEVANT:
            is_sharing_restriction = False
            is_license_notice = False
        else:
            is_sharing_restriction = False
            is_license_notice = False
        if invalidate:
            is_sharing_restriction = False
        decision = {
            "candidate_id": candidate_id,
            "label": label,
            "confidence": 0.99,
            "is_sharing_restriction": is_sharing_restriction,
            "is_license_notice": is_license_notice,
            "is_known_license": False,
            "known_license": None,
            "evidence": marker,
            "rationale": f"scripted judgment for {marker}",
        }
        decision.update(semantic_override)
        return decision


def _base_rows() -> list[dict[str, object]]:
    restriction = (
        "[R-KEEP] ACME CONFIDENTIAL and proprietary source code. "
        "Do not share, disclose, copy, or distribute it outside ACME."
    )
    return [
        # A high numeric score must not hide a ScanCode miss. The structured
        # contains flag, which also accounts for match coverage, is authoritative.
        _row("restriction", restriction, score=100.0),
        _row(
            "missed-license",
            "[L-KEEP]\nCopyright 2024 Acme.\nThis software is licensed under "
            "the Acme Research License; permission is granted to use and copy it. "
            "The software is provided without warranty of any kind, express or implied.",
            score=100.0,
        ),
        _row(
            "irrelevant",
            "[I-KEEP] Initialize the parser and return the first syntax node.",
            score=0.0,
        ),
        # Exact comment text is duplicated under different provenance. It must
        # be judged and emitted once, not allowed to dominate the training set.
        _row("restriction-duplicate", restriction, repo="fork/of-example"),
        # Boilerplate that differs only in attribution/year is one semantic
        # template and must not leak across examples or splits.
        _row(
            "missed-license-template-duplicate",
            "[L-KEEP]\nCopyright 2025 Other Corp.\nThis software is licensed under "
            "the Acme Research License; permission is granted to use and copy it. "
            "The software is provided without warranty of any kind, express or implied.",
            score=100.0,
            repo="another/fork",
        ),
        # Conversely, a low numeric score must not admit a notice that ScanCode
        # has already recognized.
        _row(
            "known-license-low-score",
            "[KNOWN-SKIP] MIT License. Permission is hereby granted, free of charge.",
            score=0.0,
            contains_license_notice=True,
        ),
    ]


class ClassifierDatasetTests(unittest.TestCase):
    maxDiff = None

    def test_bounded_pool_keeps_best_exact_hash_pair_representative(self) -> None:
        low = _candidate_fixture(
            "low",
            comment_hash="shared-comment",
            template_hash="shared-template",
            priority=1.0,
        )
        high = _candidate_fixture(
            "high",
            comment_hash="shared-comment",
            template_hash="shared-template",
            priority=9.0,
        )
        other = _candidate_fixture(
            "other",
            comment_hash="comment-other",
            template_hash="other-template",
            priority=5.0,
        )
        memberships = []
        for order in ((low, other, high), (high, other, low)):
            pool = _BoundedCandidatePool(limit=2)
            for candidate in order:
                pool.add(candidate)
            memberships.append([candidate.example_id for candidate in pool.ranked()])
        self.assertEqual(memberships, [["high", "other"], ["high", "other"]])

    def test_bounded_pool_keeps_crossed_hash_alternatives_deterministically(self) -> None:
        candidates = (
            _candidate_fixture(
                "a",
                comment_hash="comment-x",
                template_hash="template-a",
                priority=8.0,
            ),
            _candidate_fixture(
                "b",
                comment_hash="comment-y",
                template_hash="template-b",
                priority=7.0,
            ),
            _candidate_fixture(
                "d",
                comment_hash="comment-y",
                template_hash="template-a",
                priority=6.0,
            ),
            _candidate_fixture(
                "c",
                comment_hash="comment-x",
                template_hash="template-b",
                priority=9.0,
            ),
        )
        memberships = []
        for order in (candidates, tuple(reversed(candidates))):
            pool = _BoundedCandidatePool(limit=4)
            for candidate in order:
                pool.add(candidate)
            memberships.append([candidate.example_id for candidate in pool.ranked()])
        self.assertEqual(memberships, [["c", "a", "b", "d"]] * 2)

    def test_bounded_pool_compacts_stale_representatives_by_active_size(self) -> None:
        pool = _BoundedCandidatePool(limit=3_000)
        for index in range(1_000):
            pool.add(
                _candidate_fixture(
                    f"candidate-{index}",
                    comment_hash="shared-comment",
                    template_hash="shared-template",
                    priority=float(index),
                )
            )
        self.assertEqual(
            [candidate.example_id for candidate in pool.ranked()],
            ["candidate-999"],
        )
        self.assertLessEqual(len(pool._heap), 64)

    def test_exact_hash_matching_keeps_a_feasible_crossed_pair(self) -> None:
        candidates = (
            _candidate_fixture(
                "a",
                comment_hash="comment-x",
                template_hash="template-a",
                priority=10.0,
            ),
            _candidate_fixture(
                "c",
                comment_hash="comment-x",
                template_hash="template-b",
                priority=9.0,
            ),
            _candidate_fixture(
                "d",
                comment_hash="comment-y",
                template_hash="template-a",
                priority=8.0,
            ),
        )
        selections = []
        for order in (candidates, tuple(reversed(candidates))):
            selections.append(
                [
                    candidate.example_id
                    for candidate in _select_exact_hash_diverse_candidates(order)
                ]
            )
        self.assertEqual(selections, [["c", "d"], ["c", "d"]])

    def test_exact_hash_matching_handles_a_long_alternating_path_iteratively(self) -> None:
        candidates = [
            _candidate_fixture(
                "comment-0000-template-0000",
                comment_hash="comment-0000",
                template_hash="template-0000",
                priority=2.0,
            )
        ]
        for index in range(1, 1_100):
            comment_hash = f"comment-{index:04d}"
            candidates.extend(
                (
                    _candidate_fixture(
                        f"{comment_hash}-previous",
                        comment_hash=comment_hash,
                        template_hash=f"template-{index - 1:04d}",
                        priority=2.0,
                    ),
                    _candidate_fixture(
                        f"{comment_hash}-own",
                        comment_hash=comment_hash,
                        template_hash=f"template-{index:04d}",
                        priority=1.0,
                    ),
                )
            )
        selected = _select_exact_hash_diverse_candidates(candidates)
        self.assertEqual(len(selected), 1_100)

    def test_scarcity_allocator_is_order_independent_and_avoids_shortfall(self) -> None:
        shared_a = _candidate_fixture(
            "shared-a",
            comment_hash="shared-comment",
            template_hash="shared-template-a",
            priority=10.0,
            dataset="a",
        )
        distinct_a = _candidate_fixture(
            "distinct-a",
            comment_hash="distinct-comment",
            template_hash="distinct-template",
            priority=5.0,
            dataset="a",
        )
        shared_b = _candidate_fixture(
            "shared-b",
            comment_hash="shared-comment",
            template_hash="shared-template-b",
            priority=10.0,
            dataset="b",
        )
        a_key = ("a", "Python", LABEL_IRRELEVANT)
        b_key = ("b", "Python", LABEL_IRRELEVANT)
        allocations = []
        for options in (
            {a_key: [shared_a, distinct_a], b_key: [shared_b]},
            {b_key: [shared_b], a_key: [shared_a, distinct_a]},
        ):
            allocated = _allocate_scarcity_aware_candidates(options, limit=1)
            allocations.append(
                {
                    key: [candidate.example_id for candidate in candidates]
                    for key, candidates in allocated.items()
                }
            )
        expected = {a_key: ["distinct-a"], b_key: ["shared-b"]}
        self.assertEqual(allocations, [expected, expected])

    def test_scarcity_allocator_uses_augmenting_paths_to_fill_feasible_quotas(self) -> None:
        keys = [
            (f"pool-{index}", "Python", LABEL_IRRELEVANT)
            for index in range(4)
        ]

        def candidate(pool: int, comment: str, priority: float) -> _Candidate:
            return _candidate_fixture(
                f"pool-{pool}-{comment}",
                comment_hash=comment,
                template_hash=f"template-{pool}-{comment}",
                priority=priority,
                dataset=f"pool-{pool}",
            )

        options = {
            keys[0]: [candidate(0, "one", 10.0), candidate(0, "three", 9.0)],
            keys[1]: [candidate(1, "one", 10.0), candidate(1, "two", 9.0)],
            keys[2]: [candidate(2, "one", 10.0), candidate(2, "two", 9.0)],
            keys[3]: [candidate(3, "three", 10.0), candidate(3, "four", 9.0)],
        }
        allocated = _allocate_scarcity_aware_candidates(options, limit=1)
        self.assertTrue(all(len(allocated[key]) == 1 for key in keys), allocated)
        selected_hashes = [
            candidate.comment_hash
            for candidates in allocated.values()
            for candidate in candidates
        ]
        self.assertEqual(len(selected_hashes), len(set(selected_hashes)))

    def test_scarcity_allocator_spreads_unavoidable_shortfalls_across_pools(self) -> None:
        keys = [
            ("pool-a", "Python", LABEL_IRRELEVANT),
            ("pool-b", "Python", LABEL_IRRELEVANT),
        ]
        options = {
            key: [
                _candidate_fixture(
                    f"{key[0]}-{comment_hash}",
                    comment_hash=comment_hash,
                    template_hash=f"{key[0]}-{comment_hash}",
                    priority=priority,
                    dataset=key[0],
                )
                for comment_hash, priority in (("x", 2.0), ("y", 1.0))
            ]
            for key in keys
        }
        allocated = _allocate_scarcity_aware_candidates(options, limit=2)
        self.assertEqual([len(allocated[key]) for key in keys], [1, 1])

    def test_scarcity_allocator_handles_a_long_alternating_path_iteratively(self) -> None:
        options = {}
        for index in range(1_100):
            key = (f"pool-{index:04d}", "Python", LABEL_IRRELEVANT)
            options[key] = [
                _candidate_fixture(
                    f"pool-{index:04d}-first",
                    comment_hash=f"comment-{index:04d}",
                    template_hash=f"first-template-{index:04d}",
                    priority=2.0,
                    dataset=key[0],
                ),
                _candidate_fixture(
                    f"pool-{index:04d}-second",
                    comment_hash=f"comment-{index + 1:04d}",
                    template_hash=f"second-template-{index:04d}",
                    priority=1.0,
                    dataset=key[0],
                ),
            ]
        final_key = ("pool-z", "Python", LABEL_IRRELEVANT)
        options[final_key] = [
            _candidate_fixture(
                "pool-z-first",
                comment_hash="comment-0000",
                template_hash="pool-z-first",
                priority=2.0,
                dataset=final_key[0],
            ),
            _candidate_fixture(
                "pool-z-second",
                comment_hash="comment-1100",
                template_hash="pool-z-second",
                priority=1.0,
                dataset=final_key[0],
            ),
        ]
        allocated = _allocate_scarcity_aware_candidates(options, limit=1)
        self.assertTrue(all(len(candidates) == 1 for candidates in allocated.values()))

    def test_max_min_limits_top_up_after_scarce_cells(self) -> None:
        keys = [
            (f"pool-{index}", "Python", LABEL_IRRELEVANT)
            for index in range(3)
        ]
        capacities = {
            keys[0]: 0,
            keys[1]: 2,
            keys[2]: 10,
        }
        self.assertEqual(
            _max_min_fair_limits(capacities, total=8),
            {
                keys[0]: 0,
                keys[1]: 2,
                keys[2]: 6,
            },
        )

    def test_global_allocator_is_unique_fair_and_deterministic(self) -> None:
        keys = [
            ("pool-a", "Python", LABEL_IRRELEVANT),
            ("pool-b", "Python", LABEL_IRRELEVANT),
        ]

        def candidate(key: tuple[str, str, str], value: str, priority: float) -> _Candidate:
            return _candidate_fixture(
                f"{key[0]}-{value}",
                comment_hash=value,
                template_hash=f"{key[0]}-{value}",
                priority=priority,
                dataset=key[0],
            )

        options = {
            keys[0]: [
                candidate(keys[0], "shared", 10.0),
                candidate(keys[0], "a-1", 9.0),
                candidate(keys[0], "a-2", 8.0),
            ],
            keys[1]: [
                candidate(keys[1], "shared", 10.0),
                candidate(keys[1], "b-1", 9.0),
                candidate(keys[1], "b-2", 8.0),
            ],
        }
        snapshots = []
        for ordered in (options, dict(reversed(tuple(options.items())))):
            allocated = _allocate_global_candidates(
                ordered,
                targets_by_label={LABEL_IRRELEVANT: 4},
            )
            snapshots.append(
                {
                    key: [candidate.example_id for candidate in allocated[key]]
                    for key in keys
                }
            )
            hashes = [
                candidate.comment_hash
                for candidates in allocated.values()
                for candidate in candidates
            ]
            self.assertEqual(len(hashes), len(set(hashes)))
            self.assertEqual([len(allocated[key]) for key in keys], [2, 2])
        self.assertEqual(snapshots[0], snapshots[1])

    def test_global_allocator_prioritizes_a_class_without_fallback(self) -> None:
        irrelevant_key = ("pool", "Python", LABEL_IRRELEVANT)
        sharing_key = ("pool", "Python", LABEL_SHARING_RESTRICTION)
        options = {
            irrelevant_key: [
                _candidate_fixture(
                    "irrelevant-x",
                    comment_hash="x",
                    template_hash="irrelevant-x",
                    priority=2.0,
                ),
                _candidate_fixture(
                    "irrelevant-y",
                    comment_hash="y",
                    template_hash="irrelevant-y",
                    priority=1.0,
                ),
            ],
            sharing_key: [
                _candidate_fixture(
                    "sharing-x",
                    comment_hash="x",
                    template_hash="sharing-x",
                    priority=2.0,
                    candidate_class=LABEL_SHARING_RESTRICTION,
                )
            ],
        }
        allocated = _allocate_global_candidates(
            options,
            targets_by_label={
                LABEL_IRRELEVANT: 2,
                LABEL_SHARING_RESTRICTION: 1,
            },
        )
        self.assertEqual(
            [candidate.example_id for candidate in allocated[sharing_key]],
            ["sharing-x"],
        )
        self.assertEqual(
            [candidate.example_id for candidate in allocated[irrelevant_key]],
            ["irrelevant-y"],
        )

    def test_global_final_selection_is_fair_after_cell_diversity(self) -> None:
        scarce = ("scarce", "Python", LABEL_IRRELEVANT)
        rich = ("rich", "Python", LABEL_IRRELEVANT)

        def row(example_id: str, dataset: str, text: str, priority: float) -> dict[str, Any]:
            return {
                "example_id": example_id,
                "dataset": dataset,
                "language": "Python",
                "candidate_class": LABEL_IRRELEVANT,
                "opening_comment": text,
                "template_hash": hashlib.sha256(
                    _normalized_comment_template(text).encode()
                ).hexdigest(),
                "selection_priority": priority,
                "matched_terms": ["scancode_zero_random_negative"],
            }

        shared_text = (
            "Technical parser documentation with enough unique material "
            "for a stable template family."
        )
        provisionally_accepted = {
            scarce: [row("scarce-1", "scarce", "Parse a custom wire format.", 1.0)],
            rich: [
                row("rich-shared-1", "rich", shared_text, 20.0),
                row("rich-shared-2", "rich", shared_text, 19.0),
                row("rich-a", "rich", "Initialize the XML parser and return its root.", 10.0),
                row("rich-b", "rich", "Compute orbital checksums before packet buffering.", 9.0),
                row("rich-c", "rich", "Render the sidebar after theme configuration loads.", 8.0),
                row("rich-d", "rich", "Release database handles during worker shutdown.", 7.0),
                row("rich-e", "rich", "Encode geographic coordinates for the cache key.", 6.0),
                row("rich-f", "rich", "Validate audio frames against the stream header.", 5.0),
            ],
        }
        selected, duplicates, excess, limits = _select_global_diverse_rows(
            provisionally_accepted,
            combinations=[("scarce", "Python"), ("rich", "Python")],
            target_per_class=5,
        )
        counts = defaultdict(int)
        for selected_row in selected:
            counts[selected_row["dataset"]] += 1
        self.assertEqual(dict(counts), {"scarce": 1, "rich": 4})
        self.assertEqual(len(duplicates), 1)
        self.assertEqual(len(excess), 3)
        self.assertEqual(limits[scarce], 1)
        self.assertEqual(limits[rich], 4)

    def test_scan_exposes_fallback_beyond_four_shared_options(self) -> None:
        shared_comments = (
            "MIT License applies to this component.",
            "Apache License applies to this component.",
            "BSD 3-Clause License applies to this component.",
            "ISC License applies to this component.",
        )
        combinations = [(f"pool-{index}", "Python") for index in range(5)]
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_directory = root / "input"
            for index, (dataset, language) in enumerate(combinations):
                rows = [
                    _row(
                        f"shared-{shared_index}",
                        comment,
                        dataset=dataset,
                        language=language,
                    )
                    for shared_index, comment in enumerate(shared_comments)
                ]
                rows.append(
                    _row(
                        "unique-fallback",
                        "This software is provided AS IS for custom component "
                        f"{index}.",
                        dataset=dataset,
                        language=language,
                    )
                )
                _write_combination(input_directory, dataset, language, rows)

            candidates, _ = _scan_candidates(
                input_directory,
                combinations=combinations,
                target_per_combination=1,
                candidate_multiplier=1,
                max_shards_per_combination=1,
                batch_size=32,
                min_comment_chars=12,
                max_comment_chars=12000,
                seed=42,
                stats=ClassifierDatasetStats(
                    input_directory=input_directory,
                    output_directory=root / "output",
                    combinations_requested=len(combinations),
                ),
            )

        license_candidates = [
            candidate
            for candidate in candidates
            if candidate.candidate_class == LABEL_MISSED_LICENSE
        ]
        self.assertEqual(len(license_candidates), len(combinations))
        self.assertEqual(
            {candidate.dataset for candidate in license_candidates},
            {dataset for dataset, _ in combinations},
        )
        self.assertEqual(
            len({candidate.comment_hash for candidate in license_candidates}),
            len(license_candidates),
        )

    def test_known_license_consensus_normalizes_spdx_name_variants(self) -> None:
        equivalent_pairs = (
            (
                "GNU General Public License v2-or-later",
                "GPL-2.0-or-later",
            ),
            ("GNU Lesser General Public License 3.0", "LGPL-3.0-only"),
            ("GNU Affero General Public License 3.0", "AGPL-3.0-only"),
            ("Apache License 2.0", "Apache-2.0"),
            (
                "GNU General Public License 3.0-or-later and BSD 3-Clause License",
                "GPL-3.0-or-later and BSD-3-Clause",
            ),
            ("GNU GPL version 2", "GPL-2.0-only"),
            ("GNU General Public License v2+", "GPL-2.0-or-later"),
            ("GPL v2 (or later)", "GPL-2.0-or-later"),
            (
                "GNU General Public License v2 with MySQL FOSS License Exception",
                "GNU General Public License version 2 with MySQL FOSS License Exception",
            ),
        )
        signatures = []
        for prose_name, spdx_name in equivalent_pairs:
            pair = []
            for name in (prose_name, spdx_name):
                pair.append(
                    _decision_semantic_signature(
                        {
                            "is_sharing_restriction": False,
                            "is_license_notice": True,
                            "is_known_license": True,
                            "known_license": name,
                        }
                    )
                )
            self.assertEqual(pair[0], pair[1])
            signatures.extend(pair)
        self.assertNotEqual(
            _decision_semantic_signature(
                {
                    "is_sharing_restriction": False,
                    "is_license_notice": True,
                    "is_known_license": True,
                    "known_license": "GPL-3.0-only",
                }
            ),
            _decision_semantic_signature(
                {
                    "is_sharing_restriction": False,
                    "is_license_notice": True,
                    "is_known_license": True,
                    "known_license": "LGPL-3.0-only",
                }
            ),
        )
        self.assertNotEqual(
            _decision_semantic_signature(
                {
                    "is_sharing_restriction": False,
                    "is_license_notice": True,
                    "is_known_license": True,
                    "known_license": "GPL-2.0+",
                }
            ),
            _decision_semantic_signature(
                {
                    "is_sharing_restriction": False,
                    "is_license_notice": True,
                    "is_known_license": True,
                    "known_license": "GPL-2.0-only",
                }
            ),
        )
        self.assertNotEqual(
            _decision_semantic_signature(
                {
                    "is_sharing_restriction": False,
                    "is_license_notice": True,
                    "is_known_license": True,
                    "known_license": "C# License",
                }
            ),
            _decision_semantic_signature(
                {
                    "is_sharing_restriction": False,
                    "is_license_notice": True,
                    "is_known_license": True,
                    "known_license": "C++ License",
                }
            ),
        )
        self.assertNotEqual(
            _decision_semantic_signature(
                {
                    "is_sharing_restriction": False,
                    "is_license_notice": True,
                    "is_known_license": True,
                    "known_license": "许可证 MIT License",
                }
            ),
            _decision_semantic_signature(
                {
                    "is_sharing_restriction": False,
                    "is_license_notice": True,
                    "is_known_license": True,
                    "known_license": "MIT License",
                }
            ),
        )
        self.assertTrue(signatures)

    def test_evidence_matching_rejects_word_internal_substrings(self) -> None:
        candidate = _Candidate(
            example_id="evidence-test",
            comment_hash="hash",
            template_hash="template",
            candidate_class=LABEL_IRRELEVANT,
            heuristic_score=0.0,
            matched_terms=[],
            selection_priority=0.0,
            dataset="ds",
            record_id="record",
            language="Python",
            path=None,
            repo=None,
            source_path="input.parquet",
            source_row_index=0,
            opening_comment="Access is UNAUTHORIZED for external users.",
            comment_license_score=0.0,
            comment_license_contains_notice=False,
            comment_license_expression=None,
            comment_license_detection="{}",
        )
        bad = {"evidence": "authorized"}
        self.assertFalse(
            _canonicalize_decision_evidence(
                bad,
                candidate,
                max_comment_chars=len(candidate.opening_comment),
            )
        )
        good = {"evidence": "unauthorized"}
        self.assertTrue(
            _canonicalize_decision_evidence(
                good,
                candidate,
                max_comment_chars=len(candidate.opening_comment),
            )
        )
        self.assertEqual(good["evidence"], "UNAUTHORIZED")
        self.assertFalse(
            _canonicalize_decision_evidence(
                {"evidence": "   "},
                candidate,
                max_comment_chars=len(candidate.opening_comment),
            )
        )

    def test_template_normalization_collapses_nonsemantic_header_variants(self) -> None:
        apache_terms = (
            "This software is licensed to you under the terms of the Apache License, "
            "Version 2.0. You may obtain a copy of the License at "
            "https://www.apache.org/licenses/LICENSE-2.0."
        )
        first = (
            "Name: app.js\nPurpose: main application\nHistory: copied text to the "
            f"clipboard\n{apache_terms}"
        )
        second = (
            "Name: Controller.js\nPurpose: controller page\nHistory: initial commit\n"
            f"{apache_terms}"
        )
        self.assertEqual(
            _normalized_comment_template(first),
            _normalized_comment_template(second),
        )

        technical_first = (
            "eslint-disable complexity; eslint-disable global-require; "
            "eslint-disable import/no-dynamic-require\n"
            "DO NOT COPY THIS FILE INTO THE example REPO.\n"
            "Instead, make it a package and let the project depend on it."
        )
        technical_second = (
            "eslint global-require: off; eslint no-sync: off; eslint "
            "require-jsdoc: off; another technical directive\n"
            "DO NOT COPY THIS FILE INTO THE example REPO.\n"
            "Instead, make it a package and let the project depend on it."
        )
        self.assertEqual(
            _normalized_comment_template(technical_first),
            _normalized_comment_template(technical_second),
        )

        short_mit_first = (
            "Licensed under the MIT License at:\nhttps://opensource.org/licenses/MIT\n"
            "@author Brian Cavalier\n@author John Hann"
        )
        short_mit_second = (
            "Licensed under the MIT License at:\nhttps://opensource.org/licenses/MIT\n"
            "@author: Brian Cavalier\n@author: John Hann"
        )
        self.assertEqual(
            _normalized_comment_template(short_mit_first),
            _normalized_comment_template(short_mit_second),
        )
        property_first = (
            "flixey.com ajax library\nThis script is the property of flixey.com, "
            "you may not copy, view, or distribute this script in any form without "
            "prior consent of the administrator."
        )
        property_second = (
            "flv.niya.cc base script\nThis script is the property of niya.cc, you "
            "may not copy, view, or distribute this script in any form without prior "
            "consent of the administrator."
        )
        self.assertTrue(
            _near_duplicate_template(
                _template_shingles(property_first),
                _template_shingles(property_second),
            )
        )
        no_comma_restriction = (
            "This script is the property of ACME. Do not distribute externally."
        )
        no_comma_irrelevant = (
            "This script is the property of Example. Public use is allowed."
        )
        self.assertNotEqual(
            _normalized_comment_template(no_comma_restriction),
            _normalized_comment_template(no_comma_irrelevant),
        )
        legal_footer = (
            "Copyright 2009-2016 by Codility Limited. All Rights Reserved. "
            "Unauthorized copying, publication or disclosure prohibited."
        )
        self.assertTrue(
            _near_duplicate_template(
                _template_shingles(
                    "A frog crosses a river using an array of leaves. " + legal_footer
                ),
                _template_shingles(
                    "Find an equilibrium index in an integer array. " + legal_footer
                ),
            )
        )

        mit_grant = (
            "Permission is hereby granted, free of charge, to any person obtaining "
            "a copy of this software under the MIT License."
        )
        mixed_prefix = (
            "Only authorized ACME employees may access this source; external "
            f"recipients are forbidden.\n{mit_grant}"
        )
        pure_license = f"File: alpha.py\nOwner: Example\n{mit_grant}"
        self.assertNotEqual(
            _normalized_comment_template(mixed_prefix),
            _normalized_comment_template(pure_license),
        )
        semantic_metadata = (
            "Purpose: ACME confidential; authorized employees only; do not share "
            f"externally\n{mit_grant}"
        )
        ordinary_metadata = f"Purpose: ordinary component\n{mit_grant}"
        self.assertNotEqual(
            _normalized_comment_template(semantic_metadata),
            _normalized_comment_template(ordinary_metadata),
        )
        license_metadata = (
            "Purpose: Licensed under GPL-2.0-only\n"
            "This module provides utilities for parsing JSON payloads."
        )
        ordinary_purpose = (
            "Purpose: ordinary component\n"
            "This module provides utilities for parsing JSON payloads."
        )
        self.assertNotEqual(
            _normalized_comment_template(license_metadata),
            _normalized_comment_template(ordinary_purpose),
        )
        mixed_suffix = (
            f"{mit_grant}\nUsage: This source is confidential and may not be "
            "shared outside ACME."
        )
        public_suffix = f"{mit_grant}\nUsage: Public use is allowed."
        self.assertNotEqual(
            _normalized_comment_template(mixed_suffix),
            _normalized_comment_template(public_suffix),
        )
        classified_suffix = (
            f"{mit_grant}\nUsage: Classified information; official use only."
        )
        self.assertNotEqual(
            _normalized_comment_template(classified_suffix),
            _normalized_comment_template(pure_license),
        )

    def test_near_template_families_share_a_split_across_repositories(self) -> None:
        common = (
            "CONFIDENTIAL. All information is proprietary. Dissemination or "
            "reproduction is strictly forbidden without prior written permission."
        )
        rows = [
            {
                "example_id": "first",
                "repo": "org/first",
                "opening_comment": f"Copyright 2024 Acme.\n{common}",
            },
            {
                "example_id": "second",
                "repo": "fork/second",
                "opening_comment": f"Copyright 2025 Other Holder.\n{common}",
            },
            {
                "example_id": "third",
                "repo": "org/first",
                "opening_comment": "Parse the payload and return its length.",
            },
        ]
        assignments = _leakage_aware_split_assignments(rows, seed=42)
        self.assertEqual(assignments["first"], assignments["second"])
        self.assertEqual(assignments["first"], assignments["third"])

    def test_boilerplate_markers_group_stock_text_but_not_license_mentions(self) -> None:
        mit_header_one = (
            "Licensed under the MIT License at: "
            "https://opensource.org/licenses/MIT. Author: One."
        )
        mit_header_two = (
            "Licensed under the MIT Licence at: "
            "https://opensource.org/licenses/MIT. Author: Two."
        )
        marker = "boilerplate:mit-reference-header"
        self.assertIn(marker, _template_family_markers(mit_header_one))
        self.assertIn(marker, _template_family_markers(mit_header_two))
        self.assertIn(
            marker,
            _template_family_markers(
                "Licensed under MIT License\n"
                "http://www.opensource.org/licenses/mit-license.php"
            ),
        )
        self.assertFalse(_template_family_markers("Supports MIT licensed plugins."))
        self.assertFalse(
            _template_family_markers("Convert GPL metadata to an SPDX identifier.")
        )

        gpl_header = (
            "This program is free software: you can redistribute it and/or modify "
            "it under the terms of the GNU General Public License as published by "
            "the Free Software Foundation."
        )
        self.assertIn(
            "boilerplate:gpl-redistribution-header",
            _template_family_markers(gpl_header),
        )
        self.assertIn(
            "boilerplate:free-software-redistribute-hope-useful",
            _template_family_markers(
                gpl_header
                + " This program is distributed in the hope that it will be useful."
            ),
        )
        decorated_proprietary = (
            "This software is the confidential and proprietary information\n"
            "* of Example Corp. You shall not disclose\n"
            "* such Confidential Information."
        )
        self.assertIn(
            "boilerplate:confidential-proprietary-nondisclosure",
            _template_family_markers(decorated_proprietary),
        )

        rows = [
            {
                "example_id": "mit-one",
                "repo": "org/one",
                "opening_comment": mit_header_one,
            },
            {
                "example_id": "mit-two",
                "repo": "org/two",
                "opening_comment": mit_header_two,
            },
        ]
        assignments = _leakage_aware_split_assignments(rows, seed=42)
        self.assertEqual(assignments["mit-one"], assignments["mit-two"])

    def test_whole_template_trigrams_catch_minimized_near_duplicates(self) -> None:
        first = (
            "The parser processes structured payloads from local files and reports "
            "malformed records to the caller with useful line and column details "
            "for debugging."
        )
        second = first.replace("structured", "serialized")
        self.assertFalse(
            _near_duplicate_template(
                _template_shingles(first),
                _template_shingles(second),
            )
        )
        self.assertTrue(
            _near_duplicate_whole_template(
                _whole_template_word_trigrams(first),
                _whole_template_word_trigrams(second),
            )
        )
        short = "one two three four five six seven eight nine ten eleven twelve thirteen"
        self.assertFalse(
            _near_duplicate_whole_template(
                _whole_template_word_trigrams(short),
                _whole_template_word_trigrams(short),
            )
        )

        rows = [
            {
                "example_id": "whole-first",
                "repo": "org/whole-first",
                "template_hash": "whole-first-template",
                "candidate_class": LABEL_IRRELEVANT,
                "selection_priority": 2.0,
                "matched_terms": ["scancode_zero_random_negative"],
                "opening_comment": first,
            },
            {
                "example_id": "whole-second",
                "repo": "org/whole-second",
                "template_hash": "whole-second-template",
                "candidate_class": LABEL_IRRELEVANT,
                "selection_priority": 1.0,
                "matched_terms": ["scancode_zero_random_negative"],
                "opening_comment": second,
            },
        ]
        selected, duplicates, excess = _select_diverse_rows(rows, limit=2)
        self.assertEqual([row["example_id"] for row in selected], ["whole-first"])
        self.assertEqual(
            [row["example_id"] for row in duplicates],
            ["whole-second"],
        )
        self.assertFalse(excess)

        assignments = _leakage_aware_split_assignments(rows, seed=42)
        self.assertEqual(assignments["whole-first"], assignments["whole-second"])

    def test_template_family_index_matches_brute_force_relations(self) -> None:
        comments = [
            (
                "The parser processes structured payloads from local files and "
                "reports malformed records to the caller with useful line and "
                "column details for debugging."
            ),
            "A completely unrelated short implementation note.",
            (
                "Confidential. All information is proprietary. Dissemination or "
                "reproduction is strictly forbidden without written permission."
            ),
            (
                "The parser processes serialized payloads from local files and "
                "reports malformed records to the caller with useful line and "
                "column details for debugging."
            ),
            (
                "Copyright 2026 Other Holder. Confidential. All information is "
                "proprietary. Dissemination or reproduction is strictly forbidden "
                "without written permission."
            ),
        ]
        index = _TemplateFamilyIndex()
        prior: list[
            tuple[
                frozenset[tuple[str, ...]],
                frozenset[tuple[str, str, str]],
            ]
        ] = []
        for comment in comments:
            family = _template_shingles(comment)
            whole_family = _whole_template_word_trigrams(comment)
            expected = {
                prior_index
                for prior_index, (prior_family, prior_whole_family) in enumerate(
                    prior
                )
                if _near_duplicate_template(family, prior_family)
                or _near_duplicate_whole_template(
                    whole_family, prior_whole_family
                )
            }
            self.assertEqual(
                index.related_indices(family, whole_family),
                expected,
            )
            index.add(family, whole_family)
            prior.append((family, whole_family))

        legacy_only = _TemplateFamilyIndex(include_whole=False)
        first_family = _template_shingles(comments[0])
        first_whole = _whole_template_word_trigrams(comments[0])
        legacy_only.add(first_family, first_whole)
        self.assertFalse(
            legacy_only.related_indices(
                _template_shingles(comments[3]),
                _whole_template_word_trigrams(comments[3]),
            )
        )

    def test_template_family_index_covers_exact_whole_template_threshold(self) -> None:
        prior = frozenset(
            (f"word-{index:02}", "shared", "trigram")
            for index in range(15)
        )
        omitted = set(sorted(prior)[:3])
        current = frozenset(
            (prior - omitted)
            | {
                (f"replacement-{index}", "unique", "trigram")
                for index in range(3)
            }
        )
        self.assertTrue(_near_duplicate_whole_template(current, prior))

        index = _TemplateFamilyIndex()
        index.add(frozenset(), prior)
        self.assertEqual(
            index.related_indices(frozenset(), current),
            {0},
        )

    def test_template_family_index_randomized_equivalence(self) -> None:
        rng = random.Random(20260717)

        def shingle_set(values: set[int], width: int) -> frozenset[tuple[str, ...]]:
            return frozenset(
                tuple([f"token-{value:03}"] * width)
                for value in values
            )

        index = _TemplateFamilyIndex()
        prior: list[
            tuple[
                frozenset[tuple[str, ...]],
                frozenset[tuple[str, str, str]],
            ]
        ] = []
        previous_family: set[int] = set()
        previous_whole: set[int] = set()
        universe = list(range(180))
        for position in range(120):
            if position and position % 3:
                family_values = set(previous_family)
                whole_values = set(previous_whole)
                for values in (family_values, whole_values):
                    remove_count = rng.randint(0, min(5, len(values)))
                    if remove_count:
                        values.difference_update(rng.sample(sorted(values), remove_count))
                    values.update(rng.sample(universe, rng.randint(0, 5)))
            else:
                family_values = set(
                    rng.sample(universe, rng.randint(1, 45))
                )
                whole_values = set(
                    rng.sample(universe, rng.randint(12, 45))
                )
            family = shingle_set(family_values, 5)
            whole_family = shingle_set(whole_values, 3)
            expected = {
                prior_index
                for prior_index, (prior_family, prior_whole_family) in enumerate(
                    prior
                )
                if _near_duplicate_template(family, prior_family)
                or _near_duplicate_whole_template(
                    whole_family, prior_whole_family
                )
            }
            self.assertEqual(
                index.related_indices(family, whole_family),
                expected,
            )
            index.add(family, whole_family)
            prior.append((family, whole_family))
            previous_family = family_values
            previous_whole = whole_values

    def test_leakage_markers_cover_minimized_published_families(self) -> None:
        rti_first = (
            "No duplications, whole or partial, manual or electronic, may be made "
            "without express written permission. This code contains trade secrets "
            "of Real-Time Innovations, Inc."
        )
        rti_second = (
            "No duplications, whole or partial, manual or electronic, may be made "
            "without express written permission. Any copies must display this "
            "notice unaltered. This code contains trade secrets of Real-Time "
            "Innovations, Inc."
        )
        riverbed_first = (
            "Copyright (c) 2017 Riverbed Technology, Inc. All rights reserved. "
            "This software is licensed under the terms and conditions of the MIT "
            "License accompanying the software (License). This software is "
            "distributed AS IS as set forth in the License."
        )
        riverbed_second = (
            "Copyright (c) 2017 Riverbed Technology, Inc. All rights reserved. "
            "This software is licensed under the terms and conditions of the MIT "
            "License set forth at https://opensource.org/licenses/MIT (License). "
            "This software is distributed AS IS as set forth in the License."
        )
        imocom_1140 = (
            "This software is the confidential and proprietary information of "
            "IMOCOM (Confidential Information). It may not be copied, reproduced, "
            "or disclosed without express written permission of IMOCOM."
        )
        china_9f77 = (
            "This software is confidential and proprietary information of China "
            "Telecom. It may not be copied or reproduced without express written "
            "permission of China Telecom."
        )
        stock_421 = (
            "This software is the confidential and proprietary information of "
            "Savant Systems LLC. You shall not disclose such Confidential "
            "Information except as permitted by the license agreement."
        )
        pentaho = (
            "All information including source code contained herein is, and remains "
            "the sole property of Pentaho Corporation. The intellectual and "
            "technical concepts contained herein are proprietary to Pentaho."
        )
        company = (
            "All information contained herein is, and remains the property of "
            "COMPANY. The intellectual and technical concepts contained herein are "
            "proprietary to COMPANY."
        )
        receipt = (
            "The receipt or possession of this source code and related information "
            "does not convey or imply any rights to reproduce, disclose or "
            "distribute its contents."
        )
        bsd_truncated = (
            "Redistribution and use in source and binary forms, with or without"
        )
        bsd_full = (
            "Redistribution and use in source and binary forms, with or without "
            "modification, are permitted provided that redistributions of source "
            "code retain the above copyright notice."
        )

        expected_markers = {
            "rti-first": (
                rti_first,
                "boilerplate:no-duplications-trade-secrets",
            ),
            "rti-second": (
                rti_second,
                "boilerplate:no-duplications-trade-secrets",
            ),
            "riverbed-first": (
                riverbed_first,
                "boilerplate:mit-terms-conditions-as-is",
            ),
            "riverbed-second": (
                riverbed_second,
                "boilerplate:mit-terms-conditions-as-is",
            ),
            "1140": (
                imocom_1140,
                "boilerplate:confidential-proprietary-information-header",
            ),
            "9f77": (
                china_9f77,
                "boilerplate:confidential-proprietary-information-header",
            ),
            "421-stock": (
                stock_421,
                "boilerplate:confidential-proprietary-information-header",
            ),
            "pentaho": (
                pentaho,
                "boilerplate:all-information-company-banner",
            ),
            "company": (
                company,
                "boilerplate:all-information-company-banner",
            ),
            "receipt": (
                receipt,
                "boilerplate:all-information-company-banner",
            ),
            "bsd-truncated": (
                bsd_truncated,
                "boilerplate:bsd-redistribution-conditions",
            ),
            "bsd-full": (
                bsd_full,
                "boilerplate:bsd-redistribution-conditions",
            ),
        }
        for example_id, (comment, marker) in expected_markers.items():
            with self.subTest(example_id=example_id):
                self.assertIn(marker, _template_family_markers(comment))

        # The extended markers apply only after judging, preserving the bounded
        # candidate pool and judge-cache identity of an existing scan.
        self.assertNotIn(
            "boilerplate:bsd-redistribution-conditions",
            _template_family_markers(bsd_truncated, include_extended=False),
        )

        generic_one = (
            "This software is licensed under Example License terms for parser "
            "plugins. Documentation explains compatibility and configuration for "
            "local development only."
        )
        generic_two = (
            "The image codec uses Other License rules. Review package metadata "
            "before enabling optional network transports in production."
        )
        self.assertFalse(_template_family_markers(generic_one))
        self.assertFalse(_template_family_markers(generic_two))

        comments = {
            example_id: comment
            for example_id, (comment, _marker) in expected_markers.items()
        }
        comments.update(
            {
                "generic-license-one": generic_one,
                "generic-license-two": generic_two,
            }
        )
        rows = [
            {
                "example_id": example_id,
                "repo": f"org/{example_id}",
                "opening_comment": comment,
                "label": LABEL_SHARING_RESTRICTION,
                "dataset": "ds",
                "language": "Python",
            }
            for example_id, comment in comments.items()
        ]
        assignments = _leakage_aware_split_assignments(rows, seed=42)

        def assert_same_split_group(*example_ids: str) -> None:
            split_groups = {assignments[example_id][1] for example_id in example_ids}
            self.assertEqual(len(split_groups), 1)

        assert_same_split_group("rti-first", "rti-second")
        assert_same_split_group("riverbed-first", "riverbed-second")
        assert_same_split_group("1140", "9f77", "421-stock")
        assert_same_split_group("pentaho", "company", "receipt")
        assert_same_split_group("bsd-truncated", "bsd-full")
        self.assertNotEqual(
            assignments["generic-license-one"][1],
            assignments["generic-license-two"][1],
        )

    def test_velocity_mit_api_headers_form_one_post_judge_family(self) -> None:
        common_terms = (
            "is licensed under the terms of the MIT License. For more details, "
            "reference the LICENSE file in the api top-level directory."
        )
        comments = {
            "velocity-events": (
                f"The Velocity API {common_terms} Provides events to handle "
                "setting up permissions for permission subjects."
            ),
            "velocity-permissions": (
                f"The Velocity API {common_terms} Provides the basic building "
                "blocks for a custom permission system."
            ),
            "limbo-api": (
                "The LimboAPI (excluding the LimboAPI plugin) " + common_terms
            ),
        }
        marker = "boilerplate:mit-api-license-file-header"
        for example_id, comment in comments.items():
            with self.subTest(example_id=example_id):
                self.assertIn(marker, _template_family_markers(comment))
                self.assertNotIn(
                    marker,
                    _template_family_markers(comment, include_extended=False),
                )

        unrelated = (
            "This parser is licensed under the terms of the MIT License. For more "
            "details, review the project documentation and package metadata."
        )
        self.assertNotIn(marker, _template_family_markers(unrelated))

        rows = [
            {
                "example_id": example_id,
                "repo": f"org/{example_id}",
                "template_hash": f"template-{example_id}",
                "candidate_class": LABEL_MISSED_LICENSE,
                "selection_priority": float(len(comments) - index),
                "matched_terms": ["licensed_under"],
                "opening_comment": comment,
                "label": LABEL_MISSED_LICENSE,
                "dataset": "the-heap",
                "language": "Java",
            }
            for index, (example_id, comment) in enumerate(comments.items())
        ]
        selected, duplicates, excess = _select_diverse_rows(rows, limit=3)
        self.assertEqual(
            [row["example_id"] for row in selected],
            ["velocity-events"],
        )
        self.assertEqual(
            [row["example_id"] for row in duplicates],
            ["velocity-permissions", "limbo-api"],
        )
        self.assertFalse(excess)

        assignments = _leakage_aware_split_assignments(rows, seed=42)
        self.assertEqual(
            len({assignment[1] for assignment in assignments.values()}),
            1,
        )

    def test_proprietary_holder_headers_form_one_post_judge_family(self) -> None:
        southpaw = (
            "Copyright 2008 Southpaw Technology. PROPRIETARY INFORMATION. This "
            "software is proprietary to Southpaw Technology, and is not to be "
            "reproduced, transmitted, or disclosed in any way without written "
            "permission."
        )
        side_effects = (
            "PROPRIETARY INFORMATION. This software is proprietary to Side Effects "
            "Software Inc., and is not to be reproduced, transmitted, or disclosed "
            "in any way without written permission. Produced by Side Effects "
            "Software in Toronto. This module defines a scheduler implementation, "
            "depends on a command-line utility, and requires configured mapped "
            "paths for Python and the rendering tools."
        )
        marker = "boilerplate:proprietary-holder-no-reproduction"
        for comment in (southpaw, side_effects):
            self.assertIn(marker, _template_family_markers(comment))
            self.assertNotIn(
                marker,
                _template_family_markers(comment, include_extended=False),
            )

        unrelated = (
            "Proprietary information: this software is proprietary to Example Corp "
            "and implements a private rendering protocol for internal services."
        )
        self.assertNotIn(marker, _template_family_markers(unrelated))

        rows = [
            {
                "example_id": example_id,
                "repo": f"org/{example_id}",
                "template_hash": f"template-{example_id}",
                "candidate_class": LABEL_SHARING_RESTRICTION,
                "selection_priority": float(2 - index),
                "matched_terms": ["proprietary_marking"],
                "opening_comment": comment,
                "label": LABEL_SHARING_RESTRICTION,
                "dataset": "the-stack",
                "language": "Python",
            }
            for index, (example_id, comment) in enumerate(
                (("southpaw", southpaw), ("side-effects", side_effects))
            )
        ]
        selected, duplicates, excess = _select_diverse_rows(rows, limit=2)
        self.assertEqual([row["example_id"] for row in selected], ["southpaw"])
        self.assertEqual(
            [row["example_id"] for row in duplicates],
            ["side-effects"],
        )
        self.assertFalse(excess)

        assignments = _leakage_aware_split_assignments(rows, seed=42)
        self.assertEqual(assignments["southpaw"], assignments["side-effects"])

    def test_gnu_hope_and_warranty_headers_form_one_post_judge_family(self) -> None:
        comments = {
            "standard-gpl": (
                "This program is free software under the terms of the GNU General "
                "Public License. This program is distributed in the hope that it "
                "will be useful, but WITHOUT ANY WARRANTY."
            ),
            "mesquite-lgpl": (
                "Mesquite is distributed in the hope that it will be useful, but "
                "WITHOUT ANY WARRANTY. This source code and its compiled class "
                "files are free and modifiable under the terms of the GNU Lesser "
                "General Public License."
            ),
            "standard-agpl": (
                "Without any warranty, this server is distributed in the hope "
                "that it will be useful. It is available under the GNU Affero "
                "General Public License."
            ),
        }
        marker = "boilerplate:gnu-hope-useful-without-warranty"
        for example_id, comment in comments.items():
            with self.subTest(example_id=example_id):
                self.assertIn(marker, _template_family_markers(comment))
                self.assertNotIn(
                    marker,
                    _template_family_markers(comment, include_extended=False),
                )

        unrelated = (
            "The team distributed the prototype in the hope that it will be useful "
            "to reviewers, but supplied it without any warranty response metadata."
        )
        self.assertNotIn(marker, _template_family_markers(unrelated))

        rows = [
            {
                "example_id": example_id,
                "repo": f"org/{example_id}",
                "template_hash": f"template-{example_id}",
                "candidate_class": LABEL_MISSED_LICENSE,
                "selection_priority": float(len(comments) - index),
                "matched_terms": ["named_license"],
                "opening_comment": comment,
                "label": LABEL_MISSED_LICENSE,
                "dataset": "the-stack",
                "language": "Python",
            }
            for index, (example_id, comment) in enumerate(comments.items())
        ]
        selected, duplicates, excess = _select_diverse_rows(rows, limit=3)
        self.assertEqual(
            [row["example_id"] for row in selected],
            ["standard-gpl"],
        )
        self.assertEqual(
            [row["example_id"] for row in duplicates],
            ["mesquite-lgpl", "standard-agpl"],
        )
        self.assertFalse(excess)

        assignments = _leakage_aware_split_assignments(rows, seed=42)
        self.assertEqual(
            len({assignment[1] for assignment in assignments.values()}),
            1,
        )

    def test_copy_modify_warnings_form_one_post_judge_family(self) -> None:
        copy_modify = "Do Not Copy Or Modify Without Permission."
        copy_distribute_modify = (
            "Do not copy, distribute, or modify without permission."
        )
        marker = "boilerplate:do-not-copy-modify-without-permission"
        for comment in (copy_modify, copy_distribute_modify):
            self.assertIn(marker, _template_family_markers(comment))
            self.assertNotIn(
                marker,
                _template_family_markers(comment, include_extended=False),
            )

        unrelated = (
            "Do not copy generated files into the build folder; modify settings "
            "only after the permission check succeeds."
        )
        self.assertNotIn(marker, _template_family_markers(unrelated))

        rows = [
            {
                "example_id": example_id,
                "repo": f"org/{example_id}",
                "template_hash": f"template-{example_id}",
                "candidate_class": LABEL_SHARING_RESTRICTION,
                "selection_priority": float(2 - index),
                "matched_terms": ["permission_required"],
                "opening_comment": comment,
                "label": LABEL_SHARING_RESTRICTION,
                "dataset": "the-stack",
                "language": "Python",
            }
            for index, (example_id, comment) in enumerate(
                (
                    ("copy-modify", copy_modify),
                    ("copy-distribute-modify", copy_distribute_modify),
                )
            )
        ]
        selected, duplicates, excess = _select_diverse_rows(rows, limit=2)
        self.assertEqual(
            [row["example_id"] for row in selected],
            ["copy-modify"],
        )
        self.assertEqual(
            [row["example_id"] for row in duplicates],
            ["copy-distribute-modify"],
        )
        self.assertFalse(excess)

        assignments = _leakage_aware_split_assignments(rows, seed=42)
        self.assertEqual(
            assignments["copy-modify"],
            assignments["copy-distribute-modify"],
        )

    def test_diverse_selection_never_backfills_a_template_duplicate(self) -> None:
        common = (
            "This script is the property of {holder}, you may not copy, view, or "
            "distribute this script without prior consent."
        )
        rows = [
            {
                "example_id": "first",
                "template_hash": "first-template",
                "candidate_class": LABEL_SHARING_RESTRICTION,
                "selection_priority": 10.0,
                "matched_terms": ["permission_required"],
                "opening_comment": common.format(holder="First Corp"),
            },
            {
                "example_id": "duplicate",
                "template_hash": "second-template",
                "candidate_class": LABEL_SHARING_RESTRICTION,
                "selection_priority": 9.0,
                "matched_terms": ["permission_required"],
                "opening_comment": common.format(holder="Second Corp"),
            },
            {
                "example_id": "distinct",
                "template_hash": "third-template",
                "candidate_class": LABEL_SHARING_RESTRICTION,
                "selection_priority": 8.0,
                "matched_terms": ["confidential_marking"],
                "opening_comment": "Trade secret: do not disclose outside Example Corp.",
            },
        ]
        selected, duplicates, excess = _select_diverse_rows(rows, limit=3)
        self.assertEqual(
            [row["example_id"] for row in selected],
            ["first", "distinct"],
        )
        self.assertEqual([row["example_id"] for row in duplicates], ["duplicate"])
        self.assertFalse(excess)

    def test_stratified_component_assignment_populates_both_eval_splits(self) -> None:
        rows = []
        for label_index, label in enumerate(
            (
                LABEL_SHARING_RESTRICTION,
                LABEL_MISSED_LICENSE,
                LABEL_IRRELEVANT,
            )
        ):
            for index in range(40):
                rows.append(
                    {
                        "example_id": f"{label}-{index}",
                        "repo": "",
                        "opening_comment": f"unique_token_{label_index}_{index}",
                        "label": label,
                        "dataset": f"dataset-{index % 8}",
                        "language": f"language-{index % 2}",
                    }
                )
        assignments = _leakage_aware_split_assignments(rows, seed=42)
        split_counts = defaultdict(int)
        label_split_counts = defaultdict(int)
        for row in rows:
            split = assignments[row["example_id"]][0]
            split_counts[split] += 1
            label_split_counts[(row["label"], split)] += 1
        self.assertGreaterEqual(split_counts["validation"], 10)
        self.assertGreaterEqual(split_counts["test"], 10)
        for label in (
            LABEL_SHARING_RESTRICTION,
            LABEL_MISSED_LICENSE,
            LABEL_IRRELEVANT,
        ):
            self.assertGreaterEqual(label_split_counts[(label, "validation")], 3)
            self.assertGreaterEqual(label_split_counts[(label, "test")], 3)

    def test_final_irrelevant_ranking_prefers_boundaries_then_longer_randoms(self) -> None:
        def row(example_id: str, text: str, terms: list[str]) -> dict[str, object]:
            return {
                "example_id": example_id,
                "candidate_class": LABEL_IRRELEVANT,
                "selection_priority": 1.0,
                "matched_terms": terms,
                "opening_comment": text,
            }

        hard = row("hard", "Internal utility; not for external use.", ["technical_external_scope"])
        short = row("short", "Parse JSON.", ["scancode_zero_random_negative"])
        long = row("long", "Technical documentation. " * 20, ["scancode_zero_random_negative"])
        ranked = sorted((short, long, hard), key=_final_selection_key, reverse=True)
        self.assertEqual([item["example_id"] for item in ranked], ["hard", "long", "short"])

    def test_rejects_rows_declaring_a_different_dataset_or_language(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_directory = root / "input"
            output_directory = root / "output"
            rows = _base_rows()[:3]
            rows[0] = dict(rows[0], dataset="undeclared")
            _write_combination(input_directory, "ds", "Python", rows)

            with self.assertRaisesRegex(
                ValueError,
                "does not match its selected dataset/language",
            ):
                build_classifier_dataset(
                    input_directory,
                    output_directory,
                    combinations=[("ds", "Python")],
                    judge_runner=lambda _prompt: "[]",
                    judge_passes=2,
                )

    def test_uses_scancode_contains_flag_dedupes_and_writes_three_sets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_directory = root / "input"
            output_directory = root / "output"
            _write_combination(input_directory, "ds", "Python", _base_rows())
            judge = _ScriptedJudge(
                {
                    "[R-KEEP]": LABEL_SHARING_RESTRICTION,
                    "[L-KEEP]": LABEL_MISSED_LICENSE,
                    "[I-KEEP]": LABEL_IRRELEVANT,
                },
                semantic_overrides={
                    "[R-KEEP]": {
                        "evidence": "acme confidential … do not share"
                    },
                    "[L-KEEP]": {
                        "is_known_license": True,
                        "known_license": "Acme Research License",
                    }
                },
            )

            build_classifier_dataset(
                input_directory,
                output_directory,
                combinations=[("ds", "Python")],
                target_per_combination=10,
                candidate_multiplier=10,
                max_shards_per_combination=1,
                judge_runner=judge,
                judge_passes=2,
                overwrite=True,
            )

            combined_path = output_directory / "dataset.parquet"
            self.assertTrue(combined_path.is_file())
            combined = _read_rows(combined_path)
            training_table = pq.read_table(output_directory / "binary-training.parquet")
            self.assertEqual(
                training_table.column_names,
                [
                    "example_id",
                    "opening_comment",
                    "binary_label",
                    "split",
                    "dataset",
                    "language",
                ],
            )
            self.assertNotIn("judge_label", training_table.column_names)
            self.assertNotIn("label", training_table.column_names)
            self.assertEqual(training_table.num_rows, len(combined))
            multiclass_table = pq.read_table(
                output_directory / "multiclass-training.parquet"
            )
            self.assertIn("label", multiclass_table.column_names)
            self.assertNotIn("binary_label", multiclass_table.column_names)
            self.assertNotIn("label_id", multiclass_table.column_names)
            self.assertEqual(len(combined), 3)
            self.assertEqual(
                {row["label"] for row in combined},
                {
                    LABEL_SHARING_RESTRICTION,
                    LABEL_MISSED_LICENSE,
                    LABEL_IRRELEVANT,
                },
            )
            self.assertEqual(
                {
                    row["label"]: row["binary_label"]
                    for row in combined
                },
                {
                    LABEL_SHARING_RESTRICTION: 1,
                    LABEL_MISSED_LICENSE: 0,
                    LABEL_IRRELEVANT: 0,
                },
            )
            self.assertEqual(
                len({row["opening_comment"] for row in combined}),
                len(combined),
            )
            self.assertNotIn(
                "known-license-low-score",
                {row["record_id"] for row in combined},
            )
            missed = next(
                row for row in combined if row["label"] == LABEL_MISSED_LICENSE
            )
            self.assertTrue(missed["is_known_license"])
            self.assertEqual(missed["known_license"], "Acme Research License")
            restriction_row = next(
                row for row in combined if row["label"] == LABEL_SHARING_RESTRICTION
            )
            self.assertIn(
                "ACME CONFIDENTIAL and proprietary source code. Do not share",
                restriction_row["judge_evidence"],
            )

            for label in (
                LABEL_SHARING_RESTRICTION,
                LABEL_MISSED_LICENSE,
                LABEL_IRRELEVANT,
            ):
                class_path = output_directory / f"{label}.parquet"
                self.assertTrue(class_path.is_file(), class_path)
                rows = _read_rows(class_path)
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["label"], label)

            candidates_path = output_directory / "candidates.parquet"
            self.assertTrue(candidates_path.is_file())
            self.assertTrue((output_directory / "rejected.parquet").is_file())
            candidates = _read_rows(candidates_path)
            self.assertEqual(len(candidates), 3)
            self.assertEqual(
                len({row["opening_comment"] for row in candidates}),
                len(candidates),
            )
            self.assertNotIn(
                "known-license-low-score",
                {row["record_id"] for row in candidates},
            )

            # Three unique eligible rows, two independent validations each.
            self.assertEqual(len(judge.calls), 6)
            self.assertEqual(judge.calls.count("[R-KEEP]"), 2)
            self.assertNotIn("[KNOWN-SKIP]", judge.calls)

    def test_global_scan_plan_is_reusable_without_rescanning_or_reordering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_directory = root / "input"
            combinations = [("alpha", "Python"), ("beta", "Python")]
            for dataset, language in combinations:
                rows = []
                for base in _base_rows()[:3]:
                    rows.append(
                        _row(
                            f"{dataset}-{base['record_id']}",
                            f"{base['opening_comment']} Source slice {dataset}.",
                            dataset=dataset,
                            language=language,
                            score=float(base["comment_license_score"]),
                        )
                    )
                _write_combination(
                    input_directory,
                    dataset,
                    language,
                    rows,
                )

            plan_directory = root / "plan"

            def forbidden_judge(_prompt: str) -> str:
                raise AssertionError("scan-only mode invoked the judge")

            plan_stats = build_classifier_dataset(
                input_directory,
                plan_directory,
                combinations=combinations,
                target_per_class=2,
                candidate_multiplier=1,
                candidate_targets={
                    LABEL_SHARING_RESTRICTION: 2,
                    LABEL_MISSED_LICENSE: 2,
                    LABEL_IRRELEVANT: 2,
                },
                max_shards_per_combination=1,
                judge_runner=forbidden_judge,
                scan_only=True,
            )
            self.assertIsNone(plan_stats.dataset_path)
            self.assertTrue((plan_directory / "candidate-plan.json").is_file())
            self.assertFalse((plan_directory / "judge-responses.jsonl").exists())
            plan_ids = [
                row["example_id"]
                for row in _read_rows(plan_directory / "candidates.parquet")
            ]
            self.assertEqual(len(plan_ids), 6)

            # A prior candidate-plan directory is a recognizable atomic
            # overwrite target.
            build_classifier_dataset(
                input_directory,
                plan_directory,
                combinations=combinations,
                target_per_class=2,
                candidate_multiplier=1,
                candidate_targets={
                    LABEL_SHARING_RESTRICTION: 2,
                    LABEL_MISSED_LICENSE: 2,
                    LABEL_IRRELEVANT: 2,
                },
                max_shards_per_combination=1,
                scan_only=True,
                overwrite=True,
            )
            self.assertEqual(
                [
                    row["example_id"]
                    for row in _read_rows(plan_directory / "candidates.parquet")
                ],
                plan_ids,
            )

            output_directory = root / "output"
            judge = _ScriptedJudge(
                {
                    "[R-KEEP]": LABEL_SHARING_RESTRICTION,
                    "[L-KEEP]": LABEL_MISSED_LICENSE,
                    "[I-KEEP]": LABEL_IRRELEVANT,
                }
            )
            build_classifier_dataset(
                input_directory,
                output_directory,
                combinations=combinations,
                target_per_class=2,
                candidate_multiplier=1,
                candidate_targets={
                    LABEL_SHARING_RESTRICTION: 2,
                    LABEL_MISSED_LICENSE: 2,
                    LABEL_IRRELEVANT: 2,
                },
                max_shards_per_combination=1,
                judge_runner=judge,
                judge_batch_size=1,
                judge_cache_epoch="large-run-2026-07",
                candidate_plan=plan_directory,
            )
            self.assertEqual(
                [
                    row["example_id"]
                    for row in _read_rows(output_directory / "candidates.parquet")
                ],
                plan_ids,
            )
            manifest = json.loads(
                (output_directory / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["format_version"], 2)
            self.assertEqual(manifest["configuration"]["target_per_class"], 2)
            self.assertEqual(
                manifest["judge"]["cache_epoch"],
                "large-run-2026-07",
            )
            self.assertIn("candidate_plan", manifest)
            self.assertEqual(len(_read_rows(output_directory / "dataset.parquet")), 6)
            self.assertTrue(
                verify_classifier_dataset(
                    output_directory,
                    require_all_classes_per_combination=False,
                )["valid"]
            )

            with self.assertRaisesRegex(ValueError, "configuration"):
                build_classifier_dataset(
                    input_directory,
                    root / "mismatch-output",
                    combinations=combinations,
                    target_per_class=3,
                    candidate_multiplier=1,
                    max_shards_per_combination=1,
                    judge_runner=judge,
                    candidate_plan=plan_directory,
                )

    def test_conflicting_cross_cell_judge_family_is_quarantined_and_topped_up(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_directory = root / "input"
            output_directory = root / "output"
            alpha_rows = [
                {
                    **row,
                    "dataset": "alpha",
                    "language": "C++",
                }
                for row in _base_rows()[:3]
            ]
            alpha_rows.append(
                _row(
                    "conflict-irrelevant",
                    "[I-CONFLICT-A] Generated API client documentation. "
                    "These API methods are for internal use only. "
                    "OpenAPI version 1.0.",
                    dataset="alpha",
                    language="C++",
                    repo="example/api-cpp",
                )
            )
            beta_rows = [
                _row(
                    "conflict-sharing",
                    "[I-CONFLICT-B] Generated API client documentation. "
                    "These API methods are for internal use only. "
                    "OpenAPI version 2.0.",
                    dataset="beta",
                    language="Ruby",
                    repo="example/api-ruby",
                )
            ]
            _write_combination(
                input_directory,
                "alpha",
                "C++",
                alpha_rows,
            )
            _write_combination(
                input_directory,
                "beta",
                "Ruby",
                beta_rows,
            )
            judge = _ScriptedJudge(
                {
                    "[R-KEEP]": LABEL_SHARING_RESTRICTION,
                    "[L-KEEP]": LABEL_MISSED_LICENSE,
                    "[I-KEEP]": LABEL_IRRELEVANT,
                    "[I-CONFLICT-A]": LABEL_IRRELEVANT,
                    "[I-CONFLICT-B]": LABEL_SHARING_RESTRICTION,
                }
            )

            build_classifier_dataset(
                input_directory,
                output_directory,
                combinations=[("alpha", "C++"), ("beta", "Ruby")],
                target_per_class=1,
                candidate_multiplier=1,
                candidate_targets={
                    LABEL_SHARING_RESTRICTION: 1,
                    LABEL_MISSED_LICENSE: 1,
                    LABEL_IRRELEVANT: 3,
                },
                max_shards_per_combination=1,
                judge_runner=judge,
                judge_passes=2,
            )

            accepted = _read_rows(output_directory / "dataset.parquet")
            self.assertEqual(
                Counter(row["label"] for row in accepted),
                {
                    LABEL_SHARING_RESTRICTION: 1,
                    LABEL_MISSED_LICENSE: 1,
                    LABEL_IRRELEVANT: 1,
                },
            )
            self.assertEqual(
                next(
                    row["record_id"]
                    for row in accepted
                    if row["label"] == LABEL_IRRELEVANT
                ),
                "irrelevant",
            )
            rejected = _read_rows(output_directory / "rejected.parquet")
            rejection_by_record = {
                row["record_id"]: row["rejection_reason"]
                for row in rejected
            }
            self.assertEqual(
                rejection_by_record["conflict-irrelevant"],
                "template_family_label_conflict",
            )
            self.assertEqual(
                rejection_by_record["conflict-sharing"],
                "judge_label_mismatch:sharing_restriction",
            )
            manifest = json.loads(
                (output_directory / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                manifest["results"][
                    "template_family_label_conflicts_excluded"
                ],
                1,
            )
            self.assertEqual(
                manifest["results"][
                    "judge_family_label_conflict_witnesses"
                ],
                1,
            )
            self.assertEqual(
                manifest["results"]["judge_family_label_conflict_pairs"],
                1,
            )
            report = verify_classifier_dataset(
                output_directory,
                require_all_classes_per_combination=False,
            )
            self.assertTrue(report["valid"])
            self.assertEqual(
                report["reviewed_family_label_conflict_witnesses"],
                1,
            )
            self.assertEqual(
                report["reviewed_family_label_conflict_quarantines"],
                1,
            )
            self.assertEqual(
                report["reviewed_family_label_conflict_pairs"],
                1,
            )

    def test_candidate_plan_rejects_changed_source_before_judging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_directory = root / "input"
            _write_combination(
                input_directory,
                "ds",
                "Python",
                _base_rows()[:3],
            )
            plan_directory = root / "plan"
            build_classifier_dataset(
                input_directory,
                plan_directory,
                combinations=[("ds", "Python")],
                target_per_class=1,
                candidate_multiplier=1,
                max_shards_per_combination=1,
                scan_only=True,
            )
            _write_combination(
                input_directory,
                "ds",
                "Python",
                [
                    *_base_rows()[:3],
                    _row(
                        "changed",
                        "Changed technical documentation for source validation.",
                    ),
                ],
            )

            def forbidden_judge(_prompt: str) -> str:
                raise AssertionError("invalid plan reached the judge")

            with self.assertRaisesRegex(
                ValueError,
                "source file selection, size, SHA-256, or fingerprint changed",
            ):
                build_classifier_dataset(
                    input_directory,
                    root / "output",
                    combinations=[("ds", "Python")],
                    target_per_class=1,
                    candidate_multiplier=1,
                    max_shards_per_combination=1,
                    judge_runner=forbidden_judge,
                    candidate_plan=plan_directory,
                )

    def test_routes_technical_internal_scope_as_a_hard_irrelevant_candidate(self) -> None:
        for lock_text in (
            "Do not copy this module into the repository; do not release the "
            "lock until installation finishes.",
            "Do not copy this module into the repository; must not release the "
            "source lock until loading finishes.",
        ):
            with self.subTest(lock_text=lock_text):
                self.assertIsNotNone(_hard_irrelevant_features(lock_text))
                self.assertIsNone(_AFFIRMATIVE_SHARING_CONTEXT.search(lock_text))
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_directory = root / "input"
            output_directory = root / "output"
            rows = _base_rows()[:3]
            rows.append(
                _row(
                    "hard-irrelevant",
                    "[I-HARD] These helper methods are for internal use only; "
                    "callers should use the public API instead.",
                )
            )
            rows.extend(
                [
                    _row(
                        "hard-positive-override",
                        "[R-HARD] This proprietary module is for ACME internal use "
                        "only and may not be disclosed externally.",
                    ),
                    _row(
                        "hard-license-override",
                        "[L-HARD] Licensed under the MIT License. DO NOT COPY this "
                        "file into the example repository; depend on the package.",
                    ),
                    _row(
                        "hard-classifier",
                        "[I-CLASSIFIER] Do not copy this classifier module into the "
                        "public repository; install the package instead.",
                    ),
                    _row(
                        "hard-third-party-module",
                        "[I-THIRD-PARTY] Do not copy this third-party module into the "
                        "repository; install it from npm instead.",
                    ),
                    _row(
                        "hard-release-lock",
                        "[I-LOCK] Do not copy this module into the repository; do not "
                        "release the lock until installation finishes.",
                    ),
                    _row(
                        "hard-release-source-lock",
                        "[I-SOURCE-LOCK] Do not copy this module into the repository; "
                        "must not release the source lock until loading finishes.",
                    ),
                    _row(
                        "hard-classes",
                        "[I-CLASSES] Utility classes for JSON, for internal use only.",
                    ),
                    _row(
                        "hard-package",
                        "[I-PACKAGE] Internal use only! If you call anything in this "
                        "package, use the public wrapper instead.",
                    ),
                    _row(
                        "hard-external-utility",
                        "[I-EXTERNAL] Internal utilities; not for external use.",
                    ),
                    _row(
                        "hard-external-mixed-positive",
                        "[R-EXTERNAL] Internal utility; not for external use. Do not "
                        "distribute outside ACME.",
                    ),
                    _row(
                        "hard-classified-override",
                        "[R-CLASSIFIED] Do not copy this classified information into "
                        "a public repository or disclose it outside ACME.",
                    ),
                ]
            )
            _write_combination(input_directory, "ds", "Python", rows)
            judge = _ScriptedJudge(
                {
                    "[R-KEEP]": LABEL_SHARING_RESTRICTION,
                    "[L-KEEP]": LABEL_MISSED_LICENSE,
                    "[I-KEEP]": LABEL_IRRELEVANT,
                    "[I-HARD]": LABEL_IRRELEVANT,
                    "[R-HARD]": LABEL_SHARING_RESTRICTION,
                    "[L-HARD]": LABEL_MISSED_LICENSE,
                    "[I-CLASSIFIER]": LABEL_IRRELEVANT,
                    "[I-THIRD-PARTY]": LABEL_IRRELEVANT,
                    "[I-LOCK]": LABEL_IRRELEVANT,
                    "[I-SOURCE-LOCK]": LABEL_IRRELEVANT,
                    "[I-CLASSES]": LABEL_IRRELEVANT,
                    "[I-PACKAGE]": LABEL_IRRELEVANT,
                    "[I-EXTERNAL]": LABEL_IRRELEVANT,
                    "[R-EXTERNAL]": LABEL_SHARING_RESTRICTION,
                    "[R-CLASSIFIED]": LABEL_SHARING_RESTRICTION,
                }
            )
            build_classifier_dataset(
                input_directory,
                output_directory,
                combinations=[("ds", "Python")],
                target_per_combination=10,
                candidate_multiplier=10,
                max_shards_per_combination=1,
                judge_runner=judge,
                judge_passes=2,
            )
            dataset = _read_rows(output_directory / "dataset.parquet")
            hard = next(row for row in dataset if row["record_id"] == "hard-irrelevant")
            self.assertEqual(hard["candidate_class"], LABEL_IRRELEVANT)
            self.assertEqual(hard["label"], LABEL_IRRELEVANT)
            self.assertIn("technical_internal_api_scope", hard["matched_terms"])
            hard_positive = next(
                row for row in dataset if row["record_id"] == "hard-positive-override"
            )
            self.assertEqual(
                hard_positive["candidate_class"], LABEL_SHARING_RESTRICTION
            )
            hard_license = next(
                row for row in dataset if row["record_id"] == "hard-license-override"
            )
            self.assertEqual(hard_license["candidate_class"], LABEL_MISSED_LICENSE)
            hard_classifier = next(
                row for row in dataset if row["record_id"] == "hard-classifier"
            )
            self.assertEqual(hard_classifier["candidate_class"], LABEL_IRRELEVANT)
            hard_third_party = next(
                row
                for row in dataset
                if row["record_id"] == "hard-third-party-module"
            )
            self.assertEqual(hard_third_party["candidate_class"], LABEL_IRRELEVANT)
            release_lock_rows = [
                row
                for row in dataset
                if row["record_id"]
                in {"hard-release-lock", "hard-release-source-lock"}
            ]
            self.assertTrue(release_lock_rows)
            self.assertTrue(
                all(
                    row["candidate_class"] == LABEL_IRRELEVANT
                    for row in release_lock_rows
                )
            )
            for record_id in (
                "hard-classes",
                "hard-package",
                "hard-external-utility",
            ):
                routed = next(row for row in dataset if row["record_id"] == record_id)
                self.assertEqual(routed["candidate_class"], LABEL_IRRELEVANT)
            hard_classified = next(
                row
                for row in dataset
                if row["record_id"] == "hard-classified-override"
            )
            self.assertEqual(
                hard_classified["candidate_class"], LABEL_SHARING_RESTRICTION
            )
            hard_external_positive = next(
                row
                for row in dataset
                if row["record_id"] == "hard-external-mixed-positive"
            )
            self.assertEqual(
                hard_external_positive["candidate_class"],
                LABEL_SHARING_RESTRICTION,
            )

    def test_requires_consensus_and_rejects_invalid_label_invariants(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_directory = root / "input"
            output_directory = root / "output"
            rows = _base_rows()[:3]
            rows.extend(
                [
                    _row(
                        "disagreement",
                        "[R-DISAGREE] Confidential source: do not distribute externally.",
                    ),
                    _row(
                        "wrong-target",
                        "[R-WRONG-TARGET] Proprietary implementation; do not share it.",
                    ),
                ]
            )
            _write_combination(input_directory, "ds", "Python", rows)
            judge = _ScriptedJudge(
                {
                    "[R-KEEP]": LABEL_SHARING_RESTRICTION,
                    "[L-KEEP]": LABEL_MISSED_LICENSE,
                    "[I-KEEP]": LABEL_IRRELEVANT,
                    "[R-DISAGREE]": [
                        LABEL_SHARING_RESTRICTION,
                        LABEL_IRRELEVANT,
                    ],
                    "[R-WRONG-TARGET]": LABEL_SHARING_RESTRICTION,
                },
                invalid_invariant_markers={"[R-WRONG-TARGET]"},
            )

            build_classifier_dataset(
                input_directory,
                output_directory,
                combinations=[("ds", "Python")],
                target_per_combination=20,
                candidate_multiplier=10,
                max_shards_per_combination=1,
                judge_runner=judge,
                judge_passes=2,
                overwrite=True,
            )

            combined = _read_rows(output_directory / "dataset.parquet")
            record_ids = {row["record_id"] for row in combined}
            self.assertEqual(
                record_ids,
                {"restriction", "missed-license", "irrelevant"},
            )
            manifest = json.loads(
                (output_directory / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["judge_passes"], 2)
            self.assertEqual(manifest["judge_disagreements"], 1)
            self.assertEqual(manifest["invalid_judge_responses"], 2)

            rejected = _read_rows(output_directory / "rejected.parquet")
            self.assertEqual(
                {row["record_id"] for row in rejected},
                {"disagreement", "wrong-target"},
            )

    def test_rejects_a_judge_batch_without_exact_candidate_id_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_directory = root / "input"
            output_directory = root / "output"
            _write_combination(input_directory, "ds", "Python", _base_rows()[:3])
            judge = _ScriptedJudge(
                {
                    "[R-KEEP]": LABEL_SHARING_RESTRICTION,
                    "[L-KEEP]": LABEL_MISSED_LICENSE,
                    "[I-KEEP]": LABEL_IRRELEVANT,
                },
                coverage_error="duplicate_first",
            )

            with self.assertRaises((ValueError, RuntimeError)):
                build_classifier_dataset(
                    input_directory,
                    output_directory,
                    combinations=[("ds", "Python")],
                    target_per_combination=10,
                    candidate_multiplier=10,
                    max_shards_per_combination=1,
                    judge_runner=judge,
                    judge_passes=2,
                    overwrite=True,
                )
            self.assertTrue(judge.calls)

    def test_keeps_repositories_in_one_split_and_reports_combination_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_directory = root / "input"
            output_directory = root / "output"
            combinations = [("ds", "Python"), ("other", "JavaScript")]
            votes: dict[str, str] = {}
            suffixes = (("alpha", "beta"), ("gamma", "delta"))
            for combo_index, (dataset, language) in enumerate(combinations):
                rows: list[dict[str, object]] = []
                for repo_index, repo in enumerate(("org/shared", "org/second")):
                    suffix = suffixes[combo_index][repo_index]
                    restriction_marker = f"[R-{suffix}]"
                    license_marker = f"[L-{suffix}]"
                    irrelevant_marker = f"[I-{suffix}]"
                    license_text = (
                        "Licensed under the Example License; permission is granted "
                        "to copy this software."
                        if repo_index == 0
                        else "This software is licensed under the Example License. "
                        "Redistribution follows its documented terms; review package "
                        "metadata before publishing modified builds."
                    )
                    rows.extend(
                        [
                            _row(
                                f"restriction-{suffix}",
                                f"{restriction_marker} Confidential: do not distribute.",
                                dataset=dataset,
                                language=language,
                                repo=repo,
                            ),
                            _row(
                                f"license-{suffix}",
                                f"{license_marker} {license_text}",
                                dataset=dataset,
                                language=language,
                                repo=repo,
                            ),
                            _row(
                                f"irrelevant-{suffix}",
                                f"{irrelevant_marker} Parse the input and return its length.",
                                dataset=dataset,
                                language=language,
                                repo=repo,
                            ),
                        ]
                    )
                    votes[restriction_marker] = LABEL_SHARING_RESTRICTION
                    votes[license_marker] = LABEL_MISSED_LICENSE
                    votes[irrelevant_marker] = LABEL_IRRELEVANT
                _write_combination(input_directory, dataset, language, rows)

            build_classifier_dataset(
                input_directory,
                output_directory,
                combinations=combinations,
                target_per_combination=20,
                candidate_multiplier=10,
                max_shards_per_combination=1,
                judge_runner=_ScriptedJudge(votes),
                judge_passes=2,
                judge_workers=2,
                overwrite=True,
            )

            combined = _read_rows(output_directory / "dataset.parquet")
            self.assertEqual(len(combined), 12)
            splits_by_repo: dict[str, set[str]] = defaultdict(set)
            for row in combined:
                self.assertIn(row["split"], {"train", "validation", "test"})
                splits_by_repo[str(row["repo"])].add(str(row["split"]))
            self.assertTrue(splits_by_repo)
            self.assertTrue(
                all(len(splits) == 1 for splits in splits_by_repo.values()),
                splits_by_repo,
            )

            manifest = json.loads(
                (output_directory / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["records_written"], 12)
            self.assertEqual(
                manifest["label_counts"],
                {
                    LABEL_SHARING_RESTRICTION: 4,
                    LABEL_MISSED_LICENSE: 4,
                    LABEL_IRRELEVANT: 4,
                },
            )
            coverage = {
                (item["dataset"], item["language"]): item
                for item in manifest["combinations"]
            }
            self.assertEqual(set(coverage), set(combinations))
            for combination in combinations:
                self.assertEqual(coverage[combination]["records_written"], 6)
                self.assertEqual(
                    coverage[combination]["label_counts"],
                    {
                        LABEL_SHARING_RESTRICTION: 2,
                        LABEL_MISSED_LICENSE: 2,
                        LABEL_IRRELEVANT: 2,
                    },
                )
            dataset_card = (output_directory / "README.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("two prompt-diverse row-level reviews", dataset_card)
            self.assertIn("same configured LLM", dataset_card)
            self.assertNotIn("independent row-level", dataset_card)
            self.assertIn("notice, name, grant, or substantive", dataset_card)
            self.assertIn("boilerplate-marker families", dataset_card)

    def test_verifier_binds_all_manifest_count_summaries_to_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_directory = root / "input"
            output_directory = root / "output"
            _write_combination(input_directory, "ds", "Python", _base_rows()[:3])
            judge = _ScriptedJudge(
                {
                    "[R-KEEP]": LABEL_SHARING_RESTRICTION,
                    "[L-KEEP]": LABEL_MISSED_LICENSE,
                    "[I-KEEP]": LABEL_IRRELEVANT,
                }
            )
            build_classifier_dataset(
                input_directory,
                output_directory,
                combinations=[("ds", "Python")],
                target_per_combination=10,
                candidate_multiplier=10,
                max_shards_per_combination=1,
                judge_runner=judge,
                judge_passes=2,
            )

            manifest_path = output_directory / "manifest.json"
            baseline = json.loads(manifest_path.read_text(encoding="utf-8"))
            cases = (
                "format_version",
                "labels",
                "binary_positive_label",
                "top_judge_passes",
                "judge_setups",
                "top_records",
                "top_labels",
                "combination_records",
                "combination_labels",
                "target_records",
                "split_counts",
                "combination_class_counts",
                "undeclared_stratum",
            )
            for case in cases:
                with self.subTest(case=case):
                    manifest = json.loads(json.dumps(baseline))
                    if case == "format_version":
                        manifest["format_version"] = 2
                    elif case == "labels":
                        manifest["labels"] = [LABEL_IRRELEVANT]
                    elif case == "binary_positive_label":
                        manifest["binary_positive_label"] = LABEL_IRRELEVANT
                    elif case == "top_judge_passes":
                        manifest["judge_passes"] = 1
                    elif case == "judge_setups":
                        manifest["judge"]["setups"] = list(
                            reversed(manifest["judge"]["setups"])
                        )
                    elif case == "top_records":
                        manifest["records_written"] = 999
                    elif case == "top_labels":
                        manifest["label_counts"][LABEL_IRRELEVANT] = 999
                    elif case == "combination_records":
                        manifest["combinations"][0]["records_written"] = 999
                    elif case == "combination_labels":
                        manifest["combinations"][0]["label_counts"][
                            LABEL_IRRELEVANT
                        ] = 999
                    elif case == "target_records":
                        manifest["results"]["target_records"] = 999
                    elif case == "split_counts":
                        manifest["results"]["split_counts"]["train"] = 999
                    elif case == "combination_class_counts":
                        manifest["results"]["combination_class_counts"]["ds/Python"][
                            LABEL_IRRELEVANT
                        ] = 999
                    elif case == "undeclared_stratum":
                        manifest["combinations"] = []
                        manifest["results"]["target_records"] = 0
                        manifest["results"]["quota_shortfalls"] = []
                    manifest_path.write_text(
                        json.dumps(manifest, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    with self.assertRaises(ValueError):
                        verify_classifier_dataset(output_directory)

            manifest_path.write_text(
                json.dumps(baseline, indent=2) + "\n",
                encoding="utf-8",
            )
            self.assertTrue(verify_classifier_dataset(output_directory)["valid"])

            incomplete = json.loads(json.dumps(baseline))
            incomplete["combinations"].append(
                {
                    "dataset": "empty",
                    "language": "Java",
                    "records_written": 0,
                    "label_counts": {label: 0 for label in (
                        LABEL_SHARING_RESTRICTION,
                        LABEL_MISSED_LICENSE,
                        LABEL_IRRELEVANT,
                    )},
                }
            )
            incomplete["results"]["target_records"] += 30
            incomplete["results"]["quota_shortfalls"].extend(
                {
                    "dataset": "empty",
                    "language": "Java",
                    "label": label,
                    "target": 10,
                    "accepted": 0,
                    "missing": 10,
                }
                for label in (
                    LABEL_SHARING_RESTRICTION,
                    LABEL_MISSED_LICENSE,
                    LABEL_IRRELEVANT,
                )
            )
            manifest_path.write_text(
                json.dumps(incomplete, indent=2) + "\n",
                encoding="utf-8",
            )
            self.assertTrue(
                verify_classifier_dataset(
                    output_directory,
                    require_all_classes_per_combination=False,
                )["valid"]
            )
            with self.assertRaises(ValueError):
                verify_classifier_dataset(output_directory)

            manifest_path.write_text(
                json.dumps(baseline, indent=2) + "\n",
                encoding="utf-8",
            )
            class_path = output_directory / f"{LABEL_SHARING_RESTRICTION}.parquet"
            original_class = class_path.read_bytes()
            corrupted_rows = pq.read_table(class_path).to_pylist()
            corrupted_rows[0]["opening_comment"] = "CORRUPTED CLASS COPY"
            pq.write_table(
                pa.Table.from_pylist(corrupted_rows),
                class_path,
            )
            corrupted_manifest = json.loads(json.dumps(baseline))
            artifact = corrupted_manifest["artifacts"][class_path.name]
            artifact["sha256"] = hashlib.sha256(class_path.read_bytes()).hexdigest()
            artifact["bytes"] = class_path.stat().st_size
            manifest_path.write_text(
                json.dumps(corrupted_manifest, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "class artifact differs"):
                verify_classifier_dataset(output_directory)
            class_path.write_bytes(original_class)
            manifest_path.write_text(
                json.dumps(baseline, indent=2) + "\n",
                encoding="utf-8",
            )

    def test_deep_verifier_checks_recorded_source_manifests_and_reports_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_directory = root / "input"
            output_directory = root / "output"
            input_directory.mkdir()
            source_manifest = input_directory / "manifest.json"
            original_source_manifest = "source-v1\n"
            source_manifest.write_text(original_source_manifest, encoding="utf-8")
            _write_combination(input_directory, "ds", "Python", _base_rows()[:3])
            judge = _ScriptedJudge(
                {
                    "[R-KEEP]": LABEL_SHARING_RESTRICTION,
                    "[L-KEEP]": LABEL_MISSED_LICENSE,
                    "[I-KEEP]": LABEL_IRRELEVANT,
                }
            )
            build_classifier_dataset(
                input_directory,
                output_directory,
                combinations=[("ds", "Python")],
                target_per_combination=10,
                candidate_multiplier=10,
                max_shards_per_combination=1,
                judge_runner=judge,
                judge_passes=2,
            )

            manifest_path = output_directory / "manifest.json"
            baseline = json.loads(manifest_path.read_text(encoding="utf-8"))
            source_entries = baseline["input"]["source_manifests"]
            self.assertEqual(len(source_entries), 1)
            self.assertEqual(source_entries[0]["path"], "manifest.json")
            self.assertEqual(source_entries[0]["size"], source_manifest.stat().st_size)
            self.assertEqual(
                source_entries[0]["sha256"],
                hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
            )

            stored_report = json.loads(
                (output_directory / "verification.json").read_text(encoding="utf-8")
            )
            self.assertTrue(stored_report["verify_source"])
            self.assertFalse(
                verify_classifier_dataset(output_directory)["verify_source"]
            )
            self.assertTrue(
                verify_classifier_dataset(
                    output_directory, verify_source=True
                )["verify_source"]
            )

            source_manifest.write_text("source-v2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source manifest checksum changed"):
                verify_classifier_dataset(output_directory, verify_source=True)

            source_manifest.write_text(
                original_source_manifest + "changed\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "source manifest size changed"):
                verify_classifier_dataset(output_directory, verify_source=True)
            source_manifest.write_text(original_source_manifest, encoding="utf-8")

            outside_manifest = root / "outside.json"
            outside_manifest.write_text("outside\n", encoding="utf-8")
            escaped = json.loads(json.dumps(baseline))
            escaped_entry = escaped["input"]["source_manifests"][0]
            escaped_entry.update(
                {
                    "path": "../outside.json",
                    "size": outside_manifest.stat().st_size,
                    "sha256": hashlib.sha256(
                        outside_manifest.read_bytes()
                    ).hexdigest(),
                }
            )
            manifest_path.write_text(
                json.dumps(escaped, indent=2) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ValueError, "source manifest escapes resolved input directory"
            ):
                verify_classifier_dataset(output_directory, verify_source=True)

            manifest_path.write_text(
                json.dumps(baseline, indent=2) + "\n", encoding="utf-8"
            )

    def test_mixed_license_and_extra_restriction_keeps_truthful_license_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_directory = root / "input"
            output_directory = root / "output"
            rows = _base_rows()[:3]
            rows.append(
                _row(
                    "mixed",
                    "[R-MIXED] MIT License. Permission is hereby granted to use this "
                    "software. Separate ACME confidentiality terms apply: do not "
                    "share or disclose this source outside ACME.",
                )
            )
            _write_combination(input_directory, "ds", "Python", rows)
            judge = _ScriptedJudge(
                {
                    "[R-KEEP]": LABEL_SHARING_RESTRICTION,
                    "[R-MIXED]": LABEL_SHARING_RESTRICTION,
                    "[L-KEEP]": LABEL_MISSED_LICENSE,
                    "[I-KEEP]": LABEL_IRRELEVANT,
                },
                semantic_overrides={
                    "[R-MIXED]": {
                        "is_license_notice": True,
                        "is_known_license": True,
                        "known_license": "MIT License",
                    }
                },
            )

            build_classifier_dataset(
                input_directory,
                output_directory,
                combinations=[("ds", "Python")],
                target_per_combination=10,
                candidate_multiplier=10,
                max_shards_per_combination=1,
                judge_runner=judge,
                judge_passes=2,
            )

            mixed = next(
                row
                for row in _read_rows(output_directory / "dataset.parquet")
                if row["record_id"] == "mixed"
            )
            self.assertEqual(mixed["label"], LABEL_SHARING_RESTRICTION)
            self.assertEqual(mixed["binary_label"], 1)
            self.assertTrue(mixed["is_sharing_restriction"])
            self.assertTrue(mixed["is_license_notice"])
            self.assertTrue(mixed["is_known_license"])
            self.assertEqual(mixed["known_license"], "MIT License")

    def test_failed_overwrite_preserves_recognized_output_and_cleans_staging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_directory = root / "input"
            output_directory = root / "output"
            _write_combination(input_directory, "ds", "Python", _base_rows()[:3])
            votes = {
                "[R-KEEP]": LABEL_SHARING_RESTRICTION,
                "[L-KEEP]": LABEL_MISSED_LICENSE,
                "[I-KEEP]": LABEL_IRRELEVANT,
            }
            build_classifier_dataset(
                input_directory,
                output_directory,
                combinations=[("ds", "Python")],
                judge_runner=_ScriptedJudge(votes),
                judge_passes=2,
            )
            original_dataset = (output_directory / "dataset.parquet").read_bytes()
            sentinel = output_directory / "keep-me.txt"
            sentinel.write_text("old output", encoding="utf-8")
            bad_evidence = {
                marker: {"evidence": "phrase absent from every comment"}
                for marker in votes
            }

            with self.assertRaises(RuntimeError):
                build_classifier_dataset(
                    input_directory,
                    output_directory,
                    combinations=[("ds", "Python")],
                    judge_runner=_ScriptedJudge(
                        votes, semantic_overrides=bad_evidence
                    ),
                    judge_passes=2,
                    overwrite=True,
                )

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "old output")
            self.assertEqual(
                (output_directory / "dataset.parquet").read_bytes(), original_dataset
            )
            self.assertFalse(
                list(root.glob(f".{output_directory.name}.tmp-*")),
                "failed staging directories should be cleaned",
            )

    def test_refuses_overlapping_or_unrecognized_output_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_directory = root / "input"
            _write_combination(input_directory, "ds", "Python", _base_rows()[:3])
            judge = _ScriptedJudge(
                {
                    "[R-KEEP]": LABEL_SHARING_RESTRICTION,
                    "[L-KEEP]": LABEL_MISSED_LICENSE,
                    "[I-KEEP]": LABEL_IRRELEVANT,
                }
            )
            for unsafe_output in (
                input_directory,
                root,
                input_directory / "nested-output",
            ):
                with self.subTest(output=unsafe_output):
                    with self.assertRaises(ValueError):
                        build_classifier_dataset(
                            input_directory,
                            unsafe_output,
                            combinations=[("ds", "Python")],
                            judge_runner=judge,
                            judge_passes=2,
                            overwrite=True,
                        )
            unknown_output = root / "unknown-output"
            unknown_output.mkdir()
            marker = unknown_output / "unrelated.txt"
            marker.write_text("do not replace", encoding="utf-8")
            with self.assertRaises(ValueError):
                build_classifier_dataset(
                    input_directory,
                    unknown_output,
                    combinations=[("ds", "Python")],
                    judge_runner=judge,
                    judge_passes=2,
                    overwrite=True,
                )
            self.assertEqual(marker.read_text(encoding="utf-8"), "do not replace")

    def test_cache_is_batch_scoped_and_retargets_candidate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            first_input = root / "first-input"
            second_input = root / "second-input"
            first_rows = _base_rows()[:3]
            second_rows: list[dict[str, object]] = []
            for row in first_rows:
                copied = dict(row)
                copied["record_id"] = f"new-{row['record_id']}"
                # Path is semantic judge context and deliberately remains stable.
                second_rows.append(copied)
            _write_combination(first_input, "ds", "Python", first_rows)
            _write_combination(second_input, "ds", "Python", second_rows)
            cache_path = root / "judge-cache.sqlite"
            votes = {
                "[R-KEEP]": LABEL_SHARING_RESTRICTION,
                "[L-KEEP]": LABEL_MISSED_LICENSE,
                "[I-KEEP]": LABEL_IRRELEVANT,
            }
            build_classifier_dataset(
                first_input,
                root / "first-output",
                combinations=[("ds", "Python")],
                judge_runner=_ScriptedJudge(votes),
                judge_passes=2,
                judge_cache_path=cache_path,
                codex_model="fixture-judge-v1",
            )

            def unexpected_judge(_prompt: str) -> str:
                raise AssertionError("identical batch should be served from cache")

            stats = build_classifier_dataset(
                second_input,
                root / "second-output",
                combinations=[("ds", "Python")],
                judge_runner=unexpected_judge,
                judge_passes=2,
                judge_cache_path=cache_path,
                codex_model="fixture-judge-v1",
            )
            self.assertEqual(stats.judge_cache_hits, 6)
            second = _read_rows(root / "second-output" / "dataset.parquet")
            for row in second:
                votes_payload = json.loads(row["judge_votes"])
                self.assertTrue(
                    all(vote["candidate_id"] == row["example_id"] for vote in votes_payload)
                )

    def test_judge_cache_resumes_at_successfully_committed_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_directory = root / "input"
            output_directory = root / "output"
            cache_path = root / "judge-cache.sqlite"
            _write_combination(input_directory, "ds", "Python", _base_rows()[:3])
            votes = {
                "[R-KEEP]": LABEL_SHARING_RESTRICTION,
                "[L-KEEP]": LABEL_MISSED_LICENSE,
                "[I-KEEP]": LABEL_IRRELEVANT,
            }
            first_judge = _ScriptedJudge(votes)
            calls = 0

            def interrupted_judge(prompt: str) -> str:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("simulated interruption")
                return first_judge(prompt)

            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                build_classifier_dataset(
                    input_directory,
                    output_directory,
                    combinations=[("ds", "Python")],
                    judge_runner=interrupted_judge,
                    judge_passes=2,
                    judge_batch_size=2,
                    judge_retries=0,
                    judge_cache_path=cache_path,
                    codex_model="fixture-judge-v1",
                )
            self.assertFalse(output_directory.exists())
            self.assertTrue(cache_path.is_file())

            resumed_judge = _ScriptedJudge(votes)
            stats = build_classifier_dataset(
                input_directory,
                output_directory,
                combinations=[("ds", "Python")],
                judge_runner=resumed_judge,
                judge_passes=2,
                judge_batch_size=2,
                judge_retries=0,
                judge_cache_path=cache_path,
                codex_model="fixture-judge-v1",
            )
            self.assertEqual(stats.judge_cache_hits, 2)
            self.assertEqual(stats.judge_cache_misses, 4)
            self.assertEqual(stats.judge_calls, 3)
            self.assertEqual(len(resumed_judge.calls), 4)

    def test_requires_exactly_two_distinct_judge_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_directory = root / "input"
            _write_combination(input_directory, "ds", "Python", _base_rows()[:3])
            with self.assertRaisesRegex(ValueError, "exactly 2"):
                build_classifier_dataset(
                    input_directory,
                    root / "output",
                    combinations=[("ds", "Python")],
                    judge_runner=lambda _prompt: "{}",
                    judge_passes=3,
                )


if __name__ == "__main__":
    unittest.main()
