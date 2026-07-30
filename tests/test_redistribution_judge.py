from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import patch

from commentminer.redistribution_judge import (
    JUDGE_LABELS,
    LABEL_AMBIGUOUS,
    LABEL_CODE_REDISTRIBUTION_INTENT,
    JUDGMENT_PROFILE_NON_LICENSE_LIMITATIONS,
    LABEL_LICENSE_ONLY,
    LABEL_NON_LICENSE_LIMITATION,
    LABEL_NON_LICENSE_LIMITATION_WITH_LICENSE,
    LIMITATION_JUDGE_LABELS,
    LABEL_OTHER,
    _judge_output_schema,
    _limitation_judge_output_schema,
    _parse_limitation_decisions,
    _parse_decisions,
    judge_redistribution_candidates,
    redistribution_judge_rubric,
    redistribution_limitation_judge_rubric,
)


def _prompt_candidates(prompt: str) -> list[dict[str, object]]:
    marker = "Untrusted candidate data:\n"
    return json.loads(prompt.split(marker, 1)[1])


def _response_for_prompt(
    prompt: str,
    labels: dict[str, str],
) -> tuple[str, dict[str, int]]:
    decisions = []
    for candidate in _prompt_candidates(prompt):
        candidate_id = str(candidate["candidate_id"])
        decisions.append(
            {
                "candidate_id": candidate_id,
                "label": labels[candidate_id],
                "confidence": 0.75,
                "evidence": str(candidate["comment"]),
                "rationale": "Fixture decision.",
            }
        )
    return json.dumps({"decisions": decisions}), {
        "input_tokens": 1,
        "cached_input_tokens": 0,
        "output_tokens": 2,
    }


def _limitation_other_decision(
    candidate: dict[str, object],
    *,
    evidence: str | None = None,
) -> dict[str, object]:
    return {
        "candidate_id": str(candidate["candidate_id"]),
        "label": LABEL_OTHER,
        "confidence": 0.75,
        "is_non_license_redistribution_limitation": False,
        "is_license_notice": False,
        "is_known_license": False,
        "known_license": None,
        "restriction_evidence": None,
        "license_evidence": None,
        "evidence": evidence or str(candidate["comment"]),
        "rationale": "Fixture decision.",
    }


def _limitation_response_for_prompt(
    prompt: str,
    *,
    invalid_evidence_ids: frozenset[str] = frozenset(),
) -> tuple[str, dict[str, int]]:
    prompt_candidates = _prompt_candidates(prompt)
    decisions = [
        _limitation_other_decision(
            candidate,
            evidence=(
                "fabricated evidence absent from the comment"
                if str(candidate["candidate_id"]) in invalid_evidence_ids
                else None
            ),
        )
        for candidate in prompt_candidates
    ]
    return json.dumps({"decisions": decisions}), {
        "input_tokens": len(prompt_candidates),
        "cached_input_tokens": 0,
        "output_tokens": 2 * len(prompt_candidates),
    }


class RedistributionJudgeDecisionTests(unittest.TestCase):
    def _assert_legacy_presentation_evidence(
        self,
        *,
        candidate_id: str,
        opening_comment: str,
        evidence: str,
        paraphrase: str,
    ) -> None:
        batch = [
            {
                "candidate_id": candidate_id,
                "opening_comment": opening_comment,
            }
        ]

        def response_for(quoted_evidence: str) -> str:
            return json.dumps(
                {
                    "decisions": [
                        {
                            "candidate_id": candidate_id,
                            "label": LABEL_NON_LICENSE_LIMITATION,
                            "confidence": 0.99,
                            "is_non_license_redistribution_limitation": True,
                            "is_license_notice": False,
                            "is_known_license": False,
                            "known_license": None,
                            "restriction_evidence": quoted_evidence,
                            "license_evidence": None,
                            "evidence": quoted_evidence,
                            "rationale": "The comment restricts source distribution.",
                        }
                    ]
                }
            )

        parsed = _parse_limitation_decisions(response_for(evidence), batch)

        self.assertEqual(parsed[0]["restriction_evidence"], evidence)
        self.assertEqual(parsed[0]["judge_evidence"], evidence)
        with self.assertRaisesRegex(
            RuntimeError,
            "restriction_evidence is not present",
        ):
            _parse_limitation_decisions(response_for(paraphrase), batch)

    def test_non_license_rubric_has_specific_unpublished_work_boundary(self) -> None:
        rubric = redistribution_limitation_judge_rubric()

        self.assertIn("specific unpublished-work check", rubric)
        self.assertIn("does not evidence actual or intended publication", rubric)
        self.assertIn("paper, specification, dataset", rubric)
        self.assertIn("not itself a software license", rubric)

    def test_rubrics_exclude_within_project_copying_guidance(self) -> None:
        broad_rubric = redistribution_judge_rubric()
        limitation_rubric = redistribution_limitation_judge_rubric()

        for rubric in (broad_rubric, limitation_rubric):
            self.assertIn("setting is only for this project", rubric)
            self.assertIn("please do not copy", rubric)
            self.assertIn("DO NOT COPY/PASTE", rubric)
            self.assertIn("View Source", rubric)
            self.assertIn("bad workaround", rubric)
            self.assertIn("do not copy this code", rubric)
            self.assertIn("do not duplicate", rubric)
            self.assertIn("call the shared helper", rubric)

        self.assertIn("external dissemination", broad_rubric)
        self.assertIn('The words "copy", "paste", "duplicate"', limitation_rubric)
        self.assertIn("never sufficient", limitation_rubric)
        self.assertIn("Attribution or plagiarism advice", limitation_rubric)

    def test_non_license_profile_replaces_joined_summary_with_grounded_axis_evidence(self) -> None:
        batch = [
            {
                "candidate_id": "mixed",
                "opening_comment": "CONFIDENTIAL. NOTICE: Licensed under MIT.",
            }
        ]
        response = json.dumps(
            {
                "decisions": [
                    {
                        "candidate_id": "mixed",
                        "label": LABEL_NON_LICENSE_LIMITATION_WITH_LICENSE,
                        "confidence": 0.99,
                        "is_non_license_redistribution_limitation": True,
                        "is_license_notice": True,
                        "is_known_license": True,
                        "known_license": "MIT",
                        "restriction_evidence": "CONFIDENTIAL",
                        "license_evidence": "Licensed under MIT",
                        "evidence": "CONFIDENTIAL; Licensed under MIT",
                        "rationale": "The confidentiality restriction is separate from MIT.",
                    }
                ]
            }
        )

        parsed = _parse_limitation_decisions(response, batch)

        self.assertEqual(parsed[0]["judge_evidence"], "CONFIDENTIAL")

    def test_non_license_profile_keeps_restriction_and_license_axes_independent(self) -> None:
        batch = [
            {
                "candidate_id": "restricted",
                "opening_comment": "Confidential source code. Do not distribute outside Acme.",
            },
            {
                "candidate_id": "license",
                "opening_comment": "Licensed under the Apache License, Version 2.0.",
            },
            {
                "candidate_id": "mixed",
                "opening_comment": (
                    "Licensed under MIT. Confidential: do not disclose to third parties."
                ),
            },
            {
                "candidate_id": "other",
                "opening_comment": "Distribute tasks across worker threads.",
            },
        ]
        response = json.dumps(
            {
                "decisions": [
                    {
                        "candidate_id": "restricted",
                        "label": LABEL_NON_LICENSE_LIMITATION,
                        "confidence": 0.99,
                        "is_non_license_redistribution_limitation": True,
                        "is_license_notice": False,
                        "is_known_license": False,
                        "known_license": None,
                        "restriction_evidence": "Do not distribute outside Acme",
                        "license_evidence": None,
                        "evidence": "Do not distribute outside Acme",
                        "rationale": "An employer confidentiality rule limits recipients.",
                    },
                    {
                        "candidate_id": "license",
                        "label": LABEL_LICENSE_ONLY,
                        "confidence": 0.99,
                        "is_non_license_redistribution_limitation": False,
                        "is_license_notice": True,
                        "is_known_license": True,
                        "known_license": "Apache-2.0",
                        "restriction_evidence": None,
                        "license_evidence": "Apache License, Version 2.0",
                        "evidence": "Apache License, Version 2.0",
                        "rationale": "This is a named software license only.",
                    },
                    {
                        "candidate_id": "mixed",
                        "label": LABEL_NON_LICENSE_LIMITATION_WITH_LICENSE,
                        "confidence": 0.98,
                        "is_non_license_redistribution_limitation": True,
                        "is_license_notice": True,
                        "is_known_license": True,
                        "known_license": "MIT",
                        "restriction_evidence": "do not disclose to third parties",
                        "license_evidence": "Licensed under MIT",
                        "evidence": "Confidential: do not disclose to third parties",
                        "rationale": "The confidentiality limit is independent of MIT.",
                    },
                    {
                        "candidate_id": "other",
                        "label": LABEL_OTHER,
                        "confidence": 0.99,
                        "is_non_license_redistribution_limitation": False,
                        "is_license_notice": False,
                        "is_known_license": False,
                        "known_license": None,
                        "restriction_evidence": None,
                        "license_evidence": None,
                        "evidence": "Distribute tasks",
                        "rationale": "This is work scheduling.",
                    },
                ]
            }
        )

        parsed = {
            row["candidate_id"]: row
            for row in _parse_limitation_decisions(response, batch)
        }

        self.assertEqual(
            _limitation_judge_output_schema()["properties"]["decisions"]["items"][
                "properties"
            ]["label"]["enum"],
            list(LIMITATION_JUDGE_LABELS),
        )
        self.assertTrue(
            parsed["restricted"]["is_non_license_redistribution_limitation"]
        )
        self.assertFalse(parsed["restricted"]["is_license_notice"])
        self.assertEqual(parsed["license"]["judge_label"], LABEL_LICENSE_ONLY)
        self.assertEqual(parsed["license"]["known_license"], "Apache-2.0")
        self.assertEqual(
            parsed["mixed"]["judge_label"],
            LABEL_NON_LICENSE_LIMITATION_WITH_LICENSE,
        )

    def test_exact_labels_map_to_boolean_invariant(self) -> None:
        batch = [
            {
                "candidate_id": "candidate-code",
                "opening_comment": "Private source code; do not distribute.",
            },
            {
                "candidate_id": "candidate-other",
                "opening_comment": "Distribute tasks across the worker pool.",
            },
            {
                "candidate_id": "candidate-ambiguous",
                "opening_comment": "Redistribution prohibited.",
            },
        ]
        response = json.dumps(
            {
                "decisions": [
                    {
                        "candidate_id": "candidate-code",
                        "label": LABEL_CODE_REDISTRIBUTION_INTENT,
                        "confidence": 0.99,
                        "evidence": "do not distribute",
                        "rationale": "The restriction applies to source code.",
                    },
                    {
                        "candidate_id": "candidate-other",
                        "label": LABEL_OTHER,
                        "confidence": 0.98,
                        "evidence": "Distribute tasks",
                        "rationale": "This describes work scheduling.",
                    },
                    {
                        "candidate_id": "candidate-ambiguous",
                        "label": LABEL_AMBIGUOUS,
                        "confidence": 0.5,
                        "evidence": "Redistribution prohibited",
                        "rationale": "The object is not identified.",
                    },
                ]
            }
        )

        parsed = {
            row["candidate_id"]: row for row in _parse_decisions(response, batch)
        }

        self.assertEqual(
            JUDGE_LABELS,
            (
                "code_redistribution_intent",
                "other",
                "ambiguous",
            ),
        )
        self.assertEqual(
            _judge_output_schema()["properties"]["decisions"]["items"][
                "properties"
            ]["label"]["enum"],
            list(JUDGE_LABELS),
        )
        self.assertIs(
            parsed["candidate-code"]["is_code_redistribution_intent"], True
        )
        self.assertIs(
            parsed["candidate-other"]["is_code_redistribution_intent"], False
        )
        self.assertIsNone(
            parsed["candidate-ambiguous"]["is_code_redistribution_intent"]
        )

    def test_rejects_noncanonical_label_and_non_source_evidence(self) -> None:
        batch = [
            {
                "candidate_id": "candidate-1",
                "opening_comment": "Do not distribute this source code.",
            }
        ]

        with self.assertRaisesRegex(RuntimeError, "invalid label"):
            _parse_decisions(
                json.dumps(
                    {
                        "decisions": [
                            {
                                "candidate_id": "candidate-1",
                                "label": "CODE_REDISTRIBUTION_INTENT",
                                "confidence": 0.9,
                                "evidence": "Do not distribute",
                                "rationale": "Wrong label spelling.",
                            }
                        ]
                    }
                ),
                batch,
            )

        with self.assertRaisesRegex(RuntimeError, "evidence is not present"):
            _parse_decisions(
                json.dumps(
                    {
                        "decisions": [
                            {
                                "candidate_id": "candidate-1",
                                "label": LABEL_CODE_REDISTRIBUTION_INTENT,
                                "confidence": 0.9,
                                "evidence": "confidential customer material",
                                "rationale": "Invented evidence.",
                            }
                        ]
                    }
                ),
                batch,
            )

    def test_accepts_evidence_across_javadoc_line_decoration(self) -> None:
        batch = [
            {
                "candidate_id": "candidate-javadoc",
                "opening_comment": (
                    "/**\r\n"
                    " * This source is confidential and <br/>\r\n"
                    " * proprietary information belonging to Acme under the license.\r\n"
                    " */"
                ),
            }
        ]
        response = json.dumps(
            {
                "decisions": [
                    {
                        "candidate_id": "candidate-javadoc",
                        "label": LABEL_CODE_REDISTRIBUTION_INTENT,
                        "confidence": 0.95,
                        "evidence": (
                            "confidential and proprietary information ... "
                            "under the license"
                        ),
                        "rationale": "The comment identifies restricted source.",
                    }
                ]
            }
        )

        parsed = _parse_decisions(response, batch)

        self.assertEqual(len(parsed), 1)
        self.assertEqual(
            parsed[0]["judge_evidence"],
            "confidential and proprietary information ... under the license",
        )

        silently_shortened = json.loads(response)
        silently_shortened["decisions"][0]["evidence"] = (
            "confidential and proprietary information under the license"
        )
        with self.assertRaisesRegex(RuntimeError, "evidence is not present"):
            _parse_decisions(json.dumps(silently_shortened), batch)

    def test_accepts_omitted_quoted_one_word_legal_alias(self) -> None:
        batch = [
            {
                "candidate_id": "candidate-defined-term",
                "opening_comment": (
                    "/** Redistribution and use of this software and associated "
                    "documentation (the\r\n"
                    ' * "Software"), with or without modification, '
                    "are permitted. */"
                ),
            }
        ]
        response = json.dumps(
            {
                "decisions": [
                    {
                        "candidate_id": "candidate-defined-term",
                        "label": LABEL_CODE_REDISTRIBUTION_INTENT,
                        "confidence": 0.99,
                        "evidence": (
                            "Redistribution and use of this software and associated "
                            "documentation, with or without modification, are permitted"
                        ),
                        "rationale": "The comment grants redistribution permission.",
                    }
                ]
            }
        )

        self.assertEqual(len(_parse_decisions(response, batch)), 1)

    def test_accepts_one_corrected_character_typo_in_grounded_evidence(self) -> None:
        batch = [
            {
                "candidate_id": "candidate-typo",
                "opening_comment": (
                    "This code cannot be reproduced or used without he express "
                    "written consent of the owner."
                ),
            }
        ]
        response = json.dumps(
            {
                "decisions": [
                    {
                        "candidate_id": "candidate-typo",
                        "label": LABEL_CODE_REDISTRIBUTION_INTENT,
                        "confidence": 0.99,
                        "evidence": (
                            "cannot be reproduced or used without the express "
                            "written consent"
                        ),
                        "rationale": "The code cannot be reproduced without consent.",
                    }
                ]
            }
        )

        self.assertEqual(len(_parse_decisions(response, batch)), 1)

    def test_accepts_cobol_fixed_format_evidence_without_sequence_or_program_id(self) -> None:
        def cobol_comment_line(sequence: int, text: str) -> str:
            self.assertLessEqual(len(text), 65)
            return f"{sequence:06d}*{text:<65}ACME0001"

        opening_comment = "\r\n".join(
            [
                cobol_comment_line(
                    100,
                    "CONFIDENTIAL SOURCE CODE. DO NOT DISCLOSE OR",
                ),
                cobol_comment_line(
                    200,
                    "DISTRIBUTE OUTSIDE ACME.",
                ),
            ]
        )
        self._assert_legacy_presentation_evidence(
            candidate_id="candidate-cobol-fixed",
            opening_comment=opening_comment,
            evidence=(
                "CONFIDENTIAL SOURCE CODE. DO NOT DISCLOSE OR "
                "DISTRIBUTE OUTSIDE ACME."
            ),
            paraphrase=(
                "CONFIDENTIAL SOURCE CODE MUST NOT BE SHARED OUTSIDE ACME."
            ),
        )

    def test_accepts_evidence_without_legacy_line_comment_prefixes(self) -> None:
        presentations = {
            "rem": "REM ",
            "comment": "COMMENT ",
            "texinfo": "@c ",
            "j": "NB. ",
            "fortran-bang": "! ",
            "fortran-bang-c": "!C ",
        }
        for name, prefix in presentations.items():
            with self.subTest(presentation=name):
                self._assert_legacy_presentation_evidence(
                    candidate_id=f"candidate-{name}",
                    opening_comment=(
                        f"{prefix}CONFIDENTIAL SOURCE CODE.\n"
                        f"{prefix}DO NOT DISTRIBUTE OUTSIDE ACME."
                    ),
                    evidence=(
                        "CONFIDENTIAL SOURCE CODE. DO NOT DISTRIBUTE OUTSIDE ACME."
                    ),
                    paraphrase=(
                        "CONFIDENTIAL SOURCE CODE MUST NOT BE SHARED OUTSIDE ACME."
                    ),
                )

    def test_accepts_forth_evidence_across_hyphenated_line_wrap(self) -> None:
        self._assert_legacy_presentation_evidence(
            candidate_id="candidate-forth",
            opening_comment=(
                "\\ PROPRIETARY AND CONFIDEN-\r\n"
                "\\ TIAL SOURCE CODE.\r\n"
                "\\ DO NOT DISTRIBUTE OUTSIDE ACME."
            ),
            evidence=(
                "PROPRIETARY AND CONFIDENTIAL SOURCE CODE. "
                "DO NOT DISTRIBUTE OUTSIDE ACME."
            ),
            paraphrase=(
                "PROPRIETARY AND CONFIDENTIAL SOURCE CODE MUST NOT BE SHARED "
                "OUTSIDE ACME."
            ),
        )

    def test_accepts_one_token_hard_wrapped_across_comment_prefixes(self) -> None:
        self._assert_legacy_presentation_evidence(
            candidate_id="candidate-comment-hard-wrap",
            opening_comment=(
                "Comment This source may not be copied without prior wr\r\n"
                "Comment itten permission."
            ),
            evidence=(
                "This source may not be copied without prior written permission."
            ),
            paraphrase=(
                "This source may only be copied with the owner's consent."
            ),
        )

    def test_positive_axis_keeps_complete_sentence_from_interleaved_evidence(self) -> None:
        batch = [
            {
                "candidate_id": "candidate-interleaved",
                "opening_comment": (
                    "Comment This source is proprietary.                 LOVEWARE\n"
                    "Comment Written permission is required before copying or "
                    "redistribution."
                ),
            }
        ]
        joined_evidence = (
            "This source is proprietary. Written permission is required before "
            "copying or redistribution."
        )
        grounded_sentence = (
            "Written permission is required before copying or redistribution."
        )
        response = {
            "decisions": [
                {
                    "candidate_id": "candidate-interleaved",
                    "label": LABEL_NON_LICENSE_LIMITATION,
                    "confidence": 0.99,
                    "is_non_license_redistribution_limitation": True,
                    "is_license_notice": False,
                    "is_known_license": False,
                    "known_license": None,
                    "restriction_evidence": joined_evidence,
                    "license_evidence": None,
                    "evidence": joined_evidence,
                    "rationale": "The comment limits copying and redistribution.",
                }
            ]
        }

        parsed = _parse_limitation_decisions(json.dumps(response), batch)

        self.assertEqual(parsed[0]["restriction_evidence"], grounded_sentence)
        self.assertEqual(parsed[0]["judge_evidence"], grounded_sentence)

        paraphrased = json.loads(json.dumps(response))
        paraphrased_clause = (
            "Copying or redistribution requires the owner's prior consent."
        )
        paraphrased["decisions"][0]["restriction_evidence"] = paraphrased_clause
        paraphrased["decisions"][0]["evidence"] = paraphrased_clause
        with self.assertRaisesRegex(
            RuntimeError,
            "restriction_evidence is not present",
        ):
            _parse_limitation_decisions(json.dumps(paraphrased), batch)

    def test_other_decision_falls_back_to_exact_best_match_excerpt(self) -> None:
        best_match_excerpt = (
            "Distribute the queued jobs among available worker threads."
        )
        batch = [
            {
                "candidate_id": "candidate-other-excerpt",
                "opening_comment": f"// {best_match_excerpt}",
                "best_match_excerpt": best_match_excerpt,
            }
        ]
        response = json.dumps(
            {
                "decisions": [
                    {
                        "candidate_id": "candidate-other-excerpt",
                        "label": LABEL_OTHER,
                        "confidence": 0.99,
                        "is_non_license_redistribution_limitation": False,
                        "is_license_notice": False,
                        "is_known_license": False,
                        "known_license": None,
                        "restriction_evidence": None,
                        "license_evidence": None,
                        "evidence": "Work is spread evenly over the worker pool.",
                        "rationale": "This describes work scheduling.",
                    }
                ]
            }
        )

        parsed = _parse_limitation_decisions(response, batch)

        self.assertEqual(parsed[0]["judge_label"], LABEL_OTHER)
        self.assertEqual(parsed[0]["judge_evidence"], best_match_excerpt)

        batch[0]["best_match_excerpt"] = "Tasks are balanced among workers."
        with self.assertRaisesRegex(RuntimeError, "evidence is not present"):
            _parse_limitation_decisions(response, batch)

    def test_accepts_underscores_used_as_prose_separators(self) -> None:
        self._assert_legacy_presentation_evidence(
            candidate_id="candidate-underscores",
            opening_comment=(
                "// CONFIDENTIAL_SOURCE_CODE__DO_NOT_DISTRIBUTE_OUTSIDE_ACME"
            ),
            evidence="CONFIDENTIAL SOURCE CODE DO NOT DISTRIBUTE OUTSIDE ACME",
            paraphrase="CONFIDENTIAL SOURCE CODE MUST NOT BE SHARED OUTSIDE ACME",
        )

    def test_accepts_org_link_label_as_grounded_evidence(self) -> None:
        self._assert_legacy_presentation_evidence(
            candidate_id="candidate-org-link",
            opening_comment=(
                "# Do not distribute "
                "[[https://intranet.example/policy][this source code]] outside Acme."
            ),
            evidence="Do not distribute this source code outside Acme.",
            paraphrase="Do not share this source code beyond Acme.",
        )

    def test_accepts_javadoc_inline_link_as_grounded_evidence(self) -> None:
        self._assert_legacy_presentation_evidence(
            candidate_id="candidate-javadoc-link",
            opening_comment=(
                "/** Do not distribute the {@link resource} outside Acme. */"
            ),
            evidence="Do not distribute the resource outside Acme.",
            paraphrase="Do not share the resource beyond Acme.",
        )

    def test_accepts_html_entities_in_grounded_evidence(self) -> None:
        self._assert_legacy_presentation_evidence(
            candidate_id="candidate-html-entity",
            opening_comment=(
                "/** Confidential source code. Do not distribute outside "
                "Acme &amp; approved affiliates. */"
            ),
            evidence=(
                "Confidential source code. Do not distribute outside "
                "Acme & approved affiliates."
            ),
            paraphrase=(
                "Confidential source code must not be shared beyond "
                "Acme and approved affiliates."
            ),
        )


class RedistributionJudgeExecutionTests(unittest.TestCase):
    def test_non_license_profile_uses_distinct_prompt_and_cache_identity(self) -> None:
        candidate = {
            "candidate_id": "candidate-license",
            "opening_comment": "Licensed under the Apache License, Version 2.0.",
        }

        def runner(prompt: str) -> tuple[str, dict[str, int]]:
            self.assertIn("Non-license redistribution-limitation", prompt)
            return json.dumps(
                {
                    "decisions": [
                        {
                            "candidate_id": "candidate-license",
                            "label": LABEL_LICENSE_ONLY,
                            "confidence": 0.99,
                            "is_non_license_redistribution_limitation": False,
                            "is_license_notice": True,
                            "is_known_license": True,
                            "known_license": "Apache-2.0",
                            "restriction_evidence": None,
                            "license_evidence": "Apache License, Version 2.0",
                            "evidence": "Apache License, Version 2.0",
                            "rationale": "A named license with no extra restriction.",
                        }
                    ]
                }
            ), {}

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            decisions, stats, configuration = judge_redistribution_candidates(
                [candidate],
                output_directory=root / "output",
                cache_path=root / "cache.sqlite",
                judgment_profile=JUDGMENT_PROFILE_NON_LICENSE_LIMITATIONS,
                batch_size=1,
                workers=2,
                max_attempts=1,
                runner=runner,
            )

        self.assertEqual(decisions["candidate-license"]["judge_label"], LABEL_LICENSE_ONLY)
        self.assertEqual(stats.judged, 1)
        self.assertEqual(
            configuration["judgment_profile"],
            JUDGMENT_PROFILE_NON_LICENSE_LIMITATIONS,
        )
        self.assertIn(
            "non-license-redistribution-limitations-v3",
            configuration["model_identity"],
        )

    def test_requires_luna_with_supported_reasoning(self) -> None:
        candidate = {
            "candidate_id": "candidate-code",
            "opening_comment": "Private source code; do not distribute.",
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            common = {
                "candidates": [candidate],
                "output_directory": root / "output",
                "cache_path": root / "cache.sqlite",
                "runner": lambda _: ("", {}),
            }
            with self.assertRaisesRegex(ValueError, "gpt-5.6-luna"):
                judge_redistribution_candidates(
                    **common,
                    codex_model="gpt-5.5",
                )
            with self.assertRaisesRegex(ValueError, "one of low, max"):
                judge_redistribution_candidates(
                    **common,
                    reasoning_effort="high",
                )
            decisions, _, configuration = judge_redistribution_candidates(
                **{**common, "candidates": []},
                reasoning_effort="low",
            )
            self.assertEqual(decisions, {})
            self.assertEqual(configuration["reasoning_effort"], "low")

    def test_batches_run_concurrently_and_second_run_uses_cache(self) -> None:
        candidates = [
            {
                "candidate_id": "candidate-code-a",
                "opening_comment": "Private source code A; do not distribute.",
                "path": "A.java",
            },
            {
                "candidate_id": "candidate-code-b",
                "opening_comment": "Private source code B; do not distribute.",
                "path": "B.java",
            },
            {
                "candidate_id": "candidate-other",
                "opening_comment": "Distribute tasks across worker threads.",
                "path": "Scheduler.java",
            },
            {
                "candidate_id": "candidate-ambiguous",
                "opening_comment": "Redistribution prohibited.",
                "path": "Unknown.java",
            },
        ]
        labels = {
            "candidate-code-a": LABEL_CODE_REDISTRIBUTION_INTENT,
            "candidate-code-b": LABEL_CODE_REDISTRIBUTION_INTENT,
            "candidate-other": LABEL_OTHER,
            "candidate-ambiguous": LABEL_AMBIGUOUS,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            cache_path = root / "judge-cache.sqlite"
            all_started = threading.Event()
            state_lock = threading.Lock()
            state = {"active": 0, "started": 0, "maximum_active": 0}

            def concurrent_runner(prompt: str) -> tuple[str, dict[str, int]]:
                with state_lock:
                    state["active"] += 1
                    state["started"] += 1
                    state["maximum_active"] = max(
                        state["maximum_active"], state["active"]
                    )
                    if state["started"] == len(candidates):
                        all_started.set()
                try:
                    if not all_started.wait(timeout=5):
                        raise AssertionError("judge batches did not execute concurrently")
                    return _response_for_prompt(prompt, labels)
                finally:
                    with state_lock:
                        state["active"] -= 1

            decisions, stats, configuration = judge_redistribution_candidates(
                candidates,
                output_directory=root / "first-run",
                cache_path=cache_path,
                batch_size=1,
                workers=4,
                max_attempts=1,
                runner=concurrent_runner,
            )

            def unexpected_runner(_: str) -> tuple[str, dict[str, int]]:
                raise AssertionError("cache hit unexpectedly invoked the runner")

            cached_decisions, cached_stats, cached_configuration = (
                judge_redistribution_candidates(
                    candidates,
                    output_directory=root / "second-run",
                    cache_path=cache_path,
                    batch_size=1,
                    workers=4,
                    max_attempts=1,
                    runner=unexpected_runner,
                )
            )

            self.assertEqual(state["maximum_active"], 4)
            self.assertEqual(stats.candidates, 4)
            self.assertEqual(stats.judged, 4)
            self.assertEqual(stats.batches, 4)
            self.assertEqual(stats.calls, 4)
            self.assertEqual(stats.cache_hits, 0)
            self.assertEqual(stats.input_tokens, 4)
            self.assertEqual(stats.output_tokens, 8)
            self.assertEqual(set(decisions), set(labels))
            self.assertEqual(configuration["model"], "gpt-5.6-luna")
            self.assertEqual(configuration["reasoning_effort"], "max")

            self.assertEqual(cached_decisions, decisions)
            self.assertEqual(cached_stats.judged, 4)
            self.assertEqual(cached_stats.batches, 4)
            self.assertEqual(cached_stats.calls, 0)
            self.assertEqual(cached_stats.cache_hits, 4)
            self.assertEqual(
                cached_configuration["model_identity"],
                configuration["model_identity"],
            )
            self.assertTrue((root / "first-run" / "judge-rubric.md").is_file())
            self.assertTrue(
                (root / "first-run" / "judge-output.schema.json").is_file()
            )
            self.assertTrue(
                (root / "first-run" / "judge-responses.jsonl").is_file()
            )
            self.assertTrue((root / "first-run" / "judge-errors.jsonl").is_file())

            regrouped_decisions, regrouped_stats, _ = (
                judge_redistribution_candidates(
                    candidates,
                    output_directory=root / "regrouped-run",
                    cache_path=cache_path,
                    batch_size=2,
                    workers=2,
                    max_attempts=1,
                    runner=unexpected_runner,
                )
            )
            self.assertEqual(regrouped_decisions, decisions)
            self.assertEqual(regrouped_stats.batches, 2)
            self.assertEqual(regrouped_stats.calls, 0)
            self.assertEqual(regrouped_stats.cache_hits, 4)

    def test_invalid_cached_decision_is_rejudged(self) -> None:
        candidate = {
            "candidate_id": "candidate-code",
            "opening_comment": "Private source code; do not distribute.",
            "path": "Private.java",
        }
        labels = {"candidate-code": LABEL_CODE_REDISTRIBUTION_INTENT}
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            cache_path = root / "judge-cache.sqlite"
            judge_redistribution_candidates(
                [candidate],
                output_directory=root / "first-run",
                cache_path=cache_path,
                batch_size=1,
                workers=1,
                max_attempts=1,
                runner=lambda prompt: _response_for_prompt(prompt, labels),
            )
            with sqlite3.connect(cache_path) as connection:
                cached = json.loads(
                    connection.execute(
                        "SELECT decision_json FROM redistribution_decisions"
                    ).fetchone()[0]
                )
                cached["judge_evidence"] = "invented evidence"
                connection.execute(
                    "UPDATE redistribution_decisions SET decision_json = ?",
                    (json.dumps(cached),),
                )

            calls = 0

            def replacement_runner(prompt: str) -> tuple[str, dict[str, int]]:
                nonlocal calls
                calls += 1
                return _response_for_prompt(prompt, labels)

            _, stats, _ = judge_redistribution_candidates(
                [candidate],
                output_directory=root / "second-run",
                cache_path=cache_path,
                batch_size=1,
                workers=1,
                max_attempts=1,
                runner=replacement_runner,
            )

            self.assertEqual(calls, 1)
            self.assertEqual(stats.cache_hits, 0)
            self.assertEqual(stats.calls, 1)

    def test_partially_cached_batch_judges_only_missing_candidates(self) -> None:
        cached_candidate = {
            "candidate_id": "candidate-cached",
            "opening_comment": "Private source code; do not distribute.",
            "path": "Cached.java",
        }
        new_candidate = {
            "candidate_id": "candidate-new",
            "opening_comment": "Distribute tasks across the worker pool.",
            "path": "New.java",
        }
        labels = {
            "candidate-cached": LABEL_CODE_REDISTRIBUTION_INTENT,
            "candidate-new": LABEL_OTHER,
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            cache_path = root / "judge-cache.sqlite"
            judge_redistribution_candidates(
                [cached_candidate],
                output_directory=root / "first-run",
                cache_path=cache_path,
                batch_size=2,
                workers=1,
                max_attempts=1,
                runner=lambda prompt: _response_for_prompt(prompt, labels),
            )

            prompts: list[str] = []

            def missing_only_runner(prompt: str) -> tuple[str, dict[str, int]]:
                prompts.append(prompt)
                return _response_for_prompt(prompt, labels)

            decisions, stats, _ = judge_redistribution_candidates(
                [cached_candidate, new_candidate],
                output_directory=root / "second-run",
                cache_path=cache_path,
                batch_size=2,
                workers=1,
                max_attempts=1,
                runner=missing_only_runner,
            )

            self.assertEqual(set(decisions), set(labels))
            self.assertEqual(stats.cache_hits, 1)
            self.assertEqual(stats.calls, 1)
            self.assertEqual(len(prompts), 1)
            self.assertEqual(
                [row["candidate_id"] for row in _prompt_candidates(prompts[0])],
                ["candidate-new"],
            )

    def test_partial_batch_evidence_failure_retries_only_invalid_candidate(self) -> None:
        candidates = [
            {
                "candidate_id": "candidate-valid-a",
                "opening_comment": "Alpha fixture comment.",
                "path": "Alpha.java",
            },
            {
                "candidate_id": "candidate-invalid",
                "opening_comment": "Beta fixture comment.",
                "path": "Beta.java",
            },
            {
                "candidate_id": "candidate-valid-c",
                "opening_comment": "Gamma fixture comment.",
                "path": "Gamma.java",
            },
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            cache_path = root / "judge-cache.sqlite"
            prompt_candidate_ids: list[list[str]] = []

            def partial_failure_runner(prompt: str) -> tuple[str, dict[str, int]]:
                candidate_ids = [
                    str(candidate["candidate_id"])
                    for candidate in _prompt_candidates(prompt)
                ]
                prompt_candidate_ids.append(candidate_ids)
                return _limitation_response_for_prompt(
                    prompt,
                    invalid_evidence_ids=(
                        frozenset({"candidate-invalid"})
                        if len(prompt_candidate_ids) == 1
                        else frozenset()
                    ),
                )

            with patch("commentminer.redistribution_judge.time.sleep"):
                decisions, stats, configuration = judge_redistribution_candidates(
                    candidates,
                    output_directory=root / "first-run",
                    cache_path=cache_path,
                    judgment_profile=JUDGMENT_PROFILE_NON_LICENSE_LIMITATIONS,
                    reasoning_effort="low",
                    batch_size=3,
                    workers=1,
                    max_attempts=2,
                    runner=partial_failure_runner,
                )

            self.assertEqual(
                prompt_candidate_ids,
                [
                    ["candidate-invalid", "candidate-valid-a", "candidate-valid-c"],
                    ["candidate-invalid"],
                ],
            )
            self.assertEqual(set(decisions), {row["candidate_id"] for row in candidates})
            self.assertEqual(stats.candidates, 3)
            self.assertEqual(stats.judged, 3)
            self.assertEqual(stats.batches, 1)
            self.assertEqual(stats.calls, 2)
            self.assertEqual(stats.retries, 1)
            self.assertEqual(stats.cache_hits, 0)
            self.assertEqual(stats.input_tokens, 4)
            self.assertEqual(stats.output_tokens, 8)
            self.assertEqual(configuration["reasoning_effort"], "low")
            self.assertEqual(
                configuration["judgment_profile"],
                JUDGMENT_PROFILE_NON_LICENSE_LIMITATIONS,
            )

            def unexpected_runner(_: str) -> tuple[str, dict[str, int]]:
                raise AssertionError("cache hit unexpectedly invoked the runner")

            cached_decisions, cached_stats, _ = judge_redistribution_candidates(
                candidates,
                output_directory=root / "cached-run",
                cache_path=cache_path,
                judgment_profile=JUDGMENT_PROFILE_NON_LICENSE_LIMITATIONS,
                reasoning_effort="low",
                batch_size=3,
                workers=1,
                max_attempts=2,
                runner=unexpected_runner,
            )

            self.assertEqual(cached_decisions, decisions)
            self.assertEqual(cached_stats.judged, 3)
            self.assertEqual(cached_stats.calls, 0)
            self.assertEqual(cached_stats.retries, 0)
            self.assertEqual(cached_stats.cache_hits, 3)

    def test_partial_batch_malformed_and_missing_decisions_retry_only_unresolved(self) -> None:
        candidates = [
            {
                "candidate_id": "candidate-valid",
                "opening_comment": "Valid fixture comment.",
                "path": "Valid.java",
            },
            {
                "candidate_id": "candidate-malformed",
                "opening_comment": "Malformed fixture comment.",
                "path": "Malformed.java",
            },
            {
                "candidate_id": "candidate-missing",
                "opening_comment": "Missing fixture comment.",
                "path": "Missing.java",
            },
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            prompt_candidate_ids: list[list[str]] = []

            def malformed_and_missing_runner(
                prompt: str,
            ) -> tuple[str, dict[str, int]]:
                prompt_candidates = _prompt_candidates(prompt)
                candidate_ids = [
                    str(candidate["candidate_id"])
                    for candidate in prompt_candidates
                ]
                prompt_candidate_ids.append(candidate_ids)
                if len(prompt_candidate_ids) == 1:
                    candidates_by_id = {
                        str(candidate["candidate_id"]): candidate
                        for candidate in prompt_candidates
                    }
                    response = {
                        "decisions": [
                            _limitation_other_decision(
                                candidates_by_id["candidate-valid"]
                            ),
                            {
                                "candidate_id": "candidate-malformed",
                                "label": "not-a-valid-label",
                            },
                        ]
                    }
                    return json.dumps(response), {
                        "input_tokens": 3,
                        "cached_input_tokens": 0,
                        "output_tokens": 4,
                    }
                return _limitation_response_for_prompt(prompt)

            with patch("commentminer.redistribution_judge.time.sleep"):
                decisions, stats, _ = judge_redistribution_candidates(
                    candidates,
                    output_directory=root / "output",
                    cache_path=root / "judge-cache.sqlite",
                    judgment_profile=JUDGMENT_PROFILE_NON_LICENSE_LIMITATIONS,
                    reasoning_effort="low",
                    batch_size=3,
                    workers=1,
                    max_attempts=2,
                    runner=malformed_and_missing_runner,
                )

            self.assertEqual(
                prompt_candidate_ids,
                [
                    ["candidate-malformed", "candidate-missing", "candidate-valid"],
                    ["candidate-malformed", "candidate-missing"],
                ],
            )
            self.assertEqual(set(decisions), {row["candidate_id"] for row in candidates})
            self.assertEqual(stats.judged, 3)
            self.assertEqual(stats.calls, 2)
            self.assertEqual(stats.retries, 1)
            self.assertEqual(stats.cache_hits, 0)
            self.assertEqual(stats.input_tokens, 5)
            self.assertEqual(stats.output_tokens, 8)


if __name__ == "__main__":
    unittest.main()
