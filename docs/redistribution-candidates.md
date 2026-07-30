# Redistribution-Intent Candidate Dataset

## Purpose and bounded source slice

This workflow builds an auditable sample of opening comments that resemble one
or more redistribution-related seed phrases, then asks a row-level Codex judge
whether each comment actually expresses an intent to control redistribution of
the code. Seed retrieval is only candidate generation: it is never treated as
the final label.

The reproducible preset selects `the-stack-v2-dedup` / `Java` from the local
Hugging Face-style tree or the `Jkatzy/code-comments` dataset at immutable
revision `0d4c83fac76705d2e2388186b628543a4916dab8`. Its
`source_files_limit=100000` bound means the first 100,000 **original source-file
rows**, identified from the retained Stack v2 provenance. It does not mean the
first 100,000 extracted comments or the first Parquet shard. Files without an
opening comment remain part of the source-file bound but cannot yield a
candidate.

## Formatting-tolerant seed retrieval

All 126 phrases in the four `SEED_TOPICS` groups are active by default. The 18
more specialized `GOVERNMENT_RESTRICTION_SEEDS` are separate and require
`--include-government-seeds`. The matcher is fuzzy and formatting-tolerant, so
Java comment decoration, punctuation, and line breaks between words do not
turn an otherwise equivalent phrase into an exact-match miss. The manifest
records the ordered seed inventory, normalization policy, and threshold used
by the run.

Every matched source occurrence is retained in `occurrences.parquet`. The
candidate table deduplicates normalized comment text for judging while keeping
the source occurrence evidence available for audit. Do not use match score or
matched seed text as a supervised target: both are retrieval metadata.

### Opt-in coverage expansion

Four additional seed families are available for broader provenance and
dissemination review. They are disabled by default so the original 126-seed
retrieval remains reproducible:

| Flag | Manifest group | Phrases | Retrieval focus |
| --- | --- | ---: | --- |
| `--include-provenance-seeds` | `proprietary_provenance` | 24 | Decompiled or reconstructed source, extraction from compiled packages, proprietary/nonpublic markings, and language suggesting copying or exposure without permission |
| `--include-funding-seeds` | `funding_dissemination` | 24 | Funding, grants, sponsorship, and procurement language near publication approval, embargo, distribution-statement, or government data-rights conditions |
| `--include-export-control-seeds` | `export_controls` | 37 | Export/re-export restrictions, required authorization, foreign-person limits, named export regimes and classifications, and no-license-required counter-signals |
| `--include-unpublished-work-seeds` | `unpublished_work` | 30 | Code-specific unpublished-work declarations, nonpublication statements, copyright-publication disclaimers, and no-public-release language |

The families can be enabled independently or together. The government family
remains a separate fifth opt-in switch because it also covers restricted
rights, classified information, and controlled-government-information language
that is broader than funding or export control.

Enabling all four new families expands the effective inventory from 126 to 238
phrases. Enabling the government family as well yields 251 rather than 259.
Five export phrases overlap the government inventory, two unpublished phrases
overlap the core inventory, and one overlaps the provenance inventory.
Retrieval deduplicates equivalent tokenized phrases and assigns shared
unpublished phrases to `unpublished_work` when that family is enabled.

The authoritative ordered lists are `PROPRIETARY_PROVENANCE_SEEDS`,
`FUNDING_DISSEMINATION_SEEDS`, and `EXPORT_CONTROL_SEEDS` in
`commentminer.topic_modelling`, alongside `UNPUBLISHED_WORK_SEEDS`; the
effective ordered inventory is also copied into every run manifest.

A match is a **review signal, not proof or a label**. In particular, a
decompiler banner does not establish that code was stolen; a proprietary
marking does not establish an unauthorized leak; a funding acknowledgement
does not by itself prevent dissemination; and an export classification or
regime name does not necessarily prohibit release. Permission, public-release,
and open-source language may occur in the same comment. Likewise, an
unpublished paper or dataset is not evidence that the source code is
restricted. Use the retained
opening comment and source provenance for review, and run an appropriate judge
or human assessment before drawing a conclusion.

To measure the expanded retrieval coverage without invoking an LLM, publish a
scan-only preview to a new directory:

```bash
SCAN_ONLY=1 \
INCLUDE_GOVERNMENT_SEEDS=1 \
INCLUDE_PROVENANCE_SEEDS=1 \
INCLUDE_FUNDING_SEEDS=1 \
INCLUDE_EXPORT_CONTROL_SEEDS=1 \
INCLUDE_UNPUBLISHED_WORK_SEEDS=1 \
OUTPUT=var/redistribution-candidate-comments-java-100k-expanded-scan \
scripts/run-redistribution-candidate-dataset.sh
```

The equivalent CLI switches are:

```bash
uv run commentminer build-redistribution-candidate-dataset \
  var/redistribution-candidate-comments-java-100k-expanded-scan \
  --input-source Jkatzy/code-comments \
  --dataset the-stack-v2-dedup \
  --language Java \
  --source-files-limit 100000 \
  --fuzzy-threshold 0.82 \
  --include-government-seeds \
  --include-provenance-seeds \
  --include-funding-seeds \
  --include-export-control-seeds \
  --include-unpublished-work-seeds \
  --scan-only \
  --revision 0d4c83fac76705d2e2388186b628543a4916dab8
```

Compare `results.candidate_count`, `results.matched_occurrences`, and
`results.seed_phrase_count` in the preview's `manifest.json` with the baseline.
Use `occurrences.parquet` to audit every match and `candidates.parquet` for the
deduplicated comments. Scan-only output has no `dataset.parquet`, judge labels,
or judge calls.

The verified local comparison on 2026-07-21 used the three earlier expansion
families plus government, without the unpublished-work family, at
the unchanged `0.82` threshold over the identical Java 100,000-file prefix:

| Measure | 126-seed baseline | 224-seed expanded scan | Increase |
| --- | ---: | ---: | ---: |
| Matched occurrences | 1,210 | 1,687 | +477 (+39.4%) |
| Unique candidates | 709 | 881 | +172 (+24.3%) |

The 172 new candidates partitioned into 153 provenance candidates, 18
funding/dissemination candidates, and one export-control candidate; their
source occurrence counts were 454, 22, and one respectively. The scan is at
`var/expanded-risk-candidates-java-100k-scan` and independently verifies as
valid. These are retrieval gains, not estimates of how many rows a semantic
judge will accept.

Enabling the unpublished-work family as well produced the following verified
scan at `var/expanded-risk-candidates-java-100k-unpublished-scan`:

| Measure | 126-seed baseline | 224-seed prior expansion | 251-seed unpublished expansion |
| --- | ---: | ---: | ---: |
| Matched occurrences | 1,210 | 1,687 | 1,688 |
| Unique candidates | 709 | 881 | 882 |

The `unpublished_work` group marked 15 occurrences across 14 unique comments.
It added one previously missed candidate: a truncated IBM header stating that
the source code is not published. The other 14 occurrences were already
retrieved through broader confidentiality, trade-secret, provenance, or
restriction phrases; the new group makes their nonpublication signal directly
queryable. ScanCode reported `contains_license_notice=false` for 13 of the 15
occurrences. This remains a scan-only comparison, so none of these rows has a
new judge label yet.

These commands do not change the materialized 709-candidate/1,210-occurrence
artifact described below. That artifact records only the original 126 active
seeds in its manifest. Keep a distinct `OUTPUT`; use `OVERWRITE=1` only when an
expanded preview itself is intentionally being replaced.

## Codex judge

Unless `--scan-only` is set, every unique candidate is assigned one of three
semantic outcomes:

- `code_redistribution_intent`: the comment actually communicates an intent to
  control distribution, redistribution, disclosure, copying, publication, or
  sharing of the code, including ordinary open-source license grants and
  redistribution conditions
- `other`: the seed-like language has another meaning, such as a technical
  description, data distribution, distributed computing, or shared memory
- `ambiguous`: the available comment is insufficient for a reliable choice

The workflow pins the model to `gpt-5.6-luna`. Reasoning effort accepts the
audited `low` and `max` profiles, with `max` as the default. The Java presets
use `max`; the five-million-comment preset explicitly uses `low`. The default
judge schedule submits batches of 64 comments through four concurrent workers,
with up to three attempts and a 900-second timeout per attempt. Model
responses, evidence, rationales, invocation settings, and batch failures are
retained for review. Comments are untrusted model input; the runner isolates
the Codex process and disables tools while judging.

These are LLM-assisted weak labels. Human-review a sample from every outcome,
especially `ambiguous`, before using the dataset for policy or compliance
decisions.

## Non-license limitation profile

Use `--judgment-profile non_license_limitations` when the target is a
redistribution limitation that does **not** arise solely from a software
license. This profile records two independent semantic facts:

- `is_non_license_redistribution_limitation`: the opening comment independently
  limits recipients, access, copying, disclosure, publication, distribution,
  or redistribution (for example, confidentiality, internal-only, customer,
  contract, proprietary-source, government, or export-control restrictions);
- `is_license_notice`: the comment contains a genuine named, standard, custom,
  open-source, or proprietary software license notice or substantive terms.

Ordinary BSD/GPL/Apache redistribution clauses and restrictive custom-license
conditions are license facts, not non-license positives. A mixed header is a
positive only when an additional restriction remains after its license text is
set aside. Technical uses such as distributed work, copying a build artifact,
repository placement, or internal-API support scope remain negatives.

The v3 rubric also applies a high-precision external-dissemination gate to
copying language. A positive must identify an external recipient, organization
boundary, public release, owner-permission requirement, confidentiality or
trade-secret boundary, publication, disclosure, or distribution context.
Within-project copy/paste, duplication, refactoring, template, code-quality,
and browser/source-view instructions are `other`; attribution or plagiarism
advice alone is not a dissemination restriction. A bare “do not copy” with no
reliable external or maintenance context is `ambiguous`, not affirmative.

The profile performs an explicit unpublished-work check. A notice that clearly
identifies the supplied code as unpublished/not published, disclaims any actual
or intended publication, or denies publication/public release is treated as
non-license nonpublication intent. Bare copyright or all-rights-reserved text
is still insufficient, and references to an unpublished paper, specification,
dataset, result, or dependency remain negative unless the comment separately
restricts the supplied code. The current rubric uses the distinct
`non-license-redistribution-limitations-v3` prompt identity. Earlier v1/v2
decisions cannot be reused by a newly judged v3 run.

The profile derives `scancode_contains_license_notice` from the structured
ScanCode JSON, not from the numeric score. `is_scancode_missed_license` is true
when the judge finds license text and ScanCode's threshold-qualified
`contains_license_notice` value is false. The source scan used 95/95
score-and-match-coverage thresholds, so a missed row can still have a nonzero
or near-95 `comment_license_score`.

In addition to the complete audit dataset, the build writes two exact filtered
views:

- `non-license-limitations.parquet` for affirmative non-license limitations,
  including mixed license-plus-restriction headers;
- `scancode-missed-licenses.parquet` for judged licenses that ScanCode did not
  qualify, with `known_license` naming the license when the judge can identify
  it.

The dedicated Java 100k preset reuses the local scored source tree, submits
four-comment batches through four parallel judges, and pins
`gpt-5.6-luna` at `max` reasoning:

```bash
scripts/run-non-license-redistribution-limitations-java-100k.sh
```

It publishes separately at
`var/non-license-redistribution-limitations-java-100k`; it does not overwrite
the earlier broad redistribution-intent artifact.

The materialized Java profile contains 111 non-license limitations and 104
ScanCode-missed licenses among 709 candidates. Its grounded decisions contain
56 restriction-only comments, 55 mixed restriction-plus-license comments, 585
license-only comments, and 13 comments with neither fact. There were no
ambiguous rows; confidence ranges from 0.84 to 1.0 with a 0.99 median.

Of the 104 missed licenses, 65 are custom or unnamed. The other 39 raw judge
names collapse to Apache-2.0 (13), MIT (8), BSD-3-Clause (7), BSD-2-Clause (4),
generic BSD-style (2), and one each for an unspecified GNU LGPL, PostgreSQL,
EPL-2.0, a DFARS unlimited-rights legend, and JasperReports License 1.0.
Seventeen missed rows have a zero ScanCode score; the maximum is 94.92. This
historical artifact records the v1 model identity and predates both the
unpublished-work and within-project-copying rubric refinements.

### Five-million-comment all-language run

The all-language preset applies the complete 251-seed inventory to an exact
5,000,000-comment subset of every locally available Stack v2 language, then
runs the non-license limitation judge:

```bash
scripts/run-non-license-redistribution-limitations-stack-v2-all-languages-5m.sh
```

The local source contains 344,269,908 opening-comment rows across 598 language
partitions. A strict normalized ScanCode threshold of `0.9` is recorded as raw
score `< 90.0`, leaving 322,393,555 eligible rows. Max-min water filling
includes every language: smaller partitions are included completely and the
remainder is divided equally among larger partitions. Within each language,
systematic midpoint ranks spread the sample across its entire ordered eligible
population instead of taking a prefix. The manifest records all per-language
capacities, allocations, 4,492 source-shard summaries, and an inventory hash.

The preset uses 32 parallel scan workers and eight parallel Codex judges. The
judge remains fixed to `gpt-5.6-luna`, but this large run explicitly selects
`low` reasoning to reduce token use. Reasoning effort is part of the cache
identity, so these decisions cannot collide with earlier `max` judgments.
Output and cache default to:

- `var/non-license-redistribution-limitations-stack-v2-all-languages-5m`
- `var/non-license-redistribution-limitations-stack-v2-all-languages-5m-judge-cache.sqlite`

The completed run examined 344,269,908 comments in 4,492 source shards and
found 322,393,555 eligible rows below the strict raw `<90.0` threshold;
21,876,353 rows were excluded. It selected exactly 5,000,000 comments across
all 598 languages. Three hundred eighty-five smaller language partitions were
selected in full. The other 213 received near-equal quotas: 116 languages at
17,027 rows and 97 at 17,026.

The complete 251-seed inventory produced 38,461 matched occurrences and 25,921
normalized unique candidates. Matches occurred in 497 languages; after
candidate deduplication, representative candidate rows span 494 languages.
Every selected candidate has a raw ScanCode score below 90.0; the maximum is
89.92.

The v2 label distribution is:

| Label | Candidates |
| --- | ---: |
| `other` | 11,130 |
| `license_only` | 8,078 |
| `non_license_redistribution_limitation` | 6,338 |
| `non_license_redistribution_limitation_with_license` | 250 |
| `ambiguous` | 125 |

`non-license-limitations.parquet` has 6,592 rows: the 6,588 affirmative labels
plus four ambiguous decisions whose restriction axis is true while the license
axis is uncertain. `scancode-missed-licenses.parquet` has 8,328 rows because
ScanCode's structured `contains_license_notice` field was false for every
LLM-identified license in that set. Of those, 5,182 identify a known license
and 3,146 are custom or unknown. The most frequent raw known names are MIT
(2,083), BSD-3-Clause (1,926), Apache-2.0 (342), BSD-2-Clause (244), the
spelling `MIT License` (160), and ISC (74).

The published manifest records eight judge workers, 406 materialization
batches, and 25,921 cache hits under
`gpt-5.6-luna:low:codex-cli 0.144.6:non-license-redistribution-limitations-v2`.
The independent verifier physically re-read all 4,492 source shards, checked
candidate/occurrence/decision identities, and matched all 12 artifact hashes;
`verification.json` reports no errors. Its source-inventory SHA-256 is
`45145460627eace869549fccac5420af1626bc773dde2b1ec937c425f24577fa`.

These counts describe the materialized v2 artifact. Inspection found that v2
could treat within-project comments such as “do not copy/paste this
configuration” as redistribution restrictions. New runs use the v3 external-
dissemination gate described above and therefore do not reuse these cached v2
decisions. Rejudge before treating the positive subset as corrected.

## Reproduce the Java 100k run

The materialized July 21, 2026 run is at
`var/redistribution-candidate-comments-java-100k`. It scanned 100,000 original
source rows (42,613 extracted comment rows from 35,895 comment-bearing files),
retained 1,210 matched occurrences representing 709 normalized unique
comments, and produced 709 grounded judgments: 701
`code_redistribution_intent` and 8 `other`, with no `ambiguous` rows. The
published pass used four workers and adaptive four-comment batches for the
last unresolved block; its manifest records 693 validated cache hits and six
Luna calls. `verification.json` reports no errors.

The default end-to-end preset performs retrieval, parallel judging, and
verification:

```bash
scripts/run-redistribution-candidate-dataset.sh
```

To inspect the retrieval dataset before spending model calls, write it to a
separate output directory:

```bash
SCAN_ONLY=1 \
OUTPUT=var/redistribution-candidate-comments-java-100k-scan \
scripts/run-redistribution-candidate-dataset.sh
```

The equivalent explicit build command is:

```bash
uv run commentminer build-redistribution-candidate-dataset \
  var/redistribution-candidate-comments-java-100k \
  --input-source Jkatzy/code-comments \
  --dataset the-stack-v2-dedup \
  --language Java \
  --source-files-limit 100000 \
  --fuzzy-threshold 0.82 \
  --batch-size 8192 \
  --judge-batch-size 64 \
  --judge-workers 4 \
  --judge-max-batch-chars 160000 \
  --judge-max-comment-chars 12000 \
  --judge-max-attempts 3 \
  --judge-timeout-seconds 900 \
  --judge-cache var/redistribution-candidate-comments-java-100k-judge-cache.sqlite \
  --codex-model gpt-5.6-luna \
  --codex-reasoning-effort max \
  --revision 0d4c83fac76705d2e2388186b628543a4916dab8

uv run commentminer verify-redistribution-candidate-dataset \
  var/redistribution-candidate-comments-java-100k
```

`--input-source` also accepts a local Hugging Face-style Parquet root. For a
remote gated revision, pass `--token-env HF_TOKEN`; the token value is read
from the environment and is not written into the command line or manifest.

The output keeps retrieval and judgments separate:

- `occurrences.parquet` contains every matched source occurrence;
- `candidates.parquet` contains normalized unique comments before judging;
- `dataset.parquet` contains one judgment per unique candidate;
- `labeled-occurrences.parquet` joins each source occurrence to its candidate's
  judgment;
- `manifest.json`, `verification.json`, and `README.md` record provenance,
  hashes, settings, counts, and dataset semantics;
- `judge-rubric.md`, the output schema, response/error logs, and the external
  SQLite cache preserve the judge audit trail.

Scan-only output contains the first two Parquet datasets and control artifacts,
but no judged datasets. The judge cache sits outside the atomically published
output so an interrupted or intentional rebuild can reuse exact compatible
decisions. Set `OVERWRITE=1` only to replace an existing recognizable output;
unrecognized directories are never replaced.

The preset supports `INPUT_SOURCE` (or `INPUT`), `OUTPUT`, `DATASET`,
`SOURCE_LANGUAGE`, `SOURCE_FILES_LIMIT`, `FUZZY_THRESHOLD`,
`ALL_LANGUAGES`, `COMMENT_ROWS_LIMIT`, `SCANCODE_SCORE_BELOW`, `SCAN_WORKERS`,
`INCLUDE_GOVERNMENT_SEEDS`, `INCLUDE_PROVENANCE_SEEDS`,
`INCLUDE_FUNDING_SEEDS`, `INCLUDE_EXPORT_CONTROL_SEEDS`,
`INCLUDE_UNPUBLISHED_WORK_SEEDS`, `SCAN_ONLY`,
`BATCH_SIZE`, `JUDGE_BATCH_SIZE`, `JUDGE_WORKERS`, `JUDGE_MAX_BATCH_CHARS`,
`JUDGE_MAX_COMMENT_CHARS`, `JUDGE_MAX_ATTEMPTS`, `JUDGE_TIMEOUT_SECONDS`,
`JUDGE_CACHE`, `CODEX_COMMAND`, `REVISION`, `TOKEN_ENV`,
`HF_CACHE_DIRECTORY`, and `OVERWRITE`. `CODEX_MODEL` remains pinned to
`gpt-5.6-luna`; `CODEX_REASONING_EFFORT` accepts the audited `low` and `max`
profiles, with `max` remaining the default.
