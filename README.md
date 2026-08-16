# CommentMiner

CommentMiner is a Hugging Face-based pipeline for mining opening comments from very large code datasets, materializing them as Parquet, and scoring the comments with ScanCode Toolkit. Source code is treated as scratch data: remote shards are processed through bounded queues, opening comments and provenance are retained, and source content is normally removed immediately after extraction.

The data path is Parquet-only:

```text
Hugging Face Parquet source
-> opening-comment Parquet runs
-> Hugging Face dataset materialization
-> ScanCode-enriched Parquet
-> score histogram and optional analysis
```

JSON files are control artifacts only: configuration, checkpoints, manifests, and analysis reports. Heterogeneous source metadata and ScanCode detection details are serialized strings inside typed Parquet columns; they are not an alternate dataset workflow.

## Requirements

- Python 3.11 or newer
- [`uv`](https://docs.astral.sh/uv/) for dependency and command execution
- A Hugging Face token for gated datasets such as The Stack, The Stack v2, and StarCoderData
- The Codex CLI, authenticated and able to access the configured model, for
  redistribution-candidate and classifier-dataset judging; it remains optional
  for topic-cluster validation

Install all dependencies, including ScanCode Toolkit, BERTopic, and SentenceTransformers:

```bash
uv sync
```

Run the tests and inspect the CLI:

```bash
uv run python -m unittest discover -s tests
uv run commentminer --help
uv run commentminer validate-config config/pipeline.example.json
```

## Supported Sources

Every configured source uses either a Hugging Face dataset repository or a
Hugging Face Storage Bucket.

| Config name | Hugging Face dataset | Input path |
| --- | --- | --- |
| `the-stack` | `bigcode/the-stack` | Language-partitioned Parquet |
| `the-stack-v2` | `bigcode/the-stack-v2` | Full-corpus Parquet metadata plus Software Heritage blobs |
| `the-stack-v3-full` | `HuggingFaceCode/stack-v3-full` | Full-corpus bucket `contents/language=*/` Parquet shards |
| `the-heap` | `AISE-TUDelft/the-heap` | Language-partitioned Parquet |
| `redpajama-v1-github` | `togethercomputer/RedPajama-Data-1T` | GitHub JSONL shards listed by the Hub manifest |
| `pile-uncopyrighted-github` | `monology/pile-uncopyrighted` | `Github` records from Zstandard JSONL shards |
| `codeparrot-clean-valid` | `codeparrot/codeparrot-clean-valid` | Cleaned validation gzip JSONL |
| `codeparrot-github-code` | `codeparrot/github-code` | Native Parquet GitHub code shards |
| `code-clippy-github` | `CodedotAI/code_clippy_github` | Deduplicated gzip JSONL GitHub code shards |

The Stack v2 deduplicated repository contains blob IDs rather than source text. CommentMiner reads its metadata from Hugging Face, fetches the corresponding gzipped content from Software Heritage S3, decodes it using `src_encoding`, extracts comments, and does not persist the source text.

The complete Stack v3 corpus is stored as a Storage Bucket rather than a
normal Hub dataset. Its dedicated runner inventories all language partitions,
then lets 64 independent processes download and extract unrelated Parquet
shards in completion order. A shard is deleted after extraction and a separate
completion marker makes restarts idempotent:

```bash
scripts/run-stack-v3-resilient.sh
```

Track a live run in an SSH-friendly terminal dashboard with:

```bash
scripts/stack-v3-dashboard.sh
```

The dashboard shows completed and active shards, rolling throughput and ETA,
scratch-download and host-network rates, pipeline and host CPU, active workers,
raw/partial/output file counts, an exact record progress bar for the active
downloaded shard wave, and an approximate corpus-wide record bar based on the
dataset card's 43.9 billion metadata entries minus 28.3 billion stubs. The
numerator is summed from cumulative per-shard checkpoints. It also shows
extracted records and comments, memory, swap, free disk, and recent runner
events. Press `q` to quit. For logs or automation
without an interactive terminal, use `scripts/stack-v3-dashboard.sh --once` or
add `--json`.

Disk I/O percentages compare `/proc/diskstats` throughput for the backing block
device with defaults of 2,000 MiB/s read and 1,000 MiB/s write. Override those
ceilings when you have measured values with `--max-disk-read-mib-s` and
`--max-disk-write-mib-s`; the dashboard also reports the kernel's device-busy
percentage independently of those configured maxima.

Shard timing covers download start through extraction finish when both events
are present in the logs, and extraction time otherwise. The dashboard shows the
latest five durations and mean, standard deviation, median, and p10-p90 spread
over the latest 100 completed shards across the five most recent Stack v3 logs.
The live-pipeline section maps live worker-owned shards back to the inventory
and displays each active language as `language (active, completed/total)`, for
example `C++ (96, 187/624)`.

Set `SHARD_WORKERS`, `LANGUAGES`, or `MAX_SHARDS` to tune or bound a run. The
extractor uses the newest pinned ML4SE toolkit registry and retains the existing
opening-comment rule: comments must start within the first 10 source rows.

Stack v3 source shards and partial downloads are unconditionally deleted after
each worker attempt. The retained comment-only Parquet files are written below
`var/output/the-stack-v3-full/`. Materialize them into a Hugging Face dataset
upload tree with:

```bash
uv run commentminer export-hf-dataset \
  config/pipeline.example.json \
  var/comment-dataset-the-stack-v3-full \
  --input-directory var/output/the-stack-v3-full \
  --workers 64
```

RedPajama V1 stores `urls/github.txt` on the Hub rather than the source shards
themselves. CommentMiner reads that manifest, streams each external JSONL shard
line by line, maps `text` and nested `meta` provenance into input records, and
infers the programming language from the source path extension. Use
`--max-files` and `--max-records` for bounded runs:

```bash
uv run commentminer mine-dataset \
  config/pipeline.example.json \
  redpajama-v1-github \
  --max-files 1 \
  --max-records 1000
```

Extension detection uses the vendored
[FORGE language mapping](https://github.com/AISE-TUDelft/FORGE-ds-intermediate/blob/861acf2095899cb5336bbf85401b4b2191686018/code/langs_extension.json).
When an extension maps to multiple languages, CommentMiner runs every mapped
language supported by `ml4setk` and deduplicates identical extracted ranges.

The Pile source reads only records with `meta.pile_set_name == "Github"`.
Because the Pile does not preserve source paths or extensions, CommentMiner
uses Pygments content-based lexer detection for these records before invoking
the matching `ml4setk` comment parser. Each roughly 11 GB compressed shard is
processed and removed before the next shard is downloaded:

```bash
uv run commentminer mine-dataset \
  config/pipeline.example.json \
  pile-uncopyrighted-github \
  --max-files 1 \
  --max-records 1000
```

CodeParrot Clean Valid and Code Clippy use the gzip-JSONL adapter, while
CodeParrot GitHub Code uses the generic Parquet adapter. All three preserve
repository, path, language, license, and size metadata when supplied upstream.
Their source paths also participate in the FORGE multi-language extension
mapping.

Non-Hugging-Face source configurations are rejected when a pipeline configuration is loaded.

## Configuration and Discovery

The main configuration is [`config/pipeline.example.json`](config/pipeline.example.json). [`config/the-stack.sample.json`](config/the-stack.sample.json) is a small The Stack example. Relative storage paths are resolved from the configuration file's directory.

Useful inspection commands:

```bash
uv run commentminer dump-config config/pipeline.example.json
uv run commentminer list-languages config/pipeline.example.json the-stack-v2-dedup
uv run commentminer plan-download \
  config/the-stack.sample.json \
  the-stack \
  --language python \
  --max-files 1
```

`plan-download` only resolves matching Hub files. `download` deliberately retains a bounded source selection:

```bash
uv run commentminer download \
  config/the-stack.sample.json \
  the-stack \
  --language python \
  --max-files 1
```

Mining defaults to direct scratch downloads and removes processed upstream shards. Use `--cache-source-files` only when retaining those source files is intentional.

## 1. Mine Opening Comments

Opening comments are extracted with `ml4setk.OpeningCommentQuery` through `ML4SEOpeningCommentExtractor`. Each mined Parquet row has this canonical schema:

| Column | Type | Meaning |
| --- | --- | --- |
| `dataset` | string | Upstream dataset name |
| `record_id` | string | Stable source-specific record identifier |
| `opening_comment` | string | Extracted opening comment |
| `language` | nullable string | Source language |
| `path` | nullable string | Source path |
| `repo` | nullable string | Source repository |
| `extracted_at` | nullable string | UTC extraction time |
| `metadata` | string | Serialized source metadata |

The original source content is excluded.

Mine a bounded The Stack slice:

```bash
uv run commentminer mine-dataset \
  config/the-stack.sample.json \
  the-stack \
  --language ampl \
  --max-files 1 \
  --max-records 25 \
  --prefetch-files 4 \
  --download-workers 4 \
  --extraction-workers 4
```

The run is stored as immutable Parquet shards plus control metadata:

```text
var/output/the-stack-ampl/<run_id>/part-*.parquet
var/output/the-stack-ampl/<run_id>/manifest.json
var/checkpoints/the-stack-ampl.json
```

Shard writes are atomic. At each checkpoint, the active Parquet shard is durable before the source cursor advances.

The main concurrency controls are:

- `--prefetch-files` and `--download-workers` for Hugging Face source shards
- `--content-download-workers` and `--content-prefetch-records` for Stack v2 Software Heritage content
- `--extraction-workers` and `--extraction-buffer` for opening-comment extraction
- `--no-tqdm` for log-only progress

Pass a gated-dataset token by naming its environment variable:

```bash
uv run commentminer mine-dataset \
  config/pipeline.example.json \
  the-stack-v2-dedup \
  --language Python \
  --max-files 1 \
  --max-records 100 \
  --token-env HF_TOKEN
```

Mine bounded slices across enabled datasets:

```bash
uv run commentminer mine-config \
  config/pipeline.example.json \
  --max-files-per-language 1 \
  --max-records-per-language 500 \
  --token-env HF_TOKEN \
  --skip-errors
```

### Stack v2 package mining

For large Stack v2 runs, fixed-size blob-ID packages balance work more evenly than whole-language tasks. Metadata shards are staged, divided into disjoint packages, and processed concurrently:

```bash
DATASET=the-stack-v2 \
TOKEN_ENV=HF_TOKEN \
PACKAGE_SIZE=10000 \
METADATA_DOWNLOAD_WORKERS=64 \
PACKAGE_WORKERS=64 \
PACKAGE_WORKER_BACKEND=process \
CONTENT_DOWNLOAD_WORKERS=2048 \
EXTRACTION_WORKERS=1 \
scripts/run-stack-v2-id-packages.sh
```

The equivalent CLI entry point is:

```bash
uv run commentminer mine-stack-v2-packages \
  config/pipeline.example.json \
  the-stack-v2 \
  --package-size 10000 \
  --metadata-download-workers 64 \
  --package-workers 64 \
  --package-worker-backend process \
  --content-download-workers 2048 \
  --extraction-workers 1 \
  --token-env HF_TOKEN
```

Package directories include the dataset, package index, and a digest, so parallel workers do not collide. Packages may finish in any order; the bounded process pool keeps 64 independent blocks active and immediately schedules another when one finishes. Process workers recycle after one package by default; pass `--package-worker-max-tasks-per-child 0` to disable recycling.

Local throughput tools are available in `scripts/benchmark-stack-v2-throughput.py` and `scripts/benchmark-stack-v2-worker-matrix.py`. The matrix benchmark writes CSV results and a JSON plan report.

## 2. Materialize the Hugging Face Dataset

Materialization reads mined Parquet runs and writes a stable `<dataset>/<language>` Parquet tree:

```bash
uv run commentminer export-hf-dataset \
  config/pipeline.example.json \
  var/comment-dataset \
  --dedupe-record-ids \
  --overwrite
```

Typical output:

```text
var/comment-dataset/
  README.md
  manifest.json
  the-stack-v2-dedup/
    Python/
      part-00000.parquet
```

For package output, reset record-ID deduplication for each package group, use parallel exporters, and create one Hugging Face configuration per source dataset with language splits:

```bash
uv run commentminer export-hf-dataset \
  config/pipeline.example.json \
  var/comment-dataset-the-stack-v2-dedup \
  --input-directory var/output \
  --dedupe-record-ids \
  --dedupe-scope input-group \
  --dataset-card-layout language-splits \
  --workers 8 \
  --overwrite
```

`--dedupe-scope input-group` removes duplicate `dataset` plus `record_id` rows among reruns of the same package. Parallel export cannot use global record-ID deduplication.

## 3. Score Comments with ScanCode

`scan-hf-comment-licenses` reads a nested Hugging Face Parquet tree, mirrors it at the output path, preserves every source column, and appends:

- `comment_license_score`: best raw ScanCode match or clue score on a `0`–`100` scale
- `comment_license_detection`: serialized detection details, threshold-qualified matches, warnings, and errors

`contains_license_notice` inside the detection is true only when a match meets both the default score and match-coverage thresholds of 95. The raw numeric score remains independent of that classification.

The default backend is the in-process ScanCode API. It performs a known MIT canary before the run, truncates comments longer than 10,000 characters with an explicit warning, and stops on engine initialization or canary failures. The CLI backend remains available through `--scanner-backend cli`.

```bash
uv run commentminer scan-hf-comment-licenses \
  var/comment-dataset-the-stack-v2-dedup \
  --output-directory var/comment-dataset-the-stack-v2-dedup-license-scan \
  --detection-cache var/code-comments-license-score-cache.sqlite \
  --datasets the-stack-v2-dedup \
  --scanner-backend api \
  --batch-size 5000 \
  --workers 8 \
  --progress-every 10
```

Use `--datasets`, `--languages`, and `--max-shards` to bound a scan. The exact-comment SQLite cache is keyed by the comment, scanner identity, backend, thresholds, and scoring policy. Checkpoints also fingerprint every input shard, so changed inputs or scanner settings invalidate stale completed work.

Prewarm the detection cache without writing scored shards:

```bash
uv run commentminer prewarm-hf-license-cache \
  var/comment-dataset-the-stack-v2-dedup \
  --detection-cache var/code-comments-license-score-cache.sqlite \
  --datasets the-stack-v2-dedup \
  --scanner-backend api \
  --batch-size 5000 \
  --workers 8
```

## 4. Inspect Scores

Build a histogram from scored Parquet shards:

```bash
uv run commentminer license-score-histogram \
  var/comment-dataset-the-stack-v2-dedup-license-scan \
  --bins 20 \
  --width 60
```

Dataset/language filters and `--max-shards` are available for bounded inspection.

## Optional Post-Score Analysis

### Topic modelling

Run guided BERTopic on comments below a ScanCode threshold. The default model uses four phrase-level seed groups for proprietary/confidential markings, non-license sharing restrictions, custom or unrecognized licenses, and customer/contract-specific terms. Ratio and percentage forms are equivalent, so `0.95` and `95` both mean 95 percent.

The input may be either a local ScanCode-enriched Parquet directory or a Hugging Face dataset ID. Dataset IDs are resolved through the standard Hugging Face cache; dataset and language filters are also used to limit the snapshot files fetched.

```bash
uv run commentminer topic-model-low-scancode Jkatzy/code-comments \
  --datasets the-stack-v2-dedup \
  --languages Python \
  --score-threshold 0.95 \
  --sharing-prefilter \
  --min-topic-size 10
```

`--sharing-prefilter` first keeps only comments containing one of the curated
sharing, confidentiality, disclosure, or permission keywords, then runs
BERTopic on that smaller candidate set. Matches are case-insensitive whole
words, so the keyword `share` does not select `shared memory`. Add or replace
the focus with repeatable custom terms:

```bash
uv run commentminer topic-model-low-scancode Jkatzy/code-comments \
  --datasets the-stack-v2-dedup \
  --languages Python \
  --prefilter-keyword confidential \
  --prefilter-keyword "do not distribute"
```

The manifest records the effective keyword list, the number of score-eligible
comments before the prefilter, and how many comments it removed. Omitting both
prefilter options preserves the unfiltered low-ScanCode workflow.

For a small first extraction, select one dataset/language subset and one Parquet shard—for example, Python from The Stack v2 deduplicated:

```bash
uv run commentminer topic-model-low-scancode \
  var/comment-dataset-the-stack-v2-dedup-license-scan \
  --datasets the-stack-v2-dedup \
  --languages Python \
  --max-shards 1 \
  --score-threshold 0.95 \
  --sharing-prefilter \
  --min-topic-size 10
```

`SHARING_PREFILTER_KEYWORDS`, `SEED_TOPICS`, and the separate `GOVERNMENT_RESTRICTION_SEEDS` list are available from `commentminer.topic_modelling` for Python callers. Pass the sharing keywords through `prefilter_keywords`; customize BERTopic through `bertopic_model_kwargs`. Seeds guide clustering; their order and names do not define the discovered BERTopic topic IDs. Do not treat a restrictive cluster as proof that publication was unauthorized.

The sibling output directory can contain:

- `topic-assignments.parquet`
- `topics.json` and `manifest.json` report metadata
- `bertopic-model` when `--save-model` is used
- optional Codex prompt, response, and validation report files

Codex validation excludes BERTopic's `-1` outlier topic.

### Redistribution-intent candidate dataset

`build-redistribution-candidate-dataset` uses the 126 `SEED_TOPICS` phrases as
formatting-tolerant fuzzy retrieval signals, preserves every matched occurrence,
and judges each unique candidate as redistribution intent, another meaning, or
ambiguous. The separate government-specific seeds are opt-in. Retrieval scores
and matched phrases are audit metadata, not labels. Ordinary open-source
license grants and redistribution clauses count as code redistribution intent;
technical uses such as data distributions and shared memory do not.

Four more opt-in retrieval families broaden coverage for proprietary
provenance (including decompiler/reconstruction signals), funding-linked
dissemination conditions, export controls, and code-specific unpublished-work
notices. Enable them with
`--include-provenance-seeds`, `--include-funding-seeds`, and
`--include-export-control-seeds`, and `--include-unpublished-work-seeds`,
respectively. The families contain 24, 24, 37, and 30 phrases, taking the
effective inventory from 126 to 238 after exact tokenized overlaps are
deduplicated. These are candidate-generation signals only: they do not prove
theft, an unauthorized leak, a funding barrier, or an export-control violation,
and they do not assign a judge label.

Run the requested bounded preset against `Jkatzy/code-comments`:

```bash
scripts/run-redistribution-candidate-dataset.sh
```

It selects `the-stack-v2-dedup` / `Java` and the first 100,000 original
source-file rows—not 100,000 comment rows—then uses batches of 64 across four
parallel judges. The judge is fixed to `gpt-5.6-luna` with `max` reasoning.
The source revision is pinned to
`0d4c83fac76705d2e2388186b628543a4916dab8`.
Use `SCAN_ONLY=1` to materialize fuzzy candidates without model calls; use a
different `OUTPUT` for that preview so the judged dataset remains separate.
The preset exposes the expanded families as `INCLUDE_PROVENANCE_SEEDS=1`,
`INCLUDE_FUNDING_SEEDS=1`, `INCLUDE_EXPORT_CONTROL_SEEDS=1`, and
`INCLUDE_UNPUBLISHED_WORK_SEEDS=1`. For example:

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

The existing 709-candidate, 1,210-occurrence artifact is unchanged and remains
the reproducible 126-seed baseline. The expanded scan must use its own output
directory; compare the `results` counts and seed inventory in each manifest.
On the identical Java 100,000-file prefix, enabling all families produced 882
candidates and 1,688 occurrences. The unpublished-work family identified 15
occurrences across 14 comments and added one candidate missed by the previous
224-seed expansion.

The materialized run contains 1,210 matched occurrences, 709 unique judged
comments, 701 redistribution-intent labels, and 8 `other` labels; its
`verification.json` report is valid with no errors.

For the narrower target—opening comments that impose a redistribution or
sharing limitation independently of a license—run:

```bash
scripts/run-non-license-redistribution-limitations-java-100k.sh
```

That profile judges non-license limitations and genuine license text as
independent facts. It writes `non-license-limitations.parquet` plus
`scancode-missed-licenses.parquet`, where a miss means the judge found license
text but ScanCode's structured `contains_license_notice` result was false.
License-only redistribution clauses are excluded from the positive limitation
subset. The preset uses four parallel `gpt-5.6-luna` judges at `max` reasoning
and publishes to a separate output directory.

The current v3 judge requires evidence of an external recipient,
dissemination, permission, confidentiality, owner-permission, publication, or
release boundary. Within-project reuse guidance—such as “do not copy/paste this
configuration,” browser/source-view instructions, refactoring reminders, or
warnings not to imitate bad code—is not a redistribution limitation.
Attribution or plagiarism advice alone is also excluded. A bare “do not copy”
without enough context is `ambiguous`, not automatically positive. The v3
prompt identity prevents older cached decisions from being silently reused.

The historical v1 materialized profile contains 111 non-license limitations
and 104 ScanCode-missed licenses among the 709 candidates. The missed set
contains 39 named-family judgments and 65 custom/unnamed license notices. See
the detailed dataset document for the family counts and score distribution.

For broad language coverage, run:

```bash
scripts/run-non-license-redistribution-limitations-stack-v2-all-languages-5m.sh
```

This preset selects exactly 5,000,000 comments with normalized ScanCode score
below `0.9` (raw score `<90`) across all 598 local Stack v2 languages. It uses
max-min language allocation, systematic within-language sampling, the complete
251-seed inventory, 32 scan workers, and eight parallel `gpt-5.6-luna` judges
at `low` reasoning. The Java presets retain their existing `max` default.

The completed verified run examined 344,269,908 comments in 4,492 shards,
found 322,393,555 below the strict score ceiling, and selected exactly
5,000,000. Fuzzy retrieval produced 38,461 occurrences and 25,921 unique
candidates. Its v2 labels were 11,130 `other`, 8,078 `license_only`, 6,338
`non_license_redistribution_limitation`, 250 mixed limitation-plus-license, and
125 `ambiguous`. The two filtered views contain 6,592 candidate limitations
and 8,328 ScanCode-missed licenses.

Those materialized labels predate the v3 within-project copy/paste correction
described above. They remain an auditable v2 artifact, but should be rejudged
with the new prompt before the positive subset is treated as corrected.

Verify an end-to-end output independently:

```bash
uv run commentminer verify-redistribution-candidate-dataset \
  var/redistribution-candidate-comments-java-100k
```

See [`docs/redistribution-candidates.md`](docs/redistribution-candidates.md)
for the sampling semantics, labels, explicit CLI, and preset overrides.

### Sharing-restriction classifier dataset

`build-classifier-dataset` turns the ScanCode-enriched comment tree into three
training classes:

- `sharing_restriction`: an extra confidentiality, proprietary, internal-use,
  controlled-information, or other non-license limit on sharing, including
  mixed comments that also contain license text
- `scancode_missed_license`: genuine software-license notices that have
  `contains_license_notice=false`
- `irrelevant`: ordinary comments that are neither of the above

The first class has `binary_label=1`; both other classes have
`binary_label=0` and remain distinguishable as hard-negative subtypes.
Candidate keywords are retrieval signals only. Every retained unique comment
must pass two distinct row-level Codex prompt setups using the configured model:
a strict semantic review and a skeptical counter-review. Both decisions must agree with the candidate class
and semantic booleans, satisfy the class invariants, and meet the confidence
threshold. License-presence fields remain independent and truthful for mixed
positive comments.
Technical boundary cases such as internal-only API notes and repository-copy or
dependency instructions are deliberately sampled into the irrelevant route;
the judges must confirm that they do not constrain who may receive the code.
Malformed, incomplete, ambiguous, low-confidence, and disagreeing decisions
go to `rejected.parquet` rather than the training set.

Run the bounded, manifest-recorded four-dataset, three-language preset and
verify it:

```bash
scripts/run-sharing-restriction-classifier-dataset.sh
```

The helper defaults to eight source/language combinations: Python and
JavaScript from The Stack and RedPajama GitHub, Python and Java from The Heap,
and Python and Java from The Stack v2 deduplicated. It scans the
lexicographically first eight `part-*.parquet` shards in each cell by default.
The supported environment overrides are `INPUT`, `OUTPUT`, `JUDGE_CACHE`,
`CODEX_MODEL`, `TARGET_PER_COMBINATION`, `CANDIDATE_MULTIPLIER`,
`MAX_SHARDS_PER_COMBINATION`, and `JUDGE_WORKERS`. Use the equivalent CLI with
repeatable `--combination DATASET:LANGUAGE` options to change the fixed matrix.
Candidate selection and splitting are seeded; remote model outputs and
concurrent response-log ordering are not guaranteed to be byte-reproducible.

The output directory contains:

- `binary-training.parquet` and `multiclass-training.parquet`, task-specific
  projections with exactly one target plus comment text, split, ID, and source
  stratum (use `opening_comment` as the model feature)
- `dataset.parquet` and one Parquet file per class, with audit metadata that
  directly reveals the label and must not be used as model features
- `candidates.parquet` and `rejected.parquet` for selection auditing
- `judge-responses.jsonl` and `judge-rubric.md`, plus an SQLite judge cache at
  the configured cache path; reuse requires the same prompt, setup, batch
  context, declared model, Codex version, settings, and cache epoch
- `manifest.json`, `verification.json`, and a dataset card

The verifier requires unique normalized comments, ScanCode-negative status,
complete judge consensus, label/boolean consistency, exact class-file counts,
content-hashed audit artifacts, and leakage-aware groups confined to one of
train, validation, or test. Repository, near-template, and recognized
boilerplate-marker families are joined transitively into the same split
component. Add `--verify-source` when the trusted original source tree is
available to re-hash every selected shard, check recorded source manifests,
and compare every accepted source row:

```bash
uv run commentminer verify-classifier-dataset \
  var/sharing-restriction-classifier-dataset \
  --verify-source \
  --write-report
```

Comments plus their dataset/language/path/ScanCode metadata are sent to the
configured model service. The Codex backend ignores user config and rules,
runs in an empty directory, and disables local, web, app, plugin, browser,
computer-use, and image tools while judging this untrusted text.

These are LLM-assisted weak labels. Inspect the saved evidence, rationales, and
rejections before using them in high-stakes policy or compliance decisions.
The output is a quota-balanced retrieval sample, not a prevalence sample; its
class and dataset/language proportions do not estimate corpus frequencies.

### Encoder capacity benchmark

Benchmark SentenceTransformer-compatible encoders against scored Parquet input:

```bash
uv run commentminer benchmark-encoding-capacity \
  var/comment-dataset-the-stack-v2-dedup-license-scan \
  --model sentence-transformers/all-MiniLM-L6-v2=22M \
  --model your-org/your-larger-encoder=0.6B \
  --max-samples 50000 \
  --initial-samples 512 \
  --sample-growth 2 \
  --batch-size 64 \
  --device cuda
```

Declared model sizes must be between 2 million and 8 billion parameters. Output consists of `encoding-capacity-report.json` and `encoding-capacity-summary.csv`. A JSON model configuration can be supplied with `--model-config`; it is configuration rather than a comment-data format.

## Production Scoring Helpers

`scripts/run-code-comments-license-scan.sh` scans the local Stack v2 export and, when enabled, the existing Hugging Face Parquet configurations for The Heap and The Stack. It shares one exact-comment cache, stages a combined Parquet tree, verifies paths, row counts, finite scores, detection consistency, threshold hits, and engine errors, and uploads only when explicitly requested.

Run and verify locally:

```bash
UPLOAD=0 scripts/run-code-comments-license-scan.sh
```

Upload is opt-in:

```bash
UPLOAD=1 REPO_ID=Jkatzy/code-comments scripts/run-code-comments-license-scan.sh
```

Companion scripts:

- `scripts/status-code-comments-license-scan.sh`: process, checkpoint, cache, output, and disk status
- `scripts/watch-code-comments-license-scan.sh`: bounded restart monitoring; upload remains opt-in
- `scripts/verify-code-comments-license-upload.sh`: local/remote Parquet coverage and remote sample validation

## Repository Layout

- `config/`: Hugging Face source and optional analysis configuration
- `src/commentminer/`: source adapters, mining, Parquet materialization, scoring, and analysis
- `scripts/`: Stack v2 operations, benchmarks, scoring, status, and verification
- `tests/`: unit and regression coverage
- `docs/`: architecture, dataset support, and scope notes
- `var/`: ignored runtime downloads, checkpoints, outputs, caches, and logs

## Current Scope

Implemented and covered by tests:

- bounded Hugging Face Parquet downloads and source iteration
- Software Heritage content hydration for both Stack v2 variants
- concurrent extraction with atomic, checkpointed Parquet output
- package-based Stack v2 processing with worker recycling
- serial and parallel Hugging Face Parquet materialization
- Parquet ScanCode scoring with API and CLI backends
- exact-comment caching, configuration-aware checkpoints, and score histograms
- Parquet-based topic modelling and encoder benchmarking
- formatting-tolerant seed retrieval and parallel redistribution-intent judging

Archive-based and other non-Hugging-Face sources are intentionally unsupported. Long-running jobs should retain the default `INFO` logging or set another level with the global `--log-level` option.
