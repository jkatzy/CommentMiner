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
  redistribution-candidate judging; it remains optional for topic-cluster
  validation

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

Every configured source uses a Hugging Face dataset repository.

| Config name | Hugging Face dataset | Input path |
| --- | --- | --- |
| `the-stack` | `bigcode/the-stack` | Language-partitioned Parquet |
| `the-stack-v2` | `bigcode/the-stack-v2` | Parquet metadata plus Software Heritage blobs |
| `the-stack-v2-dedup` | `bigcode/the-stack-v2-dedup` | Deduplicated Parquet metadata plus Software Heritage blobs |
| `starcoderdata` | `bigcode/starcoderdata` | Language-partitioned Parquet |
| `the-heap` | `AISE-TUDelft/the-heap` | Language-partitioned Parquet |

The Stack v2 repositories contain blob IDs rather than source text. CommentMiner reads their metadata from Hugging Face, fetches the corresponding gzipped content from Software Heritage S3, decodes it using `src_encoding`, extracts comments, and does not persist the source text. This sidecar content lookup is part of the two Stack v2 Hugging Face configurations.

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
DATASET=the-stack-v2-dedup \
TOKEN_ENV=HF_TOKEN \
PACKAGE_SIZE=10000 \
PACKAGE_WORKERS=64 \
PACKAGE_WORKER_BACKEND=process \
CONTENT_DOWNLOAD_WORKERS=2048 \
scripts/run-stack-v2-id-packages.sh
```

The equivalent CLI entry point is:

```bash
uv run commentminer mine-stack-v2-packages \
  config/pipeline.example.json \
  the-stack-v2-dedup \
  --package-size 10000 \
  --package-workers 64 \
  --package-worker-backend process \
  --content-download-workers 2048 \
  --token-env HF_TOKEN
```

Package directories include the dataset, package index, and a digest, so parallel workers do not collide. Process workers recycle after one package by default; pass `--package-worker-max-tasks-per-child 0` to disable recycling.

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

Run BERTopic on comments below a ScanCode threshold. Ratio and percentage forms are equivalent, so `0.95` and `95` both mean 95 percent:

```bash
uv run commentminer topic-model-low-scancode \
  var/comment-dataset-the-stack-v2-dedup-license-scan \
  --score-threshold 0.95 \
  --min-topic-size 10 \
  --save-model \
  --judge-with-codex
```

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
