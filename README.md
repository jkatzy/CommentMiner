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
- The Codex CLI only for optional topic-cluster validation

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

Archive-based and other non-Hugging-Face sources are intentionally unsupported. Long-running jobs should retain the default `INFO` logging or set another level with the global `--log-level` option.
