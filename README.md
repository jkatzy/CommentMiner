# CommentMiner

CommentMiner mines opening comments from very large code datasets without retaining the source code, normalizes the comments into a common schema, and carries them through export and ScanCode license scoring. The pipeline is designed for corpora that are larger than local storage: downloads are bounded, progress is checkpointed, source shards are normally deleted after processing, and generated datasets are sharded for resumable work.

## Requirements

- Python 3.11 or newer
- [`uv`](https://docs.astral.sh/uv/) for the locked environment
- GNU `sort` when using JSONL comment deduplication
- A Hugging Face token for gated datasets such as The Stack, The Stack v2, and StarCoderData
- The Codex CLI only when using optional Codex topic-cluster validation

Create the environment and install every runtime dependency, including ScanCode Toolkit, BERTopic, and SentenceTransformers:

```bash
uv sync
```

Run the tests:

```bash
uv run python -m unittest discover -s tests
```

Inspect the CLI and validate the example configuration:

```bash
uv run commentminer --help
uv run commentminer validate-config config/pipeline.example.json
```

## Repository Layout

- `config/pipeline.example.json`: all supported source adapters and production-oriented defaults
- `config/the-stack.sample.json`: smaller The Stack example
- `src/commentminer/`: mining, export, deduplication, scoring, topic-modelling, and benchmark code
- `scripts/`: long-running Stack v2 and production scoring helpers
- `tests/`: unit and regression coverage for the pipeline components
- `docs/`: problem statement, architecture notes, and dataset notes
- `var/`: ignored runtime downloads, checkpoints, outputs, caches, and logs

## Workflows Through ScanCode Scoring

There are two supported routes. Topic modelling and encoder benchmarking are optional branches after scoring; they are not required stages of either route.

### JSONL route

```text
remote dataset shards
-> opening-comment JSONL runs
-> optional aggregation
-> optional normalized-comment deduplication
-> ScanCode-enriched JSONL
-> score histogram / optional analysis
```

This route uses `mine-dataset` or `mine-config`, followed by `aggregate-comment-runs`, `deduplicate-comment-run`, and `scan-comment-licenses` as needed.

### Hugging Face Parquet route

```text
Stack v2 metadata + Software Heritage content
-> package-based opening-comment JSONL runs
-> nested Hugging Face Parquet export
-> cached, checkpointed ScanCode scoring
-> verified scored Parquet dataset
-> score histogram / optional analysis
```

This is the production path for the large combined dataset. It uses `mine-stack-v2-packages`, `export-hf-dataset`, and `scan-hf-comment-licenses`. The repository also includes scripts that combine the scored Stack v2 export with legacy Parquet configurations and optionally upload the verified result.

## Supported Sources

The example configuration enables six sources:

| Config name | Upstream dataset | Adapter |
| --- | --- | --- |
| `the-stack` | `bigcode/the-stack` | Hugging Face Parquet |
| `the-stack-v2` | `bigcode/the-stack-v2` | Parquet metadata plus Software Heritage S3 content |
| `the-stack-v2-dedup` | `bigcode/the-stack-v2-dedup` | Deduplicated Parquet metadata plus Software Heritage S3 content |
| `starcoderdata` | `bigcode/starcoderdata` | Hugging Face Parquet |
| `the-heap` | `AISE-TUDelft/the-heap` | Hugging Face Parquet |
| `redpajama-github` | `togethercomputer/RedPajama-Data-1T` | Streamed URL-list JSONL |

The Pile GitHub component is not enabled because its archive layout needs a separate bounded-storage adapter.

The Stack v2 datasets contain Software Heritage blob IDs rather than source text. CommentMiner fetches gzipped content from `http://softwareheritage.s3.amazonaws.com/content/{blob_id}`, decodes it using `src_encoding`, extracts the opening comment, and does not persist the source text. Stack v2 configurations retain the canonical fields and configured `metadata_columns`, rather than every upstream column.

## Configuration and Discovery

Useful inspection commands:

```bash
uv run commentminer dump-config config/pipeline.example.json
uv run commentminer list-languages config/pipeline.example.json the-stack-v2-dedup
uv run commentminer plan-download config/the-stack.sample.json the-stack --language python --max-files 1
```

`plan-download` resolves the remote files without downloading them. `download` explicitly retains selected upstream files locally:

```bash
uv run commentminer download \
  config/the-stack.sample.json \
  the-stack \
  --language python \
  --max-files 1
```

Mining normally uses scratch downloads, checkpoints each source position, and removes processed source shards. Pass `--cache-source-files` only when the upstream files should be retained.

## Mining Opening Comments

Opening comments are extracted through `ml4setk.OpeningCommentQuery` via `ML4SEOpeningCommentExtractor`. Output records contain:

- `opening_comment`
- canonical `dataset`, `record_id`, `language`, `path`, and `repo` fields
- `extracted_at`
- selected non-content metadata under `metadata`

The original source-code content is excluded.

Mine a bounded The Stack language slice:

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

That language-specific run is written under:

```text
var/output/the-stack-ampl/<run_id>/part-*.jsonl
var/output/the-stack-ampl/<run_id>/manifest.json
```

The mining pipeline overlaps bounded shard download, row streaming, and extraction work. The main controls are:

- `--prefetch-files` and `--download-workers` for source shards
- `--content-download-workers` and `--content-prefetch-records` for Software Heritage content
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

Mine a bounded matrix across enabled datasets and languages:

```bash
uv run commentminer mine-config \
  config/pipeline.example.json \
  --max-files-per-language 1 \
  --max-records-per-language 500 \
  --token-env HF_TOKEN \
  --skip-errors
```

### Stack v2 package mining

For large Stack v2 jobs, fixed-size ID packages balance work better than whole-language jobs. Metadata shards are staged first, divided into disjoint blob-ID packages, and processed concurrently:

```bash
DATASET=the-stack-v2-dedup \
TOKEN_ENV=HF_TOKEN \
PACKAGE_SIZE=10000 \
PACKAGE_WORKERS=64 \
PACKAGE_WORKER_BACKEND=process \
CONTENT_DOWNLOAD_WORKERS=2048 \
scripts/run-stack-v2-id-packages.sh
```

The equivalent CLI begins with:

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

Package output directories include the source dataset, package index, and package digest. Process workers recycle after one package by default to release Python and PyArrow memory. Set `PACKAGE_WORKER_MAX_TASKS_PER_CHILD=0` (or CLI value `0`) to disable recycling, or choose another positive package count.

For a local Stack v2 metadata throughput benchmark that includes Software Heritage reads:

```bash
uv run python scripts/benchmark-stack-v2-throughput.py
```

Use `--content-mode stub` only to isolate local Parquet and extraction overhead.

## Exporting a Hugging Face Dataset

Export mined JSONL runs into a stable nested dataset tree:

```bash
uv run commentminer export-hf-dataset \
  config/pipeline.example.json \
  var/comment-dataset \
  --dedupe-record-ids \
  --overwrite
```

The default dataset card declares one configuration per dataset/language pair:

```text
var/comment-dataset/
  README.md
  manifest.json
  the-stack-v2-dedup/
    Python/
      part-00000.parquet
```

For package outputs, deduplicate IDs independently within each package group, export in parallel, and create one Hugging Face configuration per source dataset with language splits:

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

`--dedupe-scope input-group` removes duplicate record IDs among reruns of the same package while allowing package groups to be exported concurrently. Parallel export cannot be combined with global record-ID deduplication. Exported Parquet uses one stable schema and stores `metadata` as a JSON string.

## JSONL Aggregation and Comment Deduplication

These stages are useful when several JSONL mining runs must be combined and deduplicated by comment text before scanning. They are not required when consuming an already deduplicated upstream dataset and exporting it directly to Parquet.

Aggregate one or more runs:

```bash
uv run commentminer aggregate-comment-runs \
  var/output/the-stack-ampl/<run_id> \
  var/output/redpajama-github-java/<run_id> \
  --dataset-name combined-comments
```

Aggregation preserves each record, records its source dataset, and writes a new run under `var/output/combined-comments/<run_id>`.

Deduplicate normalized comments:

```bash
uv run commentminer deduplicate-comment-run \
  var/output/combined-comments/<run_id> \
  --dataset-name combined-comments-deduplicated \
  --hash-workers 16 \
  --hash-batch-size 10000 \
  --sort-parallelism 16
```

Deduplication removes non-alphanumeric characters, hashes the normalized comment with SHA-256 in worker processes, externally sorts the hashes, and groups equal comments. Each result keeps one representative `opening_comment`, `normalized_comment_hash`, `occurrence_count`, and the metadata for every occurrence. A missing or failed external `sort` command fails the run.

## ScanCode License Scoring

`uv sync` installs ScanCode Toolkit. Two scanner backends are available:

- `api`: in-process ScanCode Python API; default for nested Hugging Face Parquet
- `cli`: ScanCode executable; default for JSONL runs

`--scancode` and `--scancode-processes` affect only the CLI backend. When using many Parquet shard workers, keep ScanCode CLI process count low to avoid oversubscription.

Before an API run starts, CommentMiner verifies the engine with a known MIT canary. API import, initialization, scan, and canary failures stop the run instead of producing valid-looking zero scores. Parquet shards must contain `opening_comment`.

### Score and detection fields

Every scored record receives:

- `comment_license_score`: the best raw ScanCode score across license matches and license clues, on a `0`–`100` scale; `0` means no score was found
- `comment_license_detection`: expressions, threshold-qualified matches, the best score, the boolean classification, and scan warnings/errors

`contains_license_notice` is `true` only when at least one license match has both `score >= 95` and `match_coverage >= 95` by default. The raw score is independent of that boolean, so a record can have `comment_license_score: 80` and `contains_license_notice: false`.

JSONL output stores `comment_license_detection` as an object. Parquet output stores it as a JSON string and stores `comment_license_score` as `float64`.

The API backend truncates comments longer than 10,000 characters before scanning and records an explicit truncation warning in `scan_errors`.

### Scan a JSONL run

```bash
uv run commentminer scan-comment-licenses \
  var/output/combined-comments-deduplicated/<dedup_run_id> \
  --scanner-backend cli \
  --scancode scancode \
  --scancode-processes 1 \
  --batch-size 500 \
  --min-license-score 95 \
  --min-match-coverage 95
```

The default output is the sibling directory `<input>-license-scan`. It preserves `part-*.jsonl`, writes an atomic output per shard, and maintains `license-scan-checkpoint.json` plus `manifest.json`.

### Scan nested Hugging Face Parquet

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

The output mirrors `<dataset>/<language>/part-*.parquet`, preserves the source columns, and appends the two scoring columns. Use `--datasets`, `--languages`, and `--max-shards` for bounded runs.

The SQLite cache reuses detections for identical opening comments across shards and runs. Its keys include the exact comment, backend, ScanCode identity, thresholds, and scoring policy version. Checkpoints include the same scan configuration and each input shard's size and modification time. Changing the backend, ScanCode installation, thresholds, API truncation policy, or an input shard invalidates the affected completed work instead of silently reusing stale scores.

Prewarm the exact-comment cache without writing output shards:

```bash
uv run commentminer prewarm-hf-license-cache \
  var/comment-dataset-the-stack-v2-dedup \
  --detection-cache var/code-comments-license-score-cache.sqlite \
  --datasets the-stack-v2-dedup \
  --scanner-backend api \
  --batch-size 5000 \
  --workers 8
```

### Inspect score distributions

The same command reads a scored JSONL directory or a nested Parquet dataset:

```bash
uv run commentminer license-score-histogram \
  var/comment-dataset-the-stack-v2-dedup-license-scan \
  --bins 20 \
  --width 60
```

Filters and `--max-shards` are also available for Parquet inspection.

## Production Combined-Dataset Scoring

`scripts/run-code-comments-license-scan.sh` is the operational workflow for combining the local Stack v2 Parquet export with the legacy configurations from `Jkatzy/code-comments`. It:

1. Optionally downloads the legacy Parquet configurations.
2. Scans Stack v2 and legacy shards through one exact-comment cache.
3. Hardlink-stages a combined local dataset.
4. Verifies exact shard paths, per-shard row counts, detection JSON, finite `0`–`100` scores, score/detection agreement, positive scores, threshold hits, and engine errors.
5. Uploads only when `UPLOAD=1` is explicitly set.
6. Verifies the uploaded Parquet file set and remote samples.

Prerequisite: create `var/comment-dataset-the-stack-v2-dedup` with `export-hf-dataset`, or override `STACK_V2_INPUT`.

Run and verify locally without uploading:

```bash
UPLOAD=0 scripts/run-code-comments-license-scan.sh
```

Upload is an explicit opt-in:

```bash
UPLOAD=1 REPO_ID=Jkatzy/code-comments scripts/run-code-comments-license-scan.sh
```

Important controls include `DOWNLOAD_LEGACY`, `SCAN_STACK_V2`, `SCAN_LEGACY`, `FINALIZE`, `WORKERS`, `BATCH_SIZE`, `SCANCODE_BACKEND`, `SCANCODE_PROCESSES`, `DETECTION_CACHE`, and all input/output path variables near the top of the script.

Companion scripts:

- `scripts/status-code-comments-license-scan.sh`: process, checkpoint, cache, output, and disk status
- `scripts/watch-code-comments-license-scan.sh`: restart a stopped runner up to a bounded count; upload remains opt-in
- `scripts/verify-code-comments-license-upload.sh`: compare local/remote Parquet coverage and validate remote samples

## Optional Post-Score Topic Modelling

Run BERTopic on comments below a ScanCode score threshold. Thresholds accept ratio or percentage notation, so `0.95` and `95` both mean 95 percent:

```bash
uv run commentminer topic-model-low-scancode \
  var/comment-dataset-the-stack-v2-dedup-license-scan \
  --score-threshold 0.95 \
  --min-topic-size 10 \
  --save-model \
  --judge-with-codex
```

The default sibling output directory ends in `-topic-modelling` and can contain:

- `topic-assignments.jsonl`
- `topics.json`
- `manifest.json`
- `bertopic-model` when `--save-model` is used
- `codex-cluster-validation-prompt.md`
- `codex-cluster-validation-response.md`
- `codex-cluster-validation.json`

Codex validation is optional and excludes BERTopic's `-1` outlier topic. Use dataset/language filters and `--max-shards` for bounded Parquet runs.

## Optional Encoder Capacity Benchmark

Benchmark how many comments candidate SentenceTransformer-compatible encoders can process on the current machine. Declared model sizes must be between 2 million and 8 billion parameters:

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

The default output ends in `-encoding-benchmark` and contains:

- `encoding-capacity-report.json`
- `encoding-capacity-summary.csv`

For a model matrix, create a JSON file such as `config/encoding-models.json`:

```json
{
  "models": [
    {"model_id": "sentence-transformers/all-MiniLM-L6-v2", "parameters": "22M"},
    {"model_id": "your-org/your-larger-encoder", "parameters": "0.6B", "revision": "main"}
  ]
}
```

Then pass `--model-config config/encoding-models.json`. The default `--max-samples 10000` bounds accidental runs; use `--all-samples` only intentionally.

## Current Status and Limitations

Implemented and covered by tests:

- bounded Hugging Face and URL-list source adapters
- Software Heritage content fetching for both Stack v2 variants
- concurrent extraction with resumable checkpoints and sharded JSONL output
- package-based Stack v2 processing with process recycling
- JSONL aggregation and multi-process normalized-comment hashing/deduplication
- serial and parallel Hugging Face export with two dataset-card layouts
- JSONL and Parquet ScanCode scoring with API/CLI backends
- exact-comment SQLite caching, configuration-aware checkpoints, and score histograms
- BERTopic low-score modelling with optional Codex cluster validation
- 2M–8B encoder capacity benchmarking

Not yet implemented:

- an archive adapter for sources such as The Pile GitHub component
- a single automated integration test that downloads, mines, exports, and scans a live upstream dataset end to end

Long-running jobs should keep the default `INFO` logging or increase verbosity with the global `--log-level` option.
