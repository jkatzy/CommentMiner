# CommentMiner

CommentMiner is a storage-aware pipeline for mining opening comments from large code datasets such as The Stack, The Stack v2, The Heap, StarCoderData, and RedPajama GitHub. The repository focuses on orchestration: reading remote dataset files, passing source records through `ml4setk.OpeningCommentQuery`, normalizing the results into one schema, writing resumable output shards, and post-processing extracted comments.

## Scope

- Stream or batch over very large code datasets without assuming the full corpus fits on disk.
- Download only bounded source subsets per language and keep only extracted comments.
- Prefetch source shards and extract comments concurrently with bounded worker pools.
- Normalize records from multiple upstream datasets into one output format.
- Checkpoint progress so interrupted runs can resume.
- Write mined comments to shard files that can later be merged into a unified dataset.

## Constraints

- Upstream datasets are larger than the available server storage.
- Different sources expose different metadata and file layouts.
- Multiple datasets need to be converted into one consistent output schema.
- Temporary artifacts must be bounded and cleaned up aggressively.

## Repository Layout

- `config/pipeline.example.json`: example pipeline configuration
- `docs/problem.md`: problem statement and success criteria
- `docs/architecture.md`: initial design notes for storage-aware processing
- `src/commentminer/`: Python package for config loading, pipeline state, and output sharding
- `tests/`: small regression tests for the baseline scaffold

## Environment

This project uses `uv` for virtual environment and package management.

Create or update the local environment:

```bash
uv sync
```

Run the tests through the managed environment:

```bash
uv run python -m unittest discover -s tests
```

Run the CLI:

```bash
uv run commentminer validate-config config/pipeline.example.json
```

Add new runtime dependencies:

```bash
uv add <package>
```

Add new development-only dependencies:

```bash
uv add --dev <package>
```

## Pipeline Overview

The current production pipeline has five stages:

1. Download or stream dataset shards from Hugging Face.
2. Extract opening comments and save only the comment plus row metadata.
3. Aggregate extracted comment runs into one combined dataset.
4. Deduplicate the aggregated comments by normalized hash.
5. Run ScanCode over the deduplicated comments to detect license notices.

The stages are intentionally separate:

- download state is checkpointed independently from mined output
- comment extraction writes a clean dataset that excludes source code content
- license scanning runs only on the extracted comments, not on the raw code files

Typical data flow:

```text
Hugging Face dataset files
-> local shard download / streaming checkpoint
-> extracted comment JSONL shards
-> aggregated comment JSONL shards
-> deduplicated comment JSONL shards
-> ScanCode-enriched JSONL shards
```

## Supported Sources

The example config includes these source types:

- `huggingface_hub` parquet shards: The Stack v1, The Heap, and StarCoderData.
- `huggingface_hub` plus `extra.content_backend: "softwareheritage_s3"`: The Stack v2 metadata shards, where source text is fetched on demand from Software Heritage S3 by `blob_id`.
- `url_list_jsonl`: RedPajama GitHub, where a Hugging Face URL list points to large public JSONL files that are streamed directly.

The Pile GitHub component is currently not enabled because the public dataset script points to a tar archive rather than parquet or JSONL files. It needs a separate archive adapter before it can be mined with the same storage guarantees.

## Stage 1: Download

The downloader is designed for large dataset repos on the Hugging Face Hub:

- downloads are tracked file-by-file in `var/checkpoints/downloads/`
- rerunning the same command resumes from the saved checkpoint
- Hugging Face cache metadata is kept under `var/hf-cache/`
- language-specific downloads are supported when the dataset config declares language-aware patterns
- for parquet-backed sources like The Stack, `streaming: true` now means shard-at-a-time processing with local shard cleanup after processing
- mining uses direct scratch downloads by default, so source shards are not retained in the Hugging Face cache unless `--cache-source-files` is passed

There are now two distinct modes:

- `commentminer download ...`: explicitly save selected Hub files locally
- `TheStackParquetSource`: download one parquet shard, iterate its rows, checkpoint progress, and delete the shard when processing finishes
- `StackV2SWHContentSource`: download one Stack v2 metadata parquet shard, fetch each row's gzipped source blob from Software Heritage S3, and keep only extracted comments

Useful commands:

```bash
uv run commentminer list-languages config/the-stack.sample.json the-stack
uv run commentminer plan-download config/the-stack.sample.json the-stack --language python --max-files 1
uv run commentminer download config/the-stack.sample.json the-stack --language python --max-files 1
```

Use `plan-download` when you want to inspect what would be fetched before running a job. Use `download` when you explicitly want local copies of the upstream shard files.

## Stage 2: Comment Extraction

Opening comment extraction is integrated through `ml4setk.OpeningCommentQuery`.
This repository uses that query directly through `ML4SEOpeningCommentExtractor`.

The mining stage is the main production step. For The Stack, it downloads one shard, extracts comments row by row, writes extracted output, checkpoints progress, and removes the local shard when processing finishes.

Mine a configured The Stack language slice and write only extracted comments plus row metadata to JSONL shards:

```bash
uv run commentminer mine-dataset config/the-stack.sample.json the-stack \
  --language ampl \
  --max-files 1 \
  --max-records 25 \
  --prefetch-files 4 \
  --download-workers 4 \
  --extraction-workers 4
```

The Stack, The Stack v2, and StarCoderData are gated on Hugging Face. Accept the dataset terms and pass a token when needed:

```bash
uv run commentminer mine-dataset config/pipeline.example.json the-stack-v2 \
  --language AMPL \
  --max-files 1 \
  --max-records 100 \
  --token-env HF_TOKEN
```

For The Stack v2, the Hugging Face parquet files contain Software Heritage IDs rather than source text. The Stack v2 adapter fetches complete source bytes from `http://softwareheritage.s3.amazonaws.com/content/{blob_id}` with aiohttp by default, decodes them with `src_encoding`, and does not persist source text. It retries transient DNS/socket/S3 failures, skips S3 reads for languages unsupported by the configured extractor, and records missing SWH objects as empty-content skips by default. `extra.content_prefetch_records` bounds queued or completed-but-not-yet-yielded rows, and should stay at least as large as `extra.content_download_workers`.

For large Stack v2 runs, prefer package-based mining so work is balanced by fixed-size ID groups instead of by language. This mode first stages the Hugging Face metadata parquet shards, splits their `blob_id` rows into disjoint packages, and then runs packages through process workers. With `--package-worker-backend process`, each package process owns a bounded aiohttp pool; `--content-download-workers` is treated as the total content concurrency budget and is divided across package workers:

```bash
PACKAGE_SIZE=10000 \
PACKAGE_WORKERS=64 \
PACKAGE_WORKER_BACKEND=process \
CONTENT_DOWNLOAD_WORKERS=2048 \
scripts/run-stack-v2-id-packages.sh
```

The equivalent CLI is `uv run commentminer mine-stack-v2-packages config/pipeline.example.json the-stack-v2 --package-size 10000 --package-workers 64 --package-worker-backend process --content-download-workers 2048 --token-env HF_TOKEN`.

Benchmark the Stack v2 processor path against a local metadata parquet shard while including Software Heritage source blob downloads:

```bash
uv run python scripts/benchmark-stack-v2-throughput.py
```

Use `--content-mode stub` only when you need to isolate local parquet and extraction overhead without measuring Software Heritage object-store throughput.

The Heap is public and uses dataset-cased language names:

```bash
uv run commentminer mine-dataset config/pipeline.example.json the-heap \
  --language ANTLR \
  --max-files 1 \
  --max-records 100
```

For large runs, keep the default `INFO` logs or raise verbosity:

```bash
uv run commentminer --log-level INFO mine-dataset config/the-stack.sample.json the-stack --language ampl --progress-every 10000
```

`mine-dataset` now shows a per-parquet-shard `tqdm` progress bar during streaming so you can track shard-level completion and ETA. Disable that with `--no-tqdm` if you want log-only output.

Mining overlaps three bounded stages for parquet-backed sources: shard downloads, parquet row streaming, and comment extraction. Use `--download-workers` with `--prefetch-files` to control concurrent shard downloads, and `--extraction-workers` with `--extraction-buffer` to control the comment extraction thread pool and queue size.

The saved output excludes the source `content` field. It keeps:

- `opening_comment`
- canonical fields such as `dataset`, `record_id`, `language`, `path`, and `repo`
- all non-content row metadata in the `metadata` object

During mining, shards are written as resumable run outputs:

```text
var/output/<dataset-language>/<run-id>/part-*.jsonl
var/output/<dataset-language>/<run-id>/manifest.json
```

For Hugging Face upload, materialize one Parquet dataset tree grouped by source dataset and language:

```bash
uv run commentminer export-hf-dataset config/pipeline.example.json var/comment-dataset \
  --dedupe-record-ids \
  --overwrite
```

That produces:

```text
var/comment-dataset/
  README.md
  manifest.json
  the-stack-v2/
    Python/
      part-00000.parquet
  the-heap/
    ANTLR/
      part-00000.parquet
```

Upload `var/comment-dataset` as the single Hugging Face dataset repository. The generated dataset card declares one config per `<dataset>__<language>` and points each config at the matching nested Parquet shards. Exported Parquet files use a stable schema and store `metadata` as a JSON string.

Mine a bounded matrix across every enabled dataset/language slice:

```bash
uv run commentminer mine-config config/pipeline.example.json \
  --max-files-per-language 1 \
  --max-records-per-language 500 \
  --prefetch-files 4 \
  --download-workers 4 \
  --extraction-workers 4 \
  --token-env HF_TOKEN \
  --skip-errors
```

For RedPajama GitHub, the source JSONL files are streamed from their public URLs and are not saved locally. RedPajama `meta.language` is repo-level metadata, so the example config infers file language from `meta.path`:

```bash
uv run commentminer mine-dataset config/pipeline.example.json redpajama-github \
  --language java \
  --max-files 1 \
  --max-records 100 \
  --max-comment-start-row 20
```

Inspect configured language choices:

```bash
uv run commentminer list-languages config/pipeline.example.json the-stack-v2
```

## Stage 3: Aggregation

Once you have extracted comment runs from one or more source datasets, combine them into one dataset before running ScanCode.

The aggregation stage:

- reads one or more extracted run directories containing `part-*.jsonl`
- writes a new aggregated run in the same JSONL shard format
- adds a `source_dataset` field to each record
- rewrites the record-level `dataset` field to the aggregated dataset name
- preserves the rest of each extracted record unchanged

Example:

```bash
uv run commentminer aggregate-comment-runs \
  var/output/the-stack/20260407T114500Z \
  var/output/redpajama-github/20260407T120500Z \
  --dataset-name combined-comments
```

By default, the aggregated run is written under:

```text
var/output/combined-comments/<run_id>
```

## Stage 4: Deduplication

Once comments are aggregated, deduplicate them before running ScanCode so downstream processing only scans one representative copy of each normalized comment.

The deduplication stage:

- removes whitespace and all other non-alphanumeric characters from each comment
- hashes the normalized comment with SHA-256
- parallelizes the normalization and hashing pass across worker processes
- sorts the hash stream with the external `sort` command
- groups matching hashes into one deduplicated record
- keeps per-occurrence metadata so you can map a deduplicated comment back to every source file
- writes `occurrence_count` for each grouped comment

Example:

```bash
uv run commentminer deduplicate-comment-run \
  var/output/combined-comments/20260407T130000Z \
  --dataset-name combined-comments-deduplicated
```

For large runs, the main throughput controls are:

```bash
uv run commentminer deduplicate-comment-run \
  var/output/combined-comments/20260407T130000Z \
  --dataset-name combined-comments-deduplicated \
  --hash-workers 16 \
  --hash-batch-size 10000 \
  --sort-parallelism 16
```

Notes:

- `--hash-workers` controls the multi-process preprocessing stage before external sorting
- `--hash-batch-size` controls how many records are sent to each worker task at a time
- `--sort-parallelism` is passed through to GNU `sort`
- if the external `sort` command is missing or fails, the run now fails immediately instead of leaving an empty output run behind

That produces a new run like:

```text
var/output/combined-comments-deduplicated/<dedup_run_id>
```

Each deduplicated record keeps:

- one representative `opening_comment`
- `normalized_comment_hash`
- `occurrence_count`
- `occurrences`, containing the metadata of every grouped comment instance

## Stage 5: License Scanning

Previously extracted comment runs can be post-processed with ScanCode to classify license notices found inside the extracted comments. In the intended workflow, this stage runs on the deduplicated dataset from Stage 4.

This step is intentionally separate from comment extraction:

- it reads an existing mined run directory containing `part-*.jsonl`
- it writes a new sibling run directory with the same shard names plus `comment_license_detection`
- it keeps a local checkpoint so reruns skip completed shards
- it does not modify the original extracted-comment run in place

The ScanCode CLI must be installed separately and available on `PATH`, or passed explicitly with `--scancode`.

Example:

```bash
uv run commentminer scan-comment-licenses \
  var/output/combined-comments-deduplicated/20260407T140000Z \
  --scancode scancode \
  --batch-size 500 \
  --min-license-score 95 \
  --min-match-coverage 95
```

The enriched output is written by default to a sibling directory named `<input-run>-license-scan`.

For example, scanning:

```text
var/output/combined-comments-deduplicated/20260407T140000Z
```

produces:

```text
var/output/combined-comments-deduplicated/20260407T140000Z-license-scan
```

The default ScanCode settings now match the existing Stack v2 pipeline:

- `scancode --quiet --license --json-pp`
- classify a comment as containing license text only when `score >= 95` and `match_coverage >= 95`

For private repos, pass a token through an environment variable:

```bash
uv run commentminer download config/the-stack.sample.json the-stack --token-env HF_TOKEN
```

## End-To-End Example

One minimal end-to-end flow for a small The Stack slice looks like this:

```bash
uv sync
uv run commentminer plan-download config/the-stack.sample.json the-stack --language ampl --max-files 1
uv run commentminer mine-dataset config/the-stack.sample.json the-stack --language ampl --max-records 25
uv run commentminer aggregate-comment-runs var/output/the-stack/<run_id> --dataset-name combined-comments
uv run commentminer deduplicate-comment-run var/output/combined-comments/<combined_run_id> --dataset-name combined-comments-deduplicated
uv run commentminer scan-comment-licenses var/output/combined-comments-deduplicated/<dedup_run_id> --scancode scancode
```

That produces:

- raw download/checkpoint state under `var/downloads/` and `var/checkpoints/`
- extracted comments under `var/output/the-stack/<run_id>/`
- aggregated comments under `var/output/combined-comments/<combined_run_id>/`
- deduplicated comments under `var/output/combined-comments-deduplicated/<dedup_run_id>/`
- ScanCode-enriched comments under `var/output/combined-comments-deduplicated/<dedup_run_id>-license-scan/`

## Baseline Workflow

1. Define dataset sources in a config file.
2. Resolve or inspect the shard files that will be downloaded.
3. Run comment extraction to emit JSONL shards plus checkpoints and run manifests.
4. Aggregate one or more extracted comment runs into one dataset.
5. Deduplicate the aggregated run by normalized comment hash.
6. Post-process the deduplicated comment shards with ScanCode.
7. Merge or analyze the resulting comment dataset.

## Current Status

The repository currently includes:

- a Python package and CLI
- checkpointed Hugging Face downloads with language-aware file selection
- concrete storage-aware adapters for Hugging Face parquet shards and URL-list JSONL streams
- shard-at-a-time parquet processing with bounded local storage
- `ml4setk.OpeningCommentQuery` integration for opening-comment extraction
- JSONL output sharding that keeps row metadata but excludes source code content
- concrete configs for The Stack v1, The Stack v2, The Heap, StarCoderData, and RedPajama GitHub
- aggregation of extracted comment runs into one combined dataset with source-dataset provenance
- deduplication of aggregated comment runs using multithreaded hashing plus external sort-based grouping
- post-processing license detection over extracted comments using ScanCode
- runtime logging plus shard-level `tqdm` progress during streaming

The main remaining adapter gap is archive-based sources such as The Pile GitHub component.
