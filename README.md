# CommentMiner

CommentMiner is a storage-aware pipeline for mining opening comments from large code datasets such as The Stack and related corpora. The repository focuses on orchestration: downloading remote dataset shards, passing source records through an external comment extractor, normalizing the results into one schema, and then post-processing the extracted comments with license detection.

The actual comment extraction logic is intentionally out of scope here. That will be provided by `ml4se-tk`, and this repository will integrate that extractor rather than reimplementing it.

## Scope

- Stream or batch over very large code datasets without assuming the full corpus fits on disk.
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

## Stage 1: Download

The downloader is designed for large dataset repos on the Hugging Face Hub:

- downloads are tracked file-by-file in `var/checkpoints/downloads/`
- rerunning the same command resumes from the saved checkpoint
- Hugging Face cache metadata is kept under `var/hf-cache/`
- language-specific downloads are supported when the dataset config declares language-aware patterns
- for parquet-backed sources like The Stack, `streaming: true` now means shard-at-a-time processing with local shard cleanup after processing

There are now two distinct modes:

- `commentminer download ...`: explicitly save selected Hub files locally
- `TheStackParquetSource`: download one parquet shard, iterate its rows, checkpoint progress, and delete the shard when processing finishes

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
uv run commentminer mine-dataset config/the-stack.sample.json the-stack --language ampl --max-records 25
```

For large runs, keep the default `INFO` logs or raise verbosity:

```bash
uv run commentminer --log-level INFO mine-dataset config/the-stack.sample.json the-stack --language ampl --progress-every 10000
```

`mine-dataset` now shows a per-parquet-shard `tqdm` progress bar during streaming so you can track shard-level completion and ETA. Disable that with `--no-tqdm` if you want log-only output.

The saved output excludes the source `content` field. It keeps:

- `opening_comment`
- canonical fields such as `dataset`, `record_id`, `language`, `path`, and `repo`
- all non-content row metadata in the `metadata` object

The mined dataset is written under:

```text
var/output/<dataset>/<run_id>/part-xxxxx.jsonl
```

Each run directory also includes a `manifest.json` describing what was processed.

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

- a `uv`-managed Python package scaffold
- checkpointed Hugging Face downloads with language-aware file selection
- shard-at-a-time The Stack parquet processing with bounded local storage
- `ml4setk.OpeningCommentQuery` integration for opening-comment extraction
- JSONL output sharding that keeps row metadata but excludes source code content
- aggregation of extracted comment runs into one combined dataset with source-dataset provenance
- deduplication of aggregated comment runs using multithreaded hashing plus external sort-based grouping
- post-processing license detection over extracted comments using ScanCode
- runtime logging plus shard-level `tqdm` progress during streaming
