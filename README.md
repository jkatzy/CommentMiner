# CommentMiner

CommentMiner is a storage-aware pipeline for mining opening comments from large code datasets such as The Stack, The Stack v2, and related corpora. The repository focuses on orchestration: reading remote dataset files, passing source records through an external comment extractor, normalizing the results into one schema, and writing resumable output shards.

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

## Hugging Face Downloads

The downloader is designed for large dataset repos on the Hugging Face Hub:

- downloads are tracked file-by-file in `var/checkpoints/downloads/`
- rerunning the same command resumes from the saved checkpoint
- Hugging Face cache metadata is kept under `var/hf-cache/`
- language-specific downloads are supported when the dataset config declares language-aware patterns
- for parquet-backed sources like The Stack, `streaming: true` now means shard-at-a-time processing with local shard cleanup after processing

There are now two distinct modes:

- `commentminer download ...`: explicitly save selected Hub files locally
- `TheStackParquetSource`: download one parquet shard, iterate its rows, checkpoint progress, and delete the shard when processing finishes

## Comment Extraction

Opening comment extraction is integrated through `ml4setk.OpeningCommentQuery`.
This repository uses that query directly through `ML4SEOpeningCommentExtractor`.

Mine a configured The Stack language slice and write only extracted comments plus
row metadata to JSONL shards:

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

Inspect configured language choices:

```bash
uv run commentminer list-languages config/pipeline.example.json the-stack-v2
```

When a dataset config does not pin a fixed `languages` list, `list-languages` now discovers language names from the Hugging Face repo layout.

Preview the files that would be downloaded:

```bash
uv run commentminer plan-download config/pipeline.example.json the-stack-v2 --language python
```

Run a resumable download:

```bash
uv run commentminer download config/pipeline.example.json the-stack-v2 --language python
```

The repository also includes a concrete The Stack sample config:

```bash
uv run commentminer list-languages config/the-stack.sample.json the-stack
uv run commentminer plan-download config/the-stack.sample.json the-stack --language befunge --max-files 1
uv run commentminer download config/the-stack.sample.json the-stack --language befunge --max-files 1
```

## License Scanning

Previously extracted comment runs can be post-processed with ScanCode to classify license notices found inside the extracted comments.

This step is intentionally separate from comment extraction:

- it reads an existing mined run directory containing `part-*.jsonl`
- it writes a new sibling run directory with the same shard names plus `comment_license_detection`
- it keeps a local checkpoint so reruns skip completed shards
- it does not modify the original extracted-comment run in place

The ScanCode CLI must be installed separately and available on `PATH`, or passed explicitly with `--scancode`.

Example:

```bash
uv run commentminer scan-comment-licenses \
  var/output/redpajama-github/20260407T114500Z \
  --scancode scancode \
  --batch-size 500 \
  --min-license-score 95 \
  --min-match-coverage 95
```

The enriched output is written by default to a sibling directory named `<input-run>-license-scan`.

The default ScanCode settings now match the existing Stack v2 pipeline:

- `scancode --quiet --license --json-pp`
- classify a comment as containing license text only when `score >= 95` and `match_coverage >= 95`
For private repos, pass a token through an environment variable:

```bash
uv run commentminer download config/pipeline.example.json the-stack-v2 --token-env HF_TOKEN
```

## Baseline Workflow

1. Define dataset sources in a config file.
2. Build a source adapter that yields normalized input records.
3. Plug in the `ml4se-tk` opening-comment extractor.
4. Run the pipeline to emit JSONL shards plus checkpoints and run manifests.
5. Merge or post-process shards into the final dataset format.

## Current Status

The repository currently includes:

- a `uv`-managed Python package scaffold
- checkpointed Hugging Face downloads with language-aware file selection
- shard-at-a-time The Stack parquet processing with bounded local storage
- `ml4setk.OpeningCommentQuery` integration for opening-comment extraction
- JSONL output sharding that keeps row metadata but excludes source code content
- runtime logging plus shard-level `tqdm` progress during streaming
