# Architecture Notes

## Design Goals

- Process very large corpora without depending on large local storage.
- Keep source-specific logic separate from extraction and output writing.
- Make long runs restartable and easy to inspect.
- Normalize outputs from different datasets into one shared schema.

## Proposed Pipeline

1. Source adapter
   Reads one upstream dataset and yields normalized input records.
2. Extraction step
   Passes normalized records through a bounded worker pool backed by the `ml4se-tk` opening-comment extractor.
3. Normalization step
   Converts extracted comments into a canonical output row.
4. Shard writer
   Writes JSONL shards with bounded size and per-run manifests.
5. Checkpoint store
   Saves resume information after a configurable number of records.
6. Merge step
   Combines shard outputs from multiple sources into one final dataset view.

## Canonical Output Fields

- `dataset`: upstream dataset name
- `record_id`: source-specific stable identifier
- `opening_comment`: extracted comment text
- `language`: normalized language if known
- `path`: source file path if known
- `repo`: source repository if known
- `extracted_at`: UTC timestamp for the pipeline run
- `metadata`: free-form source metadata that survives normalization

## Storage Strategy

- Prefer streaming or chunked iteration over full downloads.
- Treat local storage as scratch space, not as a permanent mirror.
- Write output early and often so temporary source artifacts can be removed.
- Keep checkpoints outside the scratch directory so cleanup does not erase resume state.
- Use small, immutable shard files instead of rewriting large outputs.

## Merge Strategy

- Normalize every dataset to the same row schema first.
- Preserve provenance so merged rows can always be traced back to a source.
- Add deduplication only after the baseline pipeline is stable.
- Keep merging separate from source extraction so reprocessing one dataset does not invalidate the rest.

## First Implementation Targets

- Add dataset adapters for the first one or two corpora.
- Add an integration wrapper for `ml4se-tk`.
- Decide whether final materialization should be JSONL-only or Arrow/Parquet as well.

## Download Baseline

The first downloader implementation targets Hugging Face dataset repos:

- enumerate repo files through the Hub API
- filter files with dataset-specific glob patterns
- support optional language-aware pattern selection
- download a bounded prefetch window of files with a persistent checkpoint
- resume cleanly after interruption by skipping already completed files

## Runtime Concurrency

- Parquet sources keep a bounded window of remote shards downloaded or in flight.
- The mining pipeline keeps a bounded queue of records submitted to comment extraction workers.
- Checkpoints and output writes are still applied in source order, so resumability does not depend on worker completion order.
