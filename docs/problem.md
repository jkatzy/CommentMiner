# Problem Statement

## Summary

Build a reproducible Hugging Face-based pipeline that mines opening comments from very large code datasets and writes them into one consolidated Parquet dataset. The repository handles source-specific download and iteration logic, calls `ml4setk` for comment extraction, and persists results in a form that can be resumed, validated, scored, and materialized across datasets.

## Why This Matters

- Large public code corpora contain useful comment data for downstream analysis and modeling.
- Existing datasets use different formats and metadata conventions.
- Server disk limits make naive full-dataset downloads impractical.
- A single repeatable pipeline is easier to validate than one-off extraction scripts.

## Core Requirements

- Read Parquet records from configured Hugging Face dataset repositories.
- Avoid full local copies whenever possible.
- Extract opening comments with an external extractor.
- Save output in a canonical schema with dataset provenance.
- Resume from checkpoints after interruptions.
- Materialize all source outputs into a Hugging Face-compatible Parquet tree.
- Append ScanCode license detections and numeric scores without changing the source rows.

## Constraints

- The Stack style datasets can exceed available storage by a large margin.
- Stack v2 requires staged Hugging Face metadata plus bounded Software Heritage blob hydration.
- Source metadata is heterogeneous and may need normalization.
- Failures are expected during long-running jobs and should not require restarting from scratch.

## Success Criteria

- A dataset adapter can process a source incrementally.
- The pipeline writes comment shards instead of one monolithic output file.
- Checkpoints make reruns resume near the previous stopping point.
- Output rows preserve dataset name, record identity, and source metadata.
- Different source datasets can be transformed into the same output schema.

## Deliberate Scope

- Hugging Face is the only configured source family.
- Parquet is the only data-plane shard format from mining through scoring and analysis.
- Record-ID deduplication is optional at Hugging Face materialization time and can be global or scoped to each input group.
- JSON remains only for configuration, checkpoints, manifests, and structured report metadata.
