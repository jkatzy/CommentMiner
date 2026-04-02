# Problem Statement

## Summary

Build a reproducible pipeline that mines opening comments from very large code datasets and writes them into one consolidated comment dataset. The repository should handle source-specific download and iteration logic, call out to `ml4se-tk` for comment extraction, and persist results in a form that can be resumed, validated, and merged across datasets.

## Why This Matters

- Large public code corpora contain useful comment data for downstream analysis and modeling.
- Existing datasets use different formats and metadata conventions.
- Server disk limits make naive full-dataset downloads impractical.
- A single repeatable pipeline is easier to validate than one-off extraction scripts.

## Core Requirements

- Read records from multiple upstream datasets.
- Avoid full local copies whenever possible.
- Extract opening comments with an external extractor.
- Save output in a canonical schema with dataset provenance.
- Resume from checkpoints after interruptions.
- Support eventual merging of all source outputs into one dataset.

## Constraints

- The Stack style datasets can exceed available storage by a large margin.
- Some sources may require streaming, partial downloads, or staged processing.
- Source metadata is heterogeneous and may need normalization.
- Failures are expected during long-running jobs and should not require restarting from scratch.

## Success Criteria

- A dataset adapter can process a source incrementally.
- The pipeline writes comment shards instead of one monolithic output file.
- Checkpoints make reruns resume near the previous stopping point.
- Output rows preserve dataset name, record identity, and source metadata.
- Different source datasets can be transformed into the same output schema.

## Open Questions

- Which source adapters should be implemented first?
- What exact `ml4se-tk` extractor interface should the integration wrapper target?
- Should the final output remain JSONL shards, Arrow/Parquet, or both?
- What deduplication policy is needed when the same file appears in multiple corpora?
