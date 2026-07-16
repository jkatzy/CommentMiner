# Dataset Support Notes

## Supported Now

| Dataset | Source config | Layout | Access notes |
| --- | --- | --- | --- |
| The Stack v1 | `the-stack` | `data/{language}/*.parquet` | Gated on Hugging Face. Accept terms and use `HF_TOKEN` for downloads. |
| The Stack v2 | `the-stack-v2` | `data/{language}/*.parquet` metadata plus `s3://softwareheritage/content/{blob_id}` source blobs | Gated on Hugging Face for metadata shards. Source blobs are fetched with unsigned S3 reads by default; bulk use still needs to follow the upstream dataset terms. Language names use dataset casing such as `AMPL` and `Python`. |
| The Stack v2 deduplicated | `the-stack-v2-dedup` | `data/{language}/*.parquet` metadata plus `s3://softwareheritage/content/{blob_id}` source blobs | Uses the same bounded Software Heritage hydration path as The Stack v2. |
| StarCoderData | `starcoderdata` | `{language}/*.parquet` | Gated on Hugging Face. Non-code subdirectories are ignored in the example config. |
| The Heap | `the-heap` | `data/{language}/*.parquet` | Public Hugging Face parquet dataset with 57 language folders. |

## Low-Storage Behavior

- Parquet mining defaults to direct scratch downloads rather than Hugging Face cache retention.
- `streaming: true` removes each local source shard after it is processed.
- Stack v2 source text is not present in the Hugging Face parquet rows. CommentMiner fetches complete gzipped Software Heritage blobs on demand with unsigned S3 reads by default, decodes them with `src_encoding`, and does not persist source text.
- The library default is 32 concurrent Stack v2 content reads. The production example explicitly selects the `aiohttp` client, 2,048 reads, and a 20,000-record prefetch bound; tune these values to the host and upstream service. Transient DNS/socket/object-store failures are retried, and languages unsupported by the extractor skip content reads.
- `content_prefetch_records` bounds queued or completed-but-not-yet-yielded Stack v2 rows and must be at least as large as `content_download_workers`.
- `scripts/benchmark-stack-v2-throughput.py` includes Software Heritage object reads by default; pass `--content-mode stub` only to isolate local parquet and extraction overhead.
- `--prefetch-files` bounds how many parquet files are downloaded or in flight at once, and `--download-workers` controls the download thread pool.
- `--extraction-workers` controls concurrent comment extraction, while `--extraction-buffer` bounds queued records so memory use stays predictable.
- `--max-files` and `--max-files-per-language` bound the source subset.
- Output Parquet shards omit source `content` and keep only `opening_comment`, provenance, and selected metadata.

## Not Yet Covered

The Pile GitHub component is exposed by the Hugging Face dataset script as a tar archive at `https://the-eye.eu/public/AI/pile_preliminary_components/github.tar`. CommentMiner does not yet include an archive streaming adapter, so that component is intentionally not listed as enabled in `config/pipeline.example.json`.

The Heap is handled by the same Parquet adapter as The Stack. Its relevant columns are `content`, `language`, `file_path`, and `repo_name`; configured duplicate markers are preserved in output metadata.

Non-Hugging-Face source adapters are intentionally out of scope. In particular, URL-list and archive-based routes are not supported by the current workflow.
