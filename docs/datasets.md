# Dataset Support Notes

## Supported Now

| Dataset | Source config | Layout | Access notes |
| --- | --- | --- | --- |
| The Stack v1 | `the-stack` | `data/{language}/*.parquet` | Gated on Hugging Face. Accept terms and use `HF_TOKEN` for downloads. |
| The Stack v2 | `the-stack-v2` | `data/{language}/*.parquet` metadata plus `s3://softwareheritage/content/{blob_id}` source blobs | Gated on Hugging Face for metadata shards. Source blobs are fetched with unsigned S3 reads by default; bulk use still needs to follow the upstream dataset terms. Language names use dataset casing such as `AMPL` and `Python`. |
| StarCoderData | `starcoderdata` | `{language}/*.parquet` | Gated on Hugging Face. Non-code subdirectories are ignored in the example config. |
| The Heap | `the-heap` | `data/{language}/*.parquet` | Public Hugging Face parquet dataset with 57 language folders. |
| RedPajama GitHub | `redpajama-github` | `urls/github.txt` points to public JSONL files | Streams JSONL over HTTP; source files are not stored locally. |

## Low-Storage Behavior

- Parquet mining defaults to direct scratch downloads rather than Hugging Face cache retention.
- `streaming: true` removes each local source shard after it is processed.
- Stack v2 source text is not present in the Hugging Face parquet rows. CommentMiner fetches complete gzipped Software Heritage blobs on demand with unsigned S3 reads by default, decodes them with `src_encoding`, and does not persist source text.
- Stack v2 defaults to the fastest measured setting here of 32 S3 content download threads, can be raised explicitly to 1024 threads, retries transient DNS/socket/S3 failures, and skips S3 reads for languages unsupported by the configured extractor.
- `content_prefetch_records` bounds queued or completed-but-not-yet-yielded Stack v2 rows and must be at least as large as `content_download_workers`.
- `scripts/benchmark-stack-v2-throughput.py` includes Software Heritage object reads by default; pass `--content-mode stub` only to isolate local parquet and extraction overhead.
- `--prefetch-files` bounds how many parquet files are downloaded or in flight at once, and `--download-workers` controls the download thread pool.
- `--extraction-workers` controls concurrent comment extraction, while `--extraction-buffer` bounds queued records so memory use stays predictable.
- `--max-files` and `--max-files-per-language` bound the source subset.
- Output JSONL shards omit source `content` and keep only `opening_comment`, provenance, and metadata.

## Not Yet Covered

The Pile GitHub component is exposed by the Hugging Face dataset script as a tar archive at `https://the-eye.eu/public/AI/pile_preliminary_components/github.tar`. CommentMiner does not yet include an archive streaming adapter, so that component is intentionally not listed as enabled in `config/pipeline.example.json`.

The Heap is handled by the same parquet adapter as The Stack. Its relevant columns are `content`, `language`, `file_path`, and `repo_name`; duplicate flags such as `exact_duplicates_stackv1`, `near_duplicates_stackv2`, and RedPajama/GitHubCode duplicate markers are preserved in output metadata.

RedPajama's GitHub subset is suitable for the current no-source-storage path because its URL list points directly to JSONL files with `text` and `meta.path` fields. Its `meta.language` field is repo-level language metadata, so CommentMiner treats it as a hint and infers per-file language from `meta.path`.

URL-list JSONL runs currently resume by URL and line number. They do not yet maintain a separate "completed URL" checkpoint, so full multi-URL sweeps may re-stream a completed URL to confirm EOF on a later rerun.
