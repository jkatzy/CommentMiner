# Dataset Support Notes

## Supported Now

| Dataset | Source config | Layout | Access notes |
| --- | --- | --- | --- |
| The Stack v1 | `the-stack` | `data/{language}/*.parquet` | Gated on Hugging Face. Accept terms and use `HF_TOKEN` for downloads. |
| The Stack v2 deduplicated | `the-stack-v2-dedup` | `data/{language}/*.parquet` metadata plus `s3://softwareheritage/content/{blob_id}` source blobs | Uses the same bounded Software Heritage hydration path as The Stack v2. |
| The Stack v3 full | `the-stack-v3-full` | Storage Bucket `contents/language=*/*.parquet` rows containing source text | `scripts/run-stack-v3-resilient.sh` runs 128 independent shard downloads by default, gates the memory-heavy extraction phase to 80 workers with 16 GiB host-memory headroom, deletes source shards after extraction, and resumes from per-shard completion markers. |
| The Heap | `the-heap` | `data/{language}/*.parquet` | Public Hugging Face parquet dataset with 57 language folders. |
| RedPajama V1 GitHub | `redpajama-v1-github` | Hub `urls/github.txt` manifest pointing to external `.jsonl` shards | Streams records without retaining the source shards. Records provide `text` and nested `meta` fields for path, repository, license, and provenance. |
| The Pile uncopyrighted GitHub | `pile-uncopyrighted-github` | `train/*.jsonl.zst`, filtered to `meta.pile_set_name == "Github"` | Source paths were not retained upstream, so language is inferred from content. Shards are about 11 GB each and are removed after processing. |
| CodeParrot Clean Valid | `codeparrot-clean-valid` | Single `*.json.gz` shard | Cleaned 61,373-record validation dataset. Paths are available; language is inferred from their extensions. |
| CodeParrot GitHub Code | `codeparrot-github-code` | `data/*.parquet` | 1,126 native Parquet shards with content, path, repository, license, and size. Language is inferred from path extensions. |
| Code Clippy GitHub | `code-clippy-github` | `github-dedup-*.json.gz` | Roughly 50,000 deduplicated gzip JSONL shards with content, path, repository, license, and two upstream feature columns. Language is inferred from path extensions. |

## Low-Storage Behavior

- Parquet mining defaults to direct scratch downloads rather than Hugging Face cache retention.
- Stack v3 full uses the bucket-specific runner because Storage Buckets cannot be enumerated by the regular dataset downloader. Source shards are independent and may complete in any order.
- Stack v3 shard processes recycle after every attempt by default. If memory falls below the configured headroom, an in-progress shard checkpoints, removes its raw file, exits its process to release Arrow/Python allocations, and is returned to the parent queue. A replacement is not launched immediately: the parent requires available memory equal to the configured floor plus 125% of the largest active worker's RSS, waits for 30 seconds of sustained recovery, and then starts shards one at a time so the estimate can react to each new worker.
- Transient Stack v3 download failures (HTTP 408, 425, 429, and 5xx responses, connection errors, and timeouts) only recycle and requeue the affected shard. Retries use exponential backoff from 10 seconds up to 5 minutes while the remaining workers continue.
- Stack v3 source text is limited to its first 250 lines before ML4SE extraction, avoiding whole-file regex scans when only opening comments are eligible. Jupyter Notebook records are exempt because their complete JSON must remain available for code-cell extraction. Truncated records carry `content_truncated=true` and `content_line_limit=250` in metadata.
- `streaming: true` removes each local source shard after it is processed.
- Stack v2 source text is not present in the Hugging Face parquet rows. CommentMiner fetches complete gzipped Software Heritage blobs on demand with unsigned S3 reads by default, decodes them with `src_encoding`, and does not persist source text.
- The library default is 32 concurrent Stack v2 content reads. The production example explicitly selects the `aiohttp` client, 2,048 reads, and a 20,000-record prefetch bound; tune these values to the host and upstream service. Transient DNS/socket/object-store failures are retried, and languages unsupported by the extractor skip content reads.
- `content_prefetch_records` bounds queued or completed-but-not-yet-yielded Stack v2 rows and must be at least as large as `content_download_workers`.
- `scripts/benchmark-stack-v2-throughput.py` includes Software Heritage object reads by default; pass `--content-mode stub` only to isolate local parquet and extraction overhead.
- `--prefetch-files` bounds how many parquet files are downloaded or in flight at once, and `--download-workers` controls the download thread pool.
- `--extraction-workers` controls concurrent comment extraction, while `--extraction-buffer` bounds queued records so memory use stays predictable.
- `--max-files` and `--max-files-per-language` bound the source subset.
- Output Parquet shards omit source `content` and keep only `opening_comment`, provenance, and selected metadata.
- RedPajama V1 GitHub shards are streamed directly from the URLs published in the Hugging Face manifest. File extensions are resolved with the vendored [FORGE language-extension mapping](https://github.com/AISE-TUDelft/FORGE-ds-intermediate/blob/861acf2095899cb5336bbf85401b4b2191686018/code/langs_extension.json). Ambiguous extensions are checked with every mapped language supported by `ml4setk`; identical extracted ranges are emitted only once.
- The Pile GitHub component lacks paths and extensions. Its adapter filters out all other Pile components, uses Pygments to infer a lexer from source content, and then runs the corresponding `ml4setk` parser. Content detection is less reliable than extension-based detection.
- Gzip JSONL inputs are decompressed record by record. CodeParrot Clean Valid and Code Clippy source shards are removed after successful or bounded processing according to the normal streaming lifecycle.

## Not Yet Covered

The Heap is handled by the same Parquet adapter as The Stack. Its relevant columns are `content`, `language`, `file_path`, and `repo_name`; configured duplicate markers are preserved in output metadata.

Non-Hugging-Face source adapters are intentionally out of scope. In particular, URL-list and archive-based routes are not supported by the current workflow.
