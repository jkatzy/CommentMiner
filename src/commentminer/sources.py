from __future__ import annotations

import asyncio
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import dataclass
import gzip
import json
import logging
from pathlib import Path, PurePosixPath
import random
from threading import Event, Lock, Thread
import time
from typing import Any, Callable, Iterator, Protocol, Sequence
from urllib.parse import urlparse

try:
    import resource
except ImportError:  # pragma: no cover - resource is unavailable on Windows.
    resource = None  # type: ignore[assignment]

import httpx
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from .config import DatasetSpec, PipelineConfig
from .downloader import HuggingFaceDownloader, RemoteFile
from .models import InputRecord


_ROW_ID_SEPARATOR = "::row::"
_URL_LINE_SEPARATOR = "::line::"
_PROCESSED_CHECKPOINT_NAMESPACE = "processed-shards"
_LOGGER = logging.getLogger(__name__)

_DEFAULT_CONTENT_COLUMNS = ["content", "text"]
_DEFAULT_LANGUAGE_COLUMNS = ["lang", "language", "gha_language", "ext", "meta.language"]
_DEFAULT_PATH_COLUMNS = [
    "max_stars_repo_path",
    "max_issues_repo_path",
    "max_forks_repo_path",
    "path",
    "meta.path",
]
_DEFAULT_REPO_COLUMNS = [
    "max_stars_repo_name",
    "max_issues_repo_name",
    "max_forks_repo_name",
    "repo_name",
    "meta.repo_name",
]
_DEFAULT_URL_STREAM_RETRIES = 5
_DEFAULT_URL_STREAM_RETRY_BACKOFF_SECONDS = 10.0
_DEFAULT_PREFETCH_FILES = 4
_DEFAULT_DOWNLOAD_WORKERS = 4
_DEFAULT_STACK_V2_CONTENT_URL_TEMPLATE = "s3://softwareheritage/content/{blob_id}"
_DEFAULT_STACK_V2_AIOHTTP_CONTENT_URL_TEMPLATE = "http://softwareheritage.s3.amazonaws.com/content/{blob_id}"
_DEFAULT_STACK_V2_CONTENT_COMPRESSION = ".gz"
_DEFAULT_STACK_V2_CONTENT_DOWNLOAD_WORKERS = 32
_DEFAULT_STACK_V2_CONTENT_PREFETCH_MULTIPLIER = 4
_DEFAULT_STACK_V2_S3_MAX_POOL_CONNECTIONS = 1024
_DEFAULT_STACK_V2_S3_READ_RETRIES = 6
_DEFAULT_STACK_V2_S3_RETRY_BACKOFF_SECONDS = 0.25
_FILENAME_LANGUAGE_ALIASES = {
    "dockerfile": "dockerfile",
    "makefile": "makefile",
}
_EXTENSION_LANGUAGE_ALIASES = {
    "adoc": "asciidoc",
    "as": "actionscript",
    "bash": "shell",
    "bat": "batchfile",
    "cc": "c++",
    "cjs": "javascript",
    "clj": "clojure",
    "cls": "apex",
    "cmake": "cmake",
    "cpp": "c++",
    "cs": "c#",
    "csh": "shell",
    "css": "css",
    "cu": "cuda",
    "cuh": "cuda",
    "cxx": "c++",
    "dart": "dart",
    "erl": "erlang",
    "ex": "elixir",
    "exs": "elixir",
    "fs": "f#",
    "fsi": "f#",
    "fsx": "f#",
    "go": "go",
    "gradle": "gradle",
    "groovy": "groovy",
    "gsp": "groovy_server_pages",
    "hh": "c++",
    "hs": "haskell",
    "html": "html",
    "hpp": "c++",
    "hxx": "c++",
    "java": "java",
    "jl": "julia",
    "js": "javascript",
    "jsx": "javascript",
    "kt": "kotlin",
    "kts": "kotlin",
    "less": "less",
    "lisp": "lisp",
    "lua": "lua",
    "mjs": "javascript",
    "mm": "objective_cpp",
    "php": "php",
    "pl": "perl",
    "pm": "perl",
    "ps1": "powershell",
    "py": "python",
    "pyw": "python",
    "rake": "ruby",
    "rb": "ruby",
    "rst": "restructuredtext",
    "rs": "rust",
    "scala": "scala",
    "scss": "scss",
    "sh": "shell",
    "sql": "sql",
    "swift": "swift",
    "ts": "typescript",
    "tsx": "tsx",
    "vb": "visual_basic_net",
    "vue": "vue",
    "zsh": "shell",
}
_COMPOUND_EXTENSION_LANGUAGE_ALIASES = {
    "blade.php": "blade",
    "d.ts": "typescript",
}
_AMBIGUOUS_EXTENSION_LANGUAGE_ALIASES = {
    "h": ["c++", "c", "objective-c"],
    "m": ["objective-c", "matlab", "mathematica"],
    "r": ["r", "rebol"],
}


@dataclass(frozen=True, slots=True)
class _TextLine:
    text: str
    next_byte_offset: int | None = None
    resumed_from_byte: bool = False


@dataclass(frozen=True, slots=True)
class ShardRowCursor:
    remote_path: str
    row_index: int

    def to_record_id(self) -> str:
        return f"{self.remote_path}{_ROW_ID_SEPARATOR}{self.row_index}"

    @classmethod
    def parse(cls, value: str) -> "ShardRowCursor":
        remote_path, separator, row_index = value.rpartition(_ROW_ID_SEPARATOR)
        if not separator:
            raise ValueError(f"Invalid shard row record id: {value}")
        return cls(remote_path=remote_path, row_index=int(row_index))


@dataclass(frozen=True, slots=True)
class UrlLineCursor:
    url: str
    line_index: int

    def to_record_id(self) -> str:
        return f"{self.url}{_URL_LINE_SEPARATOR}{self.line_index}"

    @classmethod
    def parse(cls, value: str) -> "UrlLineCursor":
        url, separator, line_index = value.rpartition(_URL_LINE_SEPARATOR)
        if not separator:
            raise ValueError(f"Invalid URL line record id: {value}")
        return cls(url=url, line_index=int(line_index))


class HuggingFaceParquetSource:
    def __init__(
        self,
        config: PipelineConfig,
        dataset: DatasetSpec,
        *,
        language: str | None = None,
        show_progress: bool = True,
        token: str | bool | None = None,
        downloader: HuggingFaceDownloader | None = None,
        max_files: int | None = None,
        prefetch_files: int | None = None,
        download_workers: int | None = None,
        cache_source_files: bool | None = None,
    ) -> None:
        self.name = _source_run_name(dataset.name, language)
        self.config = config
        self.dataset = dataset
        self.language = language
        self.show_progress = show_progress
        self.token = token
        self.downloader = downloader or HuggingFaceDownloader()
        self.max_files = max_files
        self.prefetch_files = _positive_int_option(
            prefetch_files,
            dataset.extra.get("prefetch_files"),
            default=_DEFAULT_PREFETCH_FILES,
        )
        self.download_workers = _positive_int_option(
            download_workers,
            dataset.extra.get("download_workers"),
            default=_DEFAULT_DOWNLOAD_WORKERS,
        )
        self.cache_source_files = _bool_option(
            cache_source_files,
            dataset.extra.get("cache_source_files"),
            default=False,
        )
        self.content_columns = _string_list_option(
            dataset.extra.get("content_columns"),
            _DEFAULT_CONTENT_COLUMNS,
        )
        self.language_columns = _string_list_option(
            dataset.extra.get("language_columns"),
            _DEFAULT_LANGUAGE_COLUMNS,
        )
        self.path_columns = _string_list_option(
            dataset.extra.get("path_columns"),
            _DEFAULT_PATH_COLUMNS,
        )
        self.repo_columns = _string_list_option(
            dataset.extra.get("repo_columns"),
            _DEFAULT_REPO_COLUMNS,
        )
        self.metadata_columns = (
            _string_list_option(dataset.extra.get("metadata_columns"), [])
            if dataset.extra.get("metadata_columns") is not None
            else None
        )

    def iter_records(self, start_after: str | None = None) -> Iterator[InputRecord]:
        self.config.ensure_directories()
        resume_cursor = ShardRowCursor.parse(start_after) if start_after else None
        _LOGGER.info(
            "Preparing parquet source iteration dataset=%s language=%s resume_from=%s max_files=%s prefetch_files=%s download_workers=%s cache_source_files=%s",
            self.dataset.name,
            self.language or "all",
            start_after,
            self.max_files,
            self.prefetch_files,
            self.download_workers,
            self.cache_source_files,
        )
        plan = self.downloader.plan_download(
            self.config,
            self.dataset,
            language=self.language,
            token=self.token,
            max_files=self.max_files,
            checkpoint_namespace=_PROCESSED_CHECKPOINT_NAMESPACE,
        )

        with _RemoteDownloadScheduler(self, plan.pending_files) as downloads:
            for remote, local_path in downloads:
                _LOGGER.info(
                    "Starting shard processing dataset=%s language=%s remote_path=%s",
                    self.dataset.name,
                    self.language or "all",
                    remote.path,
                )
                fully_processed = False
                try:
                    yield from self._iter_file_records(remote, local_path, resume_cursor)
                    fully_processed = True
                finally:
                    if fully_processed:
                        self.downloader.mark_file_completed(
                            self.config,
                            self.dataset,
                            remote.path,
                            language=self.language,
                            checkpoint_namespace=_PROCESSED_CHECKPOINT_NAMESPACE,
                        )
                        resume_cursor = None
                        _LOGGER.info(
                            "Finished shard processing dataset=%s language=%s remote_path=%s",
                            self.dataset.name,
                            self.language or "all",
                            remote.path,
                        )
                    if self.dataset.streaming:
                        self.downloader.remove_local_file(
                            self.config,
                            self.dataset,
                            remote.path,
                            language=self.language,
                        )

    def _download_remote(self, remote: RemoteFile) -> Path:
        return self.downloader.download_remote_file(
            self.config,
            self.dataset,
            remote.path,
            language=self.language,
            token=self.token,
            use_cache=self.cache_source_files,
        )

    def _iter_file_records(
        self,
        remote: RemoteFile,
        local_path: Any,
        resume_cursor: ShardRowCursor | None,
    ) -> Iterator[InputRecord]:
        start_row = self._start_row_for_remote(remote, resume_cursor)
        stop_row = self._stop_row_for_remote(remote)

        parquet_file = pq.ParquetFile(local_path)
        _LOGGER.info(
            "Reading parquet shard dataset=%s language=%s remote_path=%s total_rows=%s start_row=%s stop_row=%s",
            self.dataset.name,
            self.language or "all",
            remote.path,
            parquet_file.metadata.num_rows if parquet_file.metadata is not None else "unknown",
            start_row,
            stop_row if stop_row is not None else "end",
        )
        progress = tqdm(
            total=self._progress_total_rows(parquet_file, stop_row),
            initial=start_row,
            desc=_progress_description(self.dataset.name, remote.path),
            unit="rows",
            dynamic_ncols=True,
            leave=False,
            disable=not self.show_progress,
        )
        row_index = 0
        try:
            columns = self._selected_parquet_columns(parquet_file)
            for batch in parquet_file.iter_batches(
                batch_size=self.dataset.batch_size,
                columns=columns,
            ):
                for row in batch.to_pylist():
                    if stop_row is not None and row_index >= stop_row:
                        return
                    if row_index < start_row:
                        row_index += 1
                        continue
                    progress.update(1)
                    yield self._row_to_input_record(remote.path, row_index, row)
                    row_index += 1
        finally:
            progress.close()

    def _start_row_for_remote(
        self,
        remote: RemoteFile,
        resume_cursor: ShardRowCursor | None,
    ) -> int:
        if resume_cursor and resume_cursor.remote_path == remote.path:
            return resume_cursor.row_index + 1
        return 0

    def _stop_row_for_remote(self, remote: RemoteFile) -> int | None:
        return None

    def _progress_total_rows(
        self,
        parquet_file: pq.ParquetFile,
        stop_row: int | None,
    ) -> int | None:
        total_rows = parquet_file.metadata.num_rows if parquet_file.metadata is not None else None
        if stop_row is None:
            return total_rows
        if total_rows is None:
            return stop_row
        return min(total_rows, stop_row)

    def _row_to_input_record(
        self,
        remote_path: str,
        row_index: int,
        row: dict[str, Any],
    ) -> InputRecord:
        content = _first_lookup(row, self.content_columns)
        language = _coerce_language(_first_lookup(row, self.language_columns)) or self.language
        path = _first_lookup(row, self.path_columns)
        repo = _first_lookup(row, self.repo_columns)
        metadata = {
            key: value
            for key, value in row.items()
            if key not in set(self.content_columns)
        }
        metadata["remote_path"] = remote_path
        metadata["row_index"] = row_index
        if self.language is not None:
            metadata["selected_language"] = self.language

        return InputRecord(
            dataset=self.dataset.name,
            record_id=ShardRowCursor(remote_path, row_index).to_record_id(),
            content=str(content) if content is not None else "",
            language=str(language) if language is not None else None,
            path=str(path) if path is not None else None,
            repo=str(repo) if repo is not None else None,
            metadata=metadata,
        )

    def _selected_parquet_columns(self, parquet_file: pq.ParquetFile) -> list[str] | None:
        if self.metadata_columns is None:
            return None
        available = set(parquet_file.schema.names)
        requested: list[str] = []
        for column in (
            self.content_columns
            + self.language_columns
            + self.path_columns
            + self.repo_columns
            + self.metadata_columns
        ):
            if "." in column or column not in available or column in requested:
                continue
            requested.append(column)
        return requested or None


class TheStackParquetSource(HuggingFaceParquetSource):
    """Backward-compatible name for the original parquet source adapter."""


class StackV2ContentFetcher(Protocol):
    def fetch(self, blob_id: str, src_encoding: str | None) -> str:
        """Return decoded source content for a Stack v2 Software Heritage blob."""


@dataclass(frozen=True, slots=True)
class _StackV2PendingContentFetch:
    row_index: int
    row: dict[str, Any]
    blob_id: str | None
    skip_reason: str | None = None


class SoftwareHeritageS3ContentFetcher:
    def __init__(
        self,
        *,
        url_template: str = _DEFAULT_STACK_V2_CONTENT_URL_TEMPLATE,
        compression: str | None = _DEFAULT_STACK_V2_CONTENT_COMPRESSION,
        aws_profile: str | None = None,
        aws_region: str | None = None,
        aws_endpoint_url: str | None = None,
        aws_unsigned: bool = True,
        s3_max_pool_connections: int = _DEFAULT_STACK_V2_S3_MAX_POOL_CONNECTIONS,
        s3_read_retries: int = _DEFAULT_STACK_V2_S3_READ_RETRIES,
        s3_retry_backoff_seconds: float = _DEFAULT_STACK_V2_S3_RETRY_BACKOFF_SECONDS,
    ) -> None:
        self.url_template = url_template
        self.compression = compression
        self.aws_profile = aws_profile
        self.aws_region = aws_region
        self.aws_endpoint_url = aws_endpoint_url
        self.aws_unsigned = aws_unsigned
        self.s3_max_pool_connections = s3_max_pool_connections
        self.s3_read_retries = s3_read_retries
        self.s3_retry_backoff_seconds = s3_retry_backoff_seconds
        self._s3_client: Any | None = None
        self._s3_client_lock = Lock()
        self._http_client: httpx.Client | None = None
        self._http_client_lock = Lock()

    @classmethod
    def from_dataset(
        cls,
        dataset: DatasetSpec,
        *,
        min_pool_connections: int | None = None,
    ) -> "SoftwareHeritageS3ContentFetcher":
        url_template = _optional_string(dataset.extra.get("swh_content_url_template"))
        default_pool_connections = max(
            _DEFAULT_STACK_V2_S3_MAX_POOL_CONNECTIONS,
            min_pool_connections or 0,
        )
        return cls(
            url_template=url_template or _DEFAULT_STACK_V2_CONTENT_URL_TEMPLATE,
            compression=_optional_string(
                dataset.extra.get(
                    "swh_content_compression",
                    _DEFAULT_STACK_V2_CONTENT_COMPRESSION,
                )
            ),
            aws_profile=_optional_string(dataset.extra.get("aws_profile")),
            aws_region=_optional_string(dataset.extra.get("aws_region")),
            aws_endpoint_url=_optional_string(dataset.extra.get("aws_endpoint_url")),
            aws_unsigned=_bool_option(dataset.extra.get("aws_unsigned"), default=True),
            s3_max_pool_connections=_positive_int_option(
                dataset.extra.get("s3_max_pool_connections"),
                dataset.extra.get("swh_s3_max_pool_connections"),
                default=default_pool_connections,
            ),
            s3_read_retries=_nonnegative_int_option(
                dataset.extra.get("s3_read_retries"),
                dataset.extra.get("swh_s3_read_retries"),
                default=_DEFAULT_STACK_V2_S3_READ_RETRIES,
            ),
            s3_retry_backoff_seconds=_nonnegative_float_option(
                dataset.extra.get("s3_retry_backoff_seconds"),
                dataset.extra.get("swh_s3_retry_backoff_seconds"),
                default=_DEFAULT_STACK_V2_S3_RETRY_BACKOFF_SECONDS,
            ),
        )

    def fetch(self, blob_id: str, src_encoding: str | None) -> str:
        url = self.url_template.format(blob_id=blob_id)
        attempt = 0
        while True:
            try:
                raw = self._read_bytes(url)
                break
            except Exception as exc:
                if (
                    attempt >= self.s3_read_retries
                    or _is_missing_content_error(exc)
                    or not _is_retryable_s3_content_error(exc)
                ):
                    raise
                attempt += 1
                sleep_seconds = self.s3_retry_backoff_seconds * (2 ** (attempt - 1))
                sleep_seconds += random.uniform(0, self.s3_retry_backoff_seconds)
                _LOGGER.warning(
                    "Retrying Stack v2 SWH content read blob_id=%s attempt=%s/%s sleep_seconds=%.2f error=%s",
                    blob_id,
                    attempt,
                    self.s3_read_retries,
                    sleep_seconds,
                    exc,
                )
                time.sleep(sleep_seconds)
        return _decode_source_bytes(raw, src_encoding)

    def _read_bytes(self, url: str) -> bytes:
        if url.startswith("s3://"):
            return self._read_s3_bytes(url)
        if url.startswith("http://") or url.startswith("https://"):
            return self._read_http_bytes(url)
        if url.startswith("file://"):
            return self._read_local_bytes(Path(url.removeprefix("file://")))
        path = Path(url)
        if path.exists():
            return self._read_local_bytes(path)
        return self._read_smart_open_bytes(url, transport_params=None)

    def _read_local_bytes(self, path: Path) -> bytes:
        if self.compression == ".gz":
            with gzip.open(path, "rb") as handle:
                return handle.read()
        return path.read_bytes()

    def _read_s3_bytes(self, url: str) -> bytes:
        parsed = urlparse(url)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        if not bucket or not key:
            raise ValueError(f"Invalid S3 URL for Stack v2 content: {url}")

        response = self._client().get_object(Bucket=bucket, Key=key)
        raw = response["Body"].read()
        if self.compression == ".gz":
            return gzip.decompress(raw)
        return raw

    def _read_http_bytes(self, url: str) -> bytes:
        response = self._http().get(url)
        if response.status_code == 404:
            raise FileNotFoundError(url)
        response.raise_for_status()
        raw = response.content
        if self.compression == ".gz":
            return gzip.decompress(raw)
        return raw

    def _read_smart_open_bytes(
        self,
        url: str,
        *,
        transport_params: dict[str, Any] | None,
    ) -> bytes:
        try:
            from smart_open import open as smart_open
        except ImportError as exc:
            raise RuntimeError(
                "Stack v2 content fetching requires smart_open with S3 support. "
                "Run `uv sync` so the `smart_open[s3]` dependency is installed."
            ) from exc

        open_kwargs: dict[str, Any] = {}
        if self.compression is not None:
            open_kwargs["compression"] = self.compression
        if transport_params is not None:
            open_kwargs["transport_params"] = transport_params

        with smart_open(url, "rb", **open_kwargs) as handle:
            return handle.read()

    def _client(self) -> Any:
        if self._s3_client is None:
            with self._s3_client_lock:
                if self._s3_client is None:
                    self._s3_client = self._build_client()
        return self._s3_client

    def _http(self) -> httpx.Client:
        if self._http_client is None:
            with self._http_client_lock:
                if self._http_client is None:
                    limits = httpx.Limits(
                        max_connections=self.s3_max_pool_connections,
                        max_keepalive_connections=self.s3_max_pool_connections,
                    )
                    self._http_client = httpx.Client(
                        follow_redirects=True,
                        limits=limits,
                        timeout=None,
                    )
        return self._http_client

    def _build_client(self) -> Any:
        try:
            import boto3
            from botocore import UNSIGNED
            from botocore.config import Config
        except ImportError as exc:
            raise RuntimeError(
                "Stack v2 Software Heritage content fetching requires boto3 and botocore. "
                "Run `uv sync` so the `smart_open[s3]` dependency group is installed."
            ) from exc

        session_kwargs: dict[str, str] = {}
        if self.aws_profile is not None:
            session_kwargs["profile_name"] = self.aws_profile
        if self.aws_region is not None:
            session_kwargs["region_name"] = self.aws_region
        client_kwargs: dict[str, Any] = {}
        if self.aws_endpoint_url is not None:
            client_kwargs["endpoint_url"] = self.aws_endpoint_url
        config_kwargs: dict[str, Any] = {
            "max_pool_connections": self.s3_max_pool_connections,
        }
        if self.aws_unsigned:
            config_kwargs["signature_version"] = UNSIGNED
        client_kwargs["config"] = Config(**config_kwargs)
        return boto3.Session(**session_kwargs).client("s3", **client_kwargs)

    def worker_kwargs(self) -> dict[str, Any]:
        return {
            "url_template": self.url_template,
            "compression": self.compression,
            "aws_profile": self.aws_profile,
            "aws_region": self.aws_region,
            "aws_endpoint_url": self.aws_endpoint_url,
            "aws_unsigned": self.aws_unsigned,
            "s3_max_pool_connections": self.s3_max_pool_connections,
            "s3_read_retries": self.s3_read_retries,
            "s3_retry_backoff_seconds": self.s3_retry_backoff_seconds,
        }


class AiohttpSoftwareHeritageContentFetcher:
    def __init__(
        self,
        *,
        url_template: str = _DEFAULT_STACK_V2_AIOHTTP_CONTENT_URL_TEMPLATE,
        compression: str | None = _DEFAULT_STACK_V2_CONTENT_COMPRESSION,
        read_retries: int = _DEFAULT_STACK_V2_S3_READ_RETRIES,
        retry_backoff_seconds: float = _DEFAULT_STACK_V2_S3_RETRY_BACKOFF_SECONDS,
        dns_cache_seconds: int = 300,
        decode_workers: int = 0,
        decode_executor: str = "inline",
    ) -> None:
        decode_executor = decode_executor.lower()
        if decode_executor not in {"inline", "thread", "process"}:
            raise ValueError(
                "decode_executor must be one of 'inline', 'thread', or 'process'"
            )
        self.url_template = url_template
        self.compression = compression
        self.read_retries = read_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.dns_cache_seconds = dns_cache_seconds
        self.decode_workers = max(0, int(decode_workers))
        self.decode_executor = decode_executor
        self._pool_lock = Lock()
        self._pool: _AiohttpSoftwareHeritageContentFetchPool | None = None

    @classmethod
    def from_dataset(cls, dataset: DatasetSpec) -> "AiohttpSoftwareHeritageContentFetcher":
        return cls(
            url_template=_optional_string(dataset.extra.get("swh_content_url_template"))
            or _DEFAULT_STACK_V2_AIOHTTP_CONTENT_URL_TEMPLATE,
            compression=_optional_string(
                dataset.extra.get(
                    "swh_content_compression",
                    _DEFAULT_STACK_V2_CONTENT_COMPRESSION,
                )
            ),
            read_retries=_nonnegative_int_option(
                dataset.extra.get("s3_read_retries"),
                dataset.extra.get("swh_s3_read_retries"),
                default=_DEFAULT_STACK_V2_S3_READ_RETRIES,
            ),
            retry_backoff_seconds=_nonnegative_float_option(
                dataset.extra.get("s3_retry_backoff_seconds"),
                dataset.extra.get("swh_s3_retry_backoff_seconds"),
                default=_DEFAULT_STACK_V2_S3_RETRY_BACKOFF_SECONDS,
            ),
            dns_cache_seconds=_positive_int_option(
                dataset.extra.get("swh_http_dns_cache_seconds"),
                default=300,
            ),
            decode_workers=_nonnegative_int_option(
                dataset.extra.get("swh_content_decode_workers"),
                dataset.extra.get("content_decode_workers"),
                default=0,
            ),
            decode_executor=(
                _optional_string(
                    dataset.extra.get("swh_content_decode_executor")
                    or dataset.extra.get("content_decode_executor")
                )
                or "inline"
            ),
        )

    def fetch(self, blob_id: str, src_encoding: str | None) -> str:
        result = self.fetch_many(
            [(blob_id, src_encoding)],
            concurrency=1,
        )[0]
        if isinstance(result, Exception):
            raise result
        return result

    def fetch_many(
        self,
        requests: Sequence[tuple[str, str | None]],
        *,
        concurrency: int,
    ) -> list[str | Exception]:
        if not requests:
            return []
        concurrency = max(1, int(concurrency))
        pool = self._fetch_pool(concurrency)
        return pool.fetch_many(requests)

    def close(self) -> None:
        with self._pool_lock:
            pool = self._pool
            self._pool = None
        if pool is not None:
            pool.close()

    def warm(self, *, concurrency: int) -> None:
        self._fetch_pool(max(1, int(concurrency)))

    def _fetch_pool(self, concurrency: int) -> "_AiohttpSoftwareHeritageContentFetchPool":
        with self._pool_lock:
            if self._pool is None:
                self._pool = _AiohttpSoftwareHeritageContentFetchPool(
                    self,
                    concurrency=concurrency,
                )
            elif concurrency > self._pool.concurrency:
                self._pool.ensure_concurrency(concurrency)
            return self._pool

    async def _fetch_many(
        self,
        requests: Sequence[tuple[str, str | None]],
        *,
        concurrency: int,
    ) -> list[str | Exception]:
        try:
            import aiohttp
        except ImportError as exc:
            raise RuntimeError(
                "Stack v2 aiohttp content fetching requires aiohttp. "
                "Run `uv sync` so the aiohttp dependency is installed."
            ) from exc

        connector = aiohttp.TCPConnector(
            limit=concurrency,
            limit_per_host=0,
            ttl_dns_cache=self.dns_cache_seconds,
            enable_cleanup_closed=True,
        )
        timeout = aiohttp.ClientTimeout(
            total=None,
            connect=30,
            sock_connect=30,
            sock_read=120,
        )
        results: list[str | Exception] = [RuntimeError("fetch not started")] * len(requests)
        queue: asyncio.Queue[tuple[int, str, str | None]] = asyncio.Queue()
        for index, (blob_id, src_encoding) in enumerate(requests):
            queue.put_nowait((index, blob_id, src_encoding))

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            async def worker() -> None:
                while True:
                    try:
                        index, blob_id, src_encoding = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    try:
                        results[index] = await self._fetch_one(
                            session,
                            blob_id,
                            src_encoding,
                        )
                    except Exception as exc:  # The caller maps errors to skip/retry policy.
                        results[index] = exc
                    finally:
                        queue.task_done()

            tasks = [
                asyncio.create_task(worker())
                for _ in range(min(concurrency, len(requests)))
            ]
            await queue.join()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        return results

    async def _fetch_one(
        self,
        session: Any,
        blob_id: str,
        src_encoding: str | None,
        *,
        decode_executor: ThreadPoolExecutor | ProcessPoolExecutor | None = None,
    ) -> str:
        url = self.url_template.format(blob_id=blob_id)
        attempt = 0
        while True:
            try:
                raw = await self._read_http_bytes(session, url)
                return await self._decode_http_bytes(
                    raw,
                    src_encoding,
                    decode_executor=decode_executor,
                )
            except Exception as exc:
                if (
                    attempt >= self.read_retries
                    or _is_missing_content_error(exc)
                    or not _is_retryable_s3_content_error(exc)
                ):
                    raise
                attempt += 1
                sleep_seconds = self.retry_backoff_seconds * (2 ** (attempt - 1))
                sleep_seconds += random.uniform(0, self.retry_backoff_seconds)
                _LOGGER.warning(
                    "Retrying Stack v2 aiohttp content read blob_id=%s attempt=%s/%s sleep_seconds=%.2f error=%s",
                    blob_id,
                    attempt,
                    self.read_retries,
                    sleep_seconds,
                    exc,
                )
                if sleep_seconds > 0:
                    await asyncio.sleep(sleep_seconds)

    async def _read_http_bytes(self, session: Any, url: str) -> bytes:
        async with session.get(url) as response:
            if response.status == 404:
                raise FileNotFoundError(url)
            response.raise_for_status()
            return await response.read()

    async def _decode_http_bytes(
        self,
        raw: bytes,
        src_encoding: str | None,
        *,
        decode_executor: ThreadPoolExecutor | ProcessPoolExecutor | None,
    ) -> str:
        if decode_executor is None:
            return _decode_stack_v2_raw_content(raw, self.compression, src_encoding)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            decode_executor,
            _decode_stack_v2_raw_content,
            raw,
            self.compression,
            src_encoding,
        )

    def worker_kwargs(self) -> dict[str, Any]:
        return {
            "url_template": self.url_template,
            "compression": self.compression,
            "read_retries": self.read_retries,
            "retry_backoff_seconds": self.retry_backoff_seconds,
            "dns_cache_seconds": self.dns_cache_seconds,
            "decode_workers": self.decode_workers,
            "decode_executor": self.decode_executor,
        }


class _AiohttpSoftwareHeritageContentFetchPool:
    def __init__(
        self,
        fetcher: AiohttpSoftwareHeritageContentFetcher,
        *,
        concurrency: int,
    ) -> None:
        self.fetcher = fetcher
        self.concurrency = 0
        self._requested_concurrency = concurrency
        self._decode_executor = self._build_decode_executor()
        self._loop = asyncio.new_event_loop()
        self._ready = Event()
        self._close_lock = Lock()
        self._closed = False
        self._startup_error: BaseException | None = None
        self._thread = Thread(
            target=self._run,
            name="commentminer-stack-v2-aiohttp-content",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait()
        if self._startup_error is not None:
            if self._decode_executor is not None:
                self._decode_executor.shutdown(wait=True, cancel_futures=True)
            raise RuntimeError("Failed to start Stack v2 aiohttp content pool") from self._startup_error

    def fetch_many(self, requests: Sequence[tuple[str, str | None]]) -> list[str | Exception]:
        if self._closed:
            raise RuntimeError("Stack v2 aiohttp content pool is closed")
        request_list = list(requests)
        future = asyncio.run_coroutine_threadsafe(
            self._submit_many(request_list),
            self._loop,
        )
        try:
            return list(future.result())
        except BaseException:
            future.cancel()
            raise

    def ensure_concurrency(self, concurrency: int) -> None:
        if concurrency <= self.concurrency:
            return
        future = asyncio.run_coroutine_threadsafe(
            self._ensure_workers(concurrency),
            self._loop,
        )
        future.result()

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        future = asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)
        try:
            future.result()
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=10)

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._start())
        except BaseException as exc:  # pragma: no cover - exercised only on aiohttp import/startup failure.
            self._startup_error = exc
            self._ready.set()
            self._loop.close()
            return
        self._ready.set()
        self._loop.run_forever()
        self._loop.close()

    async def _start(self) -> None:
        try:
            import aiohttp
        except ImportError as exc:
            raise RuntimeError(
                "Stack v2 aiohttp content fetching requires aiohttp. "
                "Run `uv sync` so the aiohttp dependency is installed."
            ) from exc

        connector = aiohttp.TCPConnector(
            limit=0,
            limit_per_host=0,
            ttl_dns_cache=self.fetcher.dns_cache_seconds,
            enable_cleanup_closed=True,
        )
        timeout = aiohttp.ClientTimeout(
            total=None,
            connect=30,
            sock_connect=30,
            sock_read=120,
        )
        self._queue: asyncio.Queue[
            tuple[asyncio.Future[str | Exception], str, str | None] | None
        ] = asyncio.Queue()
        self._workers: list[asyncio.Task[None]] = []
        self._session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        await self._ensure_workers(self._requested_concurrency)

    async def _ensure_workers(self, concurrency: int) -> None:
        while len(self._workers) < concurrency:
            self._workers.append(asyncio.create_task(self._worker()))
        self.concurrency = len(self._workers)

    async def _submit_many(
        self,
        requests: Sequence[tuple[str, str | None]],
    ) -> list[str | Exception]:
        loop = asyncio.get_running_loop()
        futures: list[asyncio.Future[str | Exception]] = []
        for blob_id, src_encoding in requests:
            future: asyncio.Future[str | Exception] = loop.create_future()
            futures.append(future)
            self._queue.put_nowait((future, blob_id, src_encoding))
        return list(await asyncio.gather(*futures))

    async def _worker(self) -> None:
        while True:
            item = await self._queue.get()
            try:
                if item is None:
                    return
                future, blob_id, src_encoding = item
                if future.cancelled():
                    continue
                try:
                    result: str | Exception = await self.fetcher._fetch_one(
                        self._session,
                        blob_id,
                        src_encoding,
                        decode_executor=self._decode_executor,
                    )
                except Exception as exc:  # The caller maps errors to skip/retry policy.
                    result = exc
                if not future.done():
                    future.set_result(result)
            finally:
                self._queue.task_done()

    async def _shutdown(self) -> None:
        for _ in self._workers:
            self._queue.put_nowait(None)
        await asyncio.gather(*self._workers, return_exceptions=True)
        await self._session.close()
        if self._decode_executor is not None:
            self._decode_executor.shutdown(wait=True, cancel_futures=True)

    def _build_decode_executor(
        self,
    ) -> ThreadPoolExecutor | ProcessPoolExecutor | None:
        if self.fetcher.decode_workers < 1 or self.fetcher.decode_executor == "inline":
            return None
        if self.fetcher.decode_executor == "process":
            return ProcessPoolExecutor(max_workers=self.fetcher.decode_workers)
        return ThreadPoolExecutor(
            max_workers=self.fetcher.decode_workers,
            thread_name_prefix="commentminer-stack-v2-decode",
        )


def _stack_v2_content_fetcher_from_dataset(
    dataset: DatasetSpec,
    *,
    min_pool_connections: int | None,
) -> StackV2ContentFetcher:
    client = _optional_string(
        dataset.extra.get("swh_content_client")
        or dataset.extra.get("content_fetch_client")
        or dataset.extra.get("content_download_client")
    )
    if client is not None and client.lower() in {"aiohttp", "async_http", "async-http"}:
        return AiohttpSoftwareHeritageContentFetcher.from_dataset(dataset)
    return SoftwareHeritageS3ContentFetcher.from_dataset(
        dataset,
        min_pool_connections=min_pool_connections,
    )


class StackV2SWHContentSource(HuggingFaceParquetSource):
    """Stack v2 adapter that hydrates source text from Software Heritage S3 blobs."""

    def __init__(
        self,
        config: PipelineConfig,
        dataset: DatasetSpec,
        *,
        language: str | None = None,
        show_progress: bool = True,
        token: str | bool | None = None,
        downloader: HuggingFaceDownloader | None = None,
        max_files: int | None = None,
        prefetch_files: int | None = None,
        download_workers: int | None = None,
        cache_source_files: bool | None = None,
        content_fetcher: StackV2ContentFetcher | None = None,
        content_executor: ThreadPoolExecutor | None = None,
        content_download_workers: int | None = None,
        content_prefetch_records: int | None = None,
        content_language_filter: Callable[[str], bool] | None = None,
        skip_missing_content: bool | None = None,
    ) -> None:
        super().__init__(
            config,
            dataset,
            language=language,
            show_progress=show_progress,
            token=token,
            downloader=downloader,
            max_files=max_files,
            prefetch_files=prefetch_files,
            download_workers=download_workers,
            cache_source_files=cache_source_files,
        )
        self.blob_id_column = _optional_string(dataset.extra.get("blob_id_column")) or "blob_id"
        self.src_encoding_column = (
            _optional_string(dataset.extra.get("src_encoding_column")) or "src_encoding"
        )
        self.content_download_workers = _positive_int_option(
            content_download_workers,
            dataset.extra.get("content_download_workers"),
            dataset.extra.get("swh_content_download_workers"),
            default=_DEFAULT_STACK_V2_CONTENT_DOWNLOAD_WORKERS,
        )
        self.content_prefetch_records = _positive_int_option(
            content_prefetch_records,
            dataset.extra.get("content_prefetch_records"),
            dataset.extra.get("swh_content_prefetch_records"),
            default=self.content_download_workers * _DEFAULT_STACK_V2_CONTENT_PREFETCH_MULTIPLIER,
        )
        self.content_fetcher = content_fetcher or _stack_v2_content_fetcher_from_dataset(
            dataset,
            min_pool_connections=self.content_download_workers,
        )
        self.content_executor = content_executor
        if isinstance(self.content_fetcher, SoftwareHeritageS3ContentFetcher):
            self.content_fetcher.s3_max_pool_connections = max(
                self.content_fetcher.s3_max_pool_connections,
                self.content_download_workers,
            )
        self.content_language_filter = content_language_filter
        self.skip_missing_content = _bool_option(
            skip_missing_content,
            dataset.extra.get("skip_missing_content"),
            dataset.extra.get("swh_skip_missing_content"),
            default=True,
        )
        if self.content_prefetch_records < self.content_download_workers:
            raise ValueError(
                "content_prefetch_records must be greater than or equal to "
                "content_download_workers"
            )
        _raise_fd_limit_for_stack_v2_content_workers(self.content_download_workers)

    def _iter_file_records(
        self,
        remote: RemoteFile,
        local_path: Any,
        resume_cursor: ShardRowCursor | None,
    ) -> Iterator[InputRecord]:
        if isinstance(self.content_fetcher, AiohttpSoftwareHeritageContentFetcher):
            yield from self._iter_file_records_aiohttp(remote, local_path, resume_cursor)
            return
        if self.content_download_workers == 1:
            yield from super()._iter_file_records(remote, local_path, resume_cursor)
            return

        start_row = self._start_row_for_remote(remote, resume_cursor)
        stop_row = self._stop_row_for_remote(remote)

        parquet_file = pq.ParquetFile(local_path)
        _LOGGER.info(
            "Reading Stack v2 parquet shard dataset=%s language=%s remote_path=%s total_rows=%s start_row=%s stop_row=%s content_download_workers=%s content_prefetch_records=%s",
            self.dataset.name,
            self.language or "all",
            remote.path,
            parquet_file.metadata.num_rows if parquet_file.metadata is not None else "unknown",
            start_row,
            stop_row if stop_row is not None else "end",
            self.content_download_workers,
            self.content_prefetch_records,
        )
        progress = tqdm(
            total=self._progress_total_rows(parquet_file, stop_row),
            initial=start_row,
            desc=_progress_description(self.dataset.name, remote.path),
            unit="rows",
            dynamic_ncols=True,
            leave=False,
            disable=not self.show_progress,
        )
        pending: dict[Future[str], _StackV2PendingContentFetch] = {}
        ready: dict[int, tuple[dict[str, Any], str, str]] = {}
        executor = self._content_download_executor()
        owns_executor = executor is not self.content_executor
        rows = self._iter_parquet_rows(parquet_file, start_row, stop_row=stop_row)
        source_exhausted = False
        next_yield_row = start_row

        def fill_pending() -> None:
            nonlocal source_exhausted
            while (
                not source_exhausted
                and len(pending) < self.content_download_workers
                and len(pending) + len(ready) < self.content_prefetch_records
            ):
                try:
                    row_index, row = next(rows)
                except StopIteration:
                    source_exhausted = True
                    break

                if not self._should_fetch_content(row):
                    ready[row_index] = (row, "", "unsupported_language")
                    continue

                blob_id, src_encoding = self._content_fetch_request(
                    remote.path,
                    row_index,
                    row,
                )
                future = executor.submit(
                    self.content_fetcher.fetch,
                    blob_id,
                    src_encoding,
                )
                pending[future] = _StackV2PendingContentFetch(
                    row_index=row_index,
                    row=row,
                    blob_id=blob_id,
                )

        try:
            fill_pending()
            while pending or ready or not source_exhausted:
                while next_yield_row in ready:
                    row, content, content_fetch_status = ready.pop(next_yield_row)
                    progress.update(1)
                    yield self._row_to_input_record_with_content(
                        remote.path,
                        next_yield_row,
                        row,
                        content,
                        content_fetch_status=content_fetch_status,
                    )
                    next_yield_row += 1
                    fill_pending()

                fill_pending()
                if next_yield_row in ready:
                    continue
                if not pending:
                    if source_exhausted:
                        return
                    continue

                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    request = pending.pop(future)
                    try:
                        content = future.result()
                    except Exception as exc:
                        content, content_fetch_status = self._content_after_fetch_error(
                            exc,
                            remote.path,
                            request.row_index,
                            request.blob_id,
                        )
                    else:
                        content_fetch_status = "fetched"
                    ready[request.row_index] = (
                        request.row,
                        content,
                        content_fetch_status,
                    )
        finally:
            progress.close()
            for future in pending:
                future.cancel()
            if owns_executor:
                executor.shutdown(wait=True, cancel_futures=True)

    def _iter_file_records_aiohttp(
        self,
        remote: RemoteFile,
        local_path: Any,
        resume_cursor: ShardRowCursor | None,
    ) -> Iterator[InputRecord]:
        start_row = self._start_row_for_remote(remote, resume_cursor)
        stop_row = self._stop_row_for_remote(remote)

        parquet_file = pq.ParquetFile(local_path)
        _LOGGER.info(
            "Reading Stack v2 parquet shard with aiohttp dataset=%s language=%s remote_path=%s total_rows=%s start_row=%s stop_row=%s content_download_workers=%s content_prefetch_records=%s",
            self.dataset.name,
            self.language or "all",
            remote.path,
            parquet_file.metadata.num_rows if parquet_file.metadata is not None else "unknown",
            start_row,
            stop_row if stop_row is not None else "end",
            self.content_download_workers,
            self.content_prefetch_records,
        )
        progress = tqdm(
            total=self._progress_total_rows(parquet_file, stop_row),
            initial=start_row,
            desc=_progress_description(self.dataset.name, remote.path),
            unit="rows",
            dynamic_ncols=True,
            leave=False,
            disable=not self.show_progress,
        )
        rows = self._iter_parquet_rows(parquet_file, start_row, stop_row=stop_row)
        try:
            while True:
                batch: list[tuple[int, dict[str, Any]]] = []
                while len(batch) < self.content_prefetch_records:
                    try:
                        batch.append(next(rows))
                    except StopIteration:
                        break
                if not batch:
                    return

                fetch_requests: list[tuple[str, str | None]] = []
                request_indexes: list[int] = []
                ready: dict[int, tuple[dict[str, Any], str, str]] = {}
                for batch_index, (row_index, row) in enumerate(batch):
                    if not self._should_fetch_content(row):
                        ready[batch_index] = (row, "", "unsupported_language")
                        continue
                    fetch_requests.append(
                        self._content_fetch_request(
                            remote.path,
                            row_index,
                            row,
                        )
                    )
                    request_indexes.append(batch_index)

                assert isinstance(self.content_fetcher, AiohttpSoftwareHeritageContentFetcher)
                fetch_results = self.content_fetcher.fetch_many(
                    fetch_requests,
                    concurrency=self.content_download_workers,
                )
                for batch_index, fetch_request, result in zip(
                    request_indexes,
                    fetch_requests,
                    fetch_results,
                    strict=True,
                ):
                    row_index, row = batch[batch_index]
                    blob_id = fetch_request[0]
                    if isinstance(result, Exception):
                        content, content_fetch_status = self._content_after_fetch_error(
                            result,
                            remote.path,
                            row_index,
                            blob_id,
                        )
                    else:
                        content = result
                        content_fetch_status = "fetched"
                    ready[batch_index] = (row, content, content_fetch_status)

                for batch_index, (row_index, _) in enumerate(batch):
                    row, content, content_fetch_status = ready[batch_index]
                    progress.update(1)
                    yield self._row_to_input_record_with_content(
                        remote.path,
                        row_index,
                        row,
                        content,
                        content_fetch_status=content_fetch_status,
                    )
        finally:
            progress.close()

    def _content_download_executor(self) -> ThreadPoolExecutor:
        if self.content_executor is not None:
            return self.content_executor
        return ThreadPoolExecutor(
            max_workers=self.content_download_workers,
            thread_name_prefix="commentminer-stack-v2-content",
        )

    def _iter_parquet_rows(
        self,
        parquet_file: pq.ParquetFile,
        start_row: int,
        *,
        stop_row: int | None = None,
    ) -> Iterator[tuple[int, dict[str, Any]]]:
        columns = self._selected_parquet_columns(parquet_file)
        row_groups, row_index = _parquet_row_groups_for_range(
            parquet_file,
            start_row,
            stop_row,
        )
        if row_groups == []:
            return
        for batch in parquet_file.iter_batches(
            batch_size=self.dataset.batch_size,
            columns=columns,
            row_groups=row_groups,
        ):
            for row in batch.to_pylist():
                if stop_row is not None and row_index >= stop_row:
                    return
                if row_index >= start_row:
                    yield row_index, row
                row_index += 1

    def _selected_parquet_columns(self, parquet_file: pq.ParquetFile) -> list[str] | None:
        columns = super()._selected_parquet_columns(parquet_file)
        if columns is None:
            return None
        available = set(parquet_file.schema.names)
        for column in (self.blob_id_column, self.src_encoding_column):
            if "." not in column and column in available and column not in columns:
                columns.append(column)
        return columns

    def _content_fetch_request(
        self,
        remote_path: str,
        row_index: int,
        row: dict[str, Any],
    ) -> tuple[str, str | None]:
        blob_id = _lookup(row, self.blob_id_column)
        if blob_id is None:
            raise ValueError(
                f"Stack v2 row is missing required blob id column "
                f"'{self.blob_id_column}' remote_path={remote_path} row_index={row_index}"
            )
        src_encoding = _lookup(row, self.src_encoding_column)
        return str(blob_id), str(src_encoding) if src_encoding is not None else None

    def _should_fetch_content(self, row: dict[str, Any]) -> bool:
        if self.content_language_filter is None:
            return True
        return any(
            self.content_language_filter(candidate)
            for candidate in self._content_language_candidates(row)
        )

    def _content_language_candidates(self, row: dict[str, Any]) -> list[str]:
        candidates: list[str] = []
        if self.language is not None:
            candidates.append(self.language)
        for column in self.language_columns:
            candidates.extend(str(value) for value in _coerce_language_values(_lookup(row, column)))
        for column in self.path_columns:
            extension = _path_extension(_lookup(row, column))
            if extension is not None:
                candidates.append(extension)

        result: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            if candidate not in seen:
                result.append(candidate)
                seen.add(candidate)
        return result

    def _content_after_fetch_error(
        self,
        exc: Exception,
        remote_path: str,
        row_index: int,
        blob_id: str | None,
    ) -> tuple[str, str]:
        if self.skip_missing_content and _is_missing_content_error(exc):
            _LOGGER.warning(
                "Skipping missing Stack v2 content blob_id=%s remote_path=%s row_index=%s error=%s",
                blob_id,
                remote_path,
                row_index,
                exc,
            )
            return "", "missing"
        raise RuntimeError(
            f"Failed to fetch Stack v2 content blob_id={blob_id} "
            f"remote_path={remote_path} row_index={row_index}"
        ) from exc

    def _row_to_input_record(
        self,
        remote_path: str,
        row_index: int,
        row: dict[str, Any],
    ) -> InputRecord:
        if not self._should_fetch_content(row):
            return self._row_to_input_record_with_content(
                remote_path,
                row_index,
                row,
                "",
                content_fetch_status="unsupported_language",
            )
        blob_id, src_encoding = self._content_fetch_request(remote_path, row_index, row)
        try:
            content = self.content_fetcher.fetch(
                blob_id,
                src_encoding,
            )
        except Exception as exc:
            content, content_fetch_status = self._content_after_fetch_error(
                exc,
                remote_path,
                row_index,
                blob_id,
            )
        else:
            content_fetch_status = "fetched"
        return self._row_to_input_record_with_content(
            remote_path,
            row_index,
            row,
            content,
            content_fetch_status=content_fetch_status,
        )

    def _row_to_input_record_with_content(
        self,
        remote_path: str,
        row_index: int,
        row: dict[str, Any],
        content: str,
        *,
        content_fetch_status: str,
    ) -> InputRecord:
        record = super()._row_to_input_record(remote_path, row_index, row)
        metadata = dict(record.metadata)
        metadata["content_backend"] = "softwareheritage_s3"
        metadata["content_fetch_status"] = content_fetch_status
        metadata["content_url_template"] = self.content_fetcher.url_template if isinstance(
            self.content_fetcher,
            SoftwareHeritageS3ContentFetcher,
        ) else "custom"
        return InputRecord(
            dataset=record.dataset,
            record_id=record.record_id,
            content=content,
            language=record.language,
            path=record.path,
            repo=record.repo,
            metadata=metadata,
        )


class UrlListJsonlSource:
    def __init__(
        self,
        config: PipelineConfig,
        dataset: DatasetSpec,
        *,
        language: str | None = None,
        show_progress: bool = True,
        token: str | bool | None = None,
        downloader: HuggingFaceDownloader | None = None,
        max_files: int | None = None,
    ) -> None:
        self.name = _source_run_name(dataset.name, language)
        self.config = config
        self.dataset = dataset
        self.language = language
        self.show_progress = show_progress
        self.token = token
        self.downloader = downloader or HuggingFaceDownloader()
        self.max_files = max_files
        self.stream_retries = _nonnegative_int_option(
            dataset.extra.get("stream_retries"),
            dataset.extra.get("url_stream_retries"),
            default=_DEFAULT_URL_STREAM_RETRIES,
        )
        self.stream_retry_backoff_seconds = _nonnegative_float_option(
            dataset.extra.get("stream_retry_backoff_seconds"),
            dataset.extra.get("url_stream_retry_backoff_seconds"),
            default=_DEFAULT_URL_STREAM_RETRY_BACKOFF_SECONDS,
        )
        self.content_columns = _string_list_option(
            dataset.extra.get("content_columns"),
            _DEFAULT_CONTENT_COLUMNS,
        )
        self.language_columns = _string_list_option(
            dataset.extra.get("language_columns"),
            _DEFAULT_LANGUAGE_COLUMNS,
        )
        self.language_hint_columns = _string_list_option(
            dataset.extra.get("language_hint_columns"),
            [],
        )
        self.infer_language_from_path = _bool_option(
            dataset.extra.get("infer_language_from_path"),
            default=False,
        )
        self.path_columns = _string_list_option(
            dataset.extra.get("path_columns"),
            _DEFAULT_PATH_COLUMNS,
        )
        self.repo_columns = _string_list_option(
            dataset.extra.get("repo_columns"),
            _DEFAULT_REPO_COLUMNS,
        )

    def iter_records(self, start_after: str | None = None) -> Iterator[InputRecord]:
        self.config.ensure_directories()
        resume_cursor = UrlLineCursor.parse(start_after) if start_after else None
        urls = self._load_urls()
        if self.max_files is not None:
            urls = urls[: self.max_files]
        _LOGGER.info(
            "Preparing URL-list JSONL source dataset=%s language=%s urls=%s resume_from=%s",
            self.dataset.name,
            self.language or "all",
            len(urls),
            start_after,
        )

        started = resume_cursor is None
        for url in urls:
            if not started:
                if resume_cursor and url == resume_cursor.url:
                    started = True
                else:
                    continue
            start_line = (
                resume_cursor.line_index + 1
                if resume_cursor and resume_cursor.url == url
                else 0
            )
            yield from self._iter_url_records(url, start_line)
            resume_cursor = None

    def _load_urls(self) -> list[str]:
        configured_urls = self.dataset.extra.get("urls")
        if configured_urls is not None:
            return [str(item) for item in configured_urls]

        url_list_path = self.dataset.extra.get("url_list_path")
        if url_list_path is None:
            raise ValueError(
                f"Dataset '{self.dataset.name}' must define extra.url_list_path or extra.urls"
            )

        local_path = self.downloader.download_remote_file(
            self.config,
            self.dataset,
            str(url_list_path),
            language=None,
            token=self.token,
            use_cache=False,
        )
        try:
            return [
                line.strip()
                for line in local_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        finally:
            if self.dataset.streaming:
                self.downloader.remove_local_file(
                    self.config,
                    self.dataset,
                    str(url_list_path),
                    language=None,
                )

    def _iter_url_records(self, url: str, start_line: int) -> Iterator[InputRecord]:
        _LOGGER.info(
            "Streaming JSONL URL dataset=%s language=%s url=%s start_line=%s stream_retries=%s",
            self.dataset.name,
            self.language or "all",
            url,
            start_line,
            self.stream_retries,
        )
        progress = tqdm(
            desc=_progress_description(self.dataset.name, url),
            initial=start_line,
            unit="rows",
            dynamic_ncols=True,
            leave=False,
            disable=not self.show_progress,
        )
        next_line_index = start_line
        next_byte_offset: int | None = None
        next_byte_line_index = 0
        retries_used = 0
        try:
            while True:
                stream_start_byte = next_byte_offset or 0
                try:
                    line_index_base: int | None = None
                    for relative_line_index, text_line in enumerate(
                        _iter_text_lines(url, start_byte=stream_start_byte)
                    ):
                        if line_index_base is None:
                            line_index_base = (
                                next_byte_line_index
                                if stream_start_byte > 0 and text_line.resumed_from_byte
                                else 0
                            )
                        line_index = line_index_base + relative_line_index
                        if text_line.next_byte_offset is not None:
                            next_byte_offset = text_line.next_byte_offset
                            next_byte_line_index = line_index + 1
                        if line_index < next_line_index:
                            continue

                        next_line_index = line_index + 1
                        progress.update(1)
                        if not text_line.text:
                            continue
                        row = json.loads(text_line.text)
                        record = self._row_to_input_record(url, line_index, row)
                        if self.language and not _language_matches(record.language, self.language):
                            continue
                        yield record
                    return
                except httpx.HTTPError as exc:
                    if not _is_retryable_http_error(exc) or retries_used >= self.stream_retries:
                        _LOGGER.exception(
                            "Streaming JSONL URL failed dataset=%s language=%s url=%s next_line=%s next_byte_offset=%s next_byte_line=%s retries_used=%s",
                            self.dataset.name,
                            self.language or "all",
                            url,
                            next_line_index,
                            next_byte_offset,
                            next_byte_line_index,
                            retries_used,
                        )
                        raise

                    retries_used += 1
                    sleep_seconds = self.stream_retry_backoff_seconds * retries_used
                    _LOGGER.warning(
                        "Retrying JSONL URL stream dataset=%s language=%s url=%s next_line=%s next_byte_offset=%s next_byte_line=%s retry=%s/%s sleep_seconds=%.1f error=%s",
                        self.dataset.name,
                        self.language or "all",
                        url,
                        next_line_index,
                        next_byte_offset,
                        next_byte_line_index,
                        retries_used,
                        self.stream_retries,
                        sleep_seconds,
                        exc,
                    )
                    if sleep_seconds > 0:
                        time.sleep(sleep_seconds)
        finally:
            progress.close()

    def _row_to_input_record(
        self,
        url: str,
        line_index: int,
        row: dict[str, Any],
    ) -> InputRecord:
        content = _first_lookup(row, self.content_columns)
        path = _first_lookup(row, self.path_columns)
        raw_language = _first_lookup(row, self.language_columns)
        language_hints = _coerce_language_values(
            _first_lookup(row, self.language_hint_columns)
        )
        if not language_hints:
            language_hints = _coerce_language_values(raw_language)
        path_language = (
            _infer_language_from_path(path, hints=language_hints)
            if self.infer_language_from_path
            else None
        )
        language = path_language or _coerce_language(raw_language)
        repo = _first_lookup(row, self.repo_columns)
        metadata = {
            key: value
            for key, value in row.items()
            if key not in set(self.content_columns)
        }
        metadata["remote_url"] = url
        metadata["line_index"] = line_index
        ext = _path_extension(path)
        if ext is not None and "ext" not in metadata:
            metadata["ext"] = ext
        if path_language is not None:
            metadata["path_language"] = path_language
        if self.language is not None and _language_matches(language, self.language):
            metadata["selected_language"] = self.language

        return InputRecord(
            dataset=self.dataset.name,
            record_id=UrlLineCursor(url, line_index).to_record_id(),
            content=str(content) if content is not None else "",
            language=str(language) if language is not None else None,
            path=str(path) if path is not None else None,
            repo=str(repo) if repo is not None else None,
            metadata=metadata,
        )


class _RemoteDownloadScheduler:
    def __init__(
        self,
        source: HuggingFaceParquetSource,
        remote_files: list[RemoteFile],
    ) -> None:
        self.source = source
        self.remote_iter = iter(remote_files)
        self.executor: ThreadPoolExecutor | None = None
        self.pending: deque[tuple[RemoteFile, Future[Path]]] = deque()

    def __enter__(self) -> "_RemoteDownloadScheduler":
        if self.source.prefetch_files > 1 and self.source.download_workers > 1:
            self.executor = ThreadPoolExecutor(
                max_workers=min(self.source.download_workers, self.source.prefetch_files),
                thread_name_prefix="commentminer-download",
            )
        self._fill()
        return self

    def __exit__(self, *_: object) -> None:
        for _, future in self.pending:
            future.cancel()
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
        while self.pending:
            remote, future = self.pending.popleft()
            if future.cancelled() or not future.done():
                continue
            try:
                future.result()
            except Exception:
                continue
            if self.source.dataset.streaming:
                self.source.downloader.remove_local_file(
                    self.source.config,
                    self.source.dataset,
                    remote.path,
                    language=self.source.language,
                )

    def __iter__(self) -> "_RemoteDownloadScheduler":
        return self

    def __next__(self) -> tuple[RemoteFile, Path]:
        if self.executor is None:
            remote = next(self.remote_iter)
            return remote, self.source._download_remote(remote)

        self._fill()
        if not self.pending:
            raise StopIteration
        remote, future = self.pending.popleft()
        return remote, future.result()

    def _fill(self) -> None:
        if self.executor is None:
            return
        while len(self.pending) < self.source.prefetch_files:
            try:
                remote = next(self.remote_iter)
            except StopIteration:
                return
            self.pending.append((remote, self.executor.submit(self.source._download_remote, remote)))


def _iter_text_lines(url: str, *, start_byte: int = 0) -> Iterator[_TextLine]:
    if url.startswith("file://"):
        path = Path(url.removeprefix("file://"))
        yield from _iter_local_text_lines(path, start_byte=start_byte)
        return

    path = Path(url)
    if path.exists():
        yield from _iter_local_text_lines(path, start_byte=start_byte)
        return

    headers = {"Accept-Encoding": "identity"}
    if start_byte > 0:
        headers["Range"] = f"bytes={start_byte}-"
    with httpx.stream("GET", url, follow_redirects=True, timeout=None, headers=headers) as response:
        if start_byte > 0 and response.status_code == 416:
            _LOGGER.info("HTTP range resume reached EOF url=%s start_byte=%s", url, start_byte)
            return
        resumed_from_byte = start_byte > 0 and response.status_code == 206
        if start_byte > 0 and not resumed_from_byte:
            _LOGGER.warning(
                "HTTP server did not honor range request url=%s start_byte=%s status_code=%s",
                url,
                start_byte,
                response.status_code,
            )
        response.raise_for_status()
        byte_offset = start_byte if resumed_from_byte else 0
        yield from _iter_byte_text_lines(
            response.iter_bytes(),
            byte_offset=byte_offset,
            resumed_from_byte=resumed_from_byte,
        )


def _iter_local_text_lines(path: Path, *, start_byte: int = 0) -> Iterator[_TextLine]:
    with path.open("rb") as handle:
        if start_byte > 0:
            handle.seek(start_byte)
        yield from _iter_byte_text_lines(
            iter(lambda: handle.read(1024 * 1024), b""),
            byte_offset=start_byte,
            resumed_from_byte=start_byte > 0,
        )


def _iter_byte_text_lines(
    chunks: Iterator[bytes],
    *,
    byte_offset: int,
    resumed_from_byte: bool,
) -> Iterator[_TextLine]:
    pending = b""
    current_offset = byte_offset
    for chunk in chunks:
        if not chunk:
            continue
        pending += chunk
        lines = pending.split(b"\n")
        pending = lines.pop()
        for line_bytes in lines:
            current_offset += len(line_bytes) + 1
            yield _TextLine(
                text=_decode_jsonl_line(line_bytes),
                next_byte_offset=current_offset,
                resumed_from_byte=resumed_from_byte,
            )

    if pending:
        current_offset += len(pending)
        yield _TextLine(
            text=_decode_jsonl_line(pending),
            next_byte_offset=current_offset,
            resumed_from_byte=resumed_from_byte,
        )


def _decode_jsonl_line(line: bytes) -> str:
    if line.endswith(b"\r"):
        line = line[:-1]
    return line.decode("utf-8")


def _decode_stack_v2_raw_content(
    raw: bytes,
    compression: str | None,
    src_encoding: str | None,
) -> str:
    if compression == ".gz":
        raw = gzip.decompress(raw)
    return _decode_source_bytes(raw, src_encoding)


def _is_retryable_http_error(exc: httpx.HTTPError) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        return status_code == 408 or status_code == 429 or status_code >= 500
    return isinstance(exc, (httpx.TimeoutException, httpx.TransportError))


def _parquet_row_groups_for_range(
    parquet_file: pq.ParquetFile,
    start_row: int,
    stop_row: int | None,
) -> tuple[list[int] | None, int]:
    metadata = parquet_file.metadata
    if metadata is None:
        return None, 0
    if start_row <= 0 and stop_row is None:
        return None, 0

    row_groups: list[int] = []
    row_group_start = 0
    first_row_index = 0
    for row_group_index in range(metadata.num_row_groups):
        row_group_rows = metadata.row_group(row_group_index).num_rows
        row_group_end = row_group_start + row_group_rows
        intersects_start = row_group_end > start_row
        intersects_stop = stop_row is None or row_group_start < stop_row
        if intersects_start and intersects_stop:
            if not row_groups:
                first_row_index = row_group_start
            row_groups.append(row_group_index)
        row_group_start = row_group_end
        if stop_row is not None and row_group_start >= stop_row:
            break
    return row_groups, first_row_index


def _is_missing_content_error(exc: Exception) -> bool:
    if isinstance(exc, FileNotFoundError):
        return True

    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        error = response.get("Error")
        if isinstance(error, dict):
            code = str(error.get("Code", ""))
            if code in {"NoSuchKey", "404", "NotFound"}:
                return True
        status_code = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if status_code == 404:
            return True

    message = str(exc)
    return "NoSuchKey" in message or "404" in message or "not found" in message.lower()


def _is_retryable_s3_content_error(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPError):
        return _is_retryable_http_error(exc)
    status = getattr(exc, "status", None)
    if status in {408, 429} or (isinstance(status, int) and status >= 500):
        return True
    status_code = getattr(exc, "status_code", None)
    if status_code in {408, 429} or (
        isinstance(status_code, int) and status_code >= 500
    ):
        return True
    if isinstance(exc, (TimeoutError, OSError)):
        return True

    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        status_code = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if status_code in {408, 429} or (
            isinstance(status_code, int) and status_code >= 500
        ):
            return True

    name = exc.__class__.__name__
    if name in {
        "ConnectTimeoutError",
        "ConnectionClosedError",
        "EndpointConnectionError",
        "HTTPClientError",
        "ProxyConnectionError",
        "ReadTimeoutError",
    }:
        return True

    message = str(exc).lower()
    return any(
        text in message
        for text in (
            "temporary failure in name resolution",
            "connection aborted",
            "connection reset",
            "connection timed out",
            "could not connect to the endpoint url",
            "max retries exceeded",
            "read timed out",
            "remote end closed connection",
            "temporarily unavailable",
        )
    )


def _first_lookup(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = _lookup(row, key)
        if value is not None:
            return value
    return None


def _lookup(row: dict[str, Any], dotted_key: str) -> Any:
    value: Any = row
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _coerce_language(value: Any) -> str | None:
    values = _coerce_language_values(value)
    return values[0] if values else None


def _coerce_language_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        values: list[str] = []
        for item in value:
            values.extend(_coerce_language_values(item))
        return values
    if isinstance(value, dict):
        for key in ("name", "language", "lang"):
            if value.get(key) is not None:
                return [str(value[key])]
        return []
    return [str(value)]


def _infer_language_from_path(raw_path: Any, *, hints: list[str]) -> str | None:
    if raw_path is None:
        return None
    path = PurePosixPath(str(raw_path))
    filename_language = _FILENAME_LANGUAGE_ALIASES.get(path.name.lower())
    if filename_language is not None:
        return filename_language

    suffixes = [
        suffix.removeprefix(".").lower()
        for suffix in path.suffixes
        if suffix and suffix != "."
    ]
    if not suffixes:
        return None

    for suffix_count in range(min(2, len(suffixes)), 1, -1):
        compound = ".".join(suffixes[-suffix_count:])
        language = _COMPOUND_EXTENSION_LANGUAGE_ALIASES.get(compound)
        if language is not None:
            return language

    extension = suffixes[-1]
    ambiguous = _AMBIGUOUS_EXTENSION_LANGUAGE_ALIASES.get(extension)
    if ambiguous is not None:
        return _select_ambiguous_language(ambiguous, hints)
    return _EXTENSION_LANGUAGE_ALIASES.get(extension, extension)


def _select_ambiguous_language(candidates: list[str], hints: list[str]) -> str:
    by_key = {_language_key(candidate): candidate for candidate in candidates}
    for hint in hints:
        match = by_key.get(_language_key(hint))
        if match is not None:
            return match
    return candidates[0]


def _path_extension(raw_path: Any) -> str | None:
    if raw_path is None:
        return None
    suffix = PurePosixPath(str(raw_path)).suffix
    if not suffix or suffix == ".":
        return None
    return suffix.removeprefix(".").lower()


def _language_matches(candidate: str | None, selected: str) -> bool:
    if candidate is None:
        return False
    return _language_key(candidate) == _language_key(selected)


def _language_key(value: str) -> str:
    return value.strip().lower().replace("_", "-").replace(" ", "-")


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    if result.lower() in {"", "none", "null", "false"}:
        return None
    return result


def _decode_source_bytes(raw: bytes, src_encoding: str | None) -> str:
    encoding = src_encoding or "utf-8"
    try:
        return raw.decode(encoding, errors="replace")
    except (LookupError, OSError) as exc:
        _LOGGER.warning(
            "Could not use Stack v2 source encoding %r; falling back to utf-8 with replacement error=%s",
            src_encoding,
            exc,
        )
        return raw.decode("utf-8", errors="replace")


def _raise_fd_limit_for_stack_v2_content_workers(content_download_workers: int) -> None:
    if resource is None:
        return
    desired_soft_limit = max(1024, content_download_workers * 4 + 512)
    try:
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (OSError, ValueError):
        return
    if soft_limit >= desired_soft_limit:
        return

    new_soft_limit = min(desired_soft_limit, hard_limit)
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft_limit, hard_limit))
    except (OSError, ValueError) as exc:
        _LOGGER.warning(
            "Unable to raise open-file limit for Stack v2 S3 concurrency "
            "soft_limit=%s desired=%s hard_limit=%s error=%s",
            soft_limit,
            desired_soft_limit,
            hard_limit,
            exc,
        )
        return

    if new_soft_limit < desired_soft_limit:
        _LOGGER.warning(
            "Open-file limit is still below the requested Stack v2 S3 concurrency target "
            "soft_limit=%s desired=%s hard_limit=%s",
            new_soft_limit,
            desired_soft_limit,
            hard_limit,
        )


def _string_list_option(value: Any, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        raise TypeError(f"Expected list[str], got {type(value).__name__}")
    return [str(item) for item in value]


def _positive_int_option(*values: Any, default: int) -> int:
    for value in values:
        if value is None:
            continue
        result = int(value)
        if result < 1:
            raise ValueError(f"Expected positive integer, got {result}")
        return result
    return default


def _nonnegative_int_option(*values: Any, default: int) -> int:
    for value in values:
        if value is None:
            continue
        result = int(value)
        if result < 0:
            raise ValueError(f"Expected non-negative integer, got {result}")
        return result
    return default


def _nonnegative_float_option(*values: Any, default: float) -> float:
    for value in values:
        if value is None:
            continue
        result = float(value)
        if result < 0:
            raise ValueError(f"Expected non-negative float, got {result}")
        return result
    return default


def _bool_option(*values: Any, default: bool) -> bool:
    for value in values:
        if value is None:
            continue
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    return default


def _source_run_name(dataset_name: str, language: str | None) -> str:
    if language is None:
        return dataset_name
    return f"{dataset_name}__{language}"


def _progress_description(dataset_name: str, remote_path: str) -> str:
    parts = remote_path.split("/")
    shard_label = "/".join(parts[-2:]) if len(parts) >= 2 else remote_path
    if len(shard_label) > 80:
        shard_label = f"...{shard_label[-77:]}"
    return f"{dataset_name}:{shard_label}"
