from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import fcntl
import fnmatch
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterator

import httpx
from huggingface_hub import HfApi, hf_hub_download, hf_hub_url
from huggingface_hub.utils import build_hf_headers

from .config import DatasetSpec, PipelineConfig


_SLUG_PATTERN = re.compile(r"[^a-zA-Z0-9]+")
_LOGGER = logging.getLogger(__name__)
_LANGUAGE_PLACEHOLDER = "__COMMENTMINER_LANGUAGE__"
_DEFAULT_DIRECT_DOWNLOAD_RETRIES = 5
_DEFAULT_DIRECT_DOWNLOAD_RETRY_BACKOFF_SECONDS = 10.0
_REMOTE_FILE_CACHE_NAMESPACE = "remote-file-cache"


def _slugify(value: str) -> str:
    slug = _SLUG_PATTERN.sub("-", value).strip("-").lower()
    return slug or "dataset"


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _matches_any(path: str, patterns: list[str]) -> bool:
    if not patterns:
        return True
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def _matches_none(path: str, patterns: list[str]) -> bool:
    return not any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def _compile_language_pattern(pattern: str) -> re.Pattern[str]:
    translated = fnmatch.translate(pattern.replace("{language}", _LANGUAGE_PLACEHOLDER))
    return re.compile(
        translated.replace(_LANGUAGE_PLACEHOLDER, r"(?P<language>[^/]+)")
    )


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


def _positive_int_option(*values: Any, default: int) -> int:
    for value in values:
        if value is None:
            continue
        result = int(value)
        if result < 1:
            raise ValueError(f"Expected positive integer, got {result}")
        return result
    return default


def _is_retryable_http_error(exc: httpx.HTTPError) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        return status_code == 408 or status_code == 429 or status_code >= 500
    return isinstance(exc, (httpx.TimeoutException, httpx.TransportError))


@dataclass(frozen=True, slots=True)
class RemoteFile:
    path: str
    size_bytes: int | None = None


@dataclass(slots=True)
class DownloadCheckpoint:
    dataset: str
    repo_id: str
    revision: str
    language: str | None = None
    completed_files: list[str] = field(default_factory=list)
    last_downloaded_file: str | None = None
    updated_at: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["completed_files"] = sorted(self.completed_files)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "DownloadCheckpoint":
        return cls(
            dataset=str(data["dataset"]),
            repo_id=str(data["repo_id"]),
            revision=str(data["revision"]),
            language=str(data["language"]) if data.get("language") is not None else None,
            completed_files=[str(item) for item in data.get("completed_files", [])],
            last_downloaded_file=str(data["last_downloaded_file"]) if data.get("last_downloaded_file") else None,
            updated_at=str(data["updated_at"]) if data.get("updated_at") is not None else None,
        )


@dataclass(frozen=True, slots=True)
class DownloadPlan:
    dataset: str
    repo_id: str
    revision: str
    language: str | None
    download_root: Path
    checkpoint_path: Path
    cache_directory: Path
    allow_patterns: list[str]
    ignore_patterns: list[str]
    matched_files: list[RemoteFile]
    pending_files: list[RemoteFile]
    completed_files: list[str]

    @property
    def matched_count(self) -> int:
        return len(self.matched_files)

    @property
    def pending_count(self) -> int:
        return len(self.pending_files)

    @property
    def completed_count(self) -> int:
        return len(self.completed_files)

    @property
    def matched_bytes(self) -> int | None:
        if any(remote.size_bytes is None for remote in self.matched_files):
            return None
        return sum(remote.size_bytes or 0 for remote in self.matched_files)

    @property
    def pending_bytes(self) -> int | None:
        if any(remote.size_bytes is None for remote in self.pending_files):
            return None
        return sum(remote.size_bytes or 0 for remote in self.pending_files)


@dataclass(frozen=True, slots=True)
class DownloadSummary:
    dataset: str
    repo_id: str
    revision: str
    language: str | None
    download_root: Path
    checkpoint_path: Path
    matched_count: int
    already_downloaded_count: int
    downloaded_count: int
    downloaded_files: list[Path]


class DownloadCheckpointStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, dataset_name: str, language: str | None = None) -> Path:
        suffix = f"-{_slugify(language)}" if language else ""
        return self.root / f"{_slugify(dataset_name)}{suffix}.json"

    def load(
        self,
        dataset_name: str,
        repo_id: str,
        revision: str,
        language: str | None = None,
    ) -> DownloadCheckpoint:
        path = self.path_for(dataset_name, language)
        if not path.exists():
            return DownloadCheckpoint(
                dataset=dataset_name,
                repo_id=repo_id,
                revision=revision,
                language=language,
            )
        raw = json.loads(path.read_text(encoding="utf-8"))
        checkpoint = DownloadCheckpoint.from_dict(raw)
        if checkpoint.repo_id != repo_id or checkpoint.revision != revision:
            return DownloadCheckpoint(
                dataset=dataset_name,
                repo_id=repo_id,
                revision=revision,
                language=language,
            )
        return checkpoint

    def save(self, checkpoint: DownloadCheckpoint) -> Path:
        checkpoint.updated_at = _utc_now()
        path = self.path_for(checkpoint.dataset, checkpoint.language)
        temp_path = path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(checkpoint.to_dict(), indent=2), encoding="utf-8")
        temp_path.replace(path)
        return path


class HuggingFaceDownloader:
    def __init__(
        self,
        *,
        api: HfApi | None = None,
        download_file: Callable[..., str] | None = None,
    ) -> None:
        self.api = api or HfApi()
        self.download_file = download_file or hf_hub_download
        self._uses_default_download_file = download_file is None

    def plan_download(
        self,
        config: PipelineConfig,
        dataset: DatasetSpec,
        *,
        language: str | None = None,
        token: str | bool | None = None,
        max_files: int | None = None,
        checkpoint_namespace: str = "downloads",
    ) -> DownloadPlan:
        repo_id = dataset.resolve_repo_id()
        allow_patterns, ignore_patterns = dataset.resolve_patterns(language)
        checkpoint_store = self.checkpoint_store(config, namespace=checkpoint_namespace)
        checkpoint = checkpoint_store.load(dataset.name, repo_id, dataset.revision, language)
        _LOGGER.info(
            "Planning download for dataset=%s repo=%s language=%s checkpoint_namespace=%s",
            dataset.name,
            repo_id,
            language or "all",
            checkpoint_namespace,
        )
        remote_files = self.list_remote_files(
            dataset,
            language=language,
            token=token,
            cache_directory=self.remote_file_cache_directory(config),
        )
        file_start_at = dataset.extra.get("file_start_at")
        if file_start_at is not None:
            remote_files = [
                remote for remote in remote_files if remote.path >= str(file_start_at)
            ]
        file_stop_at = dataset.extra.get("file_stop_at")
        if file_stop_at is not None:
            remote_files = [
                remote for remote in remote_files if remote.path <= str(file_stop_at)
            ]
        partition_count = int(dataset.extra.get("file_partition_count", 1))
        partition_index = int(dataset.extra.get("file_partition_index", 0))
        if partition_count < 1:
            raise ValueError("file_partition_count must be at least 1")
        if not 0 <= partition_index < partition_count:
            raise ValueError(
                "file_partition_index must be between 0 and "
                f"{partition_count - 1}"
            )
        if partition_count > 1:
            remote_files = [
                remote
                for index, remote in enumerate(remote_files)
                if index % partition_count == partition_index
            ]
        completed = set(checkpoint.completed_files)
        remote_paths = {remote.path for remote in remote_files}
        pending_files = [remote for remote in remote_files if remote.path not in completed]
        if max_files is not None:
            pending_files = pending_files[:max_files]

        return DownloadPlan(
            dataset=dataset.name,
            repo_id=repo_id,
            revision=dataset.revision,
            language=language,
            download_root=self._download_root(config, dataset, language),
            checkpoint_path=checkpoint_store.path_for(dataset.name, language),
            cache_directory=config.storage.huggingface_cache_directory,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            matched_files=remote_files,
            pending_files=pending_files,
            completed_files=sorted(remote_paths.intersection(completed)),
        )

    def list_remote_files(
        self,
        dataset: DatasetSpec,
        *,
        language: str | None = None,
        token: str | bool | None = None,
        cache_directory: Path | None = None,
    ) -> list[RemoteFile]:
        if dataset.source != "huggingface_hub":
            raise ValueError(f"Dataset '{dataset.name}' is not configured for Hugging Face downloads")

        repo_id = dataset.resolve_repo_id()
        allow_patterns, ignore_patterns = dataset.resolve_patterns(language)
        _LOGGER.info(
            "Listing remote files for dataset=%s repo=%s language=%s",
            dataset.name,
            repo_id,
            language or "all",
        )
        all_remote_files = self._load_or_list_remote_files(
            dataset,
            token=token,
            cache_directory=cache_directory,
        )
        remote_files = []
        for remote in all_remote_files:
            path = remote.path
            if not _matches_any(path, allow_patterns):
                continue
            if not _matches_none(path, ignore_patterns):
                continue
            remote_files.append(remote)

        remote_files.sort(key=lambda item: item.path)
        _LOGGER.info(
            "Resolved %s matching remote files for dataset=%s language=%s",
            len(remote_files),
            dataset.name,
            language or "all",
        )
        return remote_files

    def list_languages(
        self,
        dataset: DatasetSpec,
        *,
        token: str | bool | None = None,
        cache_directory: Path | None = None,
    ) -> list[str]:
        configured_languages = dataset.available_languages()
        if configured_languages:
            return configured_languages
        if dataset.source != "huggingface_hub" or not dataset.supports_language_selection():
            return []

        discovery_patterns = dataset.language_discovery_patterns()
        if not discovery_patterns:
            return []

        _LOGGER.info(
            "Discovering languages for dataset=%s repo=%s",
            dataset.name,
            dataset.resolve_repo_id(),
        )
        remote_files = self.list_remote_files(
            dataset,
            token=token,
            cache_directory=cache_directory,
        )
        compiled_patterns = [_compile_language_pattern(pattern) for pattern in discovery_patterns]
        languages: set[str] = set()
        for remote in remote_files:
            for pattern in compiled_patterns:
                match = pattern.match(remote.path)
                if match is None:
                    continue
                languages.add(match.group("language"))
                break
        discovered = sorted(languages)
        _LOGGER.info(
            "Discovered %s languages for dataset=%s",
            len(discovered),
            dataset.name,
        )
        return discovered

    def remote_file_cache_directory(self, config: PipelineConfig) -> Path:
        return config.storage.working_directory / _REMOTE_FILE_CACHE_NAMESPACE

    def _load_or_list_remote_files(
        self,
        dataset: DatasetSpec,
        *,
        token: str | bool | None,
        cache_directory: Path | None,
    ) -> list[RemoteFile]:
        if cache_directory is None:
            return self._list_repo_tree_files(dataset, token=token)

        cache_directory.mkdir(parents=True, exist_ok=True)
        cache_path = self._remote_file_cache_path(cache_directory, dataset)
        cached = self._load_remote_file_cache(cache_path, dataset)
        if cached is not None:
            _LOGGER.info(
                "Using cached remote file listing dataset=%s repo=%s path=%s files=%s",
                dataset.name,
                dataset.resolve_repo_id(),
                cache_path,
                len(cached),
            )
            return cached

        lock_path = cache_path.with_suffix(f"{cache_path.suffix}.lock")
        with lock_path.open("w", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            cached = self._load_remote_file_cache(cache_path, dataset)
            if cached is not None:
                _LOGGER.info(
                    "Using cached remote file listing dataset=%s repo=%s path=%s files=%s",
                    dataset.name,
                    dataset.resolve_repo_id(),
                    cache_path,
                    len(cached),
                )
                return cached

            remote_files = self._list_repo_tree_files(dataset, token=token)
            self._write_remote_file_cache(cache_path, dataset, remote_files)
            return remote_files

    def _list_repo_tree_files(
        self,
        dataset: DatasetSpec,
        *,
        token: str | bool | None,
    ) -> list[RemoteFile]:
        tree = self.api.list_repo_tree(
            repo_id=dataset.resolve_repo_id(),
            repo_type=dataset.repo_type,
            revision=dataset.revision,
            recursive=True,
            token=token,
        )

        remote_files: list[RemoteFile] = []
        for entry in tree:
            path = getattr(entry, "path", None)
            size_bytes = getattr(entry, "size", None)
            if path is None or size_bytes is None:
                continue
            remote_files.append(RemoteFile(path=str(path), size_bytes=int(size_bytes)))
        remote_files.sort(key=lambda item: item.path)
        return remote_files

    def _remote_file_cache_path(self, cache_directory: Path, dataset: DatasetSpec) -> Path:
        cache_key = "__".join(
            [
                dataset.name,
                dataset.resolve_repo_id(),
                dataset.repo_type,
                dataset.revision,
            ]
        )
        return cache_directory / f"{_slugify(cache_key)}.json"

    def _load_remote_file_cache(
        self,
        cache_path: Path,
        dataset: DatasetSpec,
    ) -> list[RemoteFile] | None:
        if not cache_path.exists():
            return None
        try:
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            if raw.get("repo_id") != dataset.resolve_repo_id():
                return None
            if raw.get("repo_type") != dataset.repo_type:
                return None
            if raw.get("revision") != dataset.revision:
                return None
            return [
                RemoteFile(
                    path=str(item["path"]),
                    size_bytes=int(item["size_bytes"]) if item.get("size_bytes") is not None else None,
                )
                for item in raw.get("files", [])
            ]
        except Exception as exc:
            _LOGGER.warning(
                "Ignoring unreadable remote file cache dataset=%s path=%s error=%s",
                dataset.name,
                cache_path,
                exc,
            )
            return None

    def _write_remote_file_cache(
        self,
        cache_path: Path,
        dataset: DatasetSpec,
        remote_files: list[RemoteFile],
    ) -> None:
        payload = {
            "dataset": dataset.name,
            "repo_id": dataset.resolve_repo_id(),
            "repo_type": dataset.repo_type,
            "revision": dataset.revision,
            "created_at": _utc_now(),
            "files": [
                {
                    "path": remote.path,
                    "size_bytes": remote.size_bytes,
                }
                for remote in remote_files
            ],
        }
        temp_path = cache_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temp_path.replace(cache_path)
        _LOGGER.info(
            "Cached remote file listing dataset=%s repo=%s path=%s files=%s",
            dataset.name,
            dataset.resolve_repo_id(),
            cache_path,
            len(remote_files),
        )

    def checkpoint_store(
        self,
        config: PipelineConfig,
        *,
        namespace: str = "downloads",
    ) -> DownloadCheckpointStore:
        return DownloadCheckpointStore(config.storage.checkpoint_directory / namespace)

    def load_checkpoint(
        self,
        config: PipelineConfig,
        dataset: DatasetSpec,
        *,
        language: str | None = None,
        checkpoint_namespace: str = "downloads",
    ) -> DownloadCheckpoint:
        repo_id = dataset.resolve_repo_id()
        return self.checkpoint_store(config, namespace=checkpoint_namespace).load(
            dataset.name,
            repo_id,
            dataset.revision,
            language,
        )

    def save_checkpoint(
        self,
        config: PipelineConfig,
        checkpoint: DownloadCheckpoint,
        *,
        checkpoint_namespace: str = "downloads",
    ) -> Path:
        return self.checkpoint_store(config, namespace=checkpoint_namespace).save(checkpoint)

    def download_remote_file(
        self,
        config: PipelineConfig,
        dataset: DatasetSpec,
        remote_path: str,
        *,
        language: str | None = None,
        token: str | bool | None = None,
        use_cache: bool = True,
    ) -> Path:
        _LOGGER.info(
            "Downloading remote file dataset=%s language=%s path=%s",
            dataset.name,
            language or "all",
            remote_path,
        )
        if not use_cache and self._uses_default_download_file:
            local_path = self._download_remote_file_direct(
                config,
                dataset,
                remote_path,
                language=language,
                token=token,
            )
            _LOGGER.info(
                "Finished direct download dataset=%s language=%s path=%s local_path=%s",
                dataset.name,
                language or "all",
                remote_path,
                local_path,
            )
            return local_path

        local_path = self.download_file(
            repo_id=dataset.resolve_repo_id(),
            filename=remote_path,
            repo_type=dataset.repo_type,
            revision=dataset.revision,
            cache_dir=config.storage.huggingface_cache_directory,
            local_dir=self._download_root(config, dataset, language),
            token=token,
        )
        _LOGGER.info(
            "Finished download dataset=%s language=%s path=%s local_path=%s",
            dataset.name,
            language or "all",
            remote_path,
            local_path,
        )
        return Path(local_path)

    def _download_remote_file_direct(
        self,
        config: PipelineConfig,
        dataset: DatasetSpec,
        remote_path: str,
        *,
        language: str | None,
        token: str | bool | None,
    ) -> Path:
        target_path = self._download_root(config, dataset, language) / remote_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.exists():
            return target_path

        tmp_path = target_path.with_name(f"{target_path.name}.incomplete")
        url = hf_hub_url(
            repo_id=dataset.resolve_repo_id(),
            filename=remote_path,
            repo_type=dataset.repo_type,
            revision=dataset.revision,
        )
        headers = build_hf_headers(token=token)
        download_retries = _nonnegative_int_option(
            dataset.extra.get("download_retries"),
            dataset.extra.get("direct_download_retries"),
            default=_DEFAULT_DIRECT_DOWNLOAD_RETRIES,
        )
        download_retry_backoff_seconds = _nonnegative_float_option(
            dataset.extra.get("download_retry_backoff_seconds"),
            dataset.extra.get("direct_download_retry_backoff_seconds"),
            default=_DEFAULT_DIRECT_DOWNLOAD_RETRY_BACKOFF_SECONDS,
        )
        retries_used = 0
        while True:
            try:
                with httpx.stream(
                    "GET",
                    url,
                    headers=headers,
                    follow_redirects=True,
                    timeout=None,
                ) as response:
                    response.raise_for_status()
                    with tmp_path.open("wb") as handle:
                        for chunk in response.iter_bytes():
                            handle.write(chunk)
                tmp_path.replace(target_path)
                return target_path
            except httpx.HTTPError as exc:
                tmp_path.unlink(missing_ok=True)
                if not _is_retryable_http_error(exc) or retries_used >= download_retries:
                    raise

                retries_used += 1
                sleep_seconds = download_retry_backoff_seconds * retries_used
                _LOGGER.warning(
                    "Retrying direct download dataset=%s language=%s path=%s retry=%s/%s sleep_seconds=%.1f error=%s",
                    dataset.name,
                    language or "all",
                    remote_path,
                    retries_used,
                    download_retries,
                    sleep_seconds,
                    exc,
                )
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
            except Exception:
                tmp_path.unlink(missing_ok=True)
                raise

    def mark_file_completed(
        self,
        config: PipelineConfig,
        dataset: DatasetSpec,
        remote_path: str,
        *,
        language: str | None = None,
        checkpoint_namespace: str = "downloads",
    ) -> Path:
        checkpoint = self.load_checkpoint(
            config,
            dataset,
            language=language,
            checkpoint_namespace=checkpoint_namespace,
        )
        if remote_path not in checkpoint.completed_files:
            checkpoint.completed_files.append(remote_path)
        checkpoint.last_downloaded_file = remote_path
        _LOGGER.info(
            "Marked file complete dataset=%s language=%s path=%s checkpoint_namespace=%s",
            dataset.name,
            language or "all",
            remote_path,
            checkpoint_namespace,
        )
        return self.save_checkpoint(
            config,
            checkpoint,
            checkpoint_namespace=checkpoint_namespace,
        )

    def remove_local_file(
        self,
        config: PipelineConfig,
        dataset: DatasetSpec,
        remote_path: str,
        *,
        language: str | None = None,
    ) -> None:
        local_path = self._download_root(config, dataset, language) / remote_path
        if not local_path.exists():
            return
        _LOGGER.info(
            "Removing local shard dataset=%s language=%s path=%s",
            dataset.name,
            language or "all",
            local_path,
        )
        local_path.unlink()

        download_root = self._download_root(config, dataset, language)
        parent = local_path.parent
        while parent != download_root and parent.exists():
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent

    def download(
        self,
        config: PipelineConfig,
        dataset: DatasetSpec,
        *,
        language: str | None = None,
        token: str | bool | None = None,
        max_files: int | None = None,
        download_workers: int | None = None,
    ) -> DownloadSummary:
        config.ensure_directories()
        worker_count = _positive_int_option(
            download_workers,
            dataset.extra.get("download_workers"),
            default=1,
        )
        plan = self.plan_download(
            config,
            dataset,
            language=language,
            token=token,
            max_files=max_files,
            checkpoint_namespace="downloads",
        )
        _LOGGER.info(
            "Starting download dataset=%s language=%s pending_files=%s download_workers=%s",
            dataset.name,
            language or "all",
            len(plan.pending_files),
            worker_count,
        )
        checkpoint_store = self.checkpoint_store(config, namespace="downloads")
        checkpoint = checkpoint_store.load(dataset.name, plan.repo_id, plan.revision, language)
        completed = set(checkpoint.completed_files)
        downloaded_paths: list[Path] = []
        plan.download_root.mkdir(parents=True, exist_ok=True)

        for remote, local_path in self._download_pending_files(
            plan,
            dataset,
            token=token,
            worker_count=worker_count,
        ):
            downloaded_paths.append(local_path)
            if remote.path not in completed:
                checkpoint.completed_files.append(remote.path)
                completed.add(remote.path)
            checkpoint.last_downloaded_file = remote.path
            checkpoint_store.save(checkpoint)

        checkpoint_path = checkpoint_store.save(checkpoint)
        _LOGGER.info(
            "Completed download dataset=%s language=%s downloaded_now=%s already_downloaded=%s",
            dataset.name,
            language or "all",
            len(downloaded_paths),
            plan.completed_count,
        )
        return DownloadSummary(
            dataset=dataset.name,
            repo_id=plan.repo_id,
            revision=plan.revision,
            language=language,
            download_root=plan.download_root,
            checkpoint_path=checkpoint_path,
            matched_count=plan.matched_count,
            already_downloaded_count=plan.completed_count,
            downloaded_count=len(downloaded_paths),
            downloaded_files=downloaded_paths,
        )

    def _download_pending_files(
        self,
        plan: DownloadPlan,
        dataset: DatasetSpec,
        *,
        token: str | bool | None,
        worker_count: int,
    ) -> Iterator[tuple[RemoteFile, Path]]:
        if worker_count == 1:
            for remote in plan.pending_files:
                yield (
                    remote,
                    self._download_plan_file(plan, dataset, remote, token=token),
                )
            return

        executor = ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="commentminer-download",
        )
        remote_iter = iter(plan.pending_files)
        pending: deque[tuple[RemoteFile, Future[Path]]] = deque()

        def fill() -> None:
            while len(pending) < worker_count:
                try:
                    remote = next(remote_iter)
                except StopIteration:
                    return
                pending.append(
                    (
                        remote,
                        executor.submit(
                            self._download_plan_file,
                            plan,
                            dataset,
                            remote,
                            token=token,
                        ),
                    )
                )

        try:
            fill()
            while pending:
                remote, future = pending.popleft()
                yield remote, future.result()
                fill()
        finally:
            for _, future in pending:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)

    def _download_plan_file(
        self,
        plan: DownloadPlan,
        dataset: DatasetSpec,
        remote: RemoteFile,
        *,
        token: str | bool | None,
    ) -> Path:
        local_path = self.download_file(
            repo_id=plan.repo_id,
            filename=remote.path,
            repo_type=dataset.repo_type,
            revision=dataset.revision,
            cache_dir=plan.cache_directory,
            local_dir=plan.download_root,
            token=token,
        )
        return Path(local_path)

    def _download_root(
        self,
        config: PipelineConfig,
        dataset: DatasetSpec,
        language: str | None,
    ) -> Path:
        root = config.storage.download_directory / _slugify(dataset.name)
        if language:
            root = root / _slugify(language)
        return root
