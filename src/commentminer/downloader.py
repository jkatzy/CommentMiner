from __future__ import annotations

import fnmatch
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Iterator

from huggingface_hub import HfApi, hf_hub_download

from .config import DatasetSpec, PipelineConfig


_SLUG_PATTERN = re.compile(r"[^a-zA-Z0-9]+")
_LOGGER = logging.getLogger(__name__)
_LANGUAGE_PLACEHOLDER = "__COMMENTMINER_LANGUAGE__"


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

    def supports_subprocess_workers(self) -> bool:
        return isinstance(self.api, HfApi) and self.download_file is hf_hub_download

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
        remote_files = self.list_remote_files(dataset, language=language, token=token)
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

    def iter_pending_files(
        self,
        config: PipelineConfig,
        dataset: DatasetSpec,
        *,
        language: str | None = None,
        token: str | bool | None = None,
        max_files: int | None = None,
        checkpoint_namespace: str = "downloads",
    ) -> Iterator[RemoteFile]:
        repo_id = dataset.resolve_repo_id()
        checkpoint_store = self.checkpoint_store(config, namespace=checkpoint_namespace)
        checkpoint = checkpoint_store.load(dataset.name, repo_id, dataset.revision, language)
        completed = set(checkpoint.completed_files)
        yielded = 0
        _LOGGER.info(
            "Streaming pending files for dataset=%s repo=%s language=%s checkpoint_namespace=%s",
            dataset.name,
            repo_id,
            language or "all",
            checkpoint_namespace,
        )
        for remote in self.iter_remote_files(dataset, language=language, token=token):
            if remote.path in completed:
                continue
            yield remote
            yielded += 1
            if max_files is not None and yielded >= max_files:
                break
        _LOGGER.info(
            "Finished streaming pending files for dataset=%s language=%s yielded=%s checkpoint_namespace=%s",
            dataset.name,
            language or "all",
            yielded,
            checkpoint_namespace,
        )

    def list_remote_files(
        self,
        dataset: DatasetSpec,
        *,
        language: str | None = None,
        token: str | bool | None = None,
    ) -> list[RemoteFile]:
        remote_files = list(self.iter_remote_files(dataset, language=language, token=token))
        remote_files.sort(key=lambda item: item.path)
        _LOGGER.info(
            "Resolved %s matching remote files for dataset=%s language=%s",
            len(remote_files),
            dataset.name,
            language or "all",
        )
        return remote_files

    def iter_remote_files(
        self,
        dataset: DatasetSpec,
        *,
        language: str | None = None,
        token: str | bool | None = None,
    ) -> Iterator[RemoteFile]:
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
        tree = self.api.list_repo_tree(
            repo_id=repo_id,
            repo_type=dataset.repo_type,
            revision=dataset.revision,
            recursive=True,
            token=token,
        )

        for entry in tree:
            path = getattr(entry, "path", None)
            size_bytes = getattr(entry, "size", None)
            if path is None or size_bytes is None:
                continue
            if not _matches_any(path, allow_patterns):
                continue
            if not _matches_none(path, ignore_patterns):
                continue
            yield RemoteFile(path=path, size_bytes=size_bytes)

    def list_languages(
        self,
        dataset: DatasetSpec,
        *,
        token: str | bool | None = None,
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
        remote_files = self.list_remote_files(dataset, token=token)
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
    ) -> Path:
        _LOGGER.info(
            "Downloading remote file dataset=%s language=%s path=%s",
            dataset.name,
            language or "all",
            remote_path,
        )
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
    ) -> DownloadSummary:
        config.ensure_directories()
        plan = self.plan_download(
            config,
            dataset,
            language=language,
            token=token,
            max_files=max_files,
            checkpoint_namespace="downloads",
        )
        _LOGGER.info(
            "Starting download dataset=%s language=%s pending_files=%s",
            dataset.name,
            language or "all",
            len(plan.pending_files),
        )
        checkpoint_store = self.checkpoint_store(config, namespace="downloads")
        checkpoint = checkpoint_store.load(dataset.name, plan.repo_id, plan.revision, language)
        completed = set(checkpoint.completed_files)
        downloaded_paths: list[Path] = []
        plan.download_root.mkdir(parents=True, exist_ok=True)

        for remote in plan.pending_files:
            local_path = self.download_file(
                repo_id=plan.repo_id,
                filename=remote.path,
                repo_type=dataset.repo_type,
                revision=dataset.revision,
                cache_dir=config.storage.huggingface_cache_directory,
                local_dir=plan.download_root,
                token=token,
            )
            downloaded_paths.append(Path(local_path))
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
