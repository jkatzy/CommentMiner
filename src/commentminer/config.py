from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


def _resolve_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (base_dir / path).resolve()


def _require_positive_int(name: str, value: int) -> int:
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f"Expected list[str], got {type(value).__name__}")
    return [str(item) for item in value]


def _as_string_list_mapping(value: Any) -> dict[str, list[str]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected mapping[str, list[str]], got {type(value).__name__}")
    return {str(key): _as_string_list(item) for key, item in value.items()}


@dataclass(slots=True)
class StorageConfig:
    working_directory: Path
    output_directory: Path
    checkpoint_directory: Path
    download_directory: Path
    huggingface_cache_directory: Path
    max_records_per_shard: int = 100_000
    max_bytes_per_shard: int = 128 * 1024 * 1024

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], base_dir: Path) -> "StorageConfig":
        working_directory = _resolve_path(base_dir, str(data["working_directory"]))
        return cls(
            working_directory=working_directory,
            output_directory=_resolve_path(base_dir, str(data["output_directory"])),
            checkpoint_directory=_resolve_path(base_dir, str(data["checkpoint_directory"])),
            download_directory=_resolve_path(
                base_dir,
                str(data.get("download_directory", working_directory / "downloads")),
            ),
            huggingface_cache_directory=_resolve_path(
                base_dir,
                str(data.get("huggingface_cache_directory", working_directory / "hf-cache")),
            ),
            max_records_per_shard=_require_positive_int(
                "max_records_per_shard",
                int(data.get("max_records_per_shard", 100_000)),
            ),
            max_bytes_per_shard=_require_positive_int(
                "max_bytes_per_shard",
                int(data.get("max_bytes_per_shard", 128 * 1024 * 1024)),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "working_directory": str(self.working_directory),
            "output_directory": str(self.output_directory),
            "checkpoint_directory": str(self.checkpoint_directory),
            "download_directory": str(self.download_directory),
            "huggingface_cache_directory": str(self.huggingface_cache_directory),
            "max_records_per_shard": self.max_records_per_shard,
            "max_bytes_per_shard": self.max_bytes_per_shard,
        }


@dataclass(slots=True)
class DatasetSpec:
    name: str
    input_uri: str | None = None
    source: str = "huggingface_hub"
    repo_id: str | None = None
    repo_type: str = "dataset"
    revision: str = "main"
    split: str = "train"
    streaming: bool = True
    enabled: bool = True
    subset: str | None = None
    batch_size: int = 1_000
    allow_patterns: list[str] = field(default_factory=list)
    ignore_patterns: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    language_patterns: dict[str, list[str]] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DatasetSpec":
        return cls(
            name=str(data["name"]),
            input_uri=str(data["input_uri"]) if data.get("input_uri") is not None else None,
            source=str(data.get("source", "huggingface_hub")),
            repo_id=str(data["repo_id"]) if data.get("repo_id") is not None else None,
            repo_type=str(data.get("repo_type", "dataset")),
            revision=str(data.get("revision", "main")),
            split=str(data.get("split", "train")),
            streaming=bool(data.get("streaming", True)),
            enabled=bool(data.get("enabled", True)),
            subset=str(data["subset"]) if data.get("subset") is not None else None,
            batch_size=_require_positive_int("batch_size", int(data.get("batch_size", 1_000))),
            allow_patterns=_as_string_list(data.get("allow_patterns")),
            ignore_patterns=_as_string_list(data.get("ignore_patterns")),
            languages=_as_string_list(data.get("languages")),
            language_patterns=_as_string_list_mapping(data.get("language_patterns")),
            extra=dict(data.get("extra", {})),
        )

    def supports_language_selection(self) -> bool:
        if self.languages or self.language_patterns:
            return True
        patterns = self.allow_patterns + self.ignore_patterns
        return any("{language}" in pattern for pattern in patterns)

    def available_languages(self) -> list[str]:
        configured = set(self.languages)
        configured.update(self.language_patterns.keys())
        return sorted(configured)

    def language_discovery_patterns(self) -> list[str]:
        patterns: list[str] = []
        for pattern in self.allow_patterns:
            if "{language}" not in pattern:
                continue
            patterns.append(self._format_pattern(pattern, language="{language}"))
        return patterns

    def resolve_repo_id(self) -> str:
        if self.repo_id:
            return self.repo_id
        if self.input_uri and "://" not in self.input_uri:
            return self.input_uri
        raise ValueError(f"Dataset '{self.name}' does not define a Hugging Face repo_id")

    def resolve_patterns(self, language: str | None = None) -> tuple[list[str], list[str]]:
        if language is not None and not self.supports_language_selection():
            raise ValueError(f"Dataset '{self.name}' does not define language-aware download patterns")
        if language is not None and self.available_languages() and language not in self.available_languages():
            available = ", ".join(self.available_languages())
            raise ValueError(f"Unsupported language '{language}' for dataset '{self.name}'. Available: {available}")

        allow_patterns = [self._format_pattern(pattern, language=language) for pattern in self.allow_patterns]
        ignore_patterns = [self._format_pattern(pattern, language=language) for pattern in self.ignore_patterns]
        if language is not None:
            allow_patterns.extend(
                self._format_pattern(pattern, language=language)
                for pattern in self.language_patterns.get(language, [])
            )
        return allow_patterns, ignore_patterns

    def _format_pattern(self, pattern: str, *, language: str | None) -> str:
        resolved = pattern
        replacements = {
            "language": language if language is not None else "*",
            "split": self.split,
            "subset": self.subset,
        }
        for key, value in replacements.items():
            placeholder = "{" + key + "}"
            if placeholder not in resolved:
                continue
            if value is None:
                raise ValueError(f"Dataset '{self.name}' pattern '{pattern}' requires '{key}' but it is unset")
            resolved = resolved.replace(placeholder, value)
        return resolved

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "source": self.source,
            "repo_type": self.repo_type,
            "revision": self.revision,
            "split": self.split,
            "streaming": self.streaming,
            "enabled": self.enabled,
            "batch_size": self.batch_size,
        }
        if self.input_uri is not None:
            payload["input_uri"] = self.input_uri
        if self.repo_id is not None:
            payload["repo_id"] = self.repo_id
        if self.subset is not None:
            payload["subset"] = self.subset
        if self.allow_patterns:
            payload["allow_patterns"] = self.allow_patterns
        if self.ignore_patterns:
            payload["ignore_patterns"] = self.ignore_patterns
        if self.languages:
            payload["languages"] = self.languages
        if self.language_patterns:
            payload["language_patterns"] = self.language_patterns
        if self.extra:
            payload["extra"] = self.extra
        return payload


@dataclass(slots=True)
class PipelineConfig:
    storage: StorageConfig
    datasets: list[DatasetSpec]
    checkpoint_interval_records: int = 1_000

    @classmethod
    def from_path(cls, path: Path) -> "PipelineConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(raw, base_dir=path.parent)

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        base_dir: Path,
    ) -> "PipelineConfig":
        datasets = [DatasetSpec.from_dict(item) for item in data.get("datasets", [])]
        unsupported_sources = sorted(
            {dataset.source for dataset in datasets if dataset.source != "huggingface_hub"}
        )
        if unsupported_sources:
            joined = ", ".join(unsupported_sources)
            raise ValueError(
                "CommentMiner supports Hugging Face dataset sources only; "
                f"unsupported source values: {joined}"
            )
        checkpoint_interval_records = _require_positive_int(
            "checkpoint_interval_records",
            int(data.get("checkpoint_interval_records", 1_000)),
        )
        return cls(
            storage=StorageConfig.from_dict(data["storage"], base_dir),
            datasets=datasets,
            checkpoint_interval_records=checkpoint_interval_records,
        )

    def get_dataset(self, name: str) -> DatasetSpec | None:
        for dataset in self.datasets:
            if dataset.name == name:
                return dataset
        return None

    def require_dataset(self, name: str) -> DatasetSpec:
        dataset = self.get_dataset(name)
        if dataset is None:
            raise KeyError(f"Unknown dataset '{name}'")
        return dataset

    def ensure_directories(self) -> None:
        self.storage.working_directory.mkdir(parents=True, exist_ok=True)
        self.storage.output_directory.mkdir(parents=True, exist_ok=True)
        self.storage.checkpoint_directory.mkdir(parents=True, exist_ok=True)
        self.storage.download_directory.mkdir(parents=True, exist_ok=True)
        self.storage.huggingface_cache_directory.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "storage": self.storage.to_dict(),
            "datasets": [dataset.to_dict() for dataset in self.datasets],
            "checkpoint_interval_records": self.checkpoint_interval_records,
        }
