from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import csv
import gc
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Callable, Iterable, Sequence

import pyarrow.parquet as pq

from .models import _json_safe


MIN_ENCODER_PARAMETERS = 2_000_000
MAX_ENCODER_PARAMETERS = 8_000_000_000
_DEFAULT_TEXT_FIELD = "opening_comment"


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


@dataclass(frozen=True, slots=True)
class EncodingModelSpec:
    model_id: str
    parameter_count: int | None = None
    revision: str | None = None


@dataclass(slots=True)
class EncodingCapacityStats:
    input_directory: Path
    output_directory: Path
    text_field: str
    input_format: str
    records_seen: int = 0
    records_without_text: int = 0
    samples_loaded: int = 0
    shards_read: int = 0
    models_benchmarked: int = 0
    report_path: Path | None = None
    summary_path: Path | None = None


@dataclass(slots=True)
class _LoadedTextSamples:
    texts: list[str]
    input_format: str
    records_seen: int = 0
    records_without_text: int = 0
    shards_read: int = 0


EncoderLoader = Callable[[EncodingModelSpec, str | None, Path | None, bool], Any]


def parse_encoding_model_spec(value: str) -> EncodingModelSpec:
    """Parse MODEL_ID or MODEL_ID=PARAMETERS for the benchmark CLI."""

    raw = value.strip()
    if not raw:
        raise ValueError("model spec cannot be empty")
    if "=" not in raw:
        return EncodingModelSpec(model_id=raw)

    model_id, parameter_value = raw.rsplit("=", 1)
    model_id = model_id.strip()
    if not model_id:
        raise ValueError(f"model spec has an empty model id: {value!r}")
    parameter_count = parse_parameter_count(parameter_value)
    _validate_parameter_count(parameter_count, model_id=model_id)
    return EncodingModelSpec(model_id=model_id, parameter_count=parameter_count)


def parse_parameter_count(value: object) -> int:
    """Parse counts like 22000000, 22M, 0.6B, or 2_000_000."""

    if isinstance(value, bool):
        raise ValueError(f"Invalid parameter count: {value!r}")
    if isinstance(value, int):
        if value < 1:
            raise ValueError(f"Parameter count must be positive, got {value}")
        return value
    if isinstance(value, float):
        if not math.isfinite(value) or value < 1:
            raise ValueError(f"Parameter count must be positive, got {value}")
        return int(value)

    cleaned = str(value).strip().replace(",", "").replace("_", "")
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([kKmMbB]?)", cleaned)
    if not match:
        raise ValueError(f"Invalid parameter count: {value!r}")
    number = float(match.group(1))
    suffix = match.group(2).lower()
    multiplier = {"": 1, "k": 1_000, "m": 1_000_000, "b": 1_000_000_000}[suffix]
    result = int(number * multiplier)
    if result < 1:
        raise ValueError(f"Parameter count must be positive, got {value!r}")
    return result


def load_encoding_model_specs(
    model_values: Sequence[str] | None = None,
    *,
    model_config: Path | None = None,
) -> list[EncodingModelSpec]:
    specs: list[EncodingModelSpec] = []
    if model_config is not None:
        payload = json.loads(model_config.read_text(encoding="utf-8"))
        raw_models = payload.get("models") if isinstance(payload, dict) else payload
        if not isinstance(raw_models, list):
            raise ValueError("model config must be a JSON list or an object with a 'models' list")
        specs.extend(_model_specs_from_config(raw_models))

    for value in model_values or []:
        specs.append(parse_encoding_model_spec(value))

    if not specs:
        raise ValueError("At least one --model or --model-config entry is required")
    return specs


def run_encoding_capacity_benchmark(
    input_directory: Path,
    model_specs: Sequence[EncodingModelSpec],
    *,
    output_directory: Path | None = None,
    text_field: str = _DEFAULT_TEXT_FIELD,
    dataset_names: Iterable[str] | None = None,
    languages: Iterable[str] | None = None,
    max_shards: int | None = None,
    max_samples: int | None = None,
    sample_counts: Sequence[int] | None = None,
    initial_samples: int = 128,
    sample_growth: float = 2.0,
    batch_size: int = 32,
    device: str | None = None,
    cache_folder: Path | None = None,
    trust_remote_code: bool = False,
    normalize_embeddings: bool = False,
    encoder_loader: EncoderLoader | None = None,
    overwrite: bool = False,
) -> EncodingCapacityStats:
    if not model_specs:
        raise ValueError("At least one model spec is required")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if initial_samples < 1:
        raise ValueError(f"initial_samples must be >= 1, got {initial_samples}")
    if sample_growth <= 1.0:
        raise ValueError(f"sample_growth must be > 1.0, got {sample_growth}")
    if max_shards is not None and max_shards < 1:
        raise ValueError(f"max_shards must be >= 1, got {max_shards}")
    if max_samples is not None and max_samples < 1:
        raise ValueError(f"max_samples must be >= 1, got {max_samples}")
    if sample_counts is not None and any(count < 1 for count in sample_counts):
        raise ValueError("sample_counts must all be >= 1")
    dataset_names_list = list(dataset_names or [])
    languages_list = list(languages or [])

    input_directory = input_directory.resolve()
    if not input_directory.exists() or not input_directory.is_dir():
        raise ValueError(f"Input comment directory does not exist: {input_directory}")
    for spec in model_specs:
        if spec.parameter_count is not None:
            _validate_parameter_count(spec.parameter_count, model_id=spec.model_id)

    output_directory = (
        output_directory
        or input_directory.parent / f"{input_directory.name}-encoding-benchmark"
    ).resolve()
    if output_directory == input_directory:
        raise ValueError("Output directory must differ from the input directory")
    if overwrite and output_directory.exists():
        import shutil

        shutil.rmtree(output_directory)
    if output_directory.exists() and any(output_directory.iterdir()):
        raise ValueError(
            f"Output directory '{output_directory}' already exists and is not empty. "
            "Use overwrite=True or choose another output directory."
        )
    output_directory.mkdir(parents=True, exist_ok=True)

    loaded = _load_text_samples(
        input_directory,
        text_field=text_field,
        dataset_names=dataset_names_list,
        languages=languages_list,
        max_shards=max_shards,
        max_samples=max_samples,
    )
    counts = _build_sample_counts(
        samples_loaded=len(loaded.texts),
        sample_counts=sample_counts,
        initial_samples=initial_samples,
        sample_growth=sample_growth,
    )

    stats = EncodingCapacityStats(
        input_directory=input_directory,
        output_directory=output_directory,
        text_field=text_field,
        input_format=loaded.input_format,
        records_seen=loaded.records_seen,
        records_without_text=loaded.records_without_text,
        samples_loaded=len(loaded.texts),
        shards_read=loaded.shards_read,
        models_benchmarked=len(model_specs),
        report_path=output_directory / "encoding-capacity-report.json",
        summary_path=output_directory / "encoding-capacity-summary.csv",
    )

    loader = encoder_loader or _load_sentence_transformer
    model_results = [
        _benchmark_model(
            spec,
            loaded.texts,
            counts,
            batch_size=batch_size,
            device=device,
            cache_folder=cache_folder,
            trust_remote_code=trust_remote_code,
            normalize_embeddings=normalize_embeddings,
            encoder_loader=loader,
        )
        for spec in model_specs
    ]
    report = {
        "created_at": _utc_now(),
        "input_directory": str(input_directory),
        "output_directory": str(output_directory),
        "input_format": loaded.input_format,
        "text_field": text_field,
        "records_seen": loaded.records_seen,
        "records_without_text": loaded.records_without_text,
        "samples_loaded": len(loaded.texts),
        "shards_read": loaded.shards_read,
        "dataset_filter": sorted(dataset_names_list),
        "language_filter": sorted(languages_list),
        "parameter_range": {
            "min": MIN_ENCODER_PARAMETERS,
            "max": MAX_ENCODER_PARAMETERS,
            "label": "2M-8B",
        },
        "sample_counts": counts,
        "batch_size": batch_size,
        "device": device,
        "cache_folder": str(cache_folder) if cache_folder else None,
        "trust_remote_code": trust_remote_code,
        "normalize_embeddings": normalize_embeddings,
        "requested_sample_counts": list(sample_counts) if sample_counts is not None else None,
        "models": model_results,
    }
    assert stats.report_path is not None
    assert stats.summary_path is not None
    stats.report_path.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")
    _write_summary_csv(stats.summary_path, model_results)
    return stats


def _model_specs_from_config(raw_models: Sequence[object]) -> list[EncodingModelSpec]:
    specs: list[EncodingModelSpec] = []
    for item in raw_models:
        if isinstance(item, str):
            specs.append(parse_encoding_model_spec(item))
            continue
        if not isinstance(item, dict):
            raise ValueError(f"Model config entries must be strings or objects, got {type(item).__name__}")

        model_id = item.get("model_id") or item.get("id") or item.get("model")
        if not model_id or not str(model_id).strip():
            raise ValueError(f"Model config entry is missing model_id: {item!r}")
        parameter_value = (
            item.get("parameters")
            if "parameters" in item
            else item.get("parameter_count")
            if "parameter_count" in item
            else item.get("params")
        )
        parameter_count = None
        if parameter_value is not None:
            parameter_count = parse_parameter_count(parameter_value)
            _validate_parameter_count(parameter_count, model_id=str(model_id))
        revision = item.get("revision")
        specs.append(
            EncodingModelSpec(
                model_id=str(model_id).strip(),
                parameter_count=parameter_count,
                revision=str(revision).strip() if revision else None,
            )
        )
    return specs


def _validate_parameter_count(parameter_count: int, *, model_id: str) -> None:
    if parameter_count < MIN_ENCODER_PARAMETERS or parameter_count > MAX_ENCODER_PARAMETERS:
        raise ValueError(
            f"Model '{model_id}' has {parameter_count:,} parameters, outside the supported "
            "2M-8B range"
        )


def _load_sentence_transformer(
    spec: EncodingModelSpec,
    device: str | None,
    cache_folder: Path | None,
    trust_remote_code: bool,
) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is not installed. Run `uv sync` before benchmarking encoders."
        ) from exc

    kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if device is not None:
        kwargs["device"] = device
    if cache_folder is not None:
        kwargs["cache_folder"] = str(cache_folder)
    if spec.revision is not None:
        kwargs["revision"] = spec.revision
    return SentenceTransformer(spec.model_id, **kwargs)


def _benchmark_model(
    spec: EncodingModelSpec,
    texts: Sequence[str],
    sample_counts: Sequence[int],
    *,
    batch_size: int,
    device: str | None,
    cache_folder: Path | None,
    trust_remote_code: bool,
    normalize_embeddings: bool,
    encoder_loader: EncoderLoader,
) -> dict[str, Any]:
    model_payload: dict[str, Any] = {
        "model_id": spec.model_id,
        "declared_parameter_count": spec.parameter_count,
        "revision": spec.revision,
        "inferred_parameter_count": None,
        "effective_parameter_count": spec.parameter_count,
        "largest_successful_sample_count": 0,
        "load_error": None,
        "steps": [],
    }
    model = None
    try:
        model = encoder_loader(spec, device, cache_folder, trust_remote_code)
        inferred_parameter_count = _count_model_parameters(model)
        if inferred_parameter_count is not None:
            _validate_parameter_count(inferred_parameter_count, model_id=spec.model_id)
            model_payload["inferred_parameter_count"] = inferred_parameter_count
            model_payload["effective_parameter_count"] = inferred_parameter_count
        elif spec.parameter_count is None:
            raise RuntimeError(
                f"Model '{spec.model_id}' did not provide a countable parameter set. "
                "Use MODEL_ID=PARAMETERS or a model config entry with parameters."
            )
    except Exception as exc:
        model_payload["load_error"] = _exception_payload(exc)
        if model is not None:
            del model
        _clear_accelerator_cache()
        return model_payload

    try:
        for sample_count in sample_counts:
            step = _benchmark_sample_count(
                model,
                texts[:sample_count],
                sample_count=sample_count,
                batch_size=batch_size,
                normalize_embeddings=normalize_embeddings,
            )
            model_payload["steps"].append(step)
            if not step["success"]:
                break
            model_payload["largest_successful_sample_count"] = sample_count
    finally:
        del model
        _clear_accelerator_cache()
    return model_payload


def _benchmark_sample_count(
    model: Any,
    texts: Sequence[str],
    *,
    sample_count: int,
    batch_size: int,
    normalize_embeddings: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        embeddings = model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings,
        )
        elapsed = time.perf_counter() - started
        shape = _embedding_shape(embeddings)
        nbytes = _embedding_nbytes(embeddings)
        del embeddings
        _clear_accelerator_cache()
        return {
            "sample_count": sample_count,
            "batch_size": batch_size,
            "success": True,
            "seconds": elapsed,
            "samples_per_second": sample_count / elapsed if elapsed > 0 else None,
            "embedding_shape": shape,
            "embedding_bytes": nbytes,
            "error": None,
        }
    except Exception as exc:
        elapsed = time.perf_counter() - started
        _clear_accelerator_cache()
        return {
            "sample_count": sample_count,
            "batch_size": batch_size,
            "success": False,
            "seconds": elapsed,
            "samples_per_second": None,
            "embedding_shape": None,
            "embedding_bytes": None,
            "error": _exception_payload(exc),
        }


def _load_text_samples(
    input_directory: Path,
    *,
    text_field: str,
    dataset_names: Iterable[str] | None,
    languages: Iterable[str] | None,
    max_shards: int | None,
    max_samples: int | None,
) -> _LoadedTextSamples:
    dataset_filter = set(dataset_names or [])
    language_filter = set(languages or [])
    jsonl_shards = sorted(
        set(input_directory.glob("part-*.jsonl"))
        | set(input_directory.glob("*/*/part-*.jsonl"))
    )
    parquet_shards = sorted(
        set(input_directory.glob("part-*.parquet"))
        | set(input_directory.glob("*/*/part-*.parquet"))
    )
    if max_shards is not None:
        jsonl_shards = jsonl_shards[:max_shards]
        parquet_shards = parquet_shards[:max_shards]
    if not jsonl_shards and not parquet_shards:
        raise ValueError(f"No JSONL or Parquet comment shards found in: {input_directory}")

    loaded = _LoadedTextSamples(
        texts=[],
        input_format=(
            "mixed"
            if jsonl_shards and parquet_shards
            else "jsonl"
            if jsonl_shards
            else "parquet"
        ),
    )
    for shard in jsonl_shards:
        _load_jsonl_texts(
            shard,
            loaded,
            text_field=text_field,
            dataset_filter=dataset_filter,
            language_filter=language_filter,
            max_samples=max_samples,
        )
        if max_samples is not None and len(loaded.texts) >= max_samples:
            return loaded

    for shard in parquet_shards:
        _load_parquet_texts(
            shard,
            loaded,
            text_field=text_field,
            dataset_filter=dataset_filter,
            language_filter=language_filter,
            max_samples=max_samples,
        )
        if max_samples is not None and len(loaded.texts) >= max_samples:
            return loaded
    return loaded


def _load_jsonl_texts(
    shard: Path,
    loaded: _LoadedTextSamples,
    *,
    text_field: str,
    dataset_filter: set[str],
    language_filter: set[str],
    max_samples: int | None,
) -> None:
    with shard.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            _maybe_add_text(
                payload,
                loaded,
                text_field=text_field,
                dataset_filter=dataset_filter,
                language_filter=language_filter,
                max_samples=max_samples,
            )
            if max_samples is not None and len(loaded.texts) >= max_samples:
                break
    loaded.shards_read += 1


def _load_parquet_texts(
    shard: Path,
    loaded: _LoadedTextSamples,
    *,
    text_field: str,
    dataset_filter: set[str],
    language_filter: set[str],
    max_samples: int | None,
) -> None:
    parquet_file = pq.ParquetFile(shard)
    schema_names = set(parquet_file.schema_arrow.names)
    if text_field not in schema_names:
        raise ValueError(f"Text field '{text_field}' not found in Parquet shard: {shard}")
    columns = [text_field]
    for optional in ("dataset", "language"):
        if optional in schema_names:
            columns.append(optional)
    for batch in parquet_file.iter_batches(columns=columns):
        for payload in batch.to_pylist():
            _maybe_add_text(
                payload,
                loaded,
                text_field=text_field,
                dataset_filter=dataset_filter,
                language_filter=language_filter,
                max_samples=max_samples,
            )
            if max_samples is not None and len(loaded.texts) >= max_samples:
                break
        if max_samples is not None and len(loaded.texts) >= max_samples:
            break
    loaded.shards_read += 1


def _maybe_add_text(
    payload: dict[str, Any],
    loaded: _LoadedTextSamples,
    *,
    text_field: str,
    dataset_filter: set[str],
    language_filter: set[str],
    max_samples: int | None,
) -> None:
    if max_samples is not None and len(loaded.texts) >= max_samples:
        return
    if dataset_filter and str(payload.get("dataset") or "") not in dataset_filter:
        return
    if language_filter and str(payload.get("language") or "") not in language_filter:
        return

    loaded.records_seen += 1
    text = payload.get(text_field)
    if text is None or not str(text).strip():
        loaded.records_without_text += 1
        return
    loaded.texts.append(str(text))


def _build_sample_counts(
    *,
    samples_loaded: int,
    sample_counts: Sequence[int] | None,
    initial_samples: int,
    sample_growth: float,
) -> list[int]:
    if samples_loaded < 1:
        return []
    if sample_counts is not None:
        counts = sorted({count for count in sample_counts if count <= samples_loaded})
        return counts or [samples_loaded]

    counts: list[int] = []
    current = min(initial_samples, samples_loaded)
    while current < samples_loaded:
        counts.append(current)
        current = max(current + 1, int(math.ceil(current * sample_growth)))
    counts.append(samples_loaded)
    return counts


def _count_model_parameters(model: Any) -> int | None:
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return None

    total = 0
    try:
        iterator = parameters()
        for parameter in iterator:
            numel = getattr(parameter, "numel", None)
            if callable(numel):
                total += int(numel())
            else:
                total += int(getattr(parameter, "size", 0))
    except Exception:
        return None
    return total if total > 0 else None


def _embedding_shape(embeddings: Any) -> list[int] | None:
    shape = getattr(embeddings, "shape", None)
    if shape is not None:
        return [int(item) for item in shape]
    try:
        first = embeddings[0] if embeddings else []
        return [len(embeddings), len(first)]
    except Exception:
        return None


def _embedding_nbytes(embeddings: Any) -> int | None:
    nbytes = getattr(embeddings, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    try:
        return len(json.dumps(_json_safe(embeddings)).encode("utf-8"))
    except Exception:
        return None


def _exception_payload(exc: Exception) -> dict[str, str]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
    }


def _clear_accelerator_cache() -> None:
    gc.collect()
    try:
        import torch
    except Exception:
        return

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            ipc_collect = getattr(torch.cuda, "ipc_collect", None)
            if callable(ipc_collect):
                ipc_collect()
    except Exception:
        pass
    mps = getattr(torch, "mps", None)
    empty_cache = getattr(mps, "empty_cache", None)
    if callable(empty_cache):
        try:
            empty_cache()
        except Exception:
            pass


def _write_summary_csv(path: Path, model_results: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model_id",
                "declared_parameter_count",
                "inferred_parameter_count",
                "effective_parameter_count",
                "largest_successful_sample_count",
                "sample_count",
                "success",
                "seconds",
                "samples_per_second",
                "embedding_shape",
                "embedding_bytes",
                "error_type",
                "error_message",
            ],
        )
        writer.writeheader()
        for model in model_results:
            steps = model.get("steps") or [None]
            for step in steps:
                error = (step or {}).get("error") or model.get("load_error") or {}
                writer.writerow(
                    {
                        "model_id": model.get("model_id"),
                        "declared_parameter_count": model.get("declared_parameter_count"),
                        "inferred_parameter_count": model.get("inferred_parameter_count"),
                        "effective_parameter_count": model.get("effective_parameter_count"),
                        "largest_successful_sample_count": model.get("largest_successful_sample_count"),
                        "sample_count": (step or {}).get("sample_count"),
                        "success": (step or {}).get("success", False),
                        "seconds": (step or {}).get("seconds"),
                        "samples_per_second": (step or {}).get("samples_per_second"),
                        "embedding_shape": json.dumps((step or {}).get("embedding_shape")),
                        "embedding_bytes": (step or {}).get("embedding_bytes"),
                        "error_type": error.get("type"),
                        "error_message": error.get("message"),
                    }
                )
