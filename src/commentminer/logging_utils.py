from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> None:
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Unsupported log level: {level}")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )

    dependency_level = logging.DEBUG if numeric_level <= logging.DEBUG else logging.WARNING
    for logger_name in ["httpx", "httpcore", "huggingface_hub"]:
        logging.getLogger(logger_name).setLevel(dependency_level)
