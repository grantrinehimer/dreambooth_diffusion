"""Centralised logging helpers (plays nicely with Accelerate)."""
from __future__ import annotations

import logging
from accelerate.logging import get_logger

_FMT = "%(asctime)s | %(levelname)-8s | %(name)s: %(message)s"
_DATEFMT = "%m/%d %H:%M:%S"


def configure(name: str = "dreambooth") -> logging.Logger:
    """Return a configured logger – silent on non‑main ranks by default."""
    logger = get_logger(name)

    # Only add handler once per rank
    if not logger.hasHandlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FMT, _DATEFMT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger