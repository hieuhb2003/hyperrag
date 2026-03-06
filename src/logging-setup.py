"""Logging configuration for HyP-DLM. Console + file handlers."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from datetime import datetime

from src.config import Config


_INITIALIZED = False


def setup_logging(config: Config) -> None:
    """Configure root logger with console + file handler."""
    global _INITIALIZED
    if _INITIALIZED:
        return
    _INITIALIZED = True

    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # File handler with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(log_dir / f"hypdlm_{timestamp}.log")
    file_handler.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, config.log_level.upper(), logging.INFO))

    # Format
    fmt = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Root logger
    root = logging.getLogger("hypdlm")
    root.setLevel(logging.DEBUG)
    root.addHandler(file_handler)
    root.addHandler(console_handler)


def get_logger(name: str) -> logging.Logger:
    """Get named logger for a module. Usage: logger = get_logger('chunker')"""
    return logging.getLogger(f"hypdlm.{name}")
