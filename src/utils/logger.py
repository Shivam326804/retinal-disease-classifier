"""
Logger Configuration Module
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional

from .config import Config


def setup_logger(name: str = "retinal_classifier", level: Optional[str] = None) -> logging.Logger:

    if level is None:
        level = Config.LOG_LEVEL

    log_level = getattr(logging, level.upper(), logging.INFO)

    logs_dir = Path(Config.LOGS_DIR)
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)

    if logger.hasHandlers():
        return logger

    logger.setLevel(log_level)
    logger.propagate = False

    detailed_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
    )

    console_formatter = logging.Formatter("%(levelname)s: %(message)s")

    # File handler
    file_handler = logging.handlers.RotatingFileHandler(
        logs_dir / f"{name}.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8"
    )

    file_handler.setFormatter(detailed_formatter)
    file_handler.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(log_level)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Global logger
app_logger = setup_logger()