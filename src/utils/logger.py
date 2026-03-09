"""
Logger Configuration Module
Sets up logging for the entire project
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional

from .config import Config


# ---------------------------------------------------
# LOGGER SETUP FUNCTION
# ---------------------------------------------------

def setup_logger(name: str = "retinal_classifier", level: Optional[str] = None) -> logging.Logger:
    """
    Setup project logger

    Args:
        name: logger name
        level: logging level

    Returns:
        configured logger
    """

    if level is None:
        level = Config.LOG_LEVEL

    log_level = getattr(logging, level.upper(), logging.INFO)

    # Ensure log directory exists
    logs_dir = Path(Config.LOGS_DIR)
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)

    # Prevent duplicate handlers (important for Streamlit)
    if logger.hasHandlers():
        return logger

    logger.setLevel(log_level)
    logger.propagate = False

    # ---------------------------------------------------
    # FORMATTERS
    # ---------------------------------------------------

    detailed_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_formatter = logging.Formatter(
        "%(levelname)s: %(message)s"
    )

    # ---------------------------------------------------
    # FILE HANDLER (ROTATING)
    # ---------------------------------------------------

    log_file = logs_dir / f"{name.replace('.', '_')}.log"

    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8"
    )

    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)

    logger.addHandler(file_handler)

    # ---------------------------------------------------
    # CONSOLE HANDLER
    # ---------------------------------------------------

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(console_handler)

    return logger


# ---------------------------------------------------
# GLOBAL LOGGER
# ---------------------------------------------------

app_logger = setup_logger("retinal_classifier")