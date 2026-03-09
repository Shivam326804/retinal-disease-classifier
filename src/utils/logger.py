"""
Logger Configuration Module
Sets up logging for the entire project
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional

from .config import Config


def setup_logger(name: str = __name__, level: Optional[str] = None) -> logging.Logger:
    """
    Setup logger with both file and console handlers

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance
    """

    # Use config level if none provided
    if level is None:
        level = Config.LOG_LEVEL

    # Ensure logs directory exists
    logs_dir: Path = Path(Config.LOGS_DIR)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger(name)

    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # ---------------- FORMATTERS ----------------

    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    simple_formatter = logging.Formatter("%(levelname)s: %(message)s")

    # ---------------- FILE HANDLER ----------------

    log_file = logs_dir / f"{name.replace('.', '_')}.log"

    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
    )

    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)

    logger.addHandler(file_handler)

    # ---------------- CONSOLE HANDLER ----------------

    console_handler = logging.StreamHandler()

    console_handler.setLevel(log_level)
    console_handler.setFormatter(simple_formatter)

    logger.addHandler(console_handler)

    return logger


# Global application logger
app_logger = setup_logger("retinal_classifier")