"""
logger.py
Centralized logging setup. Import get_logger() everywhere instead of using print().
"""

import logging
import sys
from pathlib import Path
from app.config import LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT, LOG_DIR


def setup_logging():
    """Configure root logger — call once at app startup."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "nfl_predictor.log", encoding="utf-8"),
    ]

    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=handlers,
    )

    # Quiet noisy third-party loggers
    for noisy in ["urllib3", "requests", "apscheduler.executors"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger. Usage: logger = get_logger(__name__)"""
    return logging.getLogger(name)
