"""
Central logging setup for Quick Aid.

Every module that wants to log calls get_logger(__name__) instead of
configuring its own handlers, so logs from app.py, medical_analyzer.py,
symptom_checker.py, and database.py all end up in one place with a
consistent format - console output for local dev, plus a rotating file
under logs/ so failures are still visible after a production process
restarts or a terminal session is gone.
"""

import logging
import os
from logging.handlers import RotatingFileHandler

LOG_DIR = os.getenv('LOG_DIR', os.path.join(os.path.dirname(__file__), 'logs'))
LOG_FILE = os.path.join(LOG_DIR, 'app.log')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()

_configured = False


def _configure_root_logger():
    global _configured
    if _configured:
        return
    _configured = True

    os.makedirs(LOG_DIR, exist_ok=True)

    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # 5 MB per file, keep 5 old files - enough history for debugging a
    # production incident without growing disk usage unbounded.
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5, encoding='utf-8'
    )
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger('quickaid')
    root_logger.setLevel(LOG_LEVEL)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Get a logger under the shared 'quickaid' namespace, configured once."""
    _configure_root_logger()
    return logging.getLogger(f'quickaid.{name}')
