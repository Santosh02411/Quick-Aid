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
import re
import os
from logging.handlers import RotatingFileHandler

from quickaid import PROJECT_ROOT

LOG_DIR = os.getenv('LOG_DIR', str(PROJECT_ROOT / 'logs'))
LOG_FILE = os.path.join(LOG_DIR, 'app.log')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()

_configured = False

# Defence in depth. The code is written NOT to log personal data, but a stray exception message or a third-party library can still carry an
# email address, a bearer/reset token or a full IP address into a log file that is kept for weeks and read by many people. This redacts them.
_EMAIL = re.compile(r'[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}')
_URL_TOKEN = re.compile(r'(/(?:reset-password|verify-email)/)[A-Za-z0-9_\-]{16,}')
_LONG_SECRET = re.compile(r'(?i)\b(token|secret|password|passwd|api[_-]?key|authorization|bearer)\b(\s*[=:]\s*|\s+)([^\s,;"\']{6,})')
_GOOGLE_KEY = re.compile(r'AIza[0-9A-Za-z_\-]{35}')
_IPV4 = re.compile(r'\b(\d{1,3}\.\d{1,3}\.\d{1,3})\.\d{1,3}\b')


def redact(text: str) -> str:
    text = _EMAIL.sub('[email]', text)
    text = _URL_TOKEN.sub(r'\1[token]', text)
    text = _GOOGLE_KEY.sub('[api-key]', text)
    text = _LONG_SECRET.sub(lambda m: f'{m.group(1)}{m.group(2)}[redacted]', text)
    return _IPV4.sub(r'\1.0', text)


class PrivacyFilter(logging.Filter):
    """Rewrites each record so emails, tokens, keys and full IPv4 addresses never reach a log file."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = redact(record.getMessage())
            record.args = ()
            if record.exc_info:
                record.exc_text = redact(logging.Formatter().formatException(record.exc_info))
                record.exc_info = None
        except Exception:        # logging must never break the request
            pass
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line, for log aggregators (LOG_FORMAT=json)."""

    def format(self, record: logging.LogRecord) -> str:
        import json
        body = {'ts': self.formatTime(record, '%Y-%m-%dT%H:%M:%S'), 'level': record.levelname, 'logger': record.name, 'message': record.getMessage()}
        if record.exc_text:
            body['exception'] = record.exc_text
        return json.dumps(body, ensure_ascii=False)


def _configure_root_logger():
    global _configured
    if _configured:
        return
    _configured = True

    os.makedirs(LOG_DIR, exist_ok=True)

    if os.getenv('LOG_FORMAT', '').lower() == 'json':
        formatter = JsonFormatter()
    else:
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

    for handler in (console_handler, file_handler):
        handler.addFilter(PrivacyFilter())

    root_logger = logging.getLogger('quickaid')
    root_logger.setLevel(LOG_LEVEL)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Get a logger under the shared 'quickaid' namespace, configured once."""
    _configure_root_logger()
    return logging.getLogger(f'quickaid.{name}')
