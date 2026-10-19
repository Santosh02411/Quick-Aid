"""Application constants and environment-driven policy. Values read from the environment are functions, so they follow
the environment at call time (tests and operators can change them without re-importing anything)."""

import os
import re

# Input limits
MAX_SYMPTOM_LENGTH = 1000
MIN_SYMPTOM_LENGTH = 3
MAX_CHAT_MESSAGE_LENGTH = 500
MAX_NOTES_LENGTH = 2000
MAX_UPLOAD_BYTES = 16 * 1024 * 1024

# Accounts
USERNAME_RE = re.compile(r'^[A-Za-z0-9_]{3,30}$')
MIN_PASSWORD_LENGTH = 8
PASSWORD_RESET_TTL_SECONDS = 60 * 60            # 1 hour
EMAIL_VERIFY_TTL_SECONDS = 60 * 60 * 24         # 24 hours

AMBULANCE_NUMBERS = {'IN': '108', 'US': '911', 'CA': '911', 'UK': '999', 'EU': '112', 'AU': '000'}

# Lists
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, '') or default)
    except ValueError:
        return default


def app_base_url() -> str:
    return os.getenv('APP_BASE_URL', '').rstrip('/')


# ---- Data retention policy (see the privacy page and README for the plain-language version) ----
def default_retention_days() -> int:
    """How long saved assessments and chats are kept for people who have not chosen. 0 = until the person deletes them."""
    return _int_env('DEFAULT_RETENTION_DAYS', 365)


def max_retention_days() -> int:
    """An operator-set ceiling nobody can exceed (0 = no ceiling)."""
    return _int_env('HISTORY_MAX_RETENTION_DAYS', 0)


def upload_max_age_minutes() -> int:
    """Any photo file still on disk after this long is a leftover from a crash and is deleted. Photos are normally deleted at once."""
    return _int_env('UPLOAD_MAX_AGE_MINUTES', 30)


def maintenance_interval_hours() -> int:
    return max(1, _int_env('MAINTENANCE_INTERVAL_HOURS', 24))


def interview_ttl_hours() -> int:
    return 24
