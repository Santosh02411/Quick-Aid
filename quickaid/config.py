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


# ---- Sessions and sign-in protection ----
def session_idle_minutes() -> int:
    """Sign people out after this long without activity (sliding)."""
    return max(5, _int_env('SESSION_IDLE_MINUTES', 60))


def session_absolute_hours() -> int:
    """Sign people out this long after they signed in, however active they are."""
    return max(1, _int_env('SESSION_ABSOLUTE_HOURS', 12))


LOCKOUT_WINDOW_SECONDS = 15 * 60
LOGIN_MAX_FAILURES_PER_ADDRESS = 5          # wrong passwords for one account from ONE address
LOGIN_MAX_FAILURES_PER_ACCOUNT = 25         # ...from anywhere (higher, so a stranger cannot easily lock the real owner out)
TWOFA_MAX_FAILURES = 5
RESET_EMAILS_PER_HOUR = 3
PENDING_2FA_SECONDS = 10 * 60               # after the password step, how long the 2FA step stays open
MAX_PASSWORD_LENGTH = 128                   # the hash is deliberately slow; refuse absurdly long input instead of burning CPU on it

# ---- External AI consent ----
AI_CONSENT_VERSION = 1                       # bump when the wording of what is sent to the AI provider changes: everyone is asked again


def require_ai_consent() -> bool:
    """When true (the default) nothing is sent to the external AI provider until the person has agreed; they get the built-in basic mode."""
    return os.getenv('REQUIRE_AI_CONSENT', 'true').strip().lower() not in ('0', 'false', 'no', 'off')


SUPPORTED_LANGUAGES = ('en', 'hi', 'kn', 'mr')
