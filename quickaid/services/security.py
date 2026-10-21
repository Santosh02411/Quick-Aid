"""Sign-in protection that needs no HTTP: password policy, lockout, security events, notifications.

Request-bound pieces (client network, session start/expiry) are in `quickaid/web/sessions.py`.
"""

import hashlib
import re
from typing import Optional

from quickaid import config
from quickaid.accounts.mailer import send_email
from quickaid.core import database as db
from quickaid.core.logging_config import get_logger

logger = get_logger('security')

# The most commonly breached passwords (lower-case). A policy of "8+ characters" alone happily accepts "password123".
COMMON_PASSWORDS = frozenset("""
password password1 password12 password123 password1234 passw0rd p@ssw0rd p@ssword pass1234 qwerty qwerty1 qwerty12 qwerty123 qwertyuiop qwerty1234
1q2w3e4r 1q2w3e4r5t 1qaz2wsx qazwsxedc zaq12wsx asdfgh asdfghjk asdfghjkl zxcvbnm abc123 abc12345 abcd1234 abcdefg1 letmein letmein1 letmein123
welcome welcome1 welcome123 admin admin123 admin1234 administrator root toor changeme changeme123 iloveyou iloveyou1 iloveyou123 sunshine monkey
monkey123 dragon dragon123 football football1 baseball master master123 shadow shadow123 superman batman trustno1 starwars hello123 hello1234
freedom whatever whatever1 whatever123 cheese123 secret secret123 test1234 test12345 testing123 default guest guest123 login login123 user1234 qwe123
qwerty12345 11111111 12345678 123456789 1234567890 123123123 121212121 987654321 0987654321 112233445 147258369 159753456 india123 india@123
bharat123 krishna123 ganesh123 sai12345 hanuman123 jaishreeram kannada123 hindi123 marathi123 admin@123 welcome@123 password@123 pass@123 test@123
""".split())
_SEQUENCES = ('0123456789', '9876543210', 'abcdefghijklmnopqrstuvwxyz', 'zyxwvutsrqponmlkjihgfedcba', 'qwertyuiopasdfghjklzxcvbnm')


def fingerprint(value: str, length: int = 10) -> str:
    """A short one-way tag, so logs and the attempts table can correlate without holding the value itself."""
    return hashlib.sha256(('quickaid-fp:' + (value or '').strip().lower()).encode('utf-8')).hexdigest()[:length]


def check_password(password: str, username: str = '', email: str = '') -> Optional[str]:
    """None when the password is acceptable, else a message saying what to change."""
    if len(password) < config.MIN_PASSWORD_LENGTH:
        return f'Password must be at least {config.MIN_PASSWORD_LENGTH} characters.'
    if len(password) > config.MAX_PASSWORD_LENGTH:
        return f'Password must be at most {config.MAX_PASSWORD_LENGTH} characters.'
    low = password.lower()
    stripped = re.sub(r'[\d!@#$%^&*_.\-]+$', '', low)
    if low in COMMON_PASSWORDS or (stripped and stripped in COMMON_PASSWORDS):
        return 'That password is on the list of the most commonly used (and most guessed) passwords. Please choose a different one.'
    if len(set(low)) <= 2:
        return 'Password cannot be made of one or two repeated characters.'
    if any(low in seq for seq in _SEQUENCES):
        return 'Password cannot be a simple sequence such as 12345678 or abcdefgh.'
    if len(username or '') >= 4 and username.lower() in low:
        return 'Password cannot contain your username.'
    local = (email or '').split('@')[0].lower()
    if len(local) >= 4 and local in low:
        return 'Password cannot contain your email name.'
    return None


# ---- lockout ----------------------------------------------------------------------------------------------------------------

def _pair(username: str, network: str) -> str:
    return fingerprint(f'{username}|{network}')


def login_locked(username: str, network: str) -> bool:
    """Too many recent failures for this account from this address, or for this account from anywhere."""
    w = config.LOCKOUT_WINDOW_SECONDS
    return (db.count_attempts('login', _pair(username, network), w) >= config.LOGIN_MAX_FAILURES_PER_ADDRESS
            or db.count_attempts('login_acct', fingerprint(username), w) >= config.LOGIN_MAX_FAILURES_PER_ACCOUNT)


def note_login_failure(username: str, network: str) -> None:
    db.record_attempt('login', _pair(username, network))
    db.record_attempt('login_acct', fingerprint(username))


def note_login_success(username: str, network: str) -> None:
    db.clear_attempts('login', _pair(username, network))


def twofa_locked(user_id: int, network: str) -> bool:
    w = config.LOCKOUT_WINDOW_SECONDS
    return (db.count_attempts('2fa', fingerprint(f'{user_id}|{network}'), w) >= config.TWOFA_MAX_FAILURES
            or db.count_attempts('2fa_acct', str(user_id), w) >= config.TWOFA_MAX_FAILURES * 3)


def note_twofa_failure(user_id: int, network: str) -> None:
    db.record_attempt('2fa', fingerprint(f'{user_id}|{network}'))
    db.record_attempt('2fa_acct', str(user_id))


def reset_email_allowed(email: str) -> bool:
    """At most RESET_EMAILS_PER_HOUR reset emails per address per hour (stops using us to flood someone's inbox)."""
    key = fingerprint(email)
    if db.count_attempts('reset', key, 3600) >= config.RESET_EMAILS_PER_HOUR:
        return False
    db.record_attempt('reset', key)
    return True


# ---- notifications ----------------------------------------------------------------------------------------------------------

def notify(user_row: dict, subject: str, body: str) -> None:
    """Tell the account owner about a security-relevant change. Best effort: never blocks or breaks the action."""
    try:
        send_email(user_row['email'], subject, body + "\n\nIf this was not you, reset your password now and turn on two-factor authentication.\n"
                   "- Quick Aid")
    except Exception:
        logger.warning("Could not send a security notification", exc_info=True)
