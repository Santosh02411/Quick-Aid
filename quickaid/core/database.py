"""
SQLite persistence for Quick Aid: user accounts + analysis history.

Uses Python's built-in sqlite3 (no extra service to run). History entries
are scoped to a signed-in user's id, since the app now requires an account
for the AI-analysis features.
"""

import sqlite3
import json
import re
import os
import hashlib
import secrets
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from werkzeug.security import generate_password_hash, check_password_hash

from quickaid import PROJECT_ROOT
from quickaid.core import crypto
from quickaid.core.logging_config import get_logger

logger = get_logger('database')

DB_PATH = os.getenv('DATABASE_PATH', str(PROJECT_ROOT / 'data' / 'quickaid.db'))

# If DATABASE_PATH points into a directory that doesn't exist yet (e.g. a
# fresh Docker volume mount), create it rather than failing on connect -
# sqlite3.connect() does not create missing parent directories itself.
_db_dir = os.path.dirname(DB_PATH)
if _db_dir:
    os.makedirs(_db_dir, exist_ok=True)


def _connect() -> sqlite3.Connection:
    """One configured connection. foreign_keys is per-connection in SQLite, so it must be set every time."""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")        # wait for a concurrent writer instead of failing instantly
    conn.execute("PRAGMA synchronous = NORMAL")         # safe with WAL, much faster than FULL
    conn.execute("PRAGMA secure_delete = ON")           # deleted rows are overwritten with zeros, not just unlinked
    return conn


@contextmanager
def get_connection():
    """A connection that commits on success and rolls back on any error (one implicit transaction per block)."""
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def transaction():
    """An EXPLICIT write transaction (`BEGIN IMMEDIATE`): takes the write lock up front, so a multi-statement change either
    happens completely or not at all, and two writers cannot interleave half-finished work. Use for anything that touches
    more than one row or table (deleting an entry together with its chats, retention sweeps, data migrations)."""
    conn = _connect()
    conn.isolation_level = None
    try:
        conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.execute("COMMIT")
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise
    finally:
        conn.close()


def init_db():
    """Bring the database up to the latest schema (versioned, transactional, backed up first). Safe on every startup."""
    from quickaid.core import migrations
    migrations.migrate(DB_PATH)
    logger.info("Database initialized at %s", DB_PATH)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

def create_user(username: str, email: str, password: str) -> Optional[Dict]:
    """
    Create a new user with a hashed password.
    Returns the created user dict, or None if the username/email is taken.
    """
    password_hash = generate_password_hash(password)
    try:
        with get_connection() as conn:
            cursor = conn.execute(
                "INSERT INTO users (username, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                (username, email, password_hash, _now())
            )
            user_id = cursor.lastrowid
        logger.info("Created new user id=%s", user_id)
        return get_user_by_id(user_id)
    except sqlite3.IntegrityError:
        logger.warning("User creation failed - username or email already taken")
        return None


def _user_dict(row) -> Optional[Dict]:
    """A user row as a dict, with the 2FA secret decrypted (it is encrypted at rest)."""
    if not row:
        return None
    d = dict(row)
    if d.get('totp_secret'):
        d['totp_secret'] = crypto.decrypt(d['totp_secret'])
    return d


def get_user_by_id(user_id: int) -> Optional[Dict]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return _user_dict(row)


def get_user_by_username(username: str) -> Optional[Dict]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    return _user_dict(row)


_DUMMY_HASH = None


def burn_password_check(password: str) -> None:
    """Spend the same time a real password check would, for usernames that do not exist (so response time does not reveal them)."""
    global _DUMMY_HASH
    if _DUMMY_HASH is None:
        _DUMMY_HASH = generate_password_hash('quickaid-timing-equaliser')
    check_password_hash(_DUMMY_HASH, password)


def verify_password(user: Dict, password: str) -> bool:
    if not user.get('has_password', 1):
        return False
    return check_password_hash(user['password_hash'], password)


def get_user_by_email(email: str) -> Optional[Dict]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    return _user_dict(row)


def list_all_users(query: Optional[str] = None) -> List[Dict]:
    """For admin tooling (the CLI and the web admin dashboard) only: username, email and account-level flags, in
    that order, newest first. Deliberately NOT routed through _user_dict() - an admin listing has no reason to
    decrypt every user's 2FA secret, and never should by default. Never includes health data of any kind; nothing
    here can be used to read a person's symptoms, photos, chats or notes."""
    sql = "SELECT id, username, email, email_verified, totp_enabled, has_password, created_at FROM users"
    params: tuple = ()
    if query:
        sql += " WHERE username LIKE ? OR email LIKE ?"
        like = f"%{query}%"
        params = (like, like)
    sql += " ORDER BY created_at DESC"
    with get_connection() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def username_taken(username: str, exclude_user_id: Optional[int] = None) -> bool:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id FROM users WHERE username = ? AND id != ?",
            (username, exclude_user_id or -1)
        ).fetchone()
    return row is not None


def email_taken(email: str, exclude_user_id: Optional[int] = None) -> bool:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id FROM users WHERE email = ? AND id != ?",
            (email, exclude_user_id or -1)
        ).fetchone()
    return row is not None


def update_username(user_id: int, new_username: str) -> bool:
    """Rename a user. Returns False if the username is already taken."""
    try:
        with get_connection() as conn:
            conn.execute("UPDATE users SET username = ? WHERE id = ?", (new_username, user_id))
        logger.info("Username updated for user_id=%s", user_id)
        return True
    except sqlite3.IntegrityError:
        return False


def update_email(user_id: int, new_email: str) -> bool:
    """Change a user's email and reset verification - they must re-verify
    the new address. Returns False if the email is already taken."""
    try:
        with get_connection() as conn:
            conn.execute(
                "UPDATE users SET email = ?, email_verified = 0 WHERE id = ?",
                (new_email, user_id)
            )
        logger.info("Email updated for user_id=%s", user_id)
        return True
    except sqlite3.IntegrityError:
        return False


def set_password(user_id: int, new_password: str) -> None:
    """Set/replace a user's password (also used the first time an
    OAuth-only account adds a password)."""
    password_hash = generate_password_hash(new_password)
    with get_connection() as conn:
        conn.execute(
            "UPDATE users SET password_hash = ?, has_password = 1, password_changed_at = ?, session_version = session_version + 1 WHERE id = ?",
            (password_hash, _now(), user_id)
        )
    logger.info("Password set/changed for user_id=%s", user_id)


def set_email_verified(user_id: int) -> None:
    with get_connection() as conn:
        conn.execute("UPDATE users SET email_verified = 1 WHERE id = ?", (user_id,))
    logger.info("Email verified for user_id=%s", user_id)


def delete_user(user_id: int) -> None:
    """Permanently delete a user and everything owned by them (history,
    tokens - all via ON DELETE CASCADE). Irreversible."""
    from quickaid.core import deletions
    with get_connection() as conn:
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    deletions.record('account', user_id=user_id)      # so restoring an older backup cannot bring this person back
    scrub(vacuum=True)
    logger.info("Deleted user_id=%s and all owned data", user_id)


# ---------------------------------------------------------------------------
# Two-factor authentication (TOTP)
# ---------------------------------------------------------------------------

def set_pending_totp_secret(user_id: int, secret: str) -> None:
    """Store a not-yet-confirmed TOTP secret (encrypted at rest; totp_enabled stays 0 until confirmed)."""
    with get_connection() as conn:
        conn.execute("UPDATE users SET totp_secret = ?, totp_enabled = 0 WHERE id = ?", (crypto.encrypt(secret), user_id))


def enable_totp(user_id: int) -> None:
    with get_connection() as conn:
        conn.execute("UPDATE users SET totp_enabled = 1, totp_last_step = NULL, session_version = session_version + 1 WHERE id = ?", (user_id,))
    logger.info("2FA enabled for user_id=%s", user_id)


def disable_totp(user_id: int) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE users SET totp_enabled = 0, totp_secret = NULL, totp_last_step = NULL, session_version = session_version + 1 WHERE id = ?",
            (user_id,)
        )
        conn.execute("DELETE FROM recovery_codes WHERE user_id = ?", (user_id,))
    logger.info("2FA disabled for user_id=%s", user_id)


# ---------------------------------------------------------------------------
# OAuth (Google sign-in)
# ---------------------------------------------------------------------------

def get_user_by_oauth(provider: str, sub: str) -> Optional[Dict]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE oauth_provider = ? AND oauth_sub = ?",
            (provider, sub)
        ).fetchone()
    return dict(row) if row else None


def link_oauth_to_user(user_id: int, provider: str, sub: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE users SET oauth_provider = ?, oauth_sub = ? WHERE id = ?",
            (provider, sub, user_id)
        )
    logger.info("Linked %s OAuth to user_id=%s", provider, user_id)


def create_oauth_user(username: str, email: str, provider: str, sub: str) -> Optional[Dict]:
    """
    Create an account for a first-time OAuth sign-in. There's no password
    the user knows yet (has_password=0) - we still fill password_hash with
    an unusable random value so the NOT NULL/hash-format invariants used
    elsewhere hold; verify_password() also short-circuits on has_password.
    The provider's email is trusted as pre-verified (Google verifies email
    ownership before issuing an ID token).
    """
    unusable_hash = generate_password_hash(secrets.token_urlsafe(32))
    try:
        with get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO users
                    (username, email, password_hash, has_password, email_verified,
                     oauth_provider, oauth_sub, created_at)
                VALUES (?, ?, ?, 0, 1, ?, ?, ?)
                """,
                (username, email, unusable_hash, provider, sub, _now())
            )
            user_id = cursor.lastrowid
        logger.info("Created new OAuth user id=%s username=%s provider=%s", user_id, username, provider)
        return get_user_by_id(user_id)
    except sqlite3.IntegrityError:
        logger.warning("OAuth user creation failed - username/email taken: %s / %s", username, email)
        return None


# ---------------------------------------------------------------------------
# Password reset / email verification tokens
# ---------------------------------------------------------------------------

def _hash_token(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode('utf-8')).hexdigest()


def create_token(user_id: int, purpose: str, ttl_seconds: int) -> str:
    """
    Create a single-use token for the given purpose ('password_reset' or
    'email_verify') and return the RAW token to email to the user. Only a
    hash of it is stored, so a database leak alone can't be used to reset
    accounts. Any previous unused tokens of the same purpose for this user
    are invalidated first, so only the newest link/code ever works.
    """
    invalidate_tokens(user_id, purpose)
    raw_token = secrets.token_urlsafe(32)
    token_hash = _hash_token(raw_token)
    expires_at = (datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)).isoformat()
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO user_tokens (user_id, purpose, token_hash, created_at, expires_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, purpose, token_hash, _now(), expires_at)
        )
    return raw_token


def get_valid_token(raw_token: str, purpose: str) -> Optional[Dict]:
    """Look up a token by its raw value; returns None if it doesn't exist,
    has already been used, or has expired."""
    token_hash = _hash_token(raw_token)
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM user_tokens WHERE token_hash = ? AND purpose = ?",
            (token_hash, purpose)
        ).fetchone()
    if row is None:
        return None
    token = dict(row)
    if token['used_at'] is not None:
        return None
    if datetime.fromisoformat(token['expires_at']) < datetime.now(timezone.utc):
        return None
    return token


def invalidate_user_tokens(user_id: int, purpose: str) -> None:
    """Make every outstanding link of this kind unusable (after a reset completes, old reset emails must stop working)."""
    with get_connection() as conn:
        conn.execute("UPDATE user_tokens SET used_at = ? WHERE user_id = ? AND purpose = ? AND used_at IS NULL", (_now(), user_id, purpose))


def neutralise_unverified_account(user_id: int) -> None:
    """A local account whose email was never verified is being claimed by the real owner via Google. Whoever registered it first may
    be an impostor who knows its password, so the password is destroyed, every session and link is revoked, and the email is now verified."""
    with get_connection() as conn:
        conn.execute("UPDATE users SET password_hash = ?, has_password = 0, email_verified = 1, session_version = session_version + 1, "
                     "totp_enabled = 0, totp_secret = NULL WHERE id = ?", (generate_password_hash(secrets.token_urlsafe(32)), user_id))
        conn.execute("UPDATE user_tokens SET used_at = ? WHERE user_id = ? AND used_at IS NULL", (_now(), user_id))
        conn.execute("DELETE FROM recovery_codes WHERE user_id = ?", (user_id,))


def consume_token(token_id: int) -> None:
    with get_connection() as conn:
        conn.execute("UPDATE user_tokens SET used_at = ? WHERE id = ?", (_now(), token_id))


def invalidate_tokens(user_id: int, purpose: str) -> None:
    """Mark all of a user's not-yet-used tokens for this purpose as used,
    so an old reset link/verification link can't be reused alongside a
    freshly requested one."""
    with get_connection() as conn:
        conn.execute(
            "UPDATE user_tokens SET used_at = ? WHERE user_id = ? AND purpose = ? AND used_at IS NULL",
            (_now(), user_id, purpose)
        )


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

def _enc_json(value) -> str:
    return crypto.encrypt(json.dumps(value))


def _dec_json(stored, default):
    """Decrypt + parse a stored value. Plaintext written by older releases still reads fine (until migration 3 encrypts it)."""
    try:
        return json.loads(crypto.decrypt(stored)) if stored else default
    except (ValueError, TypeError):
        return default


def _is_emergency(analysis: Dict) -> bool:
    em = analysis.get('emergency_alert', {})
    return bool(em.get('alert')) if isinstance(em, dict) else bool(em)


def save_image_analysis(user_id: int, original_filename: str, analysis: Dict) -> int:
    """Persist one image-analysis result for this user. Returns the new row id. Health text is stored encrypted."""
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO image_analyses
                (user_id, created_at, original_filename, detected_conditions, confidence, urgency,
                 recommendations, safety_tips, disclaimer, analysis_json, emergency_alert)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, _now(), crypto.encrypt(original_filename or ''), _enc_json(analysis.get('detected_conditions', [])),
             analysis.get('confidence'), analysis.get('urgency'), _enc_json(analysis.get('recommendations', [])),
             _enc_json(analysis.get('safety_tips', [])), analysis.get('disclaimer'), crypto.encrypt(json.dumps(analysis)),
             1 if _is_emergency(analysis) else 0)
        )
        return cur.lastrowid


def save_symptom_analysis(user_id: int, symptom_text: str, analysis: Dict) -> int:
    """Persist one symptom-check result for this user. Returns the new row id. Health text is stored encrypted."""
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO symptom_analyses
                (user_id, created_at, symptom_text, detected_symptoms, possible_conditions, urgency_level, emergency_alert,
                 recommendations, safety_tips, disclaimer, analysis_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, _now(), crypto.encrypt(symptom_text), _enc_json(analysis.get('detected_symptoms', [])),
             _enc_json(analysis.get('possible_conditions', [])), analysis.get('urgency_level'), 1 if _is_emergency(analysis) else 0,
             _enc_json(analysis.get('recommendations', [])), _enc_json(analysis.get('safety_tips', [])), analysis.get('disclaimer'),
             crypto.encrypt(json.dumps(analysis)))
        )
        return cur.lastrowid


def _row_to_image_dict(row: sqlite3.Row) -> Dict:
    return {
        'id': row['id'], 'type': 'image', 'created_at': row['created_at'],
        'original_filename': crypto.decrypt(row['original_filename']),
        'detected_conditions': _dec_json(row['detected_conditions'], []), 'confidence': row['confidence'], 'urgency': row['urgency'],
        'emergency_alert': bool(row['emergency_alert']) if 'emergency_alert' in row.keys() else False,
        'recommendations': _dec_json(row['recommendations'], []), 'safety_tips': _dec_json(row['safety_tips'], []),
        'disclaimer': row['disclaimer'],
        'has_notes': bool(row['notes']) if 'notes' in row.keys() else False,
    }


def _row_to_symptom_dict(row: sqlite3.Row) -> Dict:
    return {
        'id': row['id'], 'type': 'symptom', 'created_at': row['created_at'], 'symptom_text': crypto.decrypt(row['symptom_text']),
        'detected_symptoms': _dec_json(row['detected_symptoms'], []), 'possible_conditions': _dec_json(row['possible_conditions'], []),
        'urgency_level': row['urgency_level'], 'emergency_alert': bool(row['emergency_alert']),
        'recommendations': _dec_json(row['recommendations'], []), 'safety_tips': _dec_json(row['safety_tips'], []),
        'disclaimer': row['disclaimer'],
        'has_notes': bool(row['notes']) if 'notes' in row.keys() else False,
    }


def get_history(user_id: int, limit: int = 50) -> List[Dict]:
    """
    Return this user's image + symptom history, newest first, combined into
    a single chronological list capped at `limit` entries.
    """
    with get_connection() as conn:
        image_rows = conn.execute(
            "SELECT * FROM image_analyses WHERE user_id = ? ORDER BY created_at DESC LIMIT ?", (user_id, limit)).fetchall()
        symptom_rows = conn.execute(
            "SELECT * FROM symptom_analyses WHERE user_id = ? ORDER BY created_at DESC LIMIT ?", (user_id, limit)).fetchall()

    combined = [_row_to_image_dict(r) for r in image_rows] + [_row_to_symptom_dict(r) for r in symptom_rows]
    combined.sort(key=lambda entry: entry['created_at'], reverse=True)
    return combined[:limit]


def delete_history(user_id: int) -> None:
    """Clear all stored history for this user."""
    with get_connection() as conn:
        # Chats about a cleared analysis contain that analysis' content, so they go too.
        conn.execute("DELETE FROM chat_conversations WHERE user_id = ? AND source_id IS NOT NULL", (user_id,))
        conn.execute("DELETE FROM image_analyses WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM symptom_analyses WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM symptom_interviews WHERE user_id = ?", (user_id,))
    from quickaid.core import deletions
    deletions.record('history', user_id=user_id)
    scrub(vacuum=True)
    logger.info("Cleared history for user_id=%s", user_id)


# ---------------------------------------------------------------------------
# Single analysis lookup (used as chat context)
# ---------------------------------------------------------------------------

ANALYSIS_TABLES = {'image': 'image_analyses', 'symptom': 'symptom_analyses'}


def get_analysis(user_id: int, analysis_type: str, analysis_id: int) -> Optional[Dict]:
    """Return one of THIS user's analyses with its full stored result, or None.

    Scoped by user_id so one account can never read another's analysis.
    """
    table = ANALYSIS_TABLES.get(analysis_type)
    if not table:
        return None
    with get_connection() as conn:
        row = conn.execute(f"SELECT * FROM {table} WHERE id = ? AND user_id = ?", (analysis_id, user_id)).fetchone()  # nosec B608: identifiers come from a fixed allow-list in this module; every value is a bound parameter
    if not row:
        return None
    entry = _row_to_image_dict(row) if analysis_type == 'image' else _row_to_symptom_dict(row)
    raw = row['analysis_json']
    if raw:
        try:
            full = json.loads(crypto.decrypt(raw))
            if isinstance(full, dict):
                entry['full'] = full
        except (json.JSONDecodeError, TypeError):
            logger.warning("Stored analysis_json for %s id=%s could not be parsed", analysis_type, analysis_id)
    return entry


def list_recent_analyses(user_id: int, limit: int = 20) -> List[Dict]:
    """Lightweight list of recent analyses for the 'attach to a new chat' picker."""
    return [
        {
            'type': e['type'], 'id': e['id'], 'created_at': e['created_at'],
            'urgency': e.get('urgency') if e['type'] == 'image' else e.get('urgency_level'),
            'title': ((e['detected_conditions'][0] if e.get('detected_conditions') else e.get('original_filename'))
                      if e['type'] == 'image'
                      else (e['possible_conditions'][0] if e.get('possible_conditions') else e.get('symptom_text', ''))),
        }
        for e in get_history(user_id, limit)
    ]


# ---------------------------------------------------------------------------
# Region preference
# ---------------------------------------------------------------------------

def get_user_region(user_id: int) -> Optional[str]:
    with get_connection() as conn:
        row = conn.execute("SELECT region FROM users WHERE id = ?", (user_id,)).fetchone()
    return row['region'] if row and row['region'] else None


def set_user_region(user_id: int, region: str) -> None:
    with get_connection() as conn:
        conn.execute("UPDATE users SET region = ? WHERE id = ?", (region, user_id))


# ---------------------------------------------------------------------------
# Chat conversations (messages and titles are encrypted at rest)
# ---------------------------------------------------------------------------

MAX_CONVERSATIONS_PER_USER = 50
MAX_MESSAGES_PER_CONVERSATION = 200


def _conversation_dict(row: sqlite3.Row, message_count: Optional[int] = None) -> Dict:
    d = {
        'id': row['id'],
        'title': crypto.decrypt(row['title']),
        'source_type': row['source_type'],
        'source_id': row['source_id'],
        'created_at': row['created_at'],
        'updated_at': row['updated_at'],
    }
    if message_count is not None:
        d['message_count'] = message_count
    return d


def create_conversation(user_id: int, title: str, source_type: Optional[str] = None,
                        source_id: Optional[int] = None) -> Dict:
    now = _now()
    with get_connection() as conn:
        # Keep storage bounded: drop the oldest chats beyond the per-user cap.
        total = conn.execute("SELECT COUNT(*) AS n FROM chat_conversations WHERE user_id = ?", (user_id,)).fetchone()['n']
        if total >= MAX_CONVERSATIONS_PER_USER:
            conn.execute(
                "DELETE FROM chat_conversations WHERE id IN (SELECT id FROM chat_conversations "
                "WHERE user_id = ? ORDER BY updated_at ASC LIMIT ?)",
                (user_id, total - MAX_CONVERSATIONS_PER_USER + 1)
            )
        cur = conn.execute(
            "INSERT INTO chat_conversations (user_id, title, source_type, source_id, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, crypto.encrypt(title[:80]), source_type, source_id, now, now)
        )
        row = conn.execute("SELECT * FROM chat_conversations WHERE id = ?", (cur.lastrowid,)).fetchone()
    return _conversation_dict(row, 0)


def get_conversation(user_id: int, conversation_id: int) -> Optional[Dict]:
    """This user's conversation, or None (also None when it belongs to someone else)."""
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM chat_conversations WHERE id = ? AND user_id = ?",
                           (conversation_id, user_id)).fetchone()
        if not row:
            return None
        n = conn.execute("SELECT COUNT(*) AS n FROM chat_messages WHERE conversation_id = ?", (conversation_id,)).fetchone()['n']
    return _conversation_dict(row, n)


def find_conversation_for_analysis(user_id: int, source_type: str, source_id: int) -> Optional[Dict]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM chat_conversations WHERE user_id = ? AND source_type = ? AND source_id = ? "
            "ORDER BY updated_at DESC LIMIT 1", (user_id, source_type, source_id)).fetchone()
        if not row:
            return None
        n = conn.execute("SELECT COUNT(*) AS n FROM chat_messages WHERE conversation_id = ?", (row['id'],)).fetchone()['n']
    return _conversation_dict(row, n)


def list_conversations(user_id: int, limit: int = MAX_CONVERSATIONS_PER_USER) -> List[Dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT c.*, (SELECT COUNT(*) FROM chat_messages m WHERE m.conversation_id = c.id) AS n "
            "FROM chat_conversations c WHERE c.user_id = ? ORDER BY c.updated_at DESC LIMIT ?",
            (user_id, limit)).fetchall()
    return [_conversation_dict(r, r['n']) for r in rows]


def add_message(conversation_id: int, role: str, content: str, flagged: bool = False) -> Dict:
    now = _now()
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO chat_messages (conversation_id, role, content, flagged, created_at) VALUES (?, ?, ?, ?, ?)",
            (conversation_id, role, crypto.encrypt(content), 1 if flagged else 0, now))
        conn.execute("UPDATE chat_conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))
    return {'id': cur.lastrowid, 'role': role, 'content': content, 'flagged': bool(flagged), 'created_at': now}


def get_messages(conversation_id: int, limit: Optional[int] = None) -> List[Dict]:
    """Messages oldest-first. With `limit`, returns the most recent `limit` messages."""
    with get_connection() as conn:
        if limit:
            rows = conn.execute(
                "SELECT * FROM (SELECT * FROM chat_messages WHERE conversation_id = ? ORDER BY id DESC LIMIT ?) ORDER BY id ASC",
                (conversation_id, limit)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM chat_messages WHERE conversation_id = ? ORDER BY id ASC",
                                (conversation_id,)).fetchall()
    return [{'id': r['id'], 'role': r['role'], 'content': crypto.decrypt(r['content']),
             'flagged': bool(r['flagged']), 'created_at': r['created_at']} for r in rows]


def rename_conversation(user_id: int, conversation_id: int, title: str) -> None:
    with get_connection() as conn:
        conn.execute("UPDATE chat_conversations SET title = ? WHERE id = ? AND user_id = ?",
                     (crypto.encrypt(title[:80]), conversation_id, user_id))


def delete_conversation(user_id: int, conversation_id: int) -> bool:
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM chat_conversations WHERE id = ? AND user_id = ?", (conversation_id, user_id))
    return cur.rowcount > 0


def delete_all_conversations(user_id: int) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM chat_conversations WHERE user_id = ?", (user_id,))


# ---------------------------------------------------------------------------
# Symptom interview sessions (state encrypted: it contains health answers)
# ---------------------------------------------------------------------------

INTERVIEW_TTL_HOURS = 24


def create_interview(user_id: int, state: Dict) -> str:
    session_id = secrets.token_urlsafe(16)
    now = _now()
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=INTERVIEW_TTL_HOURS)).isoformat()
    with get_connection() as conn:
        conn.execute("DELETE FROM symptom_interviews WHERE updated_at < ?", (cutoff,))
        conn.execute(
            "INSERT INTO symptom_interviews (id, user_id, state, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, user_id, crypto.encrypt(json.dumps(state)), 'active', now, now))
    return session_id


def get_interview(user_id: int, session_id: str) -> Optional[Dict]:
    """Returns {'state': dict, 'status': str} for this user's session, or None."""
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM symptom_interviews WHERE id = ? AND user_id = ?",
                           (session_id, user_id)).fetchone()
    if not row:
        return None
    try:
        state = json.loads(crypto.decrypt(row['state']))
    except (json.JSONDecodeError, TypeError):
        return None
    return {'state': state, 'status': row['status']}


def update_interview(user_id: int, session_id: str, state: Dict, status: str = 'active') -> None:
    with get_connection() as conn:
        conn.execute("UPDATE symptom_interviews SET state = ?, status = ?, updated_at = ? WHERE id = ? AND user_id = ?",
                     (crypto.encrypt(json.dumps(state)), status, _now(), session_id, user_id))


# ---------------------------------------------------------------------------
# AI telemetry (no user text, images or identities are ever stored here)
# ---------------------------------------------------------------------------

def _b(v):
    return None if v is None else (1 if v else 0)


def record_ai_call(rec: Dict) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO ai_calls (id, analysis_id, ts, kind, model, prompt_version, ok, valid, latency_ms, error_type, "
            "input_tokens, output_tokens, cost_usd) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (rec['id'], rec.get('analysis_id'), rec['ts'], rec['kind'], rec.get('model'), rec.get('prompt_version'), _b(rec['ok']),
             _b(rec.get('valid')), rec.get('latency_ms'), rec.get('error_type'), rec.get('input_tokens', 0), rec.get('output_tokens', 0),
             rec.get('cost_usd')))


def record_ai_analysis(rec: Dict) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO ai_analyses (id, ts, kind, mode, urgency, emergency, safety_raised, status, reliability, error, fallback, "
            "ai_attempted, latency_ms, calls, input_tokens, output_tokens, cost_usd, model, prompt_version) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (rec['id'], rec['ts'], rec['kind'], rec.get('mode'), rec.get('urgency'), _b(rec.get('emergency')) or 0, _b(rec.get('safety_raised')) or 0,
             rec.get('status'), rec.get('reliability'), _b(rec.get('error')) or 0, _b(rec.get('fallback')) or 0, _b(rec.get('ai_attempted')) or 0,
             rec.get('latency_ms'), rec.get('calls', 0), rec.get('input_tokens', 0), rec.get('output_tokens', 0), rec.get('cost_usd'),
             rec.get('model'), rec.get('prompt_version')))


def fetch_ai_calls(since_iso: str) -> List[Dict]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM ai_calls WHERE ts >= ? ORDER BY ts", (since_iso,)).fetchall()
    return [dict(r) for r in rows]


def fetch_ai_analyses(since_iso: str) -> List[Dict]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM ai_analyses WHERE ts >= ? ORDER BY ts", (since_iso,)).fetchall()
    return [dict(r) for r in rows]


def purge_telemetry(days: int) -> int:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    with get_connection() as conn:
        a = conn.execute("DELETE FROM ai_calls WHERE ts < ?", (cutoff,)).rowcount
        b = conn.execute("DELETE FROM ai_analyses WHERE ts < ?", (cutoff,)).rowcount
    return a + b


# ---------------------------------------------------------------------------
# Secure deletion
# ---------------------------------------------------------------------------

def scrub(vacuum: bool = False) -> None:
    """Make deletions physical. `secure_delete` already zeroes freed content inside the database; this also truncates the write-ahead
    log (which can still hold the old rows) and, with vacuum=True, rewrites the file so no freed pages remain. Never raises."""
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)
        try:
            conn.execute("PRAGMA secure_delete = ON")
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            if vacuum:
                conn.execute("VACUUM")
        finally:
            conn.close()
    except sqlite3.Error:
        logger.warning("Could not scrub the database after a deletion", exc_info=True)


# ---------------------------------------------------------------------------
# Sessions, 2FA recovery, attempt tracking, security events, language, AI consent
# ---------------------------------------------------------------------------

def bump_session_version(user_id: int) -> int:
    """Invalidate every signed-in session of this user (all devices). Returns the new version."""
    with get_connection() as conn:
        conn.execute("UPDATE users SET session_version = session_version + 1 WHERE id = ?", (user_id,))
        return conn.execute("SELECT session_version FROM users WHERE id = ?", (user_id,)).fetchone()[0]


def claim_totp_step(user_id: int, step: int) -> bool:
    """Accept a 2FA time-step once only (a captured code cannot be replayed within its validity window)."""
    with get_connection() as conn:
        cur = conn.execute("UPDATE users SET totp_last_step = ? WHERE id = ? AND (totp_last_step IS NULL OR totp_last_step < ?)", (step, user_id, step))
        return cur.rowcount == 1


def _hash_code(code: str) -> str:
    return hashlib.sha256(re.sub(r'[^A-Z0-9]', '', code.upper()).encode('utf-8')).hexdigest()


def create_recovery_codes(user_id: int, count: int = 10) -> List[str]:
    """Replace this user's recovery codes with `count` fresh single-use ones. Only hashes are stored: the plain codes are returned ONCE."""
    alphabet = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'          # no 0/O/1/I: they get misread when copied by hand
    codes = [''.join(secrets.choice(alphabet) for _ in range(5)) + '-' + ''.join(secrets.choice(alphabet) for _ in range(5)) for _ in range(count)]
    with transaction() as conn:
        conn.execute("DELETE FROM recovery_codes WHERE user_id = ?", (user_id,))
        conn.executemany("INSERT INTO recovery_codes (user_id, code_hash, created_at) VALUES (?, ?, ?)", [(user_id, _hash_code(c), _now()) for c in codes])
    return codes


def consume_recovery_code(user_id: int, code: str) -> bool:
    """Use up one recovery code (atomically: two simultaneous tries cannot both succeed)."""
    if not code or len(re.sub(r'[^A-Za-z0-9]', '', code)) != 10:
        return False
    with get_connection() as conn:
        cur = conn.execute("UPDATE recovery_codes SET used_at = ? WHERE user_id = ? AND code_hash = ? AND used_at IS NULL", (_now(), user_id, _hash_code(code)))
        return cur.rowcount == 1


def recovery_codes_remaining(user_id: int) -> int:
    with get_connection() as conn:
        return conn.execute("SELECT COUNT(*) FROM recovery_codes WHERE user_id = ? AND used_at IS NULL", (user_id,)).fetchone()[0]


def record_attempt(scope: str, key: str) -> None:
    with get_connection() as conn:
        conn.execute("INSERT INTO auth_attempts (scope, key, created_at) VALUES (?, ?, ?)", (scope, key, _now()))


def count_attempts(scope: str, key: str, window_seconds: int) -> int:
    since = (datetime.now(timezone.utc) - timedelta(seconds=window_seconds)).isoformat()
    with get_connection() as conn:
        return conn.execute("SELECT COUNT(*) FROM auth_attempts WHERE scope = ? AND key = ? AND created_at >= ?", (scope, key, since)).fetchone()[0]


def clear_attempts(scope: str, key: str) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM auth_attempts WHERE scope = ? AND key = ?", (scope, key))


def add_security_event(user_id: Optional[int], event: str, detail: str = '', network: str = '') -> None:
    with get_connection() as conn:
        conn.execute("INSERT INTO security_events (user_id, event, detail, network, created_at) VALUES (?, ?, ?, ?, ?)",
                     (user_id, event, detail[:200], network[:64], _now()))


def list_security_events(user_id: int, limit: int = 15) -> List[Dict]:
    with get_connection() as conn:
        rows = conn.execute("SELECT event, detail, network, created_at FROM security_events WHERE user_id = ? ORDER BY id DESC LIMIT ?", (user_id, limit)).fetchall()
    return [dict(r) for r in rows]


def purge_security_data(days: int = 180) -> int:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    attempts_cutoff = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    with get_connection() as conn:
        a = conn.execute("DELETE FROM security_events WHERE created_at < ?", (cutoff,)).rowcount
        b = conn.execute("DELETE FROM auth_attempts WHERE created_at < ?", (attempts_cutoff,)).rowcount
        c = conn.execute("DELETE FROM recovery_codes WHERE used_at IS NOT NULL AND used_at < ?", (cutoff,)).rowcount
    return a + b + c


def set_user_language(user_id: int, language: Optional[str]) -> None:
    with get_connection() as conn:
        conn.execute("UPDATE users SET language = ? WHERE id = ?", (language, user_id))


def set_ai_consent(user_id: int, version: Optional[int]) -> None:
    """Record the person's choice about sending text/photos to the external AI provider: the wording version they agreed to, 0 = they said no (remembered,
    so they are not asked again), None = never asked."""
    with get_connection() as conn:
        conn.execute("UPDATE users SET ai_consent_version = ?, ai_consent_at = ? WHERE id = ?", (version, _now() if version is not None else None, user_id))
