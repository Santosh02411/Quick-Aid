"""
SQLite persistence for Quick Aid: user accounts + analysis history.

Uses Python's built-in sqlite3 (no extra service to run). History entries
are scoped to a signed-in user's id, since the app now requires an account
for the AI-analysis features.
"""

import sqlite3
import json
import os
import hashlib
import secrets
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from werkzeug.security import generate_password_hash, check_password_hash

from logging_config import get_logger

logger = get_logger('database')

DB_PATH = os.getenv('DATABASE_PATH', os.path.join(os.path.dirname(__file__), 'quickaid.db'))

# If DATABASE_PATH points into a directory that doesn't exist yet (e.g. a
# fresh Docker volume mount), create it rather than failing on connect -
# sqlite3.connect() does not create missing parent directories itself.
_db_dir = os.path.dirname(DB_PATH)
if _db_dir:
    os.makedirs(_db_dir, exist_ok=True)


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create tables if they don't already exist, then apply any schema
    migrations needed to bring an older existing database up to date.
    Safe to call on every startup."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                has_password INTEGER NOT NULL DEFAULT 1,
                email_verified INTEGER NOT NULL DEFAULT 0,
                oauth_provider TEXT,
                oauth_sub TEXT,
                totp_secret TEXT,
                totp_enabled INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS image_analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                created_at TEXT NOT NULL,
                original_filename TEXT,
                detected_conditions TEXT NOT NULL,
                confidence TEXT,
                urgency TEXT,
                recommendations TEXT NOT NULL,
                safety_tips TEXT,
                disclaimer TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS symptom_analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                created_at TEXT NOT NULL,
                symptom_text TEXT NOT NULL,
                detected_symptoms TEXT,
                possible_conditions TEXT,
                urgency_level TEXT,
                emergency_alert INTEGER,
                recommendations TEXT NOT NULL,
                safety_tips TEXT,
                disclaimer TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                purpose TEXT NOT NULL,
                token_hash TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                used_at TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_image_user ON image_analyses(user_id, created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_symptom_user ON symptom_analyses(user_id, created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_user_tokens_lookup ON user_tokens(purpose, token_hash)")
        # Column migrations must run BEFORE any index that references a
        # possibly-new column (e.g. oauth_provider on a pre-OAuth database).
        _migrate_users_table(conn)
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_oauth "
            "ON users(oauth_provider, oauth_sub) WHERE oauth_provider IS NOT NULL"
        )
    logger.info("Database initialized at %s", DB_PATH)


def _migrate_users_table(conn):
    """Add columns introduced after the initial release to a users table
    that predates them - CREATE TABLE IF NOT EXISTS above is a no-op on an
    existing table, so new columns have to be bolted on explicitly."""
    existing_columns = {row['name'] for row in conn.execute("PRAGMA table_info(users)")}
    migrations = [
        ("has_password", "ALTER TABLE users ADD COLUMN has_password INTEGER NOT NULL DEFAULT 1"),
        ("email_verified", "ALTER TABLE users ADD COLUMN email_verified INTEGER NOT NULL DEFAULT 0"),
        ("oauth_provider", "ALTER TABLE users ADD COLUMN oauth_provider TEXT"),
        ("oauth_sub", "ALTER TABLE users ADD COLUMN oauth_sub TEXT"),
        ("totp_secret", "ALTER TABLE users ADD COLUMN totp_secret TEXT"),
        ("totp_enabled", "ALTER TABLE users ADD COLUMN totp_enabled INTEGER NOT NULL DEFAULT 0"),
    ]
    for column_name, ddl in migrations:
        if column_name not in existing_columns:
            conn.execute(ddl)
            logger.info("Migrated users table: added column %s", column_name)


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
        logger.info("Created new user id=%s username=%s", user_id, username)
        return get_user_by_id(user_id)
    except sqlite3.IntegrityError:
        logger.warning("User creation failed - username or email already taken: %s / %s", username, email)
        return None


def get_user_by_id(user_id: int) -> Optional[Dict]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return dict(row) if row else None


def get_user_by_username(username: str) -> Optional[Dict]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    return dict(row) if row else None


def verify_password(user: Dict, password: str) -> bool:
    if not user.get('has_password', 1):
        return False
    return check_password_hash(user['password_hash'], password)


def get_user_by_email(email: str) -> Optional[Dict]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    return dict(row) if row else None


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
        logger.info("Username updated for user_id=%s -> %s", user_id, new_username)
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
            "UPDATE users SET password_hash = ?, has_password = 1 WHERE id = ?",
            (password_hash, user_id)
        )
    logger.info("Password set/changed for user_id=%s", user_id)


def set_email_verified(user_id: int) -> None:
    with get_connection() as conn:
        conn.execute("UPDATE users SET email_verified = 1 WHERE id = ?", (user_id,))
    logger.info("Email verified for user_id=%s", user_id)


def delete_user(user_id: int) -> None:
    """Permanently delete a user and everything owned by them (history,
    tokens - all via ON DELETE CASCADE). Irreversible."""
    with get_connection() as conn:
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    logger.info("Deleted user_id=%s and all owned data", user_id)


# ---------------------------------------------------------------------------
# Two-factor authentication (TOTP)
# ---------------------------------------------------------------------------

def set_pending_totp_secret(user_id: int, secret: str) -> None:
    """Store a not-yet-confirmed TOTP secret (totp_enabled stays 0 until
    the user proves they can generate a valid code for it)."""
    with get_connection() as conn:
        conn.execute(
            "UPDATE users SET totp_secret = ?, totp_enabled = 0 WHERE id = ?",
            (secret, user_id)
        )


def enable_totp(user_id: int) -> None:
    with get_connection() as conn:
        conn.execute("UPDATE users SET totp_enabled = 1 WHERE id = ?", (user_id,))
    logger.info("2FA enabled for user_id=%s", user_id)


def disable_totp(user_id: int) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE users SET totp_enabled = 0, totp_secret = NULL WHERE id = ?",
            (user_id,)
        )
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

def save_image_analysis(user_id: int, original_filename: str, analysis: Dict) -> None:
    """Persist one image-analysis result for this user."""
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO image_analyses
                (user_id, created_at, original_filename, detected_conditions,
                 confidence, urgency, recommendations, safety_tips, disclaimer)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                _now(),
                original_filename,
                json.dumps(analysis.get('detected_conditions', [])),
                analysis.get('confidence'),
                analysis.get('urgency'),
                json.dumps(analysis.get('recommendations', [])),
                json.dumps(analysis.get('safety_tips', [])),
                analysis.get('disclaimer'),
            )
        )


def save_symptom_analysis(user_id: int, symptom_text: str, analysis: Dict) -> None:
    """Persist one symptom-check result for this user."""
    emergency = analysis.get('emergency_alert', {})
    is_emergency = bool(emergency.get('alert')) if isinstance(emergency, dict) else bool(emergency)

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO symptom_analyses
                (user_id, created_at, symptom_text, detected_symptoms,
                 possible_conditions, urgency_level, emergency_alert,
                 recommendations, safety_tips, disclaimer)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                _now(),
                symptom_text,
                json.dumps(analysis.get('detected_symptoms', [])),
                json.dumps(analysis.get('possible_conditions', [])),
                analysis.get('urgency_level'),
                1 if is_emergency else 0,
                json.dumps(analysis.get('recommendations', [])),
                json.dumps(analysis.get('safety_tips', [])),
                analysis.get('disclaimer'),
            )
        )


def _row_to_image_dict(row: sqlite3.Row) -> Dict:
    return {
        'id': row['id'],
        'type': 'image',
        'created_at': row['created_at'],
        'original_filename': row['original_filename'],
        'detected_conditions': json.loads(row['detected_conditions'] or '[]'),
        'confidence': row['confidence'],
        'urgency': row['urgency'],
        'recommendations': json.loads(row['recommendations'] or '[]'),
        'safety_tips': json.loads(row['safety_tips'] or '[]'),
        'disclaimer': row['disclaimer'],
    }


def _row_to_symptom_dict(row: sqlite3.Row) -> Dict:
    return {
        'id': row['id'],
        'type': 'symptom',
        'created_at': row['created_at'],
        'symptom_text': row['symptom_text'],
        'detected_symptoms': json.loads(row['detected_symptoms'] or '[]'),
        'possible_conditions': json.loads(row['possible_conditions'] or '[]'),
        'urgency_level': row['urgency_level'],
        'emergency_alert': bool(row['emergency_alert']),
        'recommendations': json.loads(row['recommendations'] or '[]'),
        'safety_tips': json.loads(row['safety_tips'] or '[]'),
        'disclaimer': row['disclaimer'],
    }


def get_history(user_id: int, limit: int = 50) -> List[Dict]:
    """
    Return this user's image + symptom history, newest first, combined into
    a single chronological list capped at `limit` entries.
    """
    with get_connection() as conn:
        image_rows = conn.execute(
            "SELECT * FROM image_analyses WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit)
        ).fetchall()
        symptom_rows = conn.execute(
            "SELECT * FROM symptom_analyses WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit)
        ).fetchall()

    combined = [_row_to_image_dict(r) for r in image_rows] + [_row_to_symptom_dict(r) for r in symptom_rows]
    combined.sort(key=lambda entry: entry['created_at'], reverse=True)
    return combined[:limit]


def delete_history(user_id: int) -> None:
    """Clear all stored history for this user."""
    with get_connection() as conn:
        conn.execute("DELETE FROM image_analyses WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM symptom_analyses WHERE user_id = ?", (user_id,))
    logger.info("Cleared history for user_id=%s", user_id)
