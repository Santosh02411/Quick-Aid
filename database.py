"""
SQLite persistence for Quick Aid: user accounts, analysis history, and
follow-up conversation threads.

Uses Python's built-in sqlite3 (no extra service to run). History entries
are scoped to a signed-in user's id, since the app now requires an account
for the AI-analysis features.
"""

import sqlite3
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
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

# Default region for accounts that predate the region column / didn't pick
# one at signup. 'INTL' deliberately avoids assuming everyone is in the US.
DEFAULT_REGION = 'INTL'

# Max follow-up turns kept per analysis (each turn = one question + one
# answer) - bounds both storage growth and the context sent to Gemini.
MAX_FOLLOW_UP_TURNS = 20


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


def _add_column_if_missing(conn, table: str, column: str, coldef: str):
    """
    SQLite has no 'ADD COLUMN IF NOT EXISTS', so check PRAGMA table_info
    first. Lets existing databases pick up new columns (region,
    follow_up_history) without needing a destructive migration.
    """
    existing = {row['name'] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coldef}")
        logger.info("Migrated schema: added %s.%s", table, column)


def init_db():
    """Create tables if they don't already exist, and migrate in any new
    columns for databases created by an earlier version of the app."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                region TEXT NOT NULL DEFAULT 'INTL'
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
                disclaimer TEXT,
                follow_up_history TEXT NOT NULL DEFAULT '[]'
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
                disclaimer TEXT,
                follow_up_history TEXT NOT NULL DEFAULT '[]'
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_image_user ON image_analyses(user_id, created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_symptom_user ON symptom_analyses(user_id, created_at)")

        # Migrate in new columns for DBs created before these existed.
        _add_column_if_missing(conn, 'users', 'region', "TEXT NOT NULL DEFAULT 'INTL'")
        _add_column_if_missing(conn, 'image_analyses', 'follow_up_history', "TEXT NOT NULL DEFAULT '[]'")
        _add_column_if_missing(conn, 'symptom_analyses', 'follow_up_history', "TEXT NOT NULL DEFAULT '[]'")

    logger.info("Database initialized at %s", DB_PATH)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

def create_user(username: str, email: str, password: str, region: str = DEFAULT_REGION) -> Optional[Dict]:
    """
    Create a new user with a hashed password.
    Returns the created user dict, or None if the username/email is taken.
    """
    password_hash = generate_password_hash(password)
    try:
        with get_connection() as conn:
            cursor = conn.execute(
                "INSERT INTO users (username, email, password_hash, created_at, region) VALUES (?, ?, ?, ?, ?)",
                (username, email, password_hash, _now(), region)
            )
            user_id = cursor.lastrowid
        logger.info("Created new user id=%s username=%s region=%s", user_id, username, region)
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
    return check_password_hash(user['password_hash'], password)


def update_user_region(user_id: int, region: str) -> None:
    with get_connection() as conn:
        conn.execute("UPDATE users SET region = ? WHERE id = ?", (region, user_id))
    logger.info("Updated region for user_id=%s to %s", user_id, region)


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

def save_image_analysis(user_id: int, original_filename: str, analysis: Dict) -> int:
    """Persist one image-analysis result for this user. Returns the new row id."""
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO image_analyses
                (user_id, created_at, original_filename, detected_conditions,
                 confidence, urgency, recommendations, safety_tips, disclaimer, follow_up_history)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, '[]')
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
        return cursor.lastrowid


def save_symptom_analysis(user_id: int, symptom_text: str, analysis: Dict) -> int:
    """Persist one symptom-check result for this user. Returns the new row id."""
    emergency = analysis.get('emergency_alert', {})
    is_emergency = bool(emergency.get('alert')) if isinstance(emergency, dict) else bool(emergency)

    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO symptom_analyses
                (user_id, created_at, symptom_text, detected_symptoms,
                 possible_conditions, urgency_level, emergency_alert,
                 recommendations, safety_tips, disclaimer, follow_up_history)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '[]')
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
        return cursor.lastrowid


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
        'follow_up_history': json.loads(row['follow_up_history'] or '[]'),
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
        'follow_up_history': json.loads(row['follow_up_history'] or '[]'),
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


# ---------------------------------------------------------------------------
# Follow-up conversation threads
# ---------------------------------------------------------------------------

_ANALYSIS_TABLES = {
    'image': 'image_analyses',
    'symptom': 'symptom_analyses',
}


def get_analysis(analysis_type: str, analysis_id: int, user_id: int) -> Optional[Dict]:
    """
    Fetch a single analysis row, scoped to the owning user - used both to
    build follow-up context and to enforce that a user can only ask
    follow-ups about their own analyses.
    """
    table = _ANALYSIS_TABLES.get(analysis_type)
    if table is None:
        return None

    with get_connection() as conn:
        row = conn.execute(
            f"SELECT * FROM {table} WHERE id = ? AND user_id = ?",
            (analysis_id, user_id)
        ).fetchone()

    if row is None:
        return None
    return _row_to_image_dict(row) if analysis_type == 'image' else _row_to_symptom_dict(row)


def append_follow_up(analysis_type: str, analysis_id: int, user_id: int, question: str, answer: str) -> Optional[List[Dict]]:
    """
    Append a question/answer turn to an analysis's follow-up thread.
    Returns the updated turn list, or None if the analysis doesn't exist
    or doesn't belong to this user.
    """
    table = _ANALYSIS_TABLES.get(analysis_type)
    if table is None:
        return None

    with get_connection() as conn:
        row = conn.execute(
            f"SELECT follow_up_history FROM {table} WHERE id = ? AND user_id = ?",
            (analysis_id, user_id)
        ).fetchone()
        if row is None:
            return None

        history = json.loads(row['follow_up_history'] or '[]')
        history.append({'role': 'user', 'content': question, 'created_at': _now()})
        history.append({'role': 'model', 'content': answer, 'created_at': _now()})
        # Keep only the most recent turns to bound context size and storage.
        history = history[-(MAX_FOLLOW_UP_TURNS * 2):]

        conn.execute(
            f"UPDATE {table} SET follow_up_history = ? WHERE id = ? AND user_id = ?",
            (json.dumps(history), analysis_id, user_id)
        )

    return history
