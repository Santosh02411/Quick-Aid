"""
Lightweight SQLite persistence for Quick Aid.

Stores a history of image analyses and symptom checks so users can look back
at previous results. Uses Python's built-in sqlite3 (no extra service or
dependency to run) and scopes history per anonymous browser session via a
signed Flask session cookie - there's no user login system, so this is the
simplest way to keep one visitor's history separate from another's without
adding full authentication.
"""

import sqlite3
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Dict, List, Optional

DB_PATH = os.getenv('DATABASE_PATH', os.path.join(os.path.dirname(__file__), 'quickaid.db'))


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create tables if they don't already exist. Safe to call on every startup."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS image_analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
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
                session_id TEXT NOT NULL,
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
        conn.execute("CREATE INDEX IF NOT EXISTS idx_image_session ON image_analyses(session_id, created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_symptom_session ON symptom_analyses(session_id, created_at)")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_image_analysis(session_id: str, original_filename: str, analysis: Dict) -> None:
    """Persist one image-analysis result for this browser session."""
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO image_analyses
                (session_id, created_at, original_filename, detected_conditions,
                 confidence, urgency, recommendations, safety_tips, disclaimer)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
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


def save_symptom_analysis(session_id: str, symptom_text: str, analysis: Dict) -> None:
    """Persist one symptom-check result for this browser session."""
    emergency = analysis.get('emergency_alert', {})
    is_emergency = bool(emergency.get('alert')) if isinstance(emergency, dict) else bool(emergency)

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO symptom_analyses
                (session_id, created_at, symptom_text, detected_symptoms,
                 possible_conditions, urgency_level, emergency_alert,
                 recommendations, safety_tips, disclaimer)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
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


def get_history(session_id: str, limit: int = 50) -> List[Dict]:
    """
    Return this session's image + symptom history, newest first, combined
    into a single chronological list capped at `limit` entries.
    """
    with get_connection() as conn:
        image_rows = conn.execute(
            "SELECT * FROM image_analyses WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
            (session_id, limit)
        ).fetchall()
        symptom_rows = conn.execute(
            "SELECT * FROM symptom_analyses WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
            (session_id, limit)
        ).fetchall()

    combined = [_row_to_image_dict(r) for r in image_rows] + [_row_to_symptom_dict(r) for r in symptom_rows]
    combined.sort(key=lambda entry: entry['created_at'], reverse=True)
    return combined[:limit]


def delete_history(session_id: str) -> None:
    """Clear all stored history for this browser session."""
    with get_connection() as conn:
        conn.execute("DELETE FROM image_analyses WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM symptom_analyses WHERE session_id = ?", (session_id,))
