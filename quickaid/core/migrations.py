"""Versioned, transactional schema migrations for the SQLite database.

How it works
------------
* `schema_migrations` records every migration that has run (version, name, checksum, when, how long).
* Each migration runs inside its OWN transaction (`BEGIN IMMEDIATE` ... `COMMIT`). If one fails, that migration is rolled
  back completely and the database stays at the previous version: there is never a half-migrated schema.
* Before the first pending migration on a database that already holds data, a consistent backup is written to
  `<data dir>/backups/pre-migration-...`. A bad migration can therefore always be undone by restoring that file.
* Migration 1 ("baseline") is idempotent: it creates what is missing and adds columns older versions lacked, so databases
  created by ANY earlier release are adopted cleanly instead of being treated as new.
* A recorded checksum of each migration's code is compared on startup: editing an already-applied migration is flagged
  (never edit one; add a new migration instead).

Rules for adding a migration: append to MIGRATIONS with the next version, keep it idempotent where cheap, never
renumber, and add a test. Command line:  python -m quickaid.core.migrations status | upgrade | check
"""

import hashlib
import inspect
import json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from quickaid.core import crypto
from quickaid.core.logging_config import get_logger

logger = get_logger('migrations')

ENC_PREFIX = crypto.PREFIX


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    apply: Callable[[sqlite3.Connection], None]

    @property
    def checksum(self) -> str:
        return hashlib.sha256(inspect.getsource(self.apply).encode('utf-8')).hexdigest()[:16]


def _script(conn: sqlite3.Connection, script: str) -> None:
    """Run several statements INSIDE the current transaction.

    Not `conn.executescript()`: that silently COMMITs any open transaction first, which would break the guarantee that a
    migration is all-or-nothing. (None of our DDL contains a ';' inside a string, so a plain split is safe.)
    """
    for statement in script.split(';'):
        if statement.strip():
            conn.execute(statement)


def _columns(conn, table: str) -> set:
    return {r[1] for r in conn.execute(f'PRAGMA table_info({table})')}


def _add_column(conn, table: str, column: str, ddl: str) -> None:
    if column not in _columns(conn, table):
        conn.execute(f'ALTER TABLE {table} ADD COLUMN {column} {ddl}')
        logger.info("Migrated %s: added column %s", table, column)


# ---------------------------------------------------------------------------
# 1. Baseline: everything that existed before migrations were versioned
# ---------------------------------------------------------------------------

def m001_baseline(conn: sqlite3.Connection) -> None:
    _script(conn, """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL UNIQUE, email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL, has_password INTEGER NOT NULL DEFAULT 1, email_verified INTEGER NOT NULL DEFAULT 0,
            oauth_provider TEXT, oauth_sub TEXT, totp_secret TEXT, totp_enabled INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS image_analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            created_at TEXT NOT NULL, original_filename TEXT, detected_conditions TEXT NOT NULL, confidence TEXT, urgency TEXT,
            recommendations TEXT NOT NULL, safety_tips TEXT, disclaimer TEXT);
        CREATE TABLE IF NOT EXISTS symptom_analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            created_at TEXT NOT NULL, symptom_text TEXT NOT NULL, detected_symptoms TEXT, possible_conditions TEXT,
            urgency_level TEXT, emergency_alert INTEGER, recommendations TEXT NOT NULL, safety_tips TEXT, disclaimer TEXT);
        CREATE TABLE IF NOT EXISTS user_tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            purpose TEXT NOT NULL, token_hash TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL, expires_at TEXT NOT NULL, used_at TEXT);
        CREATE TABLE IF NOT EXISTS chat_conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            title TEXT NOT NULL, source_type TEXT, source_id INTEGER, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id INTEGER NOT NULL REFERENCES chat_conversations(id) ON DELETE CASCADE,
            role TEXT NOT NULL, content TEXT NOT NULL, flagged INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS symptom_interviews (
            id TEXT PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE, state TEXT NOT NULL,
            status TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS ai_analyses (
            id TEXT PRIMARY KEY, ts TEXT NOT NULL, kind TEXT NOT NULL, mode TEXT, urgency TEXT, emergency INTEGER NOT NULL DEFAULT 0,
            safety_raised INTEGER NOT NULL DEFAULT 0, status TEXT, reliability TEXT, error INTEGER NOT NULL DEFAULT 0,
            fallback INTEGER NOT NULL DEFAULT 0, ai_attempted INTEGER NOT NULL DEFAULT 0, latency_ms REAL, calls INTEGER NOT NULL DEFAULT 0,
            input_tokens INTEGER NOT NULL DEFAULT 0, output_tokens INTEGER NOT NULL DEFAULT 0, cost_usd REAL, model TEXT, prompt_version TEXT);
        CREATE TABLE IF NOT EXISTS ai_calls (
            id TEXT PRIMARY KEY, analysis_id TEXT, ts TEXT NOT NULL, kind TEXT NOT NULL, model TEXT, prompt_version TEXT, ok INTEGER NOT NULL,
            valid INTEGER, latency_ms REAL, error_type TEXT, input_tokens INTEGER NOT NULL DEFAULT 0, output_tokens INTEGER NOT NULL DEFAULT 0,
            cost_usd REAL);
    """)
    # columns that older releases did not have (CREATE TABLE IF NOT EXISTS is a no-op on an existing table)
    for col, ddl in (('has_password', 'INTEGER NOT NULL DEFAULT 1'), ('email_verified', 'INTEGER NOT NULL DEFAULT 0'),
                     ('oauth_provider', 'TEXT'), ('oauth_sub', 'TEXT'), ('totp_secret', 'TEXT'),
                     ('totp_enabled', 'INTEGER NOT NULL DEFAULT 0'), ('region', 'TEXT')):
        _add_column(conn, 'users', col, ddl)
    for table in ('image_analyses', 'symptom_analyses'):
        _add_column(conn, table, 'analysis_json', 'TEXT')
    _script(conn, """
        CREATE INDEX IF NOT EXISTS idx_ai_calls_ts ON ai_calls(ts);
        CREATE INDEX IF NOT EXISTS idx_ai_analyses_ts ON ai_analyses(ts);
        CREATE INDEX IF NOT EXISTS idx_chat_user ON chat_conversations(user_id, updated_at);
        CREATE INDEX IF NOT EXISTS idx_chat_source ON chat_conversations(user_id, source_type, source_id);
        CREATE INDEX IF NOT EXISTS idx_chat_msgs ON chat_messages(conversation_id, id);
        CREATE INDEX IF NOT EXISTS idx_interview_user ON symptom_interviews(user_id, updated_at);
        CREATE INDEX IF NOT EXISTS idx_image_user ON image_analyses(user_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_symptom_user ON symptom_analyses(user_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_user_tokens_lookup ON user_tokens(purpose, token_hash);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_users_oauth ON users(oauth_provider, oauth_sub) WHERE oauth_provider IS NOT NULL;
    """)


# ---------------------------------------------------------------------------
# 2. Personal notes, an emergency flag on photo analyses
# ---------------------------------------------------------------------------

def m002_history_notes_and_flags(conn: sqlite3.Connection) -> None:
    for table in ('image_analyses', 'symptom_analyses'):
        _add_column(conn, table, 'notes', 'TEXT')                 # encrypted at rest
        _add_column(conn, table, 'notes_updated_at', 'TEXT')
    _add_column(conn, 'image_analyses', 'emergency_alert', 'INTEGER NOT NULL DEFAULT 0')


# ---------------------------------------------------------------------------
# 3. Encrypt the plaintext copies of health text that early releases stored
# ---------------------------------------------------------------------------

_LEGACY_COLUMNS = {
    'image_analyses': ('original_filename', 'detected_conditions', 'recommendations', 'safety_tips'),
    'symptom_analyses': ('symptom_text', 'detected_symptoms', 'possible_conditions', 'recommendations', 'safety_tips'),
}


def m003_encrypt_legacy_history_columns(conn: sqlite3.Connection) -> None:
    """Rows saved by older releases keep symptoms, conditions and advice in readable columns. Encrypt them in place.

    Idempotent: only values not already encrypted are touched, so it is safe to re-run. Also backfills the new
    `emergency_alert` flag on photo analyses from the stored full analysis.
    """
    for table, cols in _LEGACY_COLUMNS.items():
        rows = conn.execute(f'SELECT id, {", ".join(cols)} FROM {table}').fetchall()
        for row in rows:
            updates = {c: crypto.encrypt(row[c]) for c in cols if row[c] and not str(row[c]).startswith(ENC_PREFIX)}
            if updates:
                sets = ', '.join(f'{c} = ?' for c in updates)
                conn.execute(f'UPDATE {table} SET {sets} WHERE id = ?', (*updates.values(), row['id']))
    for row in conn.execute('SELECT id, analysis_json FROM image_analyses WHERE analysis_json IS NOT NULL').fetchall():
        try:
            full = json.loads(crypto.decrypt(row['analysis_json']))
        except (ValueError, TypeError):
            continue
        if bool((full.get('emergency_alert') or {}).get('alert')):
            conn.execute('UPDATE image_analyses SET emergency_alert = 1 WHERE id = ?', (row['id'],))


# ---------------------------------------------------------------------------
# 4. Indexes for filtering, sorting and the retention sweeps
# ---------------------------------------------------------------------------

def m004_history_and_maintenance_indexes(conn: sqlite3.Connection) -> None:
    _script(conn, """
        CREATE INDEX IF NOT EXISTS idx_symptom_urgency ON symptom_analyses(user_id, urgency_level, created_at);
        CREATE INDEX IF NOT EXISTS idx_symptom_emergency ON symptom_analyses(user_id, emergency_alert, created_at);
        CREATE INDEX IF NOT EXISTS idx_image_urgency ON image_analyses(user_id, urgency, created_at);
        CREATE INDEX IF NOT EXISTS idx_image_emergency ON image_analyses(user_id, emergency_alert, created_at);
        CREATE INDEX IF NOT EXISTS idx_symptom_created ON symptom_analyses(created_at);
        CREATE INDEX IF NOT EXISTS idx_image_created ON image_analyses(created_at);
        CREATE INDEX IF NOT EXISTS idx_user_tokens_expiry ON user_tokens(expires_at);
        CREATE INDEX IF NOT EXISTS idx_interview_updated ON symptom_interviews(updated_at);
        CREATE INDEX IF NOT EXISTS idx_chat_updated ON chat_conversations(updated_at);
        CREATE INDEX IF NOT EXISTS idx_ai_calls_analysis ON ai_calls(analysis_id);
    """)


# ---------------------------------------------------------------------------
# 5. Per-user retention choice + shared maintenance bookkeeping
# ---------------------------------------------------------------------------

def m005_retention_and_maintenance(conn: sqlite3.Connection) -> None:
    _add_column(conn, 'users', 'retention_days', 'INTEGER')       # NULL = use the system default, 0 = keep until I delete
    _script(conn, """
        CREATE TABLE IF NOT EXISTS maintenance_state (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL);
    """)


MIGRATIONS: List[Migration] = [
    Migration(1, 'baseline', m001_baseline),
    Migration(2, 'history_notes_and_flags', m002_history_notes_and_flags),
    Migration(3, 'encrypt_legacy_history_columns', m003_encrypt_legacy_history_columns),
    Migration(4, 'history_and_maintenance_indexes', m004_history_and_maintenance_indexes),
    Migration(5, 'retention_and_maintenance', m005_retention_and_maintenance),
]
LATEST_VERSION = MIGRATIONS[-1].version


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=30, isolation_level=None)      # we manage transactions explicitly
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA foreign_keys = ON')
    return conn


def _ensure_table(conn) -> None:
    conn.execute('CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, name TEXT NOT NULL, checksum TEXT NOT NULL, '
                 'applied_at TEXT NOT NULL, duration_ms REAL)')


def applied_versions(conn) -> Dict[int, sqlite3.Row]:
    _ensure_table(conn)
    return {r['version']: r for r in conn.execute('SELECT * FROM schema_migrations ORDER BY version')}


def _has_data(conn) -> bool:
    row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'").fetchone()
    return bool(row and conn.execute('SELECT 1 FROM users LIMIT 1').fetchone())


def status(path: str) -> Dict:
    conn = _connect(path)
    try:
        applied = applied_versions(conn)
        return {'path': path, 'current': max(applied) if applied else 0, 'latest': LATEST_VERSION,
                'pending': [m.version for m in MIGRATIONS if m.version not in applied],
                'applied': [{'version': v, 'name': r['name'], 'applied_at': r['applied_at']} for v, r in applied.items()],
                'modified': [v for v, r in applied.items() if v <= LATEST_VERSION and r['checksum'] != next(m.checksum for m in MIGRATIONS if m.version == v)]}
    finally:
        conn.close()


def migrate(path: str, target: Optional[int] = None, backup: bool = True) -> List[int]:
    """Apply every pending migration up to `target` (default: latest). Returns the versions applied."""
    target = target or LATEST_VERSION
    conn = _connect(path)
    applied_now: List[int] = []
    try:
        conn.execute('PRAGMA busy_timeout = 30000')
        try:
            conn.execute('PRAGMA journal_mode = WAL')
        except sqlite3.DatabaseError:
            pass
        applied = applied_versions(conn)
        for v, row in applied.items():
            known = next((m for m in MIGRATIONS if m.version == v), None)
            if known and row['checksum'] != known.checksum:
                logger.warning("Migration %d (%s) was edited after it was applied. Never edit applied migrations; add a new one.", v, known.name)
        pending = [m for m in MIGRATIONS if m.version not in applied and m.version <= target]
        if not pending:
            return applied_now
        # Back up any database that already holds people's data, INCLUDING one created by an earlier release that has no
        # schema_migrations table yet (those are exactly the databases that most need a way back).
        if backup and _has_data(conn):
            from quickaid.core import backup as backup_mod
            conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
            label = f'pre-migration-v{max(applied) if applied else 0}-to-v{pending[-1].version}'
            try:
                backup_mod.create_backup(path, label=label, keep=None, compress=True)
            except Exception:
                logger.error("Pre-migration backup failed; refusing to migrate without one", exc_info=True)
                raise
        for m in pending:
            t0 = time.perf_counter()
            conn.execute('BEGIN IMMEDIATE')
            try:
                m.apply(conn)
                conn.execute('INSERT INTO schema_migrations (version, name, checksum, applied_at, duration_ms) VALUES (?, ?, ?, ?, ?)',
                             (m.version, m.name, m.checksum, datetime.now(timezone.utc).isoformat(), (time.perf_counter() - t0) * 1000))
                conn.execute('COMMIT')
            except Exception:
                conn.execute('ROLLBACK')
                logger.error("Migration %d (%s) failed and was rolled back; database left at version %d",
                             m.version, m.name, max(list(applied) + applied_now or [0]), exc_info=True)
                raise
            applied_now.append(m.version)
            logger.info("Applied migration %d: %s (%.0f ms)", m.version, m.name, (time.perf_counter() - t0) * 1000)
        return applied_now
    finally:
        conn.close()


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    from quickaid.core import database
    cmd = argv[0] if argv else 'status'
    if cmd == 'upgrade':
        done = migrate(database.DB_PATH)
        print(f'applied: {done}' if done else 'already up to date')
        return 0
    st = status(database.DB_PATH)
    print(f"database: {st['path']}\ncurrent version: {st['current']}  latest: {st['latest']}  pending: {st['pending'] or 'none'}")
    for a in st['applied']:
        print(f"  v{a['version']:<3} {a['name']:<40} {a['applied_at']}")
    if st['modified']:
        print('WARNING: applied migrations were edited:', st['modified'])
    return 1 if (cmd == 'check' and (st['pending'] or st['modified'])) else 0


if __name__ == '__main__':
    sys.exit(main())
