"""Deletion tombstones: so restoring a backup can never bring deleted people or entries back.

A backup is a photograph of the past. If someone deletes an entry (or their whole account) AFTER a backup was taken and the backup is
restored later, that entry would silently reappear. To prevent it, every deletion appends one line to an append-only log kept
OUTSIDE the database (`<database file>.deletions.log` next to it, or DELETION_LOG). After a restore, `replay()` deletes everything the log says was
deleted since. The log holds only ids and timestamps, never health text.
"""

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from quickaid.core.logging_config import get_logger

logger = get_logger('deletions')
KEEP_DAYS = 400      # longer than any backup is kept


def log_path(db_path: Optional[str] = None) -> Path:
    if os.getenv('DELETION_LOG'):
        return Path(os.environ['DELETION_LOG'])
    from quickaid.core import database
    db = Path(db_path or database.DB_PATH).resolve()
    return db.with_name(db.name + '.deletions.log')         # one log PER database, so deletions can never replay into a different one


def record(event: str, db_path: Optional[str] = None, **fields) -> None:
    """event: 'account' | 'entry' | 'history'. Never raises: failing to log must not block a person's deletion."""
    try:
        path = log_path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({'type': event, 'at': datetime.now(timezone.utc).isoformat(), **fields}, sort_keys=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    except OSError:
        logger.error("Could not write the deletion log", exc_info=True)


def replay(db_path: str) -> Dict[str, int]:
    """Re-apply every logged deletion to `db_path` (used right after a restore). Idempotent."""
    counts = {'accounts': 0, 'entries': 0, 'histories': 0}
    path = log_path(db_path)
    if not path.exists():
        return counts
    tables = {'symptom': 'symptom_analyses', 'image': 'image_analyses'}
    conn = sqlite3.connect(db_path)
    conn.execute('PRAGMA foreign_keys = ON')
    try:
        for raw in path.read_text(encoding='utf-8').splitlines():
            try:
                ev = json.loads(raw)
            except ValueError:
                continue
            if ev.get('type') == 'account':
                counts['accounts'] += conn.execute('DELETE FROM users WHERE id = ?', (ev['user_id'],)).rowcount
            elif ev.get('type') == 'entry' and ev.get('kind') in tables:
                n = conn.execute(f"DELETE FROM {tables[ev['kind']]} WHERE id = ? AND user_id = ?", (ev['id'], ev['user_id'])).rowcount  # nosec B608: identifiers come from a fixed allow-list in this module; every value is a bound parameter
                conn.execute('DELETE FROM chat_conversations WHERE user_id = ? AND source_type = ? AND source_id = ?', (ev['user_id'], ev['kind'], ev['id']))
                counts['entries'] += n
            elif ev.get('type') == 'history':
                for t in tables.values():
                    counts['histories'] += conn.execute(f'DELETE FROM {t} WHERE user_id = ? AND created_at <= ?', (ev['user_id'], ev['at'])).rowcount  # nosec B608: identifiers come from a fixed allow-list in this module; every value is a bound parameter
                conn.execute('DELETE FROM chat_conversations WHERE user_id = ? AND created_at <= ?', (ev['user_id'], ev['at']))
        conn.commit()
    finally:
        conn.close()
    if any(counts.values()):
        logger.warning("Restore: re-applied %s deletion(s) that happened after the backup was taken", sum(counts.values()))
    return counts


def prune(days: int = KEEP_DAYS, db_path: Optional[str] = None) -> int:
    path = log_path(db_path)
    if not path.exists():
        return 0
    cutoff = datetime.now(timezone.utc).timestamp() - days * 86400
    keep, dropped = [], 0
    for raw in path.read_text(encoding='utf-8').splitlines():
        try:
            at = datetime.fromisoformat(json.loads(raw)['at']).timestamp()
        except (ValueError, KeyError):
            continue
        if at >= cutoff:
            keep.append(raw)
        else:
            dropped += 1
    if dropped:
        path.write_text('\n'.join(keep) + ('\n' if keep else ''), encoding='utf-8')
    return dropped
