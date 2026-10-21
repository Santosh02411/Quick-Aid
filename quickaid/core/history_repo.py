"""History, notes, export and retention queries.

EVERY function takes the owning `user_id` and puts it in the SQL `WHERE` clause, together with the entry id. There is no way
to ask this module for "entry 17"; only for "entry 17 that belongs to user 4". A request for someone else's entry returns
the same "not found" as a missing one, so ids cannot be probed.

Health text is encrypted at rest, so free-text search decrypts a user's own candidate rows in memory (bounded by MAX_SCAN)
instead of using SQL LIKE. Filters on non-sensitive columns (type, urgency, emergency flag, date) run in SQL.
"""

import json
import math
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence

from quickaid.core import crypto, database
from quickaid.core.logging_config import get_logger

logger = get_logger('history_repo')

TABLES = {'image': 'image_analyses', 'symptom': 'symptom_analyses'}
URGENCY_COLUMN = {'image': 'urgency', 'symptom': 'urgency_level'}
URGENCY_RANK = {'high': 0, 'medium': 1, 'low': 2}
MAX_SCAN = 5000            # most rows a text search will decrypt for one user in one request
MAX_NOTES_LENGTH = 2000
SORTS = ('newest', 'oldest', 'urgency')


def _iso_start(day: str) -> str:
    return f'{day}T00:00:00'


def _iso_end(day: str) -> str:
    return f'{day}T23:59:59.999999'


def _where(user_id: int, urgencies: Optional[Sequence[str]], emergency: Optional[bool], date_from: Optional[str], date_to: Optional[str],
           urgency_col: str):
    clauses, params = ['user_id = ?'], [user_id]
    if urgencies:
        clauses.append(f"{urgency_col} IN ({','.join('?' * len(urgencies))})")
        params += list(urgencies)
    if emergency is not None:
        clauses.append('emergency_alert = ?')
        params.append(1 if emergency else 0)
    if date_from:
        clauses.append('created_at >= ?')
        params.append(_iso_start(date_from))
    if date_to:
        clauses.append('created_at <= ?')
        params.append(_iso_end(date_to))
    return ' AND '.join(clauses), params


def _union(user_id, types, urgencies, emergency, date_from, date_to):
    """The SQL that lists matching (type, id, created_at, urgency, emergency) rows across both history tables."""
    parts, params = [], []
    for t in types:
        where, p = _where(user_id, urgencies, emergency, date_from, date_to, URGENCY_COLUMN[t])
        parts.append(f"SELECT '{t}' AS type, id, created_at, {URGENCY_COLUMN[t]} AS urgency, emergency_alert AS emergency "  # nosec B608: identifiers come from a fixed allow-list in this module; every value is a bound parameter
                     f"FROM {TABLES[t]} WHERE {where}")
        params += p
    return ' UNION ALL '.join(parts), params


def _order(sort: str) -> str:
    if sort == 'oldest':
        return 'created_at ASC, id ASC'
    if sort == 'urgency':
        return "CASE urgency WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END, created_at DESC, id DESC"
    return 'created_at DESC, id DESC'


def _load_rows(conn, keys: List[Dict]) -> List[Dict]:
    """Full entries for [(type,id)...], in the order given, scoped to nothing but the ids we just selected for this user."""
    out: Dict[tuple, Dict] = {}
    for t, fn in (('image', database._row_to_image_dict), ('symptom', database._row_to_symptom_dict)):
        ids = [k['id'] for k in keys if k['type'] == t]
        if ids:
            for r in conn.execute(f"SELECT * FROM {TABLES[t]} WHERE id IN ({','.join('?' * len(ids))})", ids):  # nosec B608: identifiers come from a fixed allow-list in this module; every value is a bound parameter
                out[(t, r['id'])] = fn(r)
    return [out[(k['type'], k['id'])] for k in keys if (k['type'], k['id']) in out]


def _haystack(entry: Dict, notes: str) -> str:
    bits = [entry['type'], entry.get('urgency') or entry.get('urgency_level') or '', notes or '']
    for key in ('symptom_text', 'original_filename'):
        bits.append(entry.get(key) or '')
    for key in ('detected_symptoms', 'possible_conditions', 'detected_conditions', 'recommendations'):
        bits += [str(x) for x in entry.get(key) or []]
    return ' '.join(bits).lower()


def search_history(user_id: int, q: Optional[str] = None, types: Optional[Sequence[str]] = None, urgencies: Optional[Sequence[str]] = None,
                   emergency: Optional[bool] = None, date_from: Optional[str] = None, date_to: Optional[str] = None,
                   sort: str = 'newest', page: int = 1, per_page: int = 20) -> Dict:
    """One page of this user's history plus totals. Filters combine with AND; `q` words must ALL appear somewhere in the entry."""
    types = [t for t in (types or ['image', 'symptom']) if t in TABLES] or ['image', 'symptom']
    sort = sort if sort in SORTS else 'newest'
    page, per_page = max(1, int(page)), max(1, min(int(per_page), 100))
    terms = [w for w in (q or '').lower().split() if w][:8]
    union, params = _union(user_id, types, urgencies, emergency, date_from, date_to)

    with database.get_connection() as conn:
        if not terms:
            total = conn.execute(f'SELECT COUNT(*) FROM ({union})', params).fetchone()[0]  # nosec B608: identifiers come from a fixed allow-list in this module; every value is a bound parameter
            keys = [dict(r) for r in conn.execute(f'SELECT type, id FROM ({union}) ORDER BY {_order(sort)} LIMIT ? OFFSET ?',  # nosec B608: identifiers come from a fixed allow-list in this module; every value is a bound parameter
                                                  params + [per_page, (page - 1) * per_page])]
            items = _load_rows(conn, keys)
        else:
            keys = [dict(r) for r in conn.execute(f'SELECT type, id, created_at, urgency FROM ({union}) ORDER BY {_order(sort)} LIMIT ?',  # nosec B608: identifiers come from a fixed allow-list in this module; every value is a bound parameter
                                                  params + [MAX_SCAN])]
            candidates = _load_rows(conn, keys)
            notes = {}
            for t in types:
                for r in conn.execute(f'SELECT id, notes FROM {TABLES[t]} WHERE user_id = ? AND notes IS NOT NULL', (user_id,)):  # nosec B608: identifiers come from a fixed allow-list in this module; every value is a bound parameter
                    notes[(t, r['id'])] = crypto.decrypt(r['notes'])
            hits = [e for e in candidates if all(w in _haystack(e, notes.get((e['type'], e['id']), '')) for w in terms)]
            total = len(hits)
            items = hits[(page - 1) * per_page: page * per_page]
    return {'items': items, 'total': total, 'page': page, 'per_page': per_page, 'pages': max(1, math.ceil(total / per_page))}


def get_entry(user_id: int, entry_type: str, entry_id: int) -> Optional[Dict]:
    """One entry with its full stored analysis, personal notes and linked conversations. None unless it belongs to `user_id`."""
    if entry_type not in TABLES:
        return None
    table = TABLES[entry_type]
    with database.get_connection() as conn:
        row = conn.execute(f'SELECT * FROM {table} WHERE id = ? AND user_id = ?', (entry_id, user_id)).fetchone()  # nosec B608: identifiers come from a fixed allow-list in this module; every value is a bound parameter
        if not row:
            return None
        entry = (database._row_to_image_dict if entry_type == 'image' else database._row_to_symptom_dict)(row)
        entry['notes'] = crypto.decrypt(row['notes']) if row['notes'] else ''
        entry['notes_updated_at'] = row['notes_updated_at']
        entry['full'] = None
        if row['analysis_json']:
            try:
                entry['full'] = json.loads(crypto.decrypt(row['analysis_json']))
            except (ValueError, TypeError):
                entry['full'] = None
        convs = conn.execute(
            'SELECT c.id, c.title, c.updated_at, (SELECT COUNT(*) FROM chat_messages m WHERE m.conversation_id = c.id) AS n '
            'FROM chat_conversations c WHERE c.user_id = ? AND c.source_type = ? AND c.source_id = ? ORDER BY c.updated_at DESC',
            (user_id, entry_type, entry_id)).fetchall()
        entry['conversations'] = [{'id': c['id'], 'title': crypto.decrypt(c['title']), 'updated_at': c['updated_at'], 'message_count': c['n']}
                                  for c in convs]
    return entry


def set_notes(user_id: int, entry_type: str, entry_id: int, notes: str) -> bool:
    """Save (or clear, when empty) the person's own note on an entry. Encrypted at rest. False if the entry is not theirs."""
    if entry_type not in TABLES:
        return False
    notes = (notes or '').strip()[:MAX_NOTES_LENGTH]
    stamp = datetime.now(timezone.utc).isoformat()
    with database.get_connection() as conn:
        cur = conn.execute(f'UPDATE {TABLES[entry_type]} SET notes = ?, notes_updated_at = ? WHERE id = ? AND user_id = ?',  # nosec B608: identifiers come from a fixed allow-list in this module; every value is a bound parameter
                           (crypto.encrypt(notes) if notes else None, stamp if notes else None, entry_id, user_id))
        return cur.rowcount > 0


def delete_entry(user_id: int, entry_type: str, entry_id: int) -> bool:
    """Delete ONE entry and the chats about it, atomically. False (and nothing changes) if it is not this user's."""
    if entry_type not in TABLES:
        return False
    with database.transaction() as conn:
        cur = conn.execute(f'DELETE FROM {TABLES[entry_type]} WHERE id = ? AND user_id = ?', (entry_id, user_id))  # nosec B608: identifiers come from a fixed allow-list in this module; every value is a bound parameter
        if cur.rowcount == 0:
            return False
        conn.execute('DELETE FROM chat_conversations WHERE user_id = ? AND source_type = ? AND source_id = ?', (user_id, entry_type, entry_id))
    from quickaid.core import deletions
    deletions.record('entry', user_id=user_id, kind=entry_type, id=entry_id)       # a restored backup must not resurrect it
    return True


def count_entries(user_id: int) -> Dict[str, int]:
    with database.get_connection() as conn:
        return {t: conn.execute(f'SELECT COUNT(*) FROM {tbl} WHERE user_id = ?', (user_id,)).fetchone()[0] for t, tbl in TABLES.items()}  # nosec B608: identifiers come from a fixed allow-list in this module; every value is a bound parameter


def search_conversations(user_id: int, q: Optional[str] = None, page: int = 1, per_page: int = 20) -> Dict:
    """This user's chats, newest first. `q` matches titles and message text (decrypted in memory)."""
    page, per_page = max(1, int(page)), max(1, min(int(per_page), 100))
    terms = [w for w in (q or '').lower().split() if w][:8]
    with database.get_connection() as conn:
        rows = conn.execute(
            'SELECT c.*, (SELECT COUNT(*) FROM chat_messages m WHERE m.conversation_id = c.id) AS n FROM chat_conversations c '
            'WHERE c.user_id = ? ORDER BY c.updated_at DESC LIMIT ?', (user_id, MAX_SCAN)).fetchall()
        convs = [database._conversation_dict(r, r['n']) for r in rows]
        if terms:
            keep = []
            for c in convs:
                text = c['title'].lower() + ' ' + ' '.join(
                    crypto.decrypt(m['content']).lower() for m in conn.execute('SELECT content FROM chat_messages WHERE conversation_id = ?', (c['id'],)))
                if all(w in text for w in terms):
                    keep.append(c)
            convs = keep
    total = len(convs)
    return {'items': convs[(page - 1) * per_page: page * per_page], 'total': total, 'page': page, 'per_page': per_page,
            'pages': max(1, math.ceil(total / per_page))}


def export_user_data(user_id: int) -> Dict:
    """Everything we hold about this user's health activity, as plain data (for the person to download or move elsewhere)."""
    with database.get_connection() as conn:
        user = conn.execute('SELECT username, email, region, retention_days, created_at FROM users WHERE id = ?', (user_id,)).fetchone()
        entries = []
        for t in TABLES:
            for r in conn.execute(f'SELECT id FROM {TABLES[t]} WHERE user_id = ? ORDER BY created_at', (user_id,)).fetchall():  # nosec B608: identifiers come from a fixed allow-list in this module; every value is a bound parameter
                entries.append(r['id'] and (t, r['id']))
        convs = conn.execute('SELECT * FROM chat_conversations WHERE user_id = ? ORDER BY created_at', (user_id,)).fetchall()
        chats = []
        for c in convs:
            msgs = conn.execute('SELECT role, content, flagged, created_at FROM chat_messages WHERE conversation_id = ? ORDER BY id', (c['id'],)).fetchall()
            chats.append({'id': c['id'], 'title': crypto.decrypt(c['title']), 'about': {'type': c['source_type'], 'id': c['source_id']} if c['source_type'] else None,
                          'created_at': c['created_at'], 'updated_at': c['updated_at'],
                          'messages': [{'role': m['role'], 'content': crypto.decrypt(m['content']), 'emergency_guidance': bool(m['flagged']), 'created_at': m['created_at']} for m in msgs]})
    records = [e for e in (get_entry(user_id, t, i) for t, i in entries) if e]
    records.sort(key=lambda e: e['created_at'])
    return {'exported_at': datetime.now(timezone.utc).isoformat(), 'format_version': 1,
            'account': dict(user) if user else {}, 'assessments': records, 'conversations': chats,
            'note': 'AI-generated estimates for general information, not diagnoses.'}


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------

def get_retention_days(user_id: int) -> Optional[int]:
    with database.get_connection() as conn:
        row = conn.execute('SELECT retention_days FROM users WHERE id = ?', (user_id,)).fetchone()
    return row['retention_days'] if row else None


def set_retention_days(user_id: int, days: Optional[int]) -> None:
    with database.get_connection() as conn:
        conn.execute('UPDATE users SET retention_days = ? WHERE id = ?', (days, user_id))


def users_with_retention_rows() -> List[Dict]:
    with database.get_connection() as conn:
        return [dict(r) for r in conn.execute('SELECT id, retention_days FROM users')]


def purge_older_than(user_id: int, cutoff_iso: str) -> Dict[str, int]:
    """Delete this user's analyses and chats last touched before `cutoff_iso`, in one transaction."""
    with database.transaction() as conn:
        n = 0
        for t, tbl in TABLES.items():
            old = [r['id'] for r in conn.execute(f'SELECT id FROM {tbl} WHERE user_id = ? AND created_at < ?', (user_id, cutoff_iso))]  # nosec B608: identifiers come from a fixed allow-list in this module; every value is a bound parameter
            for i in old:
                conn.execute('DELETE FROM chat_conversations WHERE user_id = ? AND source_type = ? AND source_id = ?', (user_id, t, i))
            if old:
                n += conn.execute(f'DELETE FROM {tbl} WHERE user_id = ? AND created_at < ?', (user_id, cutoff_iso)).rowcount  # nosec B608: identifiers come from a fixed allow-list in this module; every value is a bound parameter
        chats = conn.execute('DELETE FROM chat_conversations WHERE user_id = ? AND updated_at < ?', (user_id, cutoff_iso)).rowcount
    return {'analyses': n, 'chats': chats}


def purge_expired_tokens(now_iso: Optional[str] = None) -> int:
    now_iso = now_iso or datetime.now(timezone.utc).isoformat()
    with database.get_connection() as conn:
        return conn.execute('DELETE FROM user_tokens WHERE expires_at < ?', (now_iso,)).rowcount


def purge_stale_interviews(hours: int = 24) -> int:
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    with database.get_connection() as conn:
        return conn.execute('DELETE FROM symptom_interviews WHERE updated_at < ?', (cutoff,)).rowcount


def claim_maintenance(key: str, min_interval_seconds: int) -> bool:
    """True for exactly ONE caller per interval, even across several worker processes (atomic compare-and-set)."""
    now = datetime.now(timezone.utc)
    threshold = (now - timedelta(seconds=min_interval_seconds)).isoformat()
    with database.transaction() as conn:
        row = conn.execute('SELECT value FROM maintenance_state WHERE key = ?', (key,)).fetchone()
        if row is None:
            conn.execute('INSERT INTO maintenance_state (key, value, updated_at) VALUES (?, ?, ?)', (key, now.isoformat(), now.isoformat()))
            return True
        if row['value'] > threshold:
            return False
        conn.execute('UPDATE maintenance_state SET value = ?, updated_at = ? WHERE key = ?', (now.isoformat(), now.isoformat(), key))
        return True


def last_maintenance(key: str) -> Optional[str]:
    with database.get_connection() as conn:
        row = conn.execute('SELECT value FROM maintenance_state WHERE key = ?', (key,)).fetchone()
    return row['value'] if row else None
