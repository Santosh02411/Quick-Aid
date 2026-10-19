"""History management: search, filter, paginate, detailed reports, personal notes, export and deletion.

Every method takes the owning `user_id`. Asking for an entry that is not yours behaves exactly like asking for one that does not
exist (404), so a signed-in user cannot discover or read anyone else's medical information by guessing ids.
"""

import csv
import io
import json
from datetime import datetime, timezone
from typing import Dict, Tuple

from quickaid.api.responses import ApiError
from quickaid.api.schemas import HistoryQuery
from quickaid.core import database as db, history_repo
from quickaid.core.logging_config import get_logger

logger = get_logger('history_service')

CSV_COLUMNS = ['type', 'id', 'created_at', 'urgency', 'emergency', 'title', 'conditions', 'recommendations', 'notes']


def _csv_safe(value) -> str:
    """Neutralise spreadsheet formula injection: a cell starting with = + - @ would run as a formula when opened in Excel."""
    text = '' if value is None else str(value)
    return "'" + text if text[:1] in ('=', '+', '-', '@', '\t', '\r') else text


class HistoryService:
    # ---- presentation
    @staticmethod
    def level(entry: Dict) -> str:
        return (entry.get('urgency') if entry['type'] == 'image' else entry.get('urgency_level')) or 'medium'

    def present(self, entry: Dict) -> Dict:
        """An entry as the list/detail UI wants it: the stored fields plus a normalised level, a title and the condition tags."""
        out = dict(entry)
        out['level'] = self.level(entry)
        conditions = entry.get('detected_conditions') if entry['type'] == 'image' else entry.get('possible_conditions')
        out['conditions'] = list(conditions or [])
        out['title'] = ((entry.get('original_filename') or 'Photo check') if entry['type'] == 'image' else (entry.get('symptom_text') or 'Symptom check'))[:160]
        out.pop('full', None) if 'full' in out and not out.get('full') else None
        return out

    # ---- home page
    def dashboard(self, user_id: int):
        """(stats, five most recent) for the signed-in home page."""
        entries = db.get_history(user_id, limit=200)
        urgent = sum(1 for e in entries if self.level(e) == 'high' or (e['type'] == 'symptom' and e.get('emergency_alert')))
        stats = {'total': len(entries), 'images': sum(1 for e in entries if e['type'] == 'image'),
                 'symptoms': sum(1 for e in entries if e['type'] == 'symptom'), 'urgent': urgent}
        return stats, entries[:5]

    # ---- queries
    def search(self, user_id: int, q: HistoryQuery) -> Dict:
        res = history_repo.search_history(user_id, q=q.q, types=q.type or None, urgencies=q.urgency or None, emergency=q.emergency,
                                          date_from=q.date_from, date_to=q.date_to, sort=q.sort, page=q.page, per_page=q.per_page)
        return {'history': [self.present(e) for e in res['items']],
                'pagination': {'page': res['page'], 'per_page': res['per_page'], 'total': res['total'], 'pages': res['pages'],
                               'has_next': res['page'] < res['pages'], 'has_prev': res['page'] > 1},
                'filters': {k: v for k, v in q.model_dump().items() if v not in ('', None, [])}}

    def detail(self, user_id: int, entry_type: str, entry_id: int) -> Dict:
        entry = history_repo.get_entry(user_id, entry_type, entry_id)
        if entry is None:
            raise ApiError('That entry was not found.', 404, 'entry_not_found')
        return self.present(entry)

    # ---- changes
    def set_notes(self, user_id: int, entry_type: str, entry_id: int, notes: str) -> Dict:
        if not history_repo.set_notes(user_id, entry_type, entry_id, notes):
            raise ApiError('That entry was not found.', 404, 'entry_not_found')
        entry = history_repo.get_entry(user_id, entry_type, entry_id)
        return {'notes': entry['notes'], 'notes_updated_at': entry['notes_updated_at']}

    def delete_entry(self, user_id: int, entry_type: str, entry_id: int) -> None:
        if not history_repo.delete_entry(user_id, entry_type, entry_id):
            raise ApiError('That entry was not found.', 404, 'entry_not_found')
        logger.info("History entry deleted: user_id=%s type=%s id=%s", user_id, entry_type, entry_id)

    def clear(self, user_id: int) -> None:
        db.delete_history(user_id)
        logger.info("History cleared for user_id=%s", user_id)

    # ---- export
    def export(self, user_id: int, fmt: str = 'json') -> Tuple[bytes, str, str]:
        """(content, mimetype, filename) for everything this user has saved."""
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d')
        data = history_repo.export_user_data(user_id)
        if fmt == 'csv':
            buf = io.StringIO()
            w = csv.writer(buf)
            w.writerow(CSV_COLUMNS)
            for e in data['assessments']:
                p = self.present(e)
                w.writerow([_csv_safe(x) for x in (p['type'], p['id'], p['created_at'], p['level'], 'yes' if p.get('emergency_alert') else 'no', p['title'],
                                                   '; '.join(p['conditions']), '; '.join(p.get('recommendations') or []), p.get('notes', ''))])
            return ('\ufeff' + buf.getvalue()).encode('utf-8'), 'text/csv; charset=utf-8', f'quickaid-history-{stamp}.csv'
        return json.dumps(data, indent=1, ensure_ascii=False).encode('utf-8'), 'application/json', f'quickaid-export-{stamp}.json'

    def export_entry(self, user_id: int, entry_type: str, entry_id: int) -> Tuple[bytes, str, str]:
        entry = self.detail(user_id, entry_type, entry_id)
        payload = {'exported_at': datetime.now(timezone.utc).isoformat(), 'format_version': 1, 'entry': entry,
                   'note': 'AI-generated estimate for general information. Not a diagnosis.'}
        return json.dumps(payload, indent=1, ensure_ascii=False).encode('utf-8'), 'application/json', f'quickaid-{entry_type}-{entry_id}.json'
