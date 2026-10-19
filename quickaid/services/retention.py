"""Data-retention policy and the maintenance sweep that enforces it.

The policy (shown to people on /privacy and in Account settings):
  * Uploaded photos      deleted immediately after analysis. A leftover older than UPLOAD_MAX_AGE_MINUTES (a crash) is swept.
  * Interview sessions   deleted after 24 hours.
  * Sign-in / reset tokens   deleted once expired.
  * Saved assessments and chats   kept for the person's chosen period (30 / 90 / 365 days, or until they delete them);
                         DEFAULT_RETENTION_DAYS applies until they choose; HISTORY_MAX_RETENTION_DAYS is an operator ceiling.
  * Telemetry            timings and counts only, no health text, deleted after TELEMETRY_RETENTION_DAYS.
  * Backups              rotate out after BACKUP_KEEP backups; a deletion therefore leaves the backups only until they rotate.
People can delete any single entry, all history, or their account at any time, immediately.
"""

import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from quickaid import config
from quickaid.api.responses import ApiError
from quickaid.core import database as db, history_repo
from quickaid.core.logging_config import get_logger

logger = get_logger('retention')

_UPLOAD_NAME = re.compile(r'^[0-9a-f]{32}\.(png|jpg|jpeg|gif|bmp|webp)$')
LABELS = {30: '30 days', 90: '90 days', 365: '1 year', 0: 'Until I delete it'}
CHECK_EVERY_SECONDS = 300          # how often a process even looks at whether the daily sweep is due


class RetentionService:
    def __init__(self):
        self._next_check = 0.0

    # ---- policy
    def effective_days(self, user_days: Optional[int]) -> int:
        """Days to keep this person's history (0 = until they delete it). Their choice, else the default, never above the ceiling."""
        days = user_days if user_days is not None else config.default_retention_days()
        cap = config.max_retention_days()
        if cap > 0:
            days = cap if days == 0 else min(days, cap)
        return days

    def choices(self) -> List[Dict]:
        cap = config.max_retention_days()
        return [{'days': d, 'label': LABELS[d]} for d in (30, 90, 365, 0) if not (cap > 0 and (d == 0 or d > cap))]

    def user_setting(self, user_id: int) -> Dict:
        chosen = history_repo.get_retention_days(user_id)
        return {'chosen_days': chosen, 'effective_days': self.effective_days(chosen), 'default_days': config.default_retention_days(),
                'using_default': chosen is None, 'choices': self.choices()}

    def set_user_days(self, user_id: int, days: Optional[int]) -> Dict:
        if days is not None and days not in {c['days'] for c in self.choices()}:
            raise ApiError('That retention period is not available.', 400, 'invalid_retention')
        history_repo.set_retention_days(user_id, days)
        logger.info("Retention changed: user_id=%s days=%s", user_id, days)
        return self.user_setting(user_id)

    def policy(self) -> Dict:
        return {
            'uploaded_photos': {'kept': 'Deleted immediately after analysis', 'leftover_sweep_minutes': config.upload_max_age_minutes()},
            'interview_sessions_hours': config.interview_ttl_hours(),
            'saved_history': {'default_days': config.default_retention_days(), 'ceiling_days': config.max_retention_days() or None,
                              'choices': self.choices()},
            'telemetry_days': int(os.getenv('TELEMETRY_RETENTION_DAYS', '90') or 90),
            'backups_kept': int(os.getenv('BACKUP_KEEP', '14') or 14),
            'sweep_interval_hours': config.maintenance_interval_hours(),
        }

    # ---- enforcement
    def clean_uploads(self, folder: str, max_age_minutes: Optional[int] = None) -> int:
        """Remove photo files left behind (normally none: every upload is deleted as soon as it is analysed)."""
        if not folder or not os.path.isdir(folder):
            return 0
        limit = time.time() - 60 * (max_age_minutes if max_age_minutes is not None else config.upload_max_age_minutes())
        removed = 0
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if _UPLOAD_NAME.match(name) and os.path.isfile(path) and not os.path.islink(path) and os.path.getmtime(path) < limit:
                try:
                    os.remove(path)
                    removed += 1
                except OSError:
                    logger.warning("Could not remove leftover upload %s", name)
        return removed

    def sweep(self, upload_folder: Optional[str] = None, now: Optional[datetime] = None) -> Dict:
        """Apply every retention rule once. Returns what was removed."""
        now = now or datetime.now(timezone.utc)
        report = {'uploads_removed': self.clean_uploads(upload_folder or '', None),
                  'interviews_purged': history_repo.purge_stale_interviews(config.interview_ttl_hours()),
                  'tokens_purged': history_repo.purge_expired_tokens(now.isoformat()), 'analyses_purged': 0, 'chats_purged': 0}
        for u in history_repo.users_with_retention_rows():
            days = self.effective_days(u['retention_days'])
            if days > 0:
                r = history_repo.purge_older_than(u['id'], (now - timedelta(days=days)).isoformat())
                report['analyses_purged'] += r['analyses']
                report['chats_purged'] += r['chats']
        report['telemetry_purged'] = db.purge_telemetry(int(os.getenv('TELEMETRY_RETENTION_DAYS', '90') or 90))
        if any(report.values()):
            logger.info("Retention sweep removed: %s", {k: v for k, v in report.items() if v})
        return report

    def run_if_due(self, upload_folder: Optional[str] = None, force: bool = False) -> Optional[Dict]:
        """Run the sweep at most once per MAINTENANCE_INTERVAL_HOURS across ALL worker processes. Never raises."""
        if not force and time.monotonic() < self._next_check:
            return None
        self._next_check = time.monotonic() + CHECK_EVERY_SECONDS
        try:
            if force or history_repo.claim_maintenance('sweep', config.maintenance_interval_hours() * 3600):
                return self.sweep(upload_folder)
        except Exception:
            logger.error("Maintenance sweep failed", exc_info=True)
        return None


def main() -> int:
    """python -m quickaid.services.retention : run the sweep now (for cron / container schedulers)."""
    db.init_db()
    print(RetentionService().sweep(os.getenv('UPLOAD_FOLDER')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
