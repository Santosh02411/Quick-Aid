"""Readiness: is this instance able to do its job RIGHT NOW? (Liveness, `/health`, only says the process answers.)

Used by the container orchestrator and load balancer to decide whether to send traffic here.
"""

import os
import shutil
import tempfile
from typing import Dict, Tuple

from quickaid.core import database, migrations


def check(upload_folder: str, min_free_mb: int = 100) -> Tuple[bool, Dict[str, Dict]]:
    checks: Dict[str, Dict] = {}

    try:                                                           # can we WRITE? (a read-only or locked database is not ready)
        with database.transaction() as conn:
            conn.execute('SELECT 1')
        checks['database_writable'] = {'ok': True}
    except Exception as e:
        checks['database_writable'] = {'ok': False, 'detail': type(e).__name__}

    try:
        st = migrations.status(database.DB_PATH)
        checks['schema_current'] = {'ok': not st['pending'] and not st['modified'], 'version': st['current'], 'pending': st['pending']}
    except Exception as e:
        checks['schema_current'] = {'ok': False, 'detail': type(e).__name__}

    try:                                                           # uploads: can a photo be written and removed?
        os.makedirs(upload_folder, exist_ok=True)
        fd, path = tempfile.mkstemp(dir=upload_folder)
        os.close(fd)
        os.remove(path)
        checks['uploads_writable'] = {'ok': True}
    except Exception as e:
        checks['uploads_writable'] = {'ok': False, 'detail': type(e).__name__}

    try:
        free = shutil.disk_usage(os.path.dirname(database.DB_PATH) or '.').free // (1024 * 1024)
        checks['disk_space'] = {'ok': free >= min_free_mb, 'free_mb': free, 'minimum_mb': min_free_mb}
    except Exception as e:
        checks['disk_space'] = {'ok': False, 'detail': type(e).__name__}

    return all(c['ok'] for c in checks.values()), checks
