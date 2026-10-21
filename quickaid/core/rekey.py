"""Re-encrypt every stored value under the CURRENT primary key. Run after rotating CHAT_ENCRYPTION_KEY (see crypto.py).

    python -m quickaid.core.rekey            # back up, re-encrypt, report
    python -m quickaid.core.rekey --dry-run  # only count what would change

Safe by construction: a verified backup is taken first; everything runs in ONE transaction (all or nothing); and it refuses to
proceed if any value cannot be decrypted with the keys it was given (that would destroy data), unless --force.
"""

import sys
from typing import Dict, Tuple

from quickaid.core import backup, crypto, database
from quickaid.core.logging_config import get_logger

logger = get_logger('rekey')

# every column that holds crypto.encrypt()-ed data
ENCRYPTED_COLUMNS: Dict[str, Tuple[str, ...]] = {
    'symptom_analyses': ('symptom_text', 'detected_symptoms', 'possible_conditions', 'recommendations', 'safety_tips', 'analysis_json', 'notes'),
    'image_analyses': ('original_filename', 'detected_conditions', 'recommendations', 'safety_tips', 'analysis_json', 'notes'),
    'chat_conversations': ('title',),
    'chat_messages': ('content',),
    'symptom_interviews': ('state',),
    'users': ('totp_secret',),
}


def rekey(dry_run: bool = False, force: bool = False, make_backup: bool = True) -> Dict:
    report = {'updated': 0, 'already_current': 0, 'undecryptable': 0, 'tables': {}}
    plan = []
    with database.get_connection() as conn:
        for table, cols in ENCRYPTED_COLUMNS.items():
            for col in cols:
                for row in conn.execute(f'SELECT id, {col} AS v FROM {table} WHERE {col} LIKE ?', (crypto.PREFIX + '%',)).fetchall():  # nosec B608: identifiers come from a fixed allow-list in this module; every value is a bound parameter
                    if crypto.decrypts_with_primary(row['v']):
                        report['already_current'] += 1
                        continue
                    plain = crypto.decrypt(row['v'])
                    if plain == crypto.UNAVAILABLE:
                        report['undecryptable'] += 1
                        continue
                    plan.append((table, col, row['id'], plain))
    if report['undecryptable'] and not force:
        raise RuntimeError(f"{report['undecryptable']} value(s) cannot be decrypted with the configured keys. Add the old key to "
                           "CHAT_ENCRYPTION_KEY_PREVIOUS first. Re-keying now would make them permanently unreadable (use --force to skip them).")
    if dry_run:
        report['would_update'] = len(plan)
        return report
    if make_backup and plan:
        backup.create_backup(database.DB_PATH, label='pre-rekey', keep=None)
    with database.transaction() as conn:
        for table, col, row_id, plain in plan:
            conn.execute(f'UPDATE {table} SET {col} = ? WHERE id = ?', (crypto.encrypt(plain), row_id))  # nosec B608: identifiers come from a fixed allow-list in this module; every value is a bound parameter
            report['tables'][table] = report['tables'].get(table, 0) + 1
    report['updated'] = len(plan)
    logger.info("Re-keyed %d value(s)", len(plan))
    return report


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    try:
        r = rekey(dry_run='--dry-run' in argv, force='--force' in argv)
    except RuntimeError as e:
        print('error:', e)
        return 1
    print(r)
    return 0


if __name__ == '__main__':
    sys.exit(main())
