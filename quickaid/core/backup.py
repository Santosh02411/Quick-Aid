"""Consistent SQLite backups, verification and restore.

* Backups use SQLite's ONLINE backup API, so they are consistent snapshots even while the app is serving requests
  (copying the .db file with `cp` can capture a half-written state, especially in WAL mode).
* Every backup is verified (integrity_check + foreign_key_check) before it is accepted, and gets a manifest with a
  SHA-256, size, schema version and per-table row counts.
* Files are written 0600 in a 0700 directory: a backup contains password hashes and 2FA secrets.
* Health text inside is encrypted, but the KEY IS NOT IN THE BACKUP. Keep CHAT_ENCRYPTION_KEY / SECRET_KEY somewhere safe
  and separate; without it, restored chats and analyses cannot be read.

Command line:
    python -m quickaid.core.backup create [--label TEXT] [--keep N] [--no-compress]
    python -m quickaid.core.backup list | verify FILE | prune --keep N | restore FILE [--force]
"""

import argparse
import gzip
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from quickaid.core.logging_config import get_logger

logger = get_logger('backup')

DEFAULT_KEEP = 14


def backup_dir_for(db_path: str, dest_dir: Optional[str] = None) -> Path:
    d = Path(dest_dir or os.getenv('BACKUP_DIR') or (Path(db_path).resolve().parent / 'backups'))
    d.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(d, 0o700)
    except OSError:
        pass
    return d


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _inspect(db_file: str) -> Dict:
    """Integrity-check a database file and summarise it. Raises ValueError when it is not healthy."""
    conn = sqlite3.connect(f'file:{db_file}?mode=ro', uri=True)
    try:
        result = conn.execute('PRAGMA integrity_check').fetchall()
        if [r[0] for r in result] != ['ok']:
            raise ValueError('integrity_check failed: ' + '; '.join(str(r[0]) for r in result[:5]))
        fk = conn.execute('PRAGMA foreign_key_check').fetchall()
        if fk:
            raise ValueError(f'foreign_key_check found {len(fk)} orphaned row(s)')
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        counts = {t: conn.execute('SELECT COUNT(*) FROM "%s"' % t.replace('"', '""')).fetchone()[0] for t in tables}  # nosec B608: identifiers come from a fixed allow-list in this module; every value is a bound parameter
        version = 0
        if 'schema_migrations' in tables:
            version = conn.execute('SELECT COALESCE(MAX(version), 0) FROM schema_migrations').fetchone()[0]
        return {'tables': counts, 'schema_version': version}
    finally:
        conn.close()


def create_backup(db_path: str, dest_dir: Optional[str] = None, label: Optional[str] = None, compress: bool = True,
                  keep: Optional[int] = DEFAULT_KEEP) -> Dict:
    """Write a verified snapshot of `db_path`. Returns the manifest (also saved next to the backup)."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f'database not found: {db_path}')
    out_dir = backup_dir_for(db_path, dest_dir)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')[:-3]
    stem = f"{label}-{stamp}" if label else f"quickaid-{stamp}"
    tmp_fd, tmp_name = tempfile.mkstemp(suffix='.db', dir=out_dir)
    os.close(tmp_fd)
    try:
        src = sqlite3.connect(db_path, timeout=30)
        dst = sqlite3.connect(tmp_name)
        try:
            src.backup(dst)                                    # consistent online snapshot
        finally:
            dst.close()
            src.close()
        info = _inspect(tmp_name)                              # refuse to keep a backup that is not healthy
        final = out_dir / (stem + ('.db.gz' if compress else '.db'))
        if compress:
            with open(tmp_name, 'rb') as f_in, gzip.open(final, 'wb', compresslevel=6) as f_out:
                shutil.copyfileobj(f_in, f_out)
        else:
            shutil.move(tmp_name, final)
        os.chmod(final, 0o600)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
    manifest = {'file': final.name, 'created_at': datetime.now(timezone.utc).isoformat(), 'label': label, 'compressed': compress,
                'size_bytes': final.stat().st_size, 'sha256': _sha256(final), **info, 'source': os.path.basename(db_path)}
    mpath = final.with_name(final.name + '.json')
    mpath.write_text(json.dumps(manifest, indent=1), encoding='utf-8')
    os.chmod(mpath, 0o600)
    logger.info("Backup created: %s (%d bytes, schema v%s)", final.name, manifest['size_bytes'], info['schema_version'])
    if keep:
        prune(db_path, keep, dest_dir)
    return manifest


def list_backups(db_path: str, dest_dir: Optional[str] = None) -> List[Dict]:
    out = []
    for m in sorted(backup_dir_for(db_path, dest_dir).glob('*.json'), reverse=True):
        try:
            out.append(json.loads(m.read_text(encoding='utf-8')))
        except (OSError, ValueError):
            continue
    return out


def prune(db_path: str, keep: int, dest_dir: Optional[str] = None) -> List[str]:
    """Delete the oldest SCHEDULED backups beyond `keep`. Pre-migration and pre-restore safety copies are never auto-deleted."""
    d = backup_dir_for(db_path, dest_dir)
    scheduled = sorted((p for p in d.glob('quickaid-*.db*') if not p.name.endswith('.json')), key=lambda p: p.name, reverse=True)
    removed = []
    for old in scheduled[keep:]:
        for f in (old, old.with_name(old.name + '.json')):
            if f.exists():
                f.unlink()
        removed.append(old.name)
    if removed:
        logger.info("Backup retention: removed %d old backup(s)", len(removed))
    return removed


def _materialize(backup_file: Path, workdir: str) -> str:
    """Return a plain .db path for a backup (decompressing into workdir when needed)."""
    if backup_file.name.endswith('.gz'):
        target = os.path.join(workdir, 'restored.db')
        with gzip.open(backup_file, 'rb') as f_in, open(target, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        return target
    return str(backup_file)


def verify_backup(backup_path: str) -> Dict:
    """Check a backup file against its manifest and run SQLite's own integrity checks. Raises ValueError on any problem."""
    f = Path(backup_path)
    if not f.exists():
        raise ValueError('backup file not found')
    mpath = f.with_name(f.name + '.json')
    if mpath.exists():
        manifest = json.loads(mpath.read_text(encoding='utf-8'))
        if _sha256(f) != manifest['sha256']:
            raise ValueError('checksum mismatch: the backup file was modified or corrupted')
    with tempfile.TemporaryDirectory() as tmp:
        return _inspect(_materialize(f, tmp))


def restore_backup(backup_path: str, db_path: str, force: bool = False) -> Dict:
    """Replace the live database with a backup. STOP THE APP FIRST.

    The backup is verified, the current database is saved as `pre-restore-...` (so a restore can itself be undone), and the
    swap is atomic. Refuses while another process holds the database unless `force`.
    """
    verify_backup(backup_path)
    saved = None
    if os.path.exists(db_path):
        if not force:
            probe = sqlite3.connect(db_path, timeout=1)
            try:
                probe.execute('BEGIN EXCLUSIVE')
                probe.execute('ROLLBACK')
            except sqlite3.OperationalError as e:
                raise RuntimeError('the database is in use: stop the app before restoring (or use force)') from e
            finally:
                probe.close()
        saved = create_backup(db_path, label='pre-restore', keep=None)
    d = Path(db_path).resolve().parent
    fd, tmp = tempfile.mkstemp(suffix='.db', dir=d)
    os.close(fd)
    try:
        with tempfile.TemporaryDirectory() as work:
            shutil.copyfile(_materialize(Path(backup_path), work), tmp)
        os.replace(tmp, db_path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    for suffix in ('-wal', '-shm'):                              # stale journal files from the old database must not be replayed
        if os.path.exists(db_path + suffix):
            os.remove(db_path + suffix)
    # Bring an older backup up to the current schema, then re-apply every deletion made since it was taken. Without this a restore
    # would silently resurrect entries and accounts that people deleted.
    from quickaid.core import deletions, migrations
    migrations.migrate(db_path, backup=False)
    replayed = deletions.replay(db_path)
    logger.warning("Database restored from %s", os.path.basename(backup_path))
    return {'restored_from': os.path.basename(backup_path), 'previous_database_saved_as': saved['file'] if saved else None,
            'deletions_reapplied': replayed, **_inspect(db_path)}


def main(argv=None) -> int:
    from quickaid.core import database
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('create')
    c.add_argument('--label')
    c.add_argument('--keep', type=int, default=int(os.getenv('BACKUP_KEEP', DEFAULT_KEEP)))
    c.add_argument('--no-compress', action='store_true')
    c.add_argument('--dir')
    sub.add_parser('list')
    v = sub.add_parser('verify')
    v.add_argument('file')
    p = sub.add_parser('prune')
    p.add_argument('--keep', type=int, required=True)
    r = sub.add_parser('restore')
    r.add_argument('file')
    r.add_argument('--force', action='store_true')
    args = ap.parse_args(argv)
    try:
        if args.cmd == 'create':
            m = create_backup(database.DB_PATH, args.dir, args.label, not args.no_compress, args.keep)
            print(f"created {m['file']}  {m['size_bytes']} bytes  schema v{m['schema_version']}  sha256 {m['sha256'][:12]}")
        elif args.cmd == 'list':
            for m in list_backups(database.DB_PATH):
                print(f"{m['file']:60s} {m['size_bytes']:>10} bytes  v{m['schema_version']}  {m['created_at']}")
        elif args.cmd == 'verify':
            info = verify_backup(args.file)
            print(f"OK: schema v{info['schema_version']}, tables: {sum(info['tables'].values())} rows across {len(info['tables'])} tables")
        elif args.cmd == 'prune':
            print('removed:', prune(database.DB_PATH, args.keep) or 'nothing')
        elif args.cmd == 'restore':
            print(json.dumps(restore_backup(args.file, database.DB_PATH, args.force), indent=1))
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        print('error:', e)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
