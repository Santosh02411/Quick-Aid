"""Versioned migrations, transactions and backups."""

import gzip
import json
import os
import sqlite3
import threading
from pathlib import Path

import pytest

from quickaid.core import backup, crypto, migrations


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv('SECRET_KEY', 'migration-test-key')
    monkeypatch.setenv('BACKUP_DIR', str(tmp_path / 'backups'))
    return tmp_path


def legacy_db(path):
    """A database exactly as release v5 left it: no schema_migrations table, plaintext health text, no notes/retention columns."""
    c = sqlite3.connect(path)
    c.executescript("""
        CREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL UNIQUE, email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL, has_password INTEGER NOT NULL DEFAULT 1, email_verified INTEGER NOT NULL DEFAULT 0,
            oauth_provider TEXT, oauth_sub TEXT, totp_secret TEXT, totp_enabled INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL, region TEXT);
        CREATE TABLE symptom_analyses (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            created_at TEXT NOT NULL, symptom_text TEXT NOT NULL, detected_symptoms TEXT, possible_conditions TEXT, urgency_level TEXT,
            emergency_alert INTEGER, recommendations TEXT NOT NULL, safety_tips TEXT, disclaimer TEXT, analysis_json TEXT);
        CREATE TABLE image_analyses (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            created_at TEXT NOT NULL, original_filename TEXT, detected_conditions TEXT NOT NULL, confidence TEXT, urgency TEXT,
            recommendations TEXT NOT NULL, safety_tips TEXT, disclaimer TEXT, analysis_json TEXT);
        INSERT INTO users (username, email, password_hash, created_at) VALUES ('old', 'old@example.com', 'x', '2026-01-01T00:00:00');
        INSERT INTO symptom_analyses (user_id, created_at, symptom_text, detected_symptoms, possible_conditions, urgency_level, emergency_alert, recommendations, safety_tips)
            VALUES (1, '2026-01-02T00:00:00', 'my secret rash', '["rash"]', '["eczema"]', 'low', 0, '["keep it clean"]', '[]');
        INSERT INTO image_analyses (user_id, created_at, original_filename, detected_conditions, urgency, recommendations, safety_tips, analysis_json)
            VALUES (1, '2026-01-03T00:00:00', 'arm.png', '["graze"]', 'low', '["wash"]', '[]', NULL);
    """)
    c.commit()
    c.close()


class TestMigrations:
    def test_a_new_database_gets_every_migration_in_order(self, env):
        path = str(env / 'new.db')
        applied = migrations.migrate(path)
        assert applied == [m.version for m in migrations.MIGRATIONS]
        st = migrations.status(path)
        assert st['current'] == st['latest'] and st['pending'] == [] and st['modified'] == []

    def test_running_again_does_nothing(self, env):
        path = str(env / 'new.db')
        migrations.migrate(path)
        assert migrations.migrate(path) == []

    def test_a_legacy_database_is_adopted_not_recreated(self, env):
        path = str(env / 'legacy.db')
        legacy_db(path)
        assert migrations.migrate(path) == [m.version for m in migrations.MIGRATIONS]
        c = sqlite3.connect(path)
        c.row_factory = sqlite3.Row
        assert c.execute('SELECT username FROM users').fetchone()['username'] == 'old'              # data survived
        cols = {r[1] for r in c.execute('PRAGMA table_info(symptom_analyses)')}
        assert {'notes', 'notes_updated_at'} <= cols and 'retention_days' in {r[1] for r in c.execute('PRAGMA table_info(users)')}

    def test_legacy_plaintext_health_text_is_encrypted_in_place(self, env):
        path = str(env / 'legacy.db')
        legacy_db(path)
        migrations.migrate(path)
        c = sqlite3.connect(path)
        raw = c.execute('SELECT symptom_text, possible_conditions, recommendations FROM symptom_analyses').fetchone()
        assert all(v.startswith(crypto.PREFIX) for v in raw) and 'secret' not in ' '.join(raw)
        assert crypto.decrypt(raw[0]) == 'my secret rash' and json.loads(crypto.decrypt(raw[1])) == ['eczema']
        img = c.execute('SELECT original_filename FROM image_analyses').fetchone()[0]
        assert crypto.decrypt(img) == 'arm.png'

    def test_the_data_migration_is_idempotent(self, env):
        path = str(env / 'legacy.db')
        legacy_db(path)
        migrations.migrate(path)
        c = sqlite3.connect(path)
        before = c.execute('SELECT symptom_text FROM symptom_analyses').fetchone()[0]
        c.close()
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        migrations.m003_encrypt_legacy_history_columns(conn)          # run it again by hand
        conn.commit()
        assert conn.execute('SELECT symptom_text FROM symptom_analyses').fetchone()[0] == before      # not double-encrypted

    def test_old_rows_still_read_correctly_through_the_app_layer(self, env, monkeypatch):
        path = str(env / 'legacy.db')
        legacy_db(path)
        monkeypatch.setenv('DATABASE_PATH', path)
        import importlib
        from quickaid.core import database
        importlib.reload(database)
        database.init_db()
        e = database.get_history(1)
        assert {x['type'] for x in e} == {'symptom', 'image'}
        assert [x for x in e if x['type'] == 'symptom'][0]['symptom_text'] == 'my secret rash'

    def test_a_database_that_holds_data_is_backed_up_first(self, env):
        path = str(env / 'legacy.db')
        legacy_db(path)
        migrations.migrate(path)
        backups = backup.list_backups(path)
        assert len(backups) == 1 and backups[0]['label'] == f'pre-migration-v0-to-v{migrations.LATEST_VERSION}'
        assert backup.verify_backup(str(env / 'backups' / backups[0]['file']))['tables']['users'] == 1

    def test_an_empty_database_is_not_backed_up(self, env):
        migrations.migrate(str(env / 'fresh.db'))
        assert backup.list_backups(str(env / 'fresh.db')) == []

    def test_a_failing_migration_rolls_back_completely(self, env, monkeypatch):
        path = str(env / 'x.db')
        migrations.migrate(path)

        def boom(conn):
            conn.execute('CREATE TABLE half_done (id INTEGER)')
            conn.execute('INSERT INTO half_done VALUES (1)')
            raise RuntimeError('migration exploded')
        monkeypatch.setattr(migrations, 'MIGRATIONS', migrations.MIGRATIONS + [migrations.Migration(99, 'boom', boom)])
        with pytest.raises(RuntimeError):
            migrations.migrate(path, target=99)
        c = sqlite3.connect(path)
        assert c.execute("SELECT name FROM sqlite_master WHERE name='half_done'").fetchone() is None     # nothing half-applied
        assert c.execute('SELECT MAX(version) FROM schema_migrations').fetchone()[0] == migrations.LATEST_VERSION

    def test_editing_an_applied_migration_is_detected(self, env):
        path = str(env / 'x.db')
        migrations.migrate(path)
        c = sqlite3.connect(path)
        c.execute("UPDATE schema_migrations SET checksum = 'tampered' WHERE version = 2")
        c.commit()
        assert migrations.status(path)['modified'] == [2]

    def test_each_migration_records_name_checksum_and_time(self, env):
        path = str(env / 'x.db')
        migrations.migrate(path)
        c = sqlite3.connect(path)
        rows = c.execute('SELECT version, name, checksum, applied_at FROM schema_migrations ORDER BY version').fetchall()
        assert [r[0] for r in rows] == [m.version for m in migrations.MIGRATIONS] and all(r[2] and r[3] for r in rows)

    def test_version_numbers_are_unique_and_contiguous(self):
        assert [m.version for m in migrations.MIGRATIONS] == list(range(1, len(migrations.MIGRATIONS) + 1))

    def test_history_indexes_exist_for_filtering_and_sorting(self, env):
        path = str(env / 'x.db')
        migrations.migrate(path)
        names = {r[0] for r in sqlite3.connect(path).execute("SELECT name FROM sqlite_master WHERE type='index'")}
        assert {'idx_symptom_urgency', 'idx_symptom_emergency', 'idx_image_urgency', 'idx_user_tokens_expiry', 'idx_interview_updated'} <= names

    def test_the_search_query_uses_an_index(self, env):
        path = str(env / 'x.db')
        migrations.migrate(path)
        plan = ' '.join(str(r) for r in sqlite3.connect(path).execute(
            "EXPLAIN QUERY PLAN SELECT id FROM symptom_analyses WHERE user_id = 1 AND urgency_level = 'high' ORDER BY created_at DESC"))
        assert 'USING INDEX' in plan.upper() or 'USING COVERING INDEX' in plan.upper()

    def test_command_line_status_and_upgrade(self, env, monkeypatch, capsys):
        path = str(env / 'cli.db')
        monkeypatch.setenv('DATABASE_PATH', path)
        import importlib
        from quickaid.core import database
        importlib.reload(database)
        assert migrations.main(['status']) == 0 and 'current version: 0' in capsys.readouterr().out
        assert migrations.main(['upgrade']) == 0
        assert migrations.main(['check']) == 0


class TestTransactions:
    def test_a_transaction_is_all_or_nothing(self, db_module):
        u = db_module.create_user('tx', 'tx@example.com', 'password123')
        with pytest.raises(RuntimeError):
            with db_module.transaction() as conn:
                conn.execute("INSERT INTO chat_conversations (user_id, title, created_at, updated_at) VALUES (?, 't', 'a', 'a')", (u['id'],))
                conn.execute("INSERT INTO chat_conversations (user_id, title, created_at, updated_at) VALUES (?, 't2', 'a', 'a')", (u['id'],))
                raise RuntimeError('fail halfway')
        assert db_module.list_conversations(u['id']) == []

    def test_a_committed_transaction_persists(self, db_module):
        u = db_module.create_user('tx2', 'tx2@example.com', 'password123')
        with db_module.transaction() as conn:
            conn.execute("INSERT INTO chat_conversations (user_id, title, created_at, updated_at) VALUES (?, 't', 'a', 'a')", (u['id'],))
        assert len(db_module.list_conversations(u['id'])) == 1

    def test_foreign_keys_are_enforced_on_every_connection(self, db_module):
        with db_module.get_connection() as conn:
            assert conn.execute('PRAGMA foreign_keys').fetchone()[0] == 1
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute("INSERT INTO symptom_analyses (user_id, created_at, symptom_text, recommendations) VALUES (999, 'a', 'x', '[]')")

    def test_wal_mode_and_busy_timeout_are_configured(self, db_module):
        assert sqlite3.connect(db_module.DB_PATH).execute('PRAGMA journal_mode').fetchone()[0] == 'wal'
        with db_module.get_connection() as conn:
            assert conn.execute('PRAGMA busy_timeout').fetchone()[0] >= 5000

    def test_concurrent_writers_do_not_corrupt_or_lose_rows(self, db_module):
        u = db_module.create_user('thr', 'thr@example.com', 'password123')
        errors = []

        def work(n):
            try:
                for i in range(15):
                    db_module.save_symptom_analysis(u['id'], f'text {n}-{i}', {'urgency_level': 'low'})
            except Exception as e:      # pragma: no cover
                errors.append(e)
        ts = [threading.Thread(target=work, args=(n,)) for n in range(4)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        assert errors == [] and len(db_module.get_history(u['id'], limit=500)) == 60

    def test_deleting_a_user_cascades_to_everything_they_own(self, db_module):
        u = db_module.create_user('gone', 'gone@example.com', 'password123')
        db_module.save_symptom_analysis(u['id'], 'x', {})
        c = db_module.create_conversation(u['id'], 't')
        db_module.add_message(c['id'], 'user', 'hi')
        db_module.create_interview(u['id'], {})
        db_module.delete_user(u['id'])
        with db_module.get_connection() as conn:
            for t in ('symptom_analyses', 'chat_conversations', 'chat_messages', 'symptom_interviews'):
                assert conn.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0] == 0


class TestBackups:
    def populated(self, db_module):
        u = db_module.create_user('bk', 'bk@example.com', 'password123')
        db_module.save_symptom_analysis(u['id'], 'private symptoms', {'urgency_level': 'low'})
        return db_module.DB_PATH

    def test_create_verify_and_manifest(self, db_module, env):
        path = self.populated(db_module)
        m = backup.create_backup(path)
        assert m['schema_version'] == migrations.LATEST_VERSION and m['tables']['users'] == 1 and len(m['sha256']) == 64
        info = backup.verify_backup(str(env / 'backups' / m['file']))
        assert info['tables']['symptom_analyses'] == 1

    def test_backups_are_private_files_in_a_private_folder(self, db_module, env):
        m = backup.create_backup(self.populated(db_module))
        f = env / 'backups' / m['file']
        assert oct(f.stat().st_mode & 0o777) == '0o600' and oct((env / 'backups').stat().st_mode & 0o777) == '0o700'

    def test_backup_is_a_consistent_snapshot_even_with_the_database_in_use(self, db_module, env):
        path = self.populated(db_module)
        writer = sqlite3.connect(path)
        writer.execute('BEGIN')
        writer.execute("INSERT INTO symptom_analyses (user_id, created_at, symptom_text, recommendations) VALUES (1, 'a', 'uncommitted', '[]')")
        m = backup.create_backup(path)                              # taken while a write is in flight
        writer.rollback()
        assert m['tables']['symptom_analyses'] == 1                  # the uncommitted row is NOT in the snapshot

    def test_uncompressed_backup_is_also_supported(self, db_module, env):
        m = backup.create_backup(self.populated(db_module), compress=False)
        assert m['file'].endswith('.db') and backup.verify_backup(str(env / 'backups' / m['file']))

    def test_a_tampered_backup_is_rejected(self, db_module, env):
        m = backup.create_backup(self.populated(db_module))
        f = env / 'backups' / m['file']
        data = bytearray(f.read_bytes())
        data[len(data) // 2] ^= 0xFF
        f.write_bytes(bytes(data))
        with pytest.raises(ValueError, match='checksum'):
            backup.verify_backup(str(f))

    def test_a_corrupt_database_never_becomes_a_backup(self, env):
        bad = env / 'bad.db'
        bad.write_bytes(b'SQLite format 3\x00' + os.urandom(4000))
        with pytest.raises(Exception):
            backup.create_backup(str(bad))
        assert list((env / 'backups').glob('*.db*')) == []

    def test_missing_database_and_missing_backup_are_clear_errors(self, env):
        with pytest.raises(FileNotFoundError):
            backup.create_backup(str(env / 'nope.db'))
        with pytest.raises(ValueError):
            backup.verify_backup(str(env / 'nope.db.gz'))

    def test_scheduled_backups_rotate_but_safety_copies_are_kept(self, db_module, env):
        path = self.populated(db_module)
        backup.create_backup(path, label='pre-migration-v1-to-v2', keep=None)
        for _ in range(5):
            backup.create_backup(path, keep=3)
        names = [m['file'] for m in backup.list_backups(path)]
        assert len([n for n in names if n.startswith('quickaid-')]) == 3 and any(n.startswith('pre-migration') for n in names)
        assert len(list((env / 'backups').glob('quickaid-*.db.gz'))) == 3 and len(list((env / 'backups').glob('quickaid-*.json'))) == 3

    def test_restore_brings_back_lost_data_and_saves_what_it_replaced(self, db_module, env):
        path = self.populated(db_module)
        m = backup.create_backup(path)
        with db_module.get_connection() as c:                          # accidental loss (a bug, a disk fault), NOT a deletion by the user
            c.execute('DELETE FROM symptom_analyses')
            c.execute('DELETE FROM users')
        assert db_module.get_user_by_id(1) is None
        result = backup.restore_backup(str(env / 'backups' / m['file']), path)
        assert result['tables']['users'] == 1 and result['previous_database_saved_as'].startswith('pre-restore')
        assert db_module.get_history(1)[0]['symptom_text'] == 'private symptoms'

    def test_restore_never_resurrects_what_people_deleted_after_the_backup(self, db_module, env):
        path = self.populated(db_module)
        keep = db_module.save_symptom_analysis(1, 'delete me later', {'urgency_level': 'low'})
        extra = db_module.create_user('leaver', 'leaver@example.com', 'password123')
        db_module.save_symptom_analysis(extra['id'], 'also private', {'urgency_level': 'low'})
        m = backup.create_backup(path)                                   # a backup that still contains all of it
        from quickaid.core import history_repo
        assert history_repo.delete_entry(1, 'symptom', keep)             # an entry deleted after the backup...
        db_module.delete_user(extra['id'])                               # ...and a whole account
        result = backup.restore_backup(str(env / 'backups' / m['file']), path)
        assert result['deletions_reapplied']['entries'] == 1 and result['deletions_reapplied']['accounts'] == 1
        assert db_module.get_user_by_id(extra['id']) is None
        assert [e['symptom_text'] for e in db_module.get_history(1)] == ['private symptoms']

    def test_a_cleared_history_stays_cleared_after_a_restore(self, db_module, env):
        path = self.populated(db_module)
        m = backup.create_backup(path)
        db_module.delete_history(1)
        backup.restore_backup(str(env / 'backups' / m['file']), path)
        assert db_module.get_history(1) == [] and db_module.get_user_by_id(1)

    def test_the_deletion_log_holds_only_ids_and_times(self, db_module, env):
        from quickaid.core import deletions, history_repo
        i = db_module.save_symptom_analysis(1 if db_module.get_user_by_id(1) else db_module.create_user('x', 'x@example.com', 'password123')['id'], 'secret words', {})
        uid = db_module.get_user_by_id(1)['id']
        history_repo.delete_entry(uid, 'symptom', i)
        text = deletions.log_path(db_module.DB_PATH).read_text()
        assert 'secret words' not in text and oct(deletions.log_path(db_module.DB_PATH).stat().st_mode & 0o777) == '0o600'

    def test_each_database_has_its_own_deletion_log(self, tmp_path):
        from quickaid.core import deletions
        assert deletions.log_path(str(tmp_path / 'a.db')) != deletions.log_path(str(tmp_path / 'b.db'))

    def test_restore_refuses_while_the_database_is_in_use(self, db_module, env):
        path = self.populated(db_module)
        m = backup.create_backup(path)
        holder = sqlite3.connect(path)
        holder.execute('BEGIN EXCLUSIVE')
        try:
            with pytest.raises(RuntimeError, match='in use'):
                backup.restore_backup(str(env / 'backups' / m['file']), path)
        finally:
            holder.rollback()

    def test_restore_refuses_a_bad_backup_and_leaves_the_database_alone(self, db_module, env):
        path = self.populated(db_module)
        m = backup.create_backup(path)
        f = env / 'backups' / m['file']
        f.write_bytes(f.read_bytes()[:-20])
        with pytest.raises(Exception):
            backup.restore_backup(str(f), path)
        assert db_module.get_user_by_id(1)['username'] == 'bk'

    def test_backups_are_compressed_and_smaller(self, db_module, env):
        path = self.populated(db_module)
        plain, packed = backup.create_backup(path, compress=False), backup.create_backup(path)
        assert packed['size_bytes'] < plain['size_bytes']
        with gzip.open(env / 'backups' / packed['file']) as g:
            assert g.read(16) == b'SQLite format 3\x00'

    def test_the_encryption_key_is_not_inside_a_backup(self, db_module, env, monkeypatch):
        monkeypatch.setenv('CHAT_ENCRYPTION_KEY', 'super-secret-key-material-xyz')
        m = backup.create_backup(self.populated(db_module))
        with gzip.open(env / 'backups' / m['file']) as g:
            assert b'super-secret-key-material-xyz' not in g.read()

    def test_command_line(self, db_module, env, capsys):
        path = self.populated(db_module)
        assert backup.main(['create', '--keep', '5']) == 0 and 'created' in capsys.readouterr().out
        assert backup.main(['list']) == 0
        name = backup.list_backups(path)[0]['file']
        assert backup.main(['verify', str(env / 'backups' / name)]) == 0
        assert backup.main(['verify', str(env / 'backups' / 'nope.db')]) == 1
