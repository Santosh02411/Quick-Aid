"""Key rotation, physical deletion, and keeping secrets out of repositories and archives."""

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from quickaid.core import backup, crypto, rekey

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
import package_release  # noqa: E402
import secret_scan  # noqa: E402

# Fake credentials, assembled at runtime so this file itself never contains a secret-shaped string.
FAKE_GOOGLE = 'AIza' + 'SyD4kN8vQ2xPz7LmWc1RtYbHf6JgEs0AaUo'
assert len(FAKE_GOOGLE) == 39
FAKE_OAUTH = 'GOC' + 'SPX-' + 'a1B2c3D4e5F6g7H8i9J0k1'
FAKE_AWS = 'AK' + 'IA' + 'ABCDEFGHIJKLMNOP'


class TestKeyRotation:
    def test_old_data_stays_readable_after_the_key_changes_and_new_writes_use_the_new_key(self, monkeypatch):
        monkeypatch.setenv('CHAT_ENCRYPTION_KEY', 'old-key-material-AAAA')
        old = crypto.encrypt('a secret note')
        monkeypatch.setenv('CHAT_ENCRYPTION_KEY', 'new-key-material-BBBB')
        assert crypto.decrypt(old) == crypto.UNAVAILABLE                            # without the previous key: unreadable
        monkeypatch.setenv('CHAT_ENCRYPTION_KEY_PREVIOUS', 'old-key-material-AAAA')
        assert crypto.decrypt(old) == 'a secret note'
        fresh = crypto.encrypt('written after rotation')
        monkeypatch.delenv('CHAT_ENCRYPTION_KEY_PREVIOUS')
        assert crypto.decrypt(fresh) == 'written after rotation'                    # new writes are under the NEW key alone

    def test_several_previous_keys_are_supported(self, monkeypatch):
        tokens = []
        for k in ('k1-aaaaaaaa', 'k2-bbbbbbbb'):
            monkeypatch.setenv('CHAT_ENCRYPTION_KEY', k)
            tokens.append(crypto.encrypt('v-' + k))
        monkeypatch.setenv('CHAT_ENCRYPTION_KEY', 'k3-cccccccc')
        monkeypatch.setenv('CHAT_ENCRYPTION_KEY_PREVIOUS', 'k1-aaaaaaaa, k2-bbbbbbbb')
        assert [crypto.decrypt(t) for t in tokens] == ['v-k1-aaaaaaaa', 'v-k2-bbbbbbbb']

    def populate(self, db_module, monkeypatch, key):
        monkeypatch.setenv('CHAT_ENCRYPTION_KEY', key)
        u = db_module.create_user('rot', 'rot@example.com', 'password123')
        db_module.set_pending_totp_secret(u['id'], 'JBSWY3DPEHPK3PXP')
        i = db_module.save_symptom_analysis(u['id'], 'private symptoms', {'urgency_level': 'low', 'summary': 's'})
        c = db_module.create_conversation(u['id'], 'private title')
        db_module.add_message(c['id'], 'user', 'private message')
        return u['id'], i

    def test_rekey_moves_everything_to_the_new_key_and_the_old_key_can_then_be_dropped(self, db_module, monkeypatch, tmp_path):
        monkeypatch.setenv('BACKUP_DIR', str(tmp_path / 'b'))
        uid, i = self.populate(db_module, monkeypatch, 'old-key-material-AAAA')
        monkeypatch.setenv('CHAT_ENCRYPTION_KEY', 'new-key-material-BBBB')
        monkeypatch.setenv('CHAT_ENCRYPTION_KEY_PREVIOUS', 'old-key-material-AAAA')
        report = rekey.rekey()
        assert report['updated'] >= 8 and report['undecryptable'] == 0
        assert backup.list_backups(db_module.DB_PATH)[0]['label'] == 'pre-rekey'
        monkeypatch.delenv('CHAT_ENCRYPTION_KEY_PREVIOUS')                         # the old key is now useless
        assert db_module.get_history(uid)[0]['symptom_text'] == 'private symptoms'
        assert db_module.get_user_by_id(uid)['totp_secret'] == 'JBSWY3DPEHPK3PXP'
        assert db_module.get_messages(db_module.list_conversations(uid)[0]['id'])[0]['content'] == 'private message'
        assert rekey.rekey()['updated'] == 0                                       # idempotent

    def test_rekey_refuses_to_destroy_data_it_cannot_read(self, db_module, monkeypatch, tmp_path):
        monkeypatch.setenv('BACKUP_DIR', str(tmp_path / 'b'))
        uid, i = self.populate(db_module, monkeypatch, 'old-key-material-AAAA')
        monkeypatch.setenv('CHAT_ENCRYPTION_KEY', 'new-key-material-BBBB')           # forgot CHAT_ENCRYPTION_KEY_PREVIOUS
        with pytest.raises(RuntimeError, match='CHAT_ENCRYPTION_KEY_PREVIOUS'):
            rekey.rekey()
        monkeypatch.setenv('CHAT_ENCRYPTION_KEY_PREVIOUS', 'old-key-material-AAAA')
        assert db_module.get_history(uid)[0]['symptom_text'] == 'private symptoms'  # nothing was changed by the refused attempt

    def test_dry_run_changes_nothing(self, db_module, monkeypatch):
        uid, i = self.populate(db_module, monkeypatch, 'old-key-material-AAAA')
        monkeypatch.setenv('CHAT_ENCRYPTION_KEY', 'new-key-material-BBBB')
        monkeypatch.setenv('CHAT_ENCRYPTION_KEY_PREVIOUS', 'old-key-material-AAAA')
        def raw():
            with db_module.get_connection() as c:
                return c.execute('SELECT symptom_text FROM symptom_analyses').fetchone()[0]
        before = raw()
        assert rekey.rekey(dry_run=True)['would_update'] >= 8
        assert raw() == before

    def test_every_encrypted_column_is_covered_by_rekey(self):
        """A column holding crypto.encrypt() output that rekey does not know about would be silently abandoned on rotation."""
        import re
        src = (ROOT / 'quickaid/core/database.py').read_text() + (ROOT / 'quickaid/core/history_repo.py').read_text()
        for table, cols in rekey.ENCRYPTED_COLUMNS.items():
            assert cols
        assert {'notes', 'analysis_json', 'title', 'content', 'state', 'totp_secret', 'symptom_text', 'original_filename'} <= {c for cols in rekey.ENCRYPTED_COLUMNS.values() for c in cols}
        assert 'crypto.encrypt' in src and re.search(r'ENCRYPTED_COLUMNS', (ROOT / 'quickaid/core/rekey.py').read_text())


class TestPhysicalDeletion:
    MARK = 'zzdeletedmarkerzz'

    def disk_bytes(self, path):
        data = Path(path).read_bytes()
        wal = Path(str(path) + '-wal')
        return data + (wal.read_bytes() if wal.exists() else b'')

    def test_secure_delete_zeroes_freed_content(self, db_module):
        import sqlite3
        u = db_module.create_user(self.MARK, 'm@example.com', 'password123')
        with db_module.get_connection() as c:                  # a plain DELETE, no vacuum: relies on secure_delete alone
            c.execute('DELETE FROM users WHERE id = ?', (u['id'],))
        db_module.scrub()
        assert self.MARK.encode() not in self.disk_bytes(db_module.DB_PATH)

    def test_control_without_secure_delete_the_text_would_survive(self, db_module):
        """Proves the test above can fail: with the protection off, deleted text remains readable in the file."""
        import sqlite3
        u = db_module.create_user(self.MARK + 'b', 'm2@example.com', 'password123')
        conn = sqlite3.connect(db_module.DB_PATH)
        conn.execute('PRAGMA secure_delete = OFF')
        conn.execute('PRAGMA journal_mode = DELETE')
        conn.execute('DELETE FROM users WHERE id = ?', (u['id'],))
        conn.commit()
        conn.close()
        assert (self.MARK + 'b').encode() in self.disk_bytes(db_module.DB_PATH)

    def test_deleting_an_account_leaves_no_trace_in_the_file(self, db_module):
        u = db_module.create_user(self.MARK + 'c', 'trace@example.com', 'password123')
        db_module.delete_user(u['id'])
        raw = self.disk_bytes(db_module.DB_PATH)
        assert (self.MARK + 'c').encode() not in raw and b'trace@example.com' not in raw

    def test_secure_remove_overwrites_before_unlinking(self, tmp_path, monkeypatch):
        from quickaid.core import secure_files
        f = tmp_path / 'photo.bin'
        f.write_bytes(b'SENSITIVE-PIXELS' * 5000)
        seen = {}
        real = os.remove
        monkeypatch.setattr(secure_files.os, 'remove', lambda p: (seen.setdefault('content', Path(p).read_bytes()), real(p)))
        assert secure_files.secure_remove(str(f)) is True
        assert seen['content'] == b'\x00' * (16 * 5000) and not f.exists()

    def test_secure_remove_of_a_missing_file_is_harmless(self, tmp_path):
        from quickaid.core import secure_files
        assert secure_files.secure_remove(str(tmp_path / 'nope')) is False

    def test_uploads_and_leftovers_are_removed_with_secure_remove(self, app, monkeypatch, tmp_path):
        import io
        import numpy as np
        from PIL import Image
        from quickaid.core import secure_files
        calls = []
        real = secure_files.secure_remove
        monkeypatch.setattr(secure_files, 'secure_remove', lambda p: (calls.append(p), real(p))[1])
        c = app.test_client()
        c.post('/register', data={'username': 'ann', 'email': 'a@example.com', 'password': 'orange-kettle-42', 'confirm_password': 'orange-kettle-42'})
        buf = io.BytesIO()
        Image.fromarray(np.clip(np.random.default_rng(1).integers(-25, 25, (300, 300, 1)) + np.array([215, 165, 145]), 0, 255).astype('uint8')).save(buf, 'PNG')
        c.post('/upload', data={'file': (io.BytesIO(buf.getvalue()), 'a.png')}, content_type='multipart/form-data')
        assert len(calls) == 1 and not os.path.exists(calls[0])


class TestSecretScanner:
    def test_it_finds_real_shaped_credentials_and_never_prints_them_in_full(self, capsys):
        text = f'key = "{FAKE_GOOGLE}"\noauth = {FAKE_OAUTH}\naws = {FAKE_AWS}\n' + '-----BEGIN ' + 'RSA PRIVATE KEY-----\n'
        found = secret_scan.scan_text('app/config.py', text)
        assert {f[1] for f in found} == {'Google API key', 'Google OAuth client secret', 'AWS access key id', 'Private key block'}
        assert secret_scan.report(found, 't') == 1
        out = capsys.readouterr().out
        assert FAKE_GOOGLE not in out and FAKE_OAUTH not in out and 'AIza… (' in out

    def test_env_files_with_real_values_are_flagged_but_examples_are_not(self):
        real = 'SECRET_KEY=' + 'f3a9' * 16 + '\nSMTP_PASSWORD=hunter2hunter2hunter2\nGEMINI_API_KEY=' + FAKE_GOOGLE + '\n'
        assert len(secret_scan.scan_text('.env', real)) == 4                                   # key shape matched twice + two real-looking assignments
        placeholder = 'SECRET_KEY=change_this_to_a_random_secret_key\nGEMINI_API_KEY=your_gemini_api_key_here\nSMTP_PASSWORD=\nPORT=5000\nGOOGLE_CLIENT_SECRET=GOCSPX-xxxxxxxxxxxxxxxxxxxx\n'
        assert secret_scan.scan_text('.env.example', placeholder) == []

    def test_obvious_placeholders_and_marked_lines_are_ignored(self):
        assert secret_scan.scan_text('x.py', 'k = "AIza' + 'x' * 35 + '"') == []
        assert secret_scan.scan_text('x.py', f'k = "{FAKE_GOOGLE}"  # secret-scan: allow') == []

    @pytest.mark.parametrize('name', ['.env', 'data/quickaid.db', 'data/.data_key', 'server.pem', 'id_rsa', 'backups/x.json', 'a/b/.git/config',
                                      'quickaid.db.deletions.log', 'credentials.json', 'x.db-wal'])
    def test_forbidden_files(self, name):
        assert secret_scan.forbidden_name(name)

    @pytest.mark.parametrize('name', ['.env.example', 'quickaid/app.py', 'README.md', 'tests/test_x.py', 'docs/DATABASE.md', 'static/css/app.css'])
    def test_allowed_files(self, name):
        assert not secret_scan.forbidden_name(name)

    def test_the_repository_itself_is_clean(self):
        assert secret_scan.scan_worktree() == []

    def test_the_ignore_rules_cover_every_secret_file_type(self):
        gi = (ROOT / '.gitignore').read_text()
        for needle in ('.env', '*.db', '.data_key', 'backups/', '*.pem', 'data/', 'uploads/', 'logs/', '*.deletions.log'):
            assert needle in gi, needle
        di = (ROOT / '.dockerignore').read_text()
        for needle in ('.env', '.git/', 'tests/'):
            assert needle in di, needle

    def test_the_example_env_documents_every_setting_without_a_real_value(self):
        text = (ROOT / '.env.example').read_text()
        assert secret_scan.scan_text('.env.example', text) == []
        for key in ('SECRET_KEY', 'GEMINI_API_KEY', 'SMTP_PASSWORD', 'GOOGLE_CLIENT_SECRET', 'CHAT_ENCRYPTION_KEY', 'ADMIN_USERNAMES'):
            assert key in text

    def git_repo(self, tmp_path, files):
        def run(*a):
            subprocess.run(['git', *a], cwd=tmp_path, check=True, capture_output=True)
        run('init', '-q')
        run('config', 'user.email', 't@example.com')
        run('config', 'user.name', 't')
        for name, content in files.items():
            p = tmp_path / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
        run('add', '-A')
        run('commit', '-q', '-m', 'init')
        return run

    def test_history_scanning_finds_a_secret_that_was_committed_then_deleted(self, tmp_path, monkeypatch):
        run = self.git_repo(tmp_path, {'.env': f'GEMINI_API_KEY={FAKE_GOOGLE}\n', 'app.py': 'print(1)\n'})
        run('rm', '-q', '--cached', '.env')
        (tmp_path / '.env').unlink()
        run('commit', '-q', '-m', 'remove env')
        monkeypatch.setattr(secret_scan, 'ROOT', tmp_path)
        found = secret_scan.scan_history()
        assert any('forbidden file was committed' in f[1] for f in found) and any(f[1] == 'Google API key' for f in found)
        assert secret_scan.scan_worktree() == []                                  # the tree is clean now; only history shows the leak

    def test_staged_scan_blocks_a_secret_before_the_commit(self, tmp_path, monkeypatch):
        run = self.git_repo(tmp_path, {'app.py': 'print(1)\n'})
        (tmp_path / 'config.py').write_text(f'KEY = "{FAKE_GOOGLE}"\n')
        run('add', 'config.py')
        monkeypatch.setattr(secret_scan, 'ROOT', tmp_path)
        assert [f[1] for f in secret_scan.scan_staged()] == ['Google API key']

    def test_the_release_packager_never_includes_env_or_git_and_refuses_secrets(self, tmp_path, monkeypatch):
        self.git_repo(tmp_path, {'app.py': 'print(1)\n', '.gitignore': '.env\n', '.env.example': 'SECRET_KEY=change_this\n'})
        (tmp_path / '.env').write_text('SECRET_KEY=' + 'ab12' * 16 + '\n')             # a real local secret that git ignores
        monkeypatch.setattr(package_release, 'ROOT', tmp_path)
        monkeypatch.setattr(secret_scan, 'ROOT', tmp_path)
        out = tmp_path / 'out'
        out.mkdir()
        assert package_release.main(['--version', '9', '--out', str(out)]) == 0
        names = zipfile.ZipFile(out / 'Quick-Aid-v9.zip').namelist()
        assert 'Quick-Aid/app.py' in names and 'Quick-Aid/.env.example' in names
        assert not any(n.endswith('/.env') or '/.git/' in n for n in names)
        # an untracked file that WOULD be shipped and contains a key must stop the release
        (tmp_path / 'oops.py').write_text(f'K = "{FAKE_GOOGLE}"\n')
        assert package_release.main(['--version', '10', '--out', str(out)]) == 1
        assert not (out / 'Quick-Aid-v10.zip').exists() and not list(out.glob('*.partial'))

    def test_an_unzipped_copy_without_git_is_still_scanned_but_local_runtime_files_are_not_flagged(self, tmp_path, monkeypatch):
        (tmp_path / 'app.py').write_text(f'KEY = "{FAKE_GOOGLE}"\n')
        (tmp_path / '.env').write_text('SECRET_KEY=' + 'ab12' * 16 + '\n')            # the person's own local file: skipped, as git would ignore it
        (tmp_path / 'data').mkdir()
        (tmp_path / 'data' / 'quickaid.db').write_bytes(b'x')
        (tmp_path / '.env.example').write_text('SECRET_KEY=change_this\n')
        monkeypatch.setattr(secret_scan, 'ROOT', tmp_path)
        assert [f[1] for f in secret_scan.scan_worktree()] == ['Google API key']        # a secret inside source IS found, even with no git

    def test_archive_scanning(self, tmp_path):
        z = tmp_path / 'x.zip'
        with zipfile.ZipFile(z, 'w') as f:
            f.writestr('Q/app.py', 'ok\n')
            f.writestr('Q/.env', 'SECRET_KEY=' + 'cd34' * 16)
            f.writestr('Q/.git/config', '[core]')
        found = secret_scan.scan_archive(str(z))
        assert any('.env' in f[0] for f in found) and any('.git' in f[0] for f in found)
