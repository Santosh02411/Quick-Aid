"""Data-retention policy and the maintenance sweep that enforces it."""

import os
import time
from datetime import datetime, timedelta, timezone

import pytest

from quickaid.core import database as db
from quickaid.core import history_repo
from quickaid.services.retention import RetentionService


def iso(days_ago):
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


def add(user_id, kind, days_ago, text='x'):
    if kind == 'symptom':
        i = db.save_symptom_analysis(user_id, text, {'urgency_level': 'low'})
        table = 'symptom_analyses'
    else:
        i = db.save_image_analysis(user_id, text, {'urgency': 'low'})
        table = 'image_analyses'
    with db.get_connection() as c:
        c.execute(f'UPDATE {table} SET created_at = ? WHERE id = ?', (iso(days_ago), i))
    return i


def count(user_id, table='symptom_analyses'):
    with db.get_connection() as c:
        return c.execute(f'SELECT COUNT(*) FROM {table} WHERE user_id = ?', (user_id,)).fetchone()[0]


@pytest.fixture()
def users(db_module, monkeypatch):
    for k in ('DEFAULT_RETENTION_DAYS', 'HISTORY_MAX_RETENTION_DAYS'):
        monkeypatch.delenv(k, raising=False)
    return db_module.create_user('ann', 'ann@example.com', 'password123')['id'], db_module.create_user('bo', 'bo@example.com', 'password123')['id']


class TestPolicyMath:
    @pytest.mark.parametrize('default,cap,chosen,expected', [
        (365, 0, None, 365), (365, 0, 30, 30), (365, 0, 0, 0), (0, 0, None, 0),          # no ceiling: choice or default
        (365, 90, None, 90), (365, 90, 30, 30), (365, 90, 0, 90), (365, 90, 365, 90),    # ceiling wins over everything
        (0, 180, None, 180), (30, 180, 90, 90)])
    def test_effective_days(self, monkeypatch, default, cap, chosen, expected):
        monkeypatch.setenv('DEFAULT_RETENTION_DAYS', str(default))
        monkeypatch.setenv('HISTORY_MAX_RETENTION_DAYS', str(cap))
        assert RetentionService().effective_days(chosen) == expected

    def test_choices_respect_the_ceiling(self, monkeypatch):
        assert [c['days'] for c in RetentionService().choices()] == [30, 90, 365, 0]
        monkeypatch.setenv('HISTORY_MAX_RETENTION_DAYS', '90')
        assert [c['days'] for c in RetentionService().choices()] == [30, 90]

    def test_garbage_environment_values_fall_back_to_safe_defaults(self, monkeypatch):
        monkeypatch.setenv('DEFAULT_RETENTION_DAYS', 'lots')
        monkeypatch.setenv('HISTORY_MAX_RETENTION_DAYS', '-')
        assert RetentionService().effective_days(None) == 365

    def test_the_default_is_one_year(self, monkeypatch):
        monkeypatch.delenv('DEFAULT_RETENTION_DAYS', raising=False)
        assert RetentionService().policy()['saved_history']['default_days'] == 365


class TestRetentionApi:
    def test_read_and_change_your_choice(self, client, registered_user):
        r = client.get('/api/privacy/retention').get_json()['retention']
        assert r['using_default'] and r['effective_days'] == 365 and [c['days'] for c in r['choices']] == [30, 90, 365, 0]
        r = client.put('/api/privacy/retention', json={'days': 90}).get_json()['retention']
        assert r['chosen_days'] == 90 and r['effective_days'] == 90 and not r['using_default']
        assert client.put('/api/privacy/retention', json={'days': None}).get_json()['retention']['using_default']
        assert client.put('/api/privacy/retention', json={'days': 0}).get_json()['retention']['effective_days'] == 0

    @pytest.mark.parametrize('body', [{'days': 7}, {'days': 'abc'}, {'days': -1}, {}, {'days': 1.5}])
    def test_invalid_choices_are_rejected(self, client, registered_user, body):
        r = client.put('/api/privacy/retention', json=body)
        assert r.status_code == 400 and r.get_json()['code'] == 'validation_error'

    def test_keep_forever_is_refused_when_the_operator_sets_a_ceiling(self, client, registered_user, monkeypatch):
        monkeypatch.setenv('HISTORY_MAX_RETENTION_DAYS', '90')
        r = client.put('/api/privacy/retention', json={'days': 0})
        assert r.status_code == 400 and r.get_json()['code'] == 'invalid_retention'

    def test_the_account_page_shows_the_choice(self, client, registered_user):
        html = client.get('/account').data.decode()
        assert 'Keep my saved checks and chats for' in html and 'Delete all my history' in html and 'Download everything' in html


class TestSweep:
    def test_default_policy_deletes_old_entries_and_keeps_recent_ones(self, users):
        ann, _ = users
        old, recent = add(ann, 'symptom', 400), add(ann, 'symptom', 100)
        add(ann, 'symptom', 5)
        report = RetentionService().sweep()
        assert count(ann) == 2 and report['analyses_purged'] == 1
        assert history_repo.get_entry(ann, 'symptom', old) is None and history_repo.get_entry(ann, 'symptom', recent)

    def test_each_person_gets_their_own_period(self, users):
        ann, bo = users
        history_repo.set_retention_days(ann, 30)
        history_repo.set_retention_days(bo, 0)
        for u in (ann, bo):
            add(u, 'symptom', 40)
            add(u, 'image', 40)
            add(u, 'symptom', 5)
        RetentionService().sweep()
        assert count(ann) == 1 and count(ann, 'image_analyses') == 0               # ann chose 30 days
        assert count(bo) == 2 and count(bo, 'image_analyses') == 1                 # bo chose "until I delete it"

    def test_the_operator_ceiling_overrides_keep_forever(self, users, monkeypatch):
        ann, _ = users
        history_repo.set_retention_days(ann, 0)
        add(ann, 'symptom', 200)
        add(ann, 'symptom', 10)
        monkeypatch.setenv('HISTORY_MAX_RETENTION_DAYS', '90')
        RetentionService().sweep()
        assert count(ann) == 1

    def test_chats_about_expired_entries_go_with_them(self, users):
        ann, bo = users
        old = add(ann, 'symptom', 400)
        new = add(ann, 'symptom', 3)
        db.create_conversation(ann, 'about old', 'symptom', old)
        keep = db.create_conversation(ann, 'about new', 'symptom', new)
        RetentionService().sweep()
        assert [c['id'] for c in db.list_conversations(ann)] == [keep['id']]

    def test_general_chats_expire_by_last_activity(self, users):
        ann, _ = users
        stale, active = db.create_conversation(ann, 'stale'), db.create_conversation(ann, 'active')
        with db.get_connection() as c:
            c.execute('UPDATE chat_conversations SET updated_at = ? WHERE id = ?', (iso(500), stale['id']))
        RetentionService().sweep()
        assert [c['id'] for c in db.list_conversations(ann)] == [active['id']]

    def test_nothing_recent_is_ever_touched(self, users):
        ann, bo = users
        for u in (ann, bo):
            for d in (0, 1, 30, 200):
                add(u, 'symptom', d)
        assert RetentionService().sweep()['analyses_purged'] == 0 and count(ann) == 4

    def test_expired_interviews_and_tokens_are_purged(self, users):
        ann, _ = users
        old, fresh = db.create_interview(ann, {'a': 1}), db.create_interview(ann, {'b': 2})
        with db.get_connection() as c:
            c.execute('UPDATE symptom_interviews SET updated_at = ? WHERE id = ?', (iso(3), old))
        live = db.create_token(ann, 'password_reset', 3600)
        dead = db.create_token(ann, 'email_verify', 3600)
        with db.get_connection() as c:
            c.execute("UPDATE user_tokens SET expires_at = ? WHERE purpose = 'email_verify'", (iso(1),))
        r = RetentionService().sweep()
        assert r['interviews_purged'] == 1 and r['tokens_purged'] == 1
        assert db.get_interview(ann, fresh) and db.get_valid_token(live, 'password_reset') and db.get_valid_token(dead, 'email_verify') is None

    def test_old_telemetry_is_purged(self, users, monkeypatch):
        monkeypatch.setenv('TELEMETRY_RETENTION_DAYS', '30')
        db.record_ai_call({'id': 'old', 'ts': iso(60), 'kind': 'symptom', 'ok': True})
        db.record_ai_call({'id': 'new', 'ts': iso(1), 'kind': 'symptom', 'ok': True})
        assert RetentionService().sweep()['telemetry_purged'] == 1


class TestTemporaryImages:
    def test_leftover_photos_are_deleted_and_everything_else_is_left_alone(self, tmp_path):
        old_name, new_name = 'a' * 32 + '.jpg', 'b' * 32 + '.png'
        for n in (old_name, new_name, 'notes.txt', '.gitkeep'):
            (tmp_path / n).write_bytes(b'x')
        long_ago = time.time() - 3 * 3600
        for n in (old_name, 'notes.txt'):
            os.utime(tmp_path / n, (long_ago, long_ago))
        (tmp_path / 'sub').mkdir()
        assert RetentionService().clean_uploads(str(tmp_path), 30) == 1
        assert not (tmp_path / old_name).exists()
        assert (tmp_path / new_name).exists() and (tmp_path / 'notes.txt').exists() and (tmp_path / '.gitkeep').exists()

    def test_symlinks_are_never_followed_or_removed(self, tmp_path):
        target = tmp_path / 'precious.txt'
        target.write_text('keep me')
        link = tmp_path / ('c' * 32 + '.jpg')
        link.symlink_to(target)
        os.utime(target, (0, 0))
        assert RetentionService().clean_uploads(str(tmp_path), 30) == 0 and target.exists() and link.is_symlink()

    def test_missing_folder_is_not_an_error(self, tmp_path):
        assert RetentionService().clean_uploads(str(tmp_path / 'nope'), 30) == 0 and RetentionService().clean_uploads('', 30) == 0

    def test_the_sweep_reports_removed_uploads(self, users, tmp_path):
        f = tmp_path / ('d' * 32 + '.webp')
        f.write_bytes(b'x')
        os.utime(f, (0, 0))
        assert RetentionService().sweep(str(tmp_path))['uploads_removed'] == 1

    def test_a_photo_is_gone_after_analysis_and_nothing_is_left_behind(self, client, registered_user, app):
        import io
        import numpy as np
        from PIL import Image
        arr = np.clip(np.random.default_rng(1).integers(-25, 25, (300, 300, 1)) + np.array([215, 165, 145]), 0, 255).astype('uint8')
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, 'PNG')
        r = client.post('/upload', data={'file': (io.BytesIO(buf.getvalue()), 'a.png')}, content_type='multipart/form-data')
        assert r.status_code == 200 and [n for n in os.listdir(app.config['UPLOAD_FOLDER']) if not n.startswith('.')] == []

    def test_the_first_request_triggers_the_daily_sweep(self, client, app):
        stale = os.path.join(app.config['UPLOAD_FOLDER'], 'e' * 32 + '.jpg')       # left behind by a "crash" before any request is made
        open(stale, 'wb').write(b'x')
        os.utime(stale, (0, 0))
        client.get('/')
        assert not os.path.exists(stale)


class TestScheduling:
    def test_only_one_caller_per_interval_wins(self, db_module):
        assert history_repo.claim_maintenance('t', 3600) is True
        assert history_repo.claim_maintenance('t', 3600) is False
        assert history_repo.claim_maintenance('t', 0) is True
        assert history_repo.last_maintenance('t')

    def test_run_if_due_is_throttled_and_force_overrides(self, db_module):
        svc = RetentionService()
        assert svc.run_if_due() is not None
        assert svc.run_if_due() is None                      # throttled in this process
        assert svc.run_if_due(force=True) is not None

    def test_a_second_worker_does_not_repeat_the_sweep(self, db_module):
        assert RetentionService().run_if_due() is not None
        assert RetentionService().run_if_due() is None       # a different process: the shared claim says it already ran

    def test_a_failing_sweep_never_breaks_the_request(self, db_module, monkeypatch):
        svc = RetentionService()
        monkeypatch.setattr(svc, 'sweep', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('disk full')))
        assert svc.run_if_due(force=True) is None

    def test_command_line_runs_the_sweep(self, db_module, capsys):
        from quickaid.services import retention
        assert retention.main() == 0 and 'uploads_removed' in capsys.readouterr().out


class TestPrivacyPage:
    def test_is_public_and_states_the_real_policy(self, app, monkeypatch):
        monkeypatch.setenv('DEFAULT_RETENTION_DAYS', '180')
        html = app.test_client().get('/privacy').data.decode()
        assert '180 days by default' in html and 'deleted immediately' in html and 'encrypted' in html and 'Gemini' in html
        assert 'Backups' in html and 'rotate out' in html                      # honest about backups
        assert 'delete' in html.lower() and 'Export' in html

    def test_reflects_an_operator_ceiling(self, app, monkeypatch):
        monkeypatch.setenv('HISTORY_MAX_RETENTION_DAYS', '90')
        assert 'never keeps them longer than 90 days' in app.test_client().get('/privacy').data.decode()

    def test_policy_endpoint_is_public(self, app):
        p = app.test_client().get('/api/privacy/policy').get_json()['policy']
        assert p['uploaded_photos']['kept'].startswith('Deleted immediately') and p['saved_history']['default_days'] == 365
