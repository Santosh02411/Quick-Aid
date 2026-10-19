"""The internal evaluation dashboard: who can open it, and what its API returns."""

import json
import shutil
from datetime import datetime, timezone

import pytest

from quickaid.evaluation import datasets, runs

ROUTES = ['/admin/evaluation', '/admin/api/overview', '/admin/api/telemetry', '/admin/api/runs/20261016-100000-offline-aaaa',
          '/admin/api/compare?a=x&b=y']


@pytest.fixture()
def admin(monkeypatch, registered_user):
    monkeypatch.setenv('ADMIN_USERNAMES', registered_user['username'])
    return registered_user


@pytest.fixture()
def eval_sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, 'RUNS_DIR', tmp_path / 'runs')
    return tmp_path


def seed_run(run_id, recall=1.0, **kw):
    rec = {'run_id': run_id, 'created': '2026-10-16T10:00:00', 'mode': 'offline', 'label': 'seed', 'model': {'name': 'rules', 'synthetic': False},
           'prompts': {'fingerprint': 'f1', 'parts': {'symptom': 'a1'}}, 'engine_version': 2, 'sets': ['symptom_cases'],
           'datasets': {'symptom_cases': {'version': '1.0.0', 'sha256': 'h'}}, 'metrics': {'emergency_recall': recall}, 'outcomes': [],
           'threshold_failures': [], 'gated': True}
    rec.update(kw)
    runs.save_run(rec)
    return rec


class TestAccessControl:
    def test_anonymous_users_cannot_reach_any_admin_route(self, client):
        for url in ROUTES:
            assert client.get(url).status_code in (302, 401), url
        assert client.post('/admin/api/run', json={}).status_code in (302, 401, 400)

    def test_signed_in_non_admins_get_404_so_the_page_is_not_revealed(self, client, registered_user, monkeypatch):
        monkeypatch.delenv('ADMIN_USERNAMES', raising=False)
        for url in ROUTES:
            assert client.get(url).status_code == 404, url
        assert client.post('/admin/api/run', json={'model': 'offline'}).status_code == 404

    def test_nobody_is_admin_when_the_setting_is_blank(self, client, registered_user, monkeypatch):
        monkeypatch.setenv('ADMIN_USERNAMES', '')
        assert client.get('/admin/evaluation').status_code == 404

    def test_admins_are_matched_case_insensitively_and_from_a_list(self, client, registered_user, monkeypatch):
        monkeypatch.setenv('ADMIN_USERNAMES', f" someone , {registered_user['username'].upper()} ")
        assert client.get('/admin/evaluation').status_code == 200

    def test_a_substring_of_an_admin_name_is_not_an_admin(self, client, registered_user, monkeypatch):
        monkeypatch.setenv('ADMIN_USERNAMES', registered_user['username'] + 'x')
        assert client.get('/admin/evaluation').status_code == 404

    def test_the_navigation_link_is_only_shown_to_admins(self, client, registered_user, monkeypatch):
        monkeypatch.delenv('ADMIN_USERNAMES', raising=False)
        assert '/admin/evaluation' not in client.get('/').data.decode()
        monkeypatch.setenv('ADMIN_USERNAMES', registered_user['username'])
        assert '/admin/evaluation' in client.get('/').data.decode()


class TestDashboardApi:
    def test_page_renders_with_its_script_and_no_inline_handlers(self, client, admin):
        html = client.get('/admin/evaluation').data.decode()
        assert 'id="adminApp"' in html and '/static/js/admin.js' in html and 'onclick=' not in html

    def test_overview_describes_configuration_datasets_and_runs(self, client, admin, eval_sandbox):
        seed_run('20261016-100000-offline-aaaa')
        d = client.get('/admin/api/overview').get_json()
        assert {'model', 'engine_version', 'prompts', 'price_configured'} <= set(d['config'])
        assert len(d['config']['prompts']['fingerprint']) == 8 and d['dataset_problems'] == []
        assert {x['name'] for x in d['datasets']} >= {'symptom_cases', 'symptom_holdout2'} and d['runs'][0]['run_id'] == '20261016-100000-offline-aaaa'

    def test_prices_are_reported_as_configured_only_when_set(self, client, admin, monkeypatch):
        monkeypatch.delenv('GEMINI_PRICE_INPUT_PER_M', raising=False)
        assert client.get('/admin/api/overview').get_json()['config']['price_configured'] is False
        monkeypatch.setenv('GEMINI_PRICE_INPUT_PER_M', '1')
        monkeypatch.setenv('GEMINI_PRICE_OUTPUT_PER_M', '2')
        assert client.get('/admin/api/overview').get_json()['config']['price_configured'] is True

    def test_telemetry_summarises_recent_calls_and_respects_the_window(self, client, admin, db_module):
        now = datetime.now(timezone.utc).isoformat(timespec='seconds')
        old = '2020-01-01T00:00:00+00:00'
        for i, ts in enumerate([now, now, old]):
            db_module.record_ai_call({'id': f'c{i}', 'analysis_id': f'a{i}', 'ts': ts, 'kind': 'symptom', 'model': 'm', 'prompt_version': 'p',
                                      'ok': i != 1, 'valid': True if i == 0 else None, 'latency_ms': 100 * (i + 1), 'error_type': None if i != 1 else 'TimeoutError',
                                      'input_tokens': 10, 'output_tokens': 5, 'cost_usd': None})
            db_module.record_ai_analysis({'id': f'a{i}', 'ts': ts, 'kind': 'symptom', 'mode': 'ai', 'ai_attempted': True, 'calls': 1, 'latency_ms': 100})
        d = client.get('/admin/api/telemetry?days=7').get_json()
        assert d['calls'] == 2 and d['api_failure_rate'] == 0.5 and d['window_days'] == 7 and d['by_model'].keys() == {'m'}

    @pytest.mark.parametrize('raw,expected', [('0', 1), ('-5', 1), ('9999', 90), ('abc', 7), ('', 7)])
    def test_window_is_clamped(self, client, admin, raw, expected):
        assert client.get(f'/admin/api/telemetry?days={raw}').get_json()['window_days'] == expected

    def test_run_detail_and_unknown_runs(self, client, admin, eval_sandbox):
        seed_run('20261016-100000-offline-aaaa')
        assert client.get('/admin/api/runs/20261016-100000-offline-aaaa').get_json()['label'] == 'seed'
        assert client.get('/admin/api/runs/20261016-100000-offline-zzzz').status_code == 404

    def test_path_traversal_through_the_run_id_is_refused(self, client, admin, eval_sandbox):
        (eval_sandbox / 'secret.json').write_text('{"run_id": "secret"}')
        for bad in ('..%2fsecret', '%2e%2e/secret', '....//secret', 'a%00b'):
            assert client.get(f'/admin/api/runs/{bad}').status_code == 404

    def test_compare_endpoint(self, client, admin, eval_sandbox):
        seed_run('20261016-100000-offline-aaaa', recall=0.9)
        seed_run('20261016-110000-offline-bbbb', recall=1.0, created='2026-10-16T11:00:00')
        c = client.get('/admin/api/compare?a=20261016-100000-offline-aaaa&b=20261016-110000-offline-bbbb').get_json()
        assert c['comparable'] and {r['metric']: r['verdict'] for r in c['metrics']}['emergency_recall'] == 'better'
        assert client.get('/admin/api/compare?a=nope&b=nope2').status_code == 404
        assert client.get('/admin/api/compare').status_code == 404


class TestStartingRuns:
    def test_an_offline_run_can_be_started_and_is_saved(self, client, admin, eval_sandbox):
        r = client.post('/admin/api/run', json={'model': 'offline', 'sets': ['symptom_cases'], 'label': '  from   the dashboard '}).get_json()
        assert r['success'] and r['threshold_failures'] == [] and runs.load_run(r['run_id'])['label'] == 'from the dashboard'

    def test_mock_runs_are_allowed_and_marked_synthetic(self, client, admin, eval_sandbox):
        r = client.post('/admin/api/run', json={'model': 'mock-flaky'}).get_json()
        assert r['success'] and runs.load_run(r['run_id'])['model']['synthetic'] is True

    @pytest.mark.parametrize('model', ['live', 'replay', 'banana', '', None, 5])
    def test_live_and_unknown_models_cannot_be_started_from_the_web(self, client, admin, eval_sandbox, model):
        resp = client.post('/admin/api/run', json={'model': model})
        assert resp.status_code == 400 and 'command line' in resp.get_json()['error']

    def test_unknown_datasets_are_dropped_and_at_least_one_is_required(self, client, admin, eval_sandbox):
        assert client.post('/admin/api/run', json={'model': 'offline', 'sets': ['../etc/passwd', 'nope']}).status_code == 400
        r = client.post('/admin/api/run', json={'model': 'offline', 'sets': ['nope', 'symptom_cases']}).get_json()
        assert runs.load_run(r['run_id'])['sets'] == ['symptom_cases']

    def test_label_length_is_capped(self, client, admin, eval_sandbox):
        r = client.post('/admin/api/run', json={'model': 'offline', 'label': 'x' * 500}).get_json()
        assert len(runs.load_run(r['run_id'])['label']) == 120

    def test_starting_runs_is_rate_limited(self, client, admin, eval_sandbox):
        codes = [client.post('/admin/api/run', json={'model': 'mock-rules'}).status_code for _ in range(5)]
        assert 429 in codes and codes[0] == 200


class TestDashboardAssets:
    def test_script_builds_the_page_without_innerhtml_or_eval(self, client):
        js = client.get('/static/js/admin.js').data.decode()
        assert 'innerHTML' not in js and 'eval(' not in js and 'document.write' not in js

    def test_dashboard_is_still_covered_by_the_security_policy(self, client, admin):
        csp = client.get('/admin/evaluation').headers['Content-Security-Policy']
        assert "script-src 'self'" in csp and 'unsafe-eval' not in csp and "'unsafe-inline'" not in csp.split('script-src')[1].split(';')[0]

    def test_writes_need_a_csrf_token_in_production_mode(self, app):
        from quickaid import app as app_module        # the test fixture disables CSRF checking, but no admin route may be exempted from it
        assert 'admin_start_run' not in {getattr(v, '__name__', '') for v in getattr(app_module.csrf, '_exempt_views', set())}
