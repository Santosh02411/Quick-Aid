"""Health/readiness, Prometheus metrics, privacy-safe error reporting, and the deployment configuration."""

import json
import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from quickaid.observability import errors, readiness

ROOT = Path(__file__).resolve().parent.parent
PW = 'orange-kettle-42'


class TestReadiness:
    def test_a_healthy_instance_is_ready_and_reports_each_check(self, client):
        j = client.get('/ready').get_json()
        assert client.get('/ready').status_code == 200 and j['status'] == 'ready' and j['success'] is True
        assert set(j['checks']) == {'database_writable', 'schema_current', 'uploads_writable', 'disk_space'} and all(c['ok'] for c in j['checks'].values())
        assert j['checks']['schema_current']['pending'] == []

    def test_liveness_and_readiness_are_different_questions(self, client, monkeypatch):
        monkeypatch.setattr(readiness.database, 'transaction', MagicMock(side_effect=RuntimeError('read-only file system')))
        assert client.get('/health').status_code == 200                      # the process is alive...
        r = client.get('/ready')
        assert r.status_code == 503 and r.get_json()['checks']['database_writable']['ok'] is False      # ...but should not receive traffic

    def test_pending_migrations_make_it_not_ready(self, client, monkeypatch):
        monkeypatch.setattr(readiness.migrations, 'status', lambda path: {'current': 5, 'latest': 6, 'pending': [6], 'modified': []})
        r = client.get('/ready')
        assert r.status_code == 503 and r.get_json()['checks']['schema_current']['pending'] == [6]

    def test_low_disk_space_makes_it_not_ready(self, client, monkeypatch):
        monkeypatch.setenv('MIN_FREE_DISK_MB', '999999999')
        r = client.get('/ready')
        assert r.status_code == 503 and r.get_json()['checks']['disk_space']['ok'] is False

    def test_an_unwritable_upload_folder_makes_it_not_ready(self, client, app, monkeypatch, tmp_path):
        blocker = tmp_path / 'afile'
        blocker.write_text('x')
        app.config['UPLOAD_FOLDER'] = str(blocker / 'sub')                    # a path under a regular file can never be created
        assert client.get('/ready').status_code == 503

    def test_the_readiness_response_never_leaks_internals(self, client, monkeypatch):
        monkeypatch.setattr(readiness.database, 'transaction', MagicMock(side_effect=RuntimeError('secret path /srv/db password=hunter2')))
        body = client.get('/ready').get_data(as_text=True)
        assert 'hunter2' not in body and '/srv/db' not in body and 'RuntimeError' in body

    def test_readiness_needs_no_login(self, app):
        assert app.test_client().get('/ready').status_code == 200 and app.test_client().get('/health').status_code == 200


class TestMetrics:
    def test_not_public(self, app, client, registered_user, monkeypatch):
        monkeypatch.delenv('METRICS_TOKEN', raising=False)
        monkeypatch.delenv('ADMIN_USERNAMES', raising=False)
        assert app.test_client().get('/metrics').status_code == 404
        assert client.get('/metrics').status_code == 404                           # signed in, but not an admin

    def test_a_scraper_token_works_and_a_wrong_one_does_not(self, app, monkeypatch):
        monkeypatch.setenv('METRICS_TOKEN', 'scrape-secret-123')
        c = app.test_client()
        assert c.get('/metrics', headers={'Authorization': 'Bearer scrape-secret-123'}).status_code == 200
        for bad in ('Bearer wrong', 'scrape-secret-123', 'Bearer ', 'bearer scrape-secret-123', ''):
            assert c.get('/metrics', headers={'Authorization': bad} if bad else {}).status_code == 404

    def test_the_token_comparison_is_constant_time(self, app, monkeypatch):
        monkeypatch.setenv('METRICS_TOKEN', 'scrape-secret-123')
        spy = MagicMock(return_value=False)
        monkeypatch.setattr('quickaid.blueprints.system.hmac.compare_digest', spy)
        app.test_client().get('/metrics', headers={'Authorization': 'Bearer x'})
        assert spy.called

    def test_an_admin_session_can_read_it(self, client, registered_user, monkeypatch):
        monkeypatch.setenv('ADMIN_USERNAMES', registered_user['username'])
        r = client.get('/metrics')
        assert r.status_code == 200 and r.mimetype.startswith('text/plain') and 'quickaid_http_requests_total' in r.get_data(as_text=True)

    def scrape(self, app, monkeypatch):
        monkeypatch.setenv('METRICS_TOKEN', 't0ken-for-tests')
        return app.test_client().get('/metrics', headers={'Authorization': 'Bearer t0ken-for-tests'}).get_data(as_text=True)

    def test_requests_are_counted_by_route_pattern_never_by_the_secret_in_the_path(self, app, monkeypatch):
        app.test_client().get('/reset-password/SECRETTOKEN-ABCDEFGHIJKLMNOP')
        body = self.scrape(app, monkeypatch)
        assert 'route="/reset-password/<token>"' in body and 'SECRETTOKEN' not in body
        assert re.search(r'quickaid_http_request_duration_seconds_bucket\{.*route="/reset-password/<token>"', body)

    def test_no_personal_data_appears_in_metrics(self, app, monkeypatch):
        c = app.test_client()
        c.post('/register', data={'username': 'metricsperson', 'email': 'metrics.person@example.com', 'password': PW, 'confirm_password': PW})
        c.post('/analyze_symptoms', json={'symptoms': 'private rash on my shoulder'})
        c.get('/history?q=shoulder')
        body = self.scrape(app, monkeypatch)
        for secret in ('metricsperson', 'metrics.person', 'private rash', 'shoulder'):
            assert secret not in body

    def test_ai_calls_analyses_and_emergencies_are_counted(self, app, monkeypatch):
        from quickaid.evaluation.clients import MockClient
        svc = app.extensions['quickaid_services']
        svc.symptom_checker.use_gemini, svc.symptom_checker.client = True, MockClient('rules')
        monkeypatch.setenv('REQUIRE_AI_CONSENT', 'false')
        c = app.test_client()
        c.post('/register', data={'username': 'ann', 'email': 'a@example.com', 'password': PW, 'confirm_password': PW})
        before = self.scrape(app, monkeypatch)
        c.post('/analyze_symptoms', json={'symptoms': 'headache and nausea for two days'})
        c.post('/analyze_symptoms', json={'symptoms': 'crushing chest pain and cannot breathe'})
        after = self.scrape(app, monkeypatch)

        def val(text, prefix):
            m = re.search(re.escape(prefix) + r'\} (\S+)', text)
            return float(m.group(1)) if m else 0.0
        assert val(after, 'quickaid_ai_calls_total{kind="symptom",outcome="ok"') == val(before, 'quickaid_ai_calls_total{kind="symptom",outcome="ok"') + 2     # one model call per analysis
        assert val(after, 'quickaid_emergencies_total{kind="symptom"') == val(before, 'quickaid_emergencies_total{kind="symptom"') + 1
        assert 'quickaid_analyses_total{kind="symptom"' in after

    def test_failures_and_invalid_output_are_counted_separately(self):
        from quickaid.observability import metrics
        key = lambda o: metrics.AI_CALLS.labels('probe', o)._value.get()          # noqa: E731
        a, b, c = key('ok'), key('failed'), key('invalid')
        metrics.SINK.on_call({'kind': 'probe', 'ok': True, 'valid': True, 'latency_ms': 10})
        metrics.SINK.on_call({'kind': 'probe', 'ok': False, 'valid': None})
        metrics.SINK.on_call({'kind': 'probe', 'ok': True, 'valid': False, 'latency_ms': 5})
        assert (key('ok'), key('failed'), key('invalid')) == (a + 1, b + 1, c + 1)

    def test_unhandled_errors_are_counted(self, app, monkeypatch):
        from quickaid.observability import metrics
        app.testing = False
        app.config['PROPAGATE_EXCEPTIONS'] = False
        app.extensions['quickaid_services'].history.dashboard = MagicMock(side_effect=RuntimeError('boom'))
        c = app.test_client()
        c.post('/register', data={'username': 'ann', 'email': 'a@example.com', 'password': PW, 'confirm_password': PW})
        n = metrics.UNHANDLED._value.get()
        assert c.get('/').status_code == 500 and metrics.UNHANDLED._value.get() == n + 1

    def test_scrape_time_gauges_are_present(self, app, monkeypatch):
        body = self.scrape(app, monkeypatch)
        for gauge in ('quickaid_uptime_seconds', 'quickaid_schema_version', 'quickaid_database_bytes'):
            assert re.search(rf'^{gauge} \S+$', body, re.M), gauge


class TestErrorReporting:
    def event(self):
        return {'message': 'failed for jane.doe@example.com', 'user': {'id': 7, 'email': 'jane.doe@example.com', 'ip_address': '203.0.113.9'},
                'server_name': 'prod-1', 'extra': {'symptoms': 'my private rash'}, 'breadcrumbs': {'values': [{'message': 'GET /history?q=rash'}]},
                'request': {'url': 'https://x.example/reset-password/AbCdEfGhIjKlMnOpQrSt?next=/a', 'method': 'POST', 'query_string': 'q=rash', 'cookies': {'session': 'abc'},
                            'headers': {'Authorization': 'Bearer s3cret', 'Cookie': 'session=abc'}, 'data': {'symptoms': 'my private rash', 'password': 'hunter2'}, 'env': {'REMOTE_ADDR': '203.0.113.9'}},
                'exception': {'values': [{'type': 'ValueError', 'value': 'bad value for victim@example.com token=abcdef123456',
                                          'stacktrace': {'frames': [{'filename': 'quickaid/app.py', 'lineno': 12, 'function': 'view', 'vars': {'text': 'my private rash'},
                                                                      'context_line': 'text = request.json["symptoms"]', 'pre_context': ['a'], 'post_context': ['b']}]}}]}}

    def test_everything_that_could_identify_a_person_or_hold_health_text_is_removed(self):
        out = errors.scrub_event(self.event())
        blob = json.dumps(out)
        for secret in ('jane.doe', 'victim@', 'private rash', 'hunter2', 's3cret', '203.0.113.9', 'abc123def456', 'AbCdEfGhIj', 'prod-1', 'symptoms'):
            assert secret not in blob, secret
        for gone in ('user', 'extra', 'breadcrumbs', 'server_name'):
            assert gone not in out
        assert set(out['request']) == {'url', 'method'} and out['request']['url'] == 'https://x.example/reset-password/[token]'

    def test_what_a_developer_needs_survives(self):
        out = errors.scrub_event(self.event())
        frame = out['exception']['values'][0]['stacktrace']['frames'][0]
        assert out['exception']['values'][0]['type'] == 'ValueError' and (frame['filename'], frame['lineno'], frame['function']) == ('quickaid/app.py', 12, 'view')
        assert out['request']['method'] == 'POST' and 'vars' not in frame and 'context_line' not in frame

    def test_scrubbing_never_mutates_or_crashes_on_odd_events(self):
        for weird in ({}, {'request': None}, {'exception': {'values': None}}, {'exception': {'values': [{'value': None}]}}, {'message': None}, {'logentry': {'message': 'a@b.co', 'params': ['x']}}):
            errors.scrub_event(weird)
        original = self.event()
        errors.scrub_event(original)
        assert original['user']['email'] == 'jane.doe@example.com'

    def test_off_unless_a_dsn_is_configured(self, monkeypatch):
        monkeypatch.delenv('SENTRY_DSN', raising=False)
        assert errors.init() is False

    def test_when_on_it_is_configured_to_send_no_personal_data(self, monkeypatch):
        import sentry_sdk
        seen = {}
        monkeypatch.setattr(sentry_sdk, 'init', lambda **kw: seen.update(kw))
        monkeypatch.delenv('SENTRY_TRACES_SAMPLE_RATE', raising=False)
        assert errors.init('https://key@o0.ingest.sentry.io/1') is True
        assert seen['send_default_pii'] is False and seen['include_local_variables'] is False and seen['max_breadcrumbs'] == 0
        assert seen['before_send'] is errors.scrub_event and seen['traces_sample_rate'] == 0.0 and seen['before_breadcrumb']({}, {}) is None

    def test_the_app_starts_without_the_sdk_installed(self, monkeypatch):
        import builtins
        real = builtins.__import__
        monkeypatch.setattr(builtins, '__import__', lambda name, *a, **k: (_ for _ in ()).throw(ImportError()) if name == 'sentry_sdk' else real(name, *a, **k))
        assert errors.init('https://key@o0.ingest.sentry.io/1') is False


class TestDeploymentConfiguration:
    def test_the_image_is_hardened(self):
        d = (ROOT / 'Dockerfile').read_text()
        assert re.search(r'^USER quickaid', d, re.M) and 'HEALTHCHECK' in d and 'nologin' in d
        assert re.search(r'^CMD \["gunicorn", "-c", "deploy/gunicorn.conf.py", "wsgi:app"\]', d, re.M)         # exec form: signals reach gunicorn
        assert 'COPY .env' not in d and '--debug' not in d and 'flask run' not in d

    def test_the_image_build_context_excludes_secrets_and_history(self):
        ignore = (ROOT / '.dockerignore').read_text().splitlines()
        for needed in ('.env', '.git/', 'tests/'):
            assert any(line.strip() == needed or line.strip().startswith(needed) for line in ignore), needed

    def test_compose_runs_the_web_service_with_least_privilege(self):
        web = yaml.safe_load((ROOT / 'docker-compose.yml').read_text())['services']['web']
        assert web['read_only'] is True and web['cap_drop'] == ['ALL'] and 'no-new-privileges:true' in web['security_opt'] and web['init'] is True
        assert web['mem_limit'] and web['pids_limit'] and '/app/uploads' in ' '.join(web['tmpfs']) and 'ready' in ' '.join(web['healthcheck']['test'])
        assert 'ports' not in web                                              # only nginx is published

    def test_gunicorn_is_configured_for_production(self):
        src = (ROOT / 'deploy/gunicorn.conf.py').read_text()
        ns = {}
        exec(compile(src, 'gunicorn.conf.py', 'exec'), ns)                    # the file must at least be valid and importable
        assert ns['accesslog'] is None and ns['max_requests'] > 0 and ns['timeout'] >= 60 and ns['worker_class'] == 'gthread'
        assert ns['limit_request_line'] <= 8190 and callable(ns['on_starting']) and callable(ns['child_exit'])

    def test_every_dependency_is_pinned_to_an_exact_version(self):
        for line in (ROOT / 'requirements.txt').read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('-r'):
                assert '==' in line, f'unpinned dependency: {line}'

    def test_monitoring_dependencies_are_declared(self):
        req = (ROOT / 'requirements.txt').read_text()
        assert 'prometheus-client==' in req and 'sentry-sdk==' in req

    @pytest.fixture()
    def ci(self):
        return yaml.safe_load((ROOT / '.github/workflows/ci.yml').read_text())

    def test_ci_has_least_privilege_and_the_jobs_that_matter(self, ci):
        assert ci['permissions'] == {'contents': 'read'} and {'lint', 'secrets', 'test', 'browser', 'dependency-audit', 'docker'} <= set(ci['jobs'])

    def test_ci_tests_two_python_versions_with_a_coverage_floor(self, ci):
        job = ci['jobs']['test']
        assert job['strategy']['matrix']['python'] == ['3.11', '3.12']
        run = ' '.join(s.get('run', '') for s in job['steps'])
        assert '--cov-fail-under=' in run and 'migrations check' in run and 'manage_datasets.py check' in run and 'run_evaluation.py' in run

    def test_ci_scans_for_secrets_with_full_history_and_audits_dependencies(self, ci):
        steps = ci['jobs']['secrets']['steps']
        assert any(s.get('with', {}).get('fetch-depth') == 0 for s in steps) and any('secret_scan.py --history' in s.get('run', '') for s in steps)
        assert any('pip-audit' in s.get('run', '') for s in ci['jobs']['dependency-audit']['steps'])

    def test_ci_smoke_tests_the_hardened_container(self, ci):
        run = ' '.join(s.get('run', '') for s in ci['jobs']['docker']['steps'])
        for needle in ('docker build', '--read-only', '--cap-drop ALL', '/ready', '/metrics', 'test ! -e /app/.env', 'test ! -e /app/.git'):
            assert needle in run, needle
