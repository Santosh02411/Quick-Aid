"""The shared API layer: standard responses, structured errors, request ids, validation, OpenAPI, and architecture."""

import ast
import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parent.parent / 'quickaid'


class TestStandardResponses:
    def test_success_carries_success_and_request_id(self, client, registered_user):
        r = client.get('/api/history')
        j = r.get_json()
        assert j['success'] is True and j['request_id'] and j['request_id'] == r.headers['X-Request-ID']

    def test_failure_has_one_shape_everywhere(self, client, registered_user):
        for r in (client.get('/api/history/symptom/999'), client.get('/api/history?page=0'), client.get('/api/nope'),
                  client.post('/analyze_symptoms', json={'symptoms': ''}), client.delete('/api/history')):
            j = r.get_json()
            assert j['success'] is False and isinstance(j['error'], str) and j['code'] and j['request_id'], r.request.path

    def test_codes_are_stable_and_machine_readable(self, client, registered_user):
        assert client.get('/api/history/symptom/999').get_json()['code'] == 'entry_not_found'
        assert client.post('/analyze_symptoms', json={'symptoms': ''}).get_json()['code'] == 'missing_symptoms'
        assert client.post('/analyze_symptoms', json={'symptoms': 'a'}).get_json()['code'] == 'symptoms_too_short'
        assert client.post('/analyze_symptoms', json={'symptoms': 'x' * 2000}).get_json()['code'] == 'symptoms_too_long'
        assert client.post('/analyze_symptoms', json={'symptoms': 5}).get_json()['code'] == 'invalid_symptoms'
        assert client.post('/api/chats/1/messages', json={'message': ''}).get_json()['code'] == 'empty_message'

    def test_a_json_array_body_is_rejected_cleanly(self, client, registered_user):
        r = client.post('/analyze_symptoms', data='[1, 2]', content_type='application/json')
        assert r.status_code == 400 and r.get_json()['code'] == 'invalid_json'

    def test_http_errors_are_json_for_apis_and_html_for_pages(self, client, registered_user):
        j = client.get('/api/does-not-exist')
        assert j.status_code == 404 and j.get_json()['code'] == 'not_found'
        m = client.put('/api/history')
        assert m.status_code == 405 and m.get_json()['code'] == 'method_not_allowed'
        page = client.get('/no-such-page')
        assert page.status_code == 404 and page.mimetype == 'text/html'

    def test_rate_limit_responses_are_standard(self, client, registered_user):
        for _ in range(16):
            client.post('/analyze_symptoms', json={'symptoms': 'headache'})
        j = client.post('/analyze_symptoms', json={'symptoms': 'headache'}).get_json()
        assert j['success'] is False and j['code'] == 'rate_limited' and 'slow down' in j['error']

    def test_csrf_failures_are_json_for_api_calls(self, client, registered_user, app):
        app.config['WTF_CSRF_ENABLED'] = True
        r = client.post('/api/history/clear')
        assert r.status_code == 400 and r.get_json()['code'] == 'csrf_failed'

    def test_unexpected_errors_never_leak_internals(self, client, registered_user, app):
        from quickaid.services.container import Services
        app.testing = False                       # in TESTING mode Flask re-raises; in production the handler responds
        app.config['PROPAGATE_EXCEPTIONS'] = False
        svc = app.extensions['quickaid_services']
        svc.history.search = MagicMock(side_effect=RuntimeError('secret internal detail: /srv/db password=hunter2'))
        r = client.get('/api/history')
        body = r.get_data(as_text=True)
        assert r.status_code == 500 and r.get_json()['code'] == 'internal_error' and r.get_json()['request_id']
        assert 'hunter2' not in body and 'Traceback' not in body and 'secret internal' not in body
        page = client.get('/history')
        assert page.status_code == 500 and 'quote this reference' in page.get_data(as_text=True) and 'hunter2' not in page.get_data(as_text=True)


class TestRequestIds:
    def test_a_valid_incoming_id_is_kept(self, client):
        assert client.get('/health', headers={'X-Request-ID': 'trace-abc_123.XYZ'}).headers['X-Request-ID'] == 'trace-abc_123.XYZ'

    @pytest.mark.parametrize('bad', ['short', 'has spaces in it', 'x' * 100, 'inject<script>1234', 'a,b;c=d-12345'])
    def test_a_hostile_or_malformed_id_is_replaced(self, client, bad):
        rid = client.get('/health', headers={'X-Request-ID': bad}).headers['X-Request-ID']
        assert rid != bad and re.fullmatch(r'[0-9a-f]{16}', rid)

    def test_ids_are_unique_per_request(self, client):
        assert len({client.get('/health').headers['X-Request-ID'] for _ in range(10)}) == 10

    def test_the_id_in_an_error_matches_the_header(self, client, registered_user):
        r = client.get('/api/history/symptom/1234')
        assert r.get_json()['request_id'] == r.headers['X-Request-ID']

    def test_health_is_standard_too(self, client):
        j = client.get('/health').get_json()
        assert j['success'] is True and j['status'] == 'ok' and 'request_id' in j


class TestOpenApi:
    @pytest.fixture()
    def spec(self, app, client, registered_user, monkeypatch):
        monkeypatch.setenv('ADMIN_USERNAMES', registered_user['username'])
        return client.get('/api/openapi.json').get_json()

    def test_the_document_is_valid_openapi(self, spec):
        from openapi_spec_validator import validate
        validate(spec)

    def test_every_json_api_route_is_documented(self, app):
        undocumented = []
        for rule in app.url_map.iter_rules():
            if rule.endpoint == 'static':
                continue
            is_api = rule.rule.startswith(('/api/', '/admin/api/')) or rule.rule in ('/upload', '/analyze_symptoms', '/health')
            if is_api and not getattr(app.view_functions[rule.endpoint], '__api_doc__', None):
                undocumented.append(rule.rule)
        assert undocumented == []

    def test_paths_methods_and_params_match_the_real_routes(self, spec):
        p = spec['paths']
        assert set(p['/api/history']) == {'get'} and 'post' in p['/api/history/clear'] and 'delete' in p['/api/history/{entry_type}/{entry_id}']
        param = {x['name']: x for x in p['/api/history/{entry_type}/{entry_id}']['get']['parameters']}
        assert param['entry_id']['schema']['type'] == 'integer' and param['entry_type']['schema']['enum'] == ['symptom', 'image']

    def test_query_parameters_come_from_the_validation_model(self, spec):
        q = {x['name']: x for x in spec['paths']['/api/history']['get']['parameters']}
        assert {'q', 'type', 'urgency', 'emergency', 'date_from', 'date_to', 'sort', 'page', 'per_page'} <= set(q)
        assert q['per_page']['schema']['maximum'] == 100 and q['sort']['schema']['enum'] == ['newest', 'oldest', 'urgency']

    def test_request_bodies_come_from_the_validation_models(self, spec):
        body = spec['paths']['/api/history/{entry_type}/{entry_id}/notes']['patch']['requestBody']['content']['application/json']['schema']
        assert body == {'$ref': '#/components/schemas/NotesBody'}
        assert spec['components']['schemas']['NotesBody']['properties']['notes']['maxLength'] == 2000
        assert spec['components']['schemas']['SymptomRequest']['properties']['symptoms']['maxLength'] == 1000

    def test_every_operation_documents_the_standard_errors(self, spec):
        for path, ops in spec['paths'].items():
            for method, op in ops.items():
                assert {'400', '404', '429', '500'} <= set(op['responses']), (path, method)
                assert op['responses']['500']['content']['application/json']['schema'] == {'$ref': '#/components/schemas/Error'}

    def test_authenticated_state_changes_declare_csrf(self, spec):
        assert spec['paths']['/api/history/clear']['post']['security'] == [{'cookieAuth': [], 'csrfToken': []}]
        assert spec['paths']['/api/history']['get']['security'] == [{'cookieAuth': []}]
        assert 'security' not in spec['paths']['/health']['get'] and '401' not in spec['paths']['/health']['get']['responses']

    def test_downloads_are_documented_as_files(self, spec):
        assert 'application/json' in spec['paths']['/api/history/export']['get']['responses']['200']['content']

    def test_operation_ids_are_unique(self, spec):
        ids = [op['operationId'] for ops in spec['paths'].values() for op in ops.values()]
        assert len(ids) == len(set(ids))

    def test_spec_generation_is_deterministic(self, app):
        from quickaid.api.openapi import build_spec
        assert build_spec(app) == build_spec(app)

    def test_docs_are_for_admins_unless_made_public(self, app, client, registered_user, monkeypatch):
        monkeypatch.delenv('ADMIN_USERNAMES', raising=False)
        monkeypatch.delenv('API_DOCS_PUBLIC', raising=False)
        assert client.get('/api/openapi.json').status_code == 404 and client.get('/api/docs').status_code == 404
        monkeypatch.setenv('API_DOCS_PUBLIC', 'true')
        assert client.get('/api/openapi.json').status_code == 200 and client.get('/api/docs').status_code == 200
        monkeypatch.setenv('API_DOCS_PUBLIC', 'false')
        monkeypatch.setenv('ADMIN_USERNAMES', registered_user['username'])
        assert client.get('/api/docs').status_code == 200
        assert app.test_client().get('/api/openapi.json').status_code == 401

    def test_docs_page_and_viewer_script(self, client, registered_user, monkeypatch):
        monkeypatch.setenv('ADMIN_USERNAMES', registered_user['username'])
        assert 'id="apiDocs"' in client.get('/api/docs').data.decode()
        js = client.get('/static/js/docs.js').data.decode()
        assert 'innerHTML' not in js and 'eval(' not in js


class TestArchitecture:
    def test_every_route_lives_in_a_blueprint(self, app):
        endpoints = {r.endpoint for r in app.url_map.iter_rules() if r.endpoint != 'static'}
        assert all('.' in e for e in endpoints)
        assert {e.split('.')[0] for e in endpoints} == {'pages', 'system', 'auth', 'symptoms', 'images', 'conversations', 'history', 'privacy', 'admin'}

    def test_the_app_module_only_wires_things_together(self):
        src = (ROOT / 'app.py').read_text()
        assert '@app.route' not in src and 'def ' in src and len(src.splitlines()) < 110

    def test_no_duplicate_routes(self, app):
        seen = set()
        for r in app.url_map.iter_rules():
            for m in r.methods - {'HEAD', 'OPTIONS'}:
                key = (re.sub(r'<[^>]+>', '<x>', r.rule), m)
                assert key not in seen, key
                seen.add(key)

    def imports(self, rel):
        tree = ast.parse((ROOT / rel).read_text())
        out = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.module:
                out.add(n.module)
                out.update(f'{n.module}.{a.name}' for a in n.names)
            elif isinstance(n, ast.Import):
                out.update(a.name for a in n.names)
        return out

    @pytest.mark.parametrize('view', ['symptoms', 'images', 'conversations', 'history', 'privacy'])
    def test_domain_views_reach_data_only_through_services(self, view):
        imp = self.imports(f'blueprints/{view}.py')
        forbidden = {'quickaid.core.database', 'quickaid.core.history_repo', 'quickaid.ai.medical_analyzer', 'quickaid.ai.symptom_checker',
                     'quickaid.ai.triage_engine', 'quickaid.ai.conversation', 'quickaid.ai.interview', 'quickaid.core'}
        assert not (imp & forbidden), imp & forbidden

    @pytest.mark.parametrize('svc', ['analysis', 'chats', 'history', 'retention', 'safety', 'container'])
    def test_services_contain_no_http_handling(self, svc):
        imp = self.imports(f'services/{svc}.py')
        assert not ({'flask.request', 'flask.jsonify', 'flask.render_template', 'flask.session', 'flask.redirect'} & imp)

    def test_services_do_not_import_views(self):
        for f in (ROOT / 'services').glob('*.py'):
            assert not any(i.startswith('quickaid.blueprints') for i in self.imports(f'services/{f.name}')), f.name

    @pytest.mark.parametrize('pkg', ['ai', 'core', 'api'])
    def test_lower_layers_never_import_upwards(self, pkg):
        for f in (ROOT / pkg).glob('*.py'):
            bad = [i for i in self.imports(f'{pkg}/{f.name}') if i.startswith(('quickaid.blueprints', 'quickaid.web', 'quickaid.app'))]
            assert not bad, (f.name, bad)

    def test_medical_safety_is_reached_through_one_facade(self):
        for f in (ROOT / 'services').glob('*.py'):
            if f.name != 'safety.py':
                assert 'quickaid.ai.triage_engine' not in self.imports(f'services/{f.name}'), f.name

    def test_each_app_instance_has_its_own_isolated_services(self, app):
        from quickaid.app import create_app
        other = create_app()
        a, b = app.extensions['quickaid_services'], other.extensions['quickaid_services']
        assert a is not b and a.symptom_checker is not b.symptom_checker and a.chat_ai is not b.chat_ai and a.history is not b.history

    def test_the_old_import_surface_still_works(self):
        from quickaid import app as app_module
        assert app_module.app and app_module.medical_analyzer and app_module.conversation_service and app_module.csrf and app_module.logger
        assert app_module.conversation_service is app_module.app.extensions['quickaid_services'].chat_ai
