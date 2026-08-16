"""
Flask test-client tests: auth flow, route protection, and request
validation (symptom length limits, image content validation).
"""

import io
import pytest


class TestAuthFlow:
    def test_register_creates_account_and_logs_in(self, client):
        resp = client.post('/register', data={
            'username': 'newuser', 'email': 'newuser@example.com', 'password': 'supersecret123'
        })
        assert resp.status_code == 302  # redirect to index on success

        # session should now be authenticated - protected route should work
        resp2 = client.get('/api/history')
        assert resp2.status_code == 200

    def test_register_duplicate_username_rejected(self, client, registered_user):
        client.post('/logout')
        resp = client.post('/register', data={
            'username': registered_user['username'],
            'email': 'different@example.com',
            'password': 'anotherpassword123'
        })
        assert resp.status_code == 409

    def test_register_short_password_rejected(self, client):
        resp = client.post('/register', data={
            'username': 'shortpw', 'email': 'shortpw@example.com', 'password': '123'
        })
        assert resp.status_code == 400

    def test_register_invalid_username_rejected(self, client):
        resp = client.post('/register', data={
            'username': 'a b!', 'email': 'bad@example.com', 'password': 'supersecret123'
        })
        assert resp.status_code == 400

    def test_register_missing_fields_rejected(self, client):
        resp = client.post('/register', data={'username': 'onlyusername'})
        assert resp.status_code == 400

    def test_login_correct_credentials(self, client, registered_user):
        client.post('/logout')
        resp = client.post('/login', data={
            'username': registered_user['username'], 'password': registered_user['password']
        })
        assert resp.status_code == 302
        assert client.get('/api/history').status_code == 200

    def test_login_wrong_password_rejected(self, client, registered_user):
        client.post('/logout')
        resp = client.post('/login', data={
            'username': registered_user['username'], 'password': 'totallywrong'
        })
        assert resp.status_code == 401

    def test_login_nonexistent_user_rejected(self, client):
        resp = client.post('/login', data={'username': 'ghost', 'password': 'whatever123'})
        assert resp.status_code == 401

    def test_logout_then_protected_route_requires_login(self, client, registered_user):
        assert client.get('/api/history').status_code == 200
        client.post('/logout')
        resp = client.get('/api/history')
        assert resp.status_code == 401


class TestRouteProtection:
    def test_upload_requires_login(self, client):
        data = {'file': (io.BytesIO(b'not a real file'), 'test.jpg')}
        resp = client.post('/upload', data=data, content_type='multipart/form-data')
        assert resp.status_code == 401

    def test_analyze_symptoms_requires_login(self, client):
        resp = client.post('/analyze_symptoms', json={'symptoms': 'headache'})
        assert resp.status_code == 401

    def test_history_page_redirects_when_logged_out(self, client):
        resp = client.get('/history')
        assert resp.status_code == 302
        assert '/login' in resp.headers.get('Location', '')

    def test_public_routes_accessible_without_login(self, client):
        assert client.get('/').status_code == 200
        assert client.get('/emergency').status_code == 200
        assert client.get('/health').status_code == 200

    def test_history_accessible_when_logged_in(self, client, registered_user):
        resp = client.get('/history')
        assert resp.status_code == 200


class TestSymptomValidation:
    def test_empty_symptoms_rejected(self, client, registered_user):
        resp = client.post('/analyze_symptoms', json={'symptoms': ''})
        assert resp.status_code == 400

    def test_too_short_symptoms_rejected(self, client, registered_user):
        resp = client.post('/analyze_symptoms', json={'symptoms': 'ab'})
        assert resp.status_code == 400

    def test_too_long_symptoms_rejected(self, client, registered_user):
        resp = client.post('/analyze_symptoms', json={'symptoms': 'a' * 2000})
        assert resp.status_code == 400

    def test_non_string_symptoms_rejected(self, client, registered_user):
        resp = client.post('/analyze_symptoms', json={'symptoms': 12345})
        assert resp.status_code == 400

    def test_valid_symptoms_accepted(self, client, registered_user):
        resp = client.post('/analyze_symptoms', json={'symptoms': 'I have a headache'})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['success'] is True


class TestImageValidation:
    def test_fake_image_rejected(self, client, registered_user):
        data = {'file': (io.BytesIO(b'this is not an image at all'), 'fake.jpg')}
        resp = client.post('/upload', data=data, content_type='multipart/form-data')
        assert resp.status_code == 400
        assert 'error' in resp.get_json()

    def test_valid_png_accepted(self, client, registered_user):
        from PIL import Image
        buf = io.BytesIO()
        Image.new('RGB', (10, 10), color='red').save(buf, format='PNG')
        buf.seek(0)

        data = {'file': (buf, 'test.png')}
        resp = client.post('/upload', data=data, content_type='multipart/form-data')
        assert resp.status_code == 200
        assert resp.get_json()['success'] is True

    def test_no_file_rejected(self, client, registered_user):
        resp = client.post('/upload', data={}, content_type='multipart/form-data')
        assert resp.status_code == 400

    def test_disallowed_extension_rejected(self, client, registered_user):
        data = {'file': (io.BytesIO(b'whatever'), 'script.exe')}
        resp = client.post('/upload', data=data, content_type='multipart/form-data')
        assert resp.status_code == 400


class TestHistoryEndpoint:
    def test_history_populated_after_symptom_check(self, client, registered_user):
        client.post('/analyze_symptoms', json={'symptoms': 'headache and nausea'})
        resp = client.get('/api/history')
        data = resp.get_json()
        assert len(data['history']) == 1
        assert data['history'][0]['type'] == 'symptom'

    def test_clear_history_empties_it(self, client, registered_user):
        client.post('/analyze_symptoms', json={'symptoms': 'headache'})
        assert len(client.get('/api/history').get_json()['history']) == 1

        client.post('/api/history/clear')
        assert len(client.get('/api/history').get_json()['history']) == 0


class TestMiscRoutes:
    def test_unknown_route_returns_404_not_500(self, client):
        resp = client.get('/this-route-does-not-exist')
        assert resp.status_code == 404

    def test_wrong_method_returns_405(self, client):
        resp = client.post('/emergency')
        assert resp.status_code == 405

    def test_health_check_reports_ok(self, client):
        resp = client.get('/health')
        data = resp.get_json()
        assert data['status'] == 'ok'
        assert data['database'] == 'ok'


class TestRateLimiting:
    """
    /analyze_symptoms is capped at 15/minute, /upload at 10/minute (see
    app.py). Each app fixture instance gets its own fresh in-memory limiter
    (the app module is reimported per test), so these don't interfere with
    other tests or each other.
    """

    def test_analyze_symptoms_rate_limited_after_15_requests(self, client, registered_user):
        statuses = []
        for _ in range(17):
            resp = client.post('/analyze_symptoms', json={'symptoms': 'headache'})
            statuses.append(resp.status_code)

        assert statuses[:15] == [200] * 15
        assert 429 in statuses[15:]

    def test_upload_rate_limited_after_10_requests(self, client, registered_user):
        from PIL import Image

        def make_upload():
            buf = io.BytesIO()
            Image.new('RGB', (10, 10), color='blue').save(buf, format='PNG')
            buf.seek(0)
            return {'file': (buf, 'test.png')}

        statuses = []
        for _ in range(12):
            resp = client.post('/upload', data=make_upload(), content_type='multipart/form-data')
            statuses.append(resp.status_code)

        assert statuses[:10] == [200] * 10
        assert 429 in statuses[10:]

    def test_rate_limit_response_is_clean_json(self, client, registered_user):
        for _ in range(16):
            client.post('/analyze_symptoms', json={'symptoms': 'headache'})

        resp = client.post('/analyze_symptoms', json={'symptoms': 'headache'})
        assert resp.status_code == 429
        data = resp.get_json()
        assert 'error' in data
