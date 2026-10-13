"""Tests for the redesigned frontend: signup validation, password UX hooks,
theme support, dashboard data and region-aware emergency numbers."""


def register(client, username='newuser', email='new@example.com', password='supersecret123', confirm=None):
    return client.post('/register', data={
        'username': username, 'email': email, 'password': password,
        'confirm_password': password if confirm is None else confirm,
    })


class TestSignupConfirmPassword:
    def test_mismatched_confirmation_rejected(self, client):
        resp = register(client, confirm='different123')
        assert resp.status_code == 400
        assert b'Passwords do not match' in resp.data

    def test_missing_confirmation_rejected(self, client):
        resp = client.post('/register', data={'username': 'abc', 'email': 'a@b.co', 'password': 'supersecret123'})
        assert resp.status_code == 400

    def test_typed_values_kept_after_error(self, client):
        resp = register(client, username='keepme', email='keep@example.com', confirm='nope12345')
        assert b'value="keepme"' in resp.data
        assert b'value="keep@example.com"' in resp.data

    def test_password_is_never_echoed_back(self, client):
        resp = register(client, password='supersecret123', confirm='nope12345')
        assert b'supersecret123' not in resp.data

    def test_matching_confirmation_creates_account(self, client):
        resp = register(client)
        assert resp.status_code in (200, 302)
        assert client.get('/history').status_code == 200  # logged in


class TestPasswordFieldMarkup:
    def test_register_page_has_eye_toggles_confirm_and_strength(self, client):
        html = client.get('/register').data.decode()
        assert html.count('data-toggle-pw') == 2
        assert 'name="confirm_password"' in html
        assert 'data-strength' in html
        assert 'data-match="password"' in html

    def test_login_page_has_eye_toggle(self, client):
        assert 'data-toggle-pw="password"' in client.get('/login').data.decode()

    def test_invalid_reset_link_renders_cleanly(self, client):
        html = client.get('/reset-password/not-a-real-token').data.decode()
        assert 'Link expired or invalid' in html

    def test_account_page_password_fields_have_toggles(self, client, registered_user):
        html = client.get('/account').data.decode()
        assert 'data-toggle-pw="new_password"' in html
        assert 'data-toggle-pw="confirm_password"' in html


class TestTheme:
    def test_every_page_loads_theme_toggle_and_assets(self, client):
        for path in ('/', '/login', '/register', '/emergency', '/forgot-password'):
            html = client.get(path).data.decode()
            assert 'data-theme-toggle' in html, path
            assert '/static/css/app.css' in html, path
            assert '/static/js/app.js' in html, path

    def test_inline_theme_script_uses_csp_nonce(self, client):
        resp = client.get('/')
        csp = resp.headers.get('Content-Security-Policy', '')
        html = resp.data.decode()
        import re
        nonce = re.search(r"nonce-([\w\-+/=]+)", csp)
        assert nonce and f'nonce="{nonce.group(1)}"' in html

    def test_stylesheet_defines_light_and_dark(self, client):
        css = client.get('/static/css/app.css')
        assert css.status_code == 200
        assert b':root[data-theme="dark"]' in css.data

    def test_no_blue_brand_colour_left_in_stylesheet(self, client):
        css = client.get('/static/css/app.css').data.decode().lower()
        for old in ('#667eea', '#764ba2', '#3498db', '#2980b9'):
            assert old not in css


class TestLandingAndDashboard:
    def test_landing_for_visitors_shows_signup(self, client):
        html = client.get('/').data.decode()
        assert 'Create free account' in html
        assert 'Checks done' not in html

    def test_dashboard_for_logged_in_users(self, client, registered_user):
        html = client.get('/').data.decode()
        assert 'Checks done' in html
        assert 'id="symptomsInput"' in html
        assert 'id="dropzone"' in html
        assert registered_user['username'] in html

    def test_dashboard_stats_count_checks(self, client, registered_user):
        client.get('/')
        client.post('/analyze_symptoms', json={'symptoms': 'I have a headache and nausea'},
                    headers={'X-CSRFToken': 'x'})
        html = client.get('/').data.decode()
        assert 'Recent checks' in html

    def test_ai_output_is_not_injected_as_html(self, client):
        js = client.get('/static/js/app.js').data.decode()
        assert 'innerHTML' not in js


class TestRegionAwareEmergencyNumbers:
    def test_default_region_is_india(self, client):
        html = client.get('/emergency').data.decode()
        assert '112' in html and '108' in html
        assert '911' not in html

    def test_emergency_page_is_public(self, client):
        assert client.get('/emergency').status_code == 200
