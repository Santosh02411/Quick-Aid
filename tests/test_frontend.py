"""Tests for the redesigned frontend: signup validation, password UX hooks,
theme support, dashboard data and region-aware emergency numbers."""


import json


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


# ---------------------------------------------------------------------------
# Explainability, reliability and honest basic-mode messaging
# ---------------------------------------------------------------------------

def png_bytes(img):
    from io import BytesIO
    buf = BytesIO()
    img.save(buf, 'PNG')
    buf.seek(0)
    return buf


class TestExplainableResults:
    def test_symptom_result_carries_the_whole_explainability_bundle(self, client, registered_user):
        r = client.post('/analyze_symptoms', json={'symptoms': 'headache and nausea'}).get_json()['analysis']
        for key in ('reported', 'possible_explanations', 'red_flags_present', 'recommendations', 'information_gaps',
                    'reliability', 'uncertainty_statement', 'safety_review', 'assessment_status', 'care_level', 'timeframe'):
            assert key in r, key
        assert r['reliability']['level'] == 'limited'          # basic mode in tests
        assert 'not a diagnosis' in r['uncertainty_statement'].lower()
        assert r['safety_review']['engine_version']

    def test_emergency_wording_is_reviewed_before_saving(self, client, registered_user):
        r = client.post('/analyze_symptoms', json={'symptoms': 'my dad is clutching his chest and is grey and sweaty'}).get_json()['analysis']
        assert r['emergency_alert']['alert'] and r['care_level'] == 'emergency'

    def test_unusable_photo_gets_a_retake_response_not_a_guess(self, client, registered_user):
        import numpy as np
        from PIL import Image
        rng = np.random.default_rng(4)
        dark = np.clip((rng.integers(-25, 25, (300, 300, 1)) + np.array([215, 165, 145])) * 0.08, 0, 255).astype('uint8')
        r = client.post('/upload', data={'file': (png_bytes(Image.fromarray(dark)), 'dark.png')},
                        content_type='multipart/form-data').get_json()
        a = r['analysis']
        assert a['assessment_status'] == 'insufficient' and a['possible_explanations'] == [] and a['quality_check']['unusable']
        assert r['analysis_id']

    def test_unrecognised_text_is_never_met_with_reassurance(self, client, registered_user):
        r = client.post('/analyze_symptoms', json={'symptoms': 'zxqv blorp wibble'})
        body = r.get_json()
        text = json.dumps(body)
        assert 'emergency' in text.lower() and 'call 112' in text.lower()

    def test_saved_analysis_keeps_the_reliability_data(self, client, registered_user):
        r = client.post('/analyze_symptoms', json={'symptoms': 'headache and nausea'}).get_json()
        from quickaid.core import database as db
        saved = db.get_analysis(1, 'symptom', r['analysis_id'])
        assert saved['full']['reliability']['level'] and saved['full']['safety_review']


class TestBasicModeHonesty:
    def test_dashboard_says_ai_is_not_connected(self, client, registered_user):
        html = client.get('/').data.decode()
        assert 'AI is not connected' in html and 'can miss serious problems' in html

    def test_result_banner_warns_that_serious_problems_can_be_missed(self, client):
        js = client.get('/static/js/app.js').data.decode()
        assert 'It can miss serious problems' in js and 'another language' in js

    def test_result_view_shows_reliability_and_safety_checks(self, client):
        js = client.get('/static/js/app.js').data.decode()
        for phrase in ('How reliable is this?', 'Safety checks applied', 'What the photo shows', 'What a photo cannot tell us',
                       'What you reported', 'What it could be', 'Warning signs present now', 'What would make this more reliable',
                       'Not enough to assess'):
            assert phrase in js, phrase

    def test_confidence_is_never_shown_as_a_percentage(self, client):
        js = client.get('/static/js/app.js').data.decode()
        assert "'Confidence: '" not in js and '%' not in js.split('function reliabilityPanel')[1].split('function safetyLog')[0]
