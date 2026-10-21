"""External-AI consent: nothing leaves the server unless the person agreed, and what is sent is minimal."""

import io
import json

import numpy as np
import pytest
from PIL import Image

from quickaid import config
from quickaid.evaluation.clients import MockClient

PW = 'orange-kettle-42'


class Recorder:
    """Looks like a Gemini client, records every request, answers with the synthetic rules-based mock."""

    def __init__(self):
        self.calls = []
        self._inner = MockClient('rules')
        self.models = self

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return self._inner.models.generate_content(**kwargs)


@pytest.fixture()
def ai(app, monkeypatch):
    """AI 'configured' with a recording client on every service, and consent REQUIRED (the production default)."""
    monkeypatch.setenv('REQUIRE_AI_CONSENT', 'true')
    svc = app.extensions['quickaid_services']
    rec = Recorder()
    for obj in (svc.symptom_checker, svc.medical_analyzer, svc.chat_ai):
        obj.use_gemini, obj.client = True, rec
    return rec


def new_user(app, name, email=None):
    c = app.test_client()
    c.post('/register', data={'username': name, 'email': email or f'{name}@example.com', 'password': PW, 'confirm_password': PW})
    return c


def photo(n=320, seed=1, exif=False):
    arr = np.clip(np.random.default_rng(seed).integers(-25, 25, (n, n, 1)) + np.array([215, 165, 145]), 0, 255).astype('uint8')
    im = Image.fromarray(arr)
    buf = io.BytesIO()
    if exif:
        e = Image.Exif()
        e[0x010F] = 'SecretCameraMaker'                  # Make
        e[0x0110] = 'SecretModel 9000'                   # Model
        e[0x0132] = '2026:01:02 03:04:05'                # DateTime
        gps = e.get_ifd(0x8825)
        gps[1], gps[2], gps[3], gps[4] = 'N', (12.0, 34.0, 56.0), 'E', (77.0, 12.0, 30.0)
        im.save(buf, 'JPEG', exif=e)
    else:
        im.save(buf, 'PNG')
    return buf.getvalue()


def use_ai(client):
    return client.put('/api/privacy/ai-consent', json={'granted': True})


class TestNothingLeavesWithoutConsent:
    def test_every_feature_stays_on_the_server_until_the_person_agrees(self, app, ai):
        c = new_user(app, 'ann')
        assert c.post('/analyze_symptoms', json={'symptoms': 'headache and nausea for two days'}).get_json()['analysis']['analysis_mode'] == 'basic'
        assert c.post('/upload', data={'file': (io.BytesIO(photo()), 'a.png')}, content_type='multipart/form-data').get_json()['analysis']['analysis_mode'] == 'basic'
        start = c.post('/api/interview/start', json={'symptoms': 'cough for a few days'}).get_json()
        assert start['mode'] == 'basic'
        cid = c.post('/api/chats', json={}).get_json()['conversation']['id']
        assert c.post(f'/api/chats/{cid}/messages', json={'message': 'Is a cough serious?'}).get_json()['mode'] == 'basic'
        assert ai.calls == []                                   # ZERO requests reached the AI client

    def test_agreeing_turns_the_ai_on_immediately(self, app, ai):
        c = new_user(app, 'ann')
        c.post('/analyze_symptoms', json={'symptoms': 'headache and nausea for two days'})
        assert ai.calls == []
        assert use_ai(c).get_json()['consent']['granted'] is True
        out = c.post('/analyze_symptoms', json={'symptoms': 'headache and nausea for two days'}).get_json()['analysis']
        assert out['analysis_mode'] == 'ai' and len(ai.calls) == 1

    def test_withdrawing_stops_it_at_once_and_is_recorded(self, app, ai):
        c = new_user(app, 'ann')
        use_ai(c)
        c.post('/analyze_symptoms', json={'symptoms': 'headache and nausea for two days'})
        n = len(ai.calls)
        c.put('/api/privacy/ai-consent', json={'granted': False})
        assert c.post('/analyze_symptoms', json={'symptoms': 'headache and nausea for two days'}).get_json()['analysis']['analysis_mode'] == 'basic'
        assert len(ai.calls) == n
        assert 'AI analysis turned off' in c.get('/account').get_data(as_text=True)

    def test_one_persons_choice_never_leaks_into_anothers_request(self, app, ai):
        a, b = new_user(app, 'ann'), new_user(app, 'bob')
        use_ai(a)
        assert a.post('/analyze_symptoms', json={'symptoms': 'headache and nausea'}).get_json()['analysis']['analysis_mode'] == 'ai'
        assert b.post('/analyze_symptoms', json={'symptoms': 'headache and nausea'}).get_json()['analysis']['analysis_mode'] == 'basic'
        assert a.post('/analyze_symptoms', json={'symptoms': 'headache and nausea'}).get_json()['analysis']['analysis_mode'] == 'ai'
        assert len(ai.calls) == 2

    def test_a_new_wording_asks_everyone_again(self, app, ai, monkeypatch):
        c = new_user(app, 'ann')
        use_ai(c)
        monkeypatch.setattr(config, 'AI_CONSENT_VERSION', config.AI_CONSENT_VERSION + 1)
        assert c.get('/api/privacy/ai-consent').get_json()['consent']['granted'] is False
        assert c.post('/analyze_symptoms', json={'symptoms': 'headache and nausea'}).get_json()['analysis']['analysis_mode'] == 'basic'

    def test_the_operator_can_turn_the_requirement_off(self, app, ai, monkeypatch):
        monkeypatch.setenv('REQUIRE_AI_CONSENT', 'false')
        c = new_user(app, 'ann')
        assert c.post('/analyze_symptoms', json={'symptoms': 'headache and nausea'}).get_json()['analysis']['analysis_mode'] == 'ai'

    def test_emergency_guidance_never_waits_for_consent_or_the_ai(self, app, ai):
        c = new_user(app, 'ann')
        out = c.post('/analyze_symptoms', json={'symptoms': 'my father has crushing chest pain and cannot breathe'}).get_json()['analysis']
        assert out['emergency_alert']['alert'] and out['care_level'] == 'emergency' and ai.calls == []

    def test_consent_is_a_property_of_the_account_not_the_browser(self, app, ai):
        c = new_user(app, 'ann')
        use_ai(c)
        c.post('/logout')
        again = app.test_client()
        again.post('/login', data={'username': 'ann', 'password': PW})
        assert again.get('/api/privacy/ai-consent').get_json()['consent']['granted'] is True

    def test_the_setting_must_be_a_real_boolean(self, app, ai):
        c = new_user(app, 'ann')
        for body in ({}, {'granted': 'yes please'}, {'granted': None}, {'granted': [1]}):
            assert c.put('/api/privacy/ai-consent', json=body).status_code == 400
        assert app.test_client().put('/api/privacy/ai-consent', json={'granted': True}).status_code == 401


class TestWhatIsSent:
    def test_no_identifiers_in_any_request_to_the_provider(self, app, ai):
        c = new_user(app, 'privatename', 'private.person@example.com')
        use_ai(c)
        c.post('/analyze_symptoms', json={'symptoms': 'headache and nausea for two days'})
        c.post('/upload', data={'file': (io.BytesIO(photo()), 'private-name.png')}, content_type='multipart/form-data')
        cid = c.post('/api/chats', json={}).get_json()['conversation']['id']
        c.patch(f'/api/history/symptom/1/notes', json={'notes': 'my secret diary note'})
        c.post(f'/api/chats/{cid}/messages', json={'message': 'is it serious?'})
        blob = json.dumps([{'model': k.get('model'), 'system': str(getattr(k.get('config'), 'system_instruction', '')),
                            'contents': [x if isinstance(x, str) else str(type(x)) for x in (k['contents'] if isinstance(k['contents'], list) else [k['contents']])]}
                           for k in ai.calls], default=str)
        assert len(ai.calls) >= 3
        for secret in ('privatename', 'private.person', 'example.com', 'secret diary', 'private-name.png', 'testuser'):
            assert secret not in blob, secret

    def test_photos_are_sent_without_metadata_and_size_limited(self, app, ai):
        c = new_user(app, 'ann')
        use_ai(c)
        big = io.BytesIO()
        Image.fromarray(np.random.default_rng(3).integers(0, 255, (3000, 2400, 3), dtype=np.uint8)).save(big, 'JPEG', exif=self._exif())
        raw = photo(exif=True)
        assert b'SecretModel' in raw                                         # the uploaded file really does carry camera + GPS details
        for data, name in ((raw, 'a.jpg'), (big.getvalue(), 'b.jpg')):
            ai.calls.clear()
            r = c.post('/upload', data={'file': (io.BytesIO(data), name)}, content_type='multipart/form-data').get_json()
            assert r['success']
            sent = [x for x in ai.calls[0]['contents'] if hasattr(x, 'size')][0]
            assert sent.info == {} and len(sent.getexif()) == 0
            assert max(sent.size) <= 2048
            buf = io.BytesIO()
            sent.save(buf, 'JPEG')
            assert b'SecretModel' not in buf.getvalue() and b'SecretCameraMaker' not in buf.getvalue() and b'Exif' not in buf.getvalue()

    @staticmethod
    def _exif():
        e = Image.Exif()
        e[0x0110] = 'SecretModel 9000'
        return e

    def test_the_photo_is_still_deleted_after_being_sent(self, app, ai):
        import os
        c = new_user(app, 'ann')
        use_ai(c)
        c.post('/upload', data={'file': (io.BytesIO(photo()), 'a.png')}, content_type='multipart/form-data')
        assert [n for n in os.listdir(app.config['UPLOAD_FOLDER']) if not n.startswith('.')] == []


class TestTheNotice:
    def test_the_dashboard_asks_clearly_until_the_person_decides(self, app, ai):
        c = new_user(app, 'ann')
        import html as htmllib
        html = htmllib.unescape(c.get('/').get_data(as_text=True))
        assert 'Allow AI analysis?' in html and "Google's Gemini" in html and 'can miss serious problems' in html and 'Keep basic mode' in html
        assert 'never sent' in html.lower() or 'never sent' in html
        use_ai(c)
        assert 'Allow AI analysis?' not in c.get('/').get_data(as_text=True)

    def test_saying_no_is_remembered_and_the_person_is_not_asked_again(self, app, ai):
        c = new_user(app, 'ann')
        assert 'Allow AI analysis?' in c.get('/').get_data(as_text=True)
        r = c.put('/api/privacy/ai-consent', json={'granted': False}).get_json()['consent']
        assert r['granted'] is False and r['decided'] is True
        assert 'Allow AI analysis?' not in c.get('/').get_data(as_text=True)          # remembered
        assert c.post('/analyze_symptoms', json={'symptoms': 'headache and nausea'}).get_json()['analysis']['analysis_mode'] == 'basic' and ai.calls == []
        c.post('/logout')
        again = app.test_client()
        again.post('/login', data={'username': 'ann', 'password': PW})
        assert 'Allow AI analysis?' not in again.get('/').get_data(as_text=True)
        assert 'data-ai-consent-toggle checked' not in again.get('/account').get_data(as_text=True)     # still changeable under Account

    def test_a_new_wording_asks_a_person_who_said_no_only_if_they_never_decided(self, app, ai, monkeypatch):
        c = new_user(app, 'ann')
        c.put('/api/privacy/ai-consent', json={'granted': False})
        monkeypatch.setattr(config, 'AI_CONSENT_VERSION', config.AI_CONSENT_VERSION + 1)
        assert 'Allow AI analysis?' not in c.get('/').get_data(as_text=True)          # a "no" stays a "no" when the wording changes

    def test_no_card_when_ai_is_not_configured_at_all(self, app, monkeypatch):
        monkeypatch.setenv('REQUIRE_AI_CONSENT', 'true')
        c = new_user(app, 'ann')
        assert 'Allow AI analysis?' not in c.get('/').get_data(as_text=True)

    def test_the_account_page_has_the_switch(self, app, ai):
        c = new_user(app, 'ann')
        assert 'Send my text and photos to the AI service' in c.get('/account').get_data(as_text=True)
        use_ai(c)
        assert 'id="aiConsentToggle" data-ai-consent-toggle checked' in c.get('/account').get_data(as_text=True)

    def test_the_privacy_page_states_what_is_and_is_not_sent(self, app):
        html = app.test_client().get('/privacy').get_data(as_text=True)
        for needle in ('Only if you agree', "Google's Gemini API", 'location, camera and time details removed', 'Never sent', 'before</strong> anything is sent',
                       'its own terms for the API plan'):
            assert needle in html, needle

    def test_the_consent_endpoint_is_documented(self, app, client, registered_user, monkeypatch):
        monkeypatch.setenv('ADMIN_USERNAMES', registered_user['username'])
        paths = client.get('/api/openapi.json').get_json()['paths']
        assert 'put' in paths['/api/privacy/ai-consent'] and 'get' in paths['/api/privacy/ai-consent']

    def test_the_script_handles_the_buttons_and_the_toggle(self, client):
        js = client.get('/static/js/app.js').data.decode()
        assert 'data-ai-consent' in js and '/api/privacy/ai-consent' in js and "api('PUT'" not in js
