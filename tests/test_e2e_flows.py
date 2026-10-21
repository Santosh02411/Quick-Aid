"""End-to-end journeys that cross feature boundaries, the way a real person (or attacker) would move through the app."""

import io
import json
import os
import re
from unittest.mock import MagicMock

import numpy as np
import pyotp
import pytest
from PIL import Image

from quickaid.evaluation.clients import FakeResponse, MockClient
from tests.test_ai_consent import Recorder

PW = 'orange-kettle-42'


def person(app, name, lang=None):
    c = app.test_client()
    assert c.post('/register', data={'username': name, 'email': f'{name}@example.com', 'password': PW, 'confirm_password': PW}).status_code == 302
    if lang:
        c.put('/api/language', json={'language': lang})
    return c


def png(seed=1, n=320):
    arr = np.clip(np.random.default_rng(seed).integers(-25, 25, (n, n, 1)) + np.array([215, 165, 145]), 0, 255).astype('uint8')
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, 'PNG')
    return buf.getvalue()


def enable_ai(app, monkeypatch, client=None):
    monkeypatch.setenv('REQUIRE_AI_CONSENT', 'true')
    svc = app.extensions['quickaid_services']
    rec = client or Recorder()
    for obj in (svc.symptom_checker, svc.medical_analyzer, svc.chat_ai):
        obj.use_gemini, obj.client = True, rec
    return rec


class TestTheWholeLife:
    """Register, decide about AI, check, read the report, ask a follow-up, organise, export, delete, come back, leave."""

    def test_a_person_from_signup_to_leaving_and_nothing_remains(self, app, monkeypatch):
        rec = enable_ai(app, monkeypatch)
        c = person(app, 'journey')
        # 1. nothing leaves the server until they choose
        first = c.post('/analyze_symptoms', json={'symptoms': 'sore throat and mild fever since yesterday'}).get_json()
        assert first['analysis']['analysis_mode'] == 'basic' and rec.calls == [] and first['analysis_id']
        # 2. they agree, and the next check uses the AI
        assert c.put('/api/privacy/ai-consent', json={'granted': True}).get_json()['consent']['granted']
        second = c.post('/analyze_symptoms', json={'symptoms': 'headache and nausea since this morning'}).get_json()
        assert second['analysis']['analysis_mode'] == 'ai' and len(rec.calls) == 1
        # 3. the report page and follow-up conversation about it
        report = c.get(f"/history/symptom/{second['analysis_id']}")
        assert report.status_code == 200 and 'rep-hero' in report.get_data(as_text=True) and 'not a diagnosis' in report.get_data(as_text=True).lower()
        chat = c.post('/api/follow_up', json={'analysis_type': 'symptom', 'analysis_id': second['analysis_id'], 'question': 'Should I worry?'}).get_json()
        assert chat['success'] and chat['conversation_id'] and chat['answer']
        # 4. organise: search, annotate, filter
        assert c.get('/api/history?q=headache').get_json()['pagination']['total'] == 1
        assert c.patch(f"/api/history/symptom/{second['analysis_id']}/notes", json={'notes': 'tell the GP'}).get_json()['success']
        assert c.get('/api/history?q=GP').get_json()['pagination']['total'] == 1
        # 5. export has everything, including the chat and the note
        data = json.loads(c.get('/api/history/export').data)
        assert len(data['assessments']) == 2 and data['conversations'] and any(a['notes'] == 'tell the GP' for a in data['assessments'])
        # 6. delete one entry: its chat goes with it, the other entry stays
        assert c.delete(f"/api/history/symptom/{second['analysis_id']}").get_json()['success']
        assert c.get('/api/chats').get_json()['conversations'] == [] and c.get('/api/history').get_json()['pagination']['total'] == 1
        # 7. sign out and back in: everything (including the AI choice) is still theirs
        c.post('/logout')
        assert c.get('/api/history').status_code == 401
        c.post('/login', data={'username': 'journey', 'password': PW})
        assert c.get('/api/history').get_json()['pagination']['total'] == 1 and c.get('/api/privacy/ai-consent').get_json()['consent']['granted']
        # 8. leaving: the account and every row of their data is gone
        c.post('/account/delete', data={'current_password': PW, 'confirm_text': 'DELETE'})
        from quickaid.core import database as db
        with db.get_connection() as conn:
            for table in ('users', 'symptom_analyses', 'chat_conversations', 'chat_messages', 'recovery_codes', 'security_events'):
                assert conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0] == 0, table
        assert app.test_client().post('/login', data={'username': 'journey', 'password': PW}).status_code == 401


class TestEmergencyHandling:
    @pytest.mark.parametrize('lang,text,heading', [
        ('en', 'my father has crushing chest pain and is sweating', 'Chest pain or pressure'),
        ('hi', 'मेरे पिता को सीने में तेज दर्द है', 'सीने में दर्द या दबाव'),
        ('mr', 'माझ्या वडिलांना छातीत दुखतंय', 'छातीत दुखणे किंवा दाब'),
        ('kn', 'ನನ್ನ ತಂದೆಗೆ ಎದೆ ನೋವು ಇದೆ', 'ಎದೆ ನೋವು ಅಥವಾ ಒತ್ತಡ')])
    def test_an_emergency_is_handled_immediately_the_same_way_through_every_door(self, app, monkeypatch, lang, text, heading):
        """One-shot check, interview, and chat: guidance in the person's language, with the number, without waiting for (or trusting) the AI."""
        rec = enable_ai(app, monkeypatch)
        c = person(app, f'emerg{lang}', lang)
        c.put('/api/privacy/ai-consent', json={'granted': True})          # even WITH consent the emergency path does not need the model
        rec.calls.clear()
        one = c.post('/analyze_symptoms', json={'symptoms': text}).get_json()['analysis']
        interview = c.post('/api/interview/start', json={'symptoms': text}).get_json()
        cid = c.post('/api/chats', json={}).get_json()['conversation']['id']
        chat = c.post(f'/api/chats/{cid}/messages', json={'message': text}).get_json()
        for result in (one, interview['analysis']):
            assert result['care_level'] == 'emergency' and result['urgency_level'] == 'high' and result['red_flags_present'] == [heading]
            assert '112' in result['recommendations'][0] and result['emergency_alert']['alert'] is True
        assert interview['status'] == 'emergency' and chat['emergency'] and '112' in chat['assistant_message']['content']
        assert len(rec.calls) == 1                                           # only the one-shot check consulted the model; interview and chat did not
        saved = c.get('/api/history').get_json()['history']
        assert len(saved) == 2 and all(e['level'] == 'high' and e['emergency_alert'] for e in saved)

    def test_an_emergency_still_works_when_the_ai_is_down_and_when_it_lies(self, app, monkeypatch):
        liar = MagicMock()
        liar.models.generate_content.return_value = FakeResponse(json.dumps({'urgency_level': 'low', 'emergency_alert': False,
                                                                              'summary': 'Nothing to worry about, you definitely have a cold.',
                                                                              'possible_explanations': [{'name': 'cold', 'likelihood': 'more likely'}]}), 10, 10)
        enable_ai(app, monkeypatch, liar)
        c = person(app, 'liar')
        c.put('/api/privacy/ai-consent', json={'granted': True})
        r = c.post('/analyze_symptoms', json={'symptoms': 'sudden crushing chest pain, cannot breathe'}).get_json()['analysis']
        assert r['care_level'] == 'emergency' and r['urgency_level'] == 'high' and r['safety_review']['urgency_raised']      # the safety net overrode the model
        assert 'definitely' not in json.dumps(r) and 'Nothing to worry' not in r['summary']
        down = MagicMock()
        down.models.generate_content.side_effect = RuntimeError('503 provider unavailable')
        app.extensions['quickaid_services'].symptom_checker.client = down
        import quickaid.ai.symptom_checker as sc
        monkeypatch.setattr(sc.time, 'sleep', lambda *_: None)
        r2 = c.post('/analyze_symptoms', json={'symptoms': 'crushing chest pain and cannot breathe'}).get_json()['analysis']
        assert r2['care_level'] == 'emergency' and r2['analysis_mode'] == 'basic' and '112' in ' '.join(r2['recommendations'])


class TestWhenTheProviderMisbehaves:
    def test_an_outage_degrades_to_basic_mode_with_an_honest_notice_and_is_recorded(self, app, monkeypatch):
        import quickaid.ai.symptom_checker as sc
        monkeypatch.setattr(sc.time, 'sleep', lambda *_: None)
        down = MagicMock()
        down.models.generate_content.side_effect = TimeoutError('upstream timeout')
        enable_ai(app, monkeypatch, down)
        c = person(app, 'outage')
        c.put('/api/privacy/ai-consent', json={'granted': True})
        r = c.post('/analyze_symptoms', json={'symptoms': 'a mild headache since yesterday'})
        a = r.get_json()['analysis']
        assert r.status_code == 200 and a['analysis_mode'] == 'basic' and a['reliability']['level'] == 'limited' and a['recommendations']
        from quickaid.core import database as db
        calls = db.fetch_ai_calls('2000-01-01')
        assert calls and all(not x['ok'] and x['error_type'] == 'TimeoutError' for x in calls)
        assert any(x['fallback'] for x in db.fetch_ai_analyses('2000-01-01'))

    def test_garbage_from_the_model_is_never_shown(self, app, monkeypatch):
        import quickaid.ai.symptom_checker as sc
        monkeypatch.setattr(sc.time, 'sleep', lambda *_: None)
        bad = MagicMock()
        bad.models.generate_content.return_value = FakeResponse('<script>alert(1)</script> {not json', 5, 5)
        enable_ai(app, monkeypatch, bad)
        c = person(app, 'garbage')
        c.put('/api/privacy/ai-consent', json={'granted': True})
        r = c.post('/analyze_symptoms', json={'symptoms': 'a mild headache since yesterday'}).get_json()['analysis']
        assert r['analysis_mode'] == 'basic' and '<script>' not in json.dumps(r)

    def test_a_crash_in_analysis_never_loses_the_photo_cleanup(self, app, monkeypatch):
        c = person(app, 'crash')
        monkeypatch.setattr(app.extensions['quickaid_services'].medical_analyzer, 'analyze_image', MagicMock(side_effect=RuntimeError('boom')))
        r = c.post('/upload', data={'file': (io.BytesIO(png()), 'a.png')}, content_type='multipart/form-data')
        assert r.status_code == 500 and r.get_json()['code'] == 'analysis_failed'
        assert [n for n in os.listdir(app.config['UPLOAD_FOLDER']) if not n.startswith('.')] == []


class TestPhotoJourney:
    def test_good_bad_and_unusable_photos(self, app):
        c = person(app, 'photos')
        ok = c.post('/upload', data={'file': (io.BytesIO(png()), 'a.png')}, content_type='multipart/form-data').get_json()
        assert ok['success'] and {'findings', 'possible_explanations', 'not_assessable'} <= set(ok['analysis'])
        dark = np.clip((np.random.default_rng(2).integers(-25, 25, (300, 300, 1)) + np.array([215, 165, 145])) * 0.05, 0, 255).astype('uint8')
        buf = io.BytesIO()
        Image.fromarray(dark).save(buf, 'PNG')
        unusable = c.post('/upload', data={'file': (io.BytesIO(buf.getvalue()), 'dark.png')}, content_type='multipart/form-data').get_json()
        assert unusable['analysis']['assessment_status'] == 'insufficient' and unusable['analysis']['possible_explanations'] == []
        bad = c.post('/upload', data={'file': (io.BytesIO(b'%PDF-1.4 not a photo'), 'x.png')}, content_type='multipart/form-data')
        assert bad.status_code == 400 and bad.get_json()['hints']
        assert c.get('/api/history').get_json()['pagination']['total'] == 2            # the rejected file left no trace
        assert [n for n in os.listdir(app.config['UPLOAD_FOLDER']) if not n.startswith('.')] == []


class TestAccessControlJourney:
    def test_a_curious_neighbour_gets_nothing_and_cannot_tell_anything_exists(self, app):
        alice, mallory = person(app, 'alice'), person(app, 'mallory')
        a = alice.post('/analyze_symptoms', json={'symptoms': 'private rash on my arm for a week'}).get_json()
        alice.post('/api/follow_up', json={'analysis_type': 'symptom', 'analysis_id': a['analysis_id'], 'question': 'Is it contagious?'})
        cid = alice.get('/api/chats').get_json()['conversations'][0]['id']
        attempts = [mallory.get(f"/api/history/symptom/{a['analysis_id']}"), mallory.get(f"/history/symptom/{a['analysis_id']}"),
                    mallory.patch(f"/api/history/symptom/{a['analysis_id']}/notes", json={'notes': 'x'}), mallory.delete(f"/api/history/symptom/{a['analysis_id']}"),
                    mallory.get(f"/api/history/symptom/{a['analysis_id']}/export"), mallory.get(f'/api/chats/{cid}'), mallory.get(f'/api/chats/{cid}/export'),
                    mallory.post(f'/api/chats/{cid}/messages', json={'message': 'hi'}), mallory.delete(f'/api/chats/{cid}'),
                    mallory.post('/api/follow_up', json={'analysis_type': 'symptom', 'analysis_id': a['analysis_id'], 'question': 'what is it?'})]
        assert {r.status_code for r in attempts} == {404}
        assert mallory.get('/api/history?q=rash').get_json()['history'] == [] and json.loads(mallory.get('/api/history/export').data)['assessments'] == []
        assert 'rash' not in mallory.get('/history').get_data(as_text=True)
        assert alice.get(f"/api/history/symptom/{a['analysis_id']}").get_json()['entry']['symptom_text'].startswith('private rash')


class TestAccountRecoveryJourney:
    def test_forgot_password_then_2fa_then_lost_phone(self, app, monkeypatch):
        sent = []
        monkeypatch.setattr('quickaid.blueprints.auth.send_email', lambda to, subject, body, *a, **k: sent.append(body))
        monkeypatch.setattr('quickaid.services.security.send_email', lambda to, subject, body, *a, **k: sent.append(body))
        c = person(app, 'recover')
        # turn on 2FA and save the recovery codes
        c.get('/account/2fa/setup')
        with c.session_transaction() as s:
            secret = s['pending_totp_secret']
        page = c.post('/account/2fa/confirm', data={'code': pyotp.TOTP(secret).now()}).get_data(as_text=True)
        codes = re.findall(r'\b[A-Z2-9]{5}-[A-Z2-9]{5}\b', page)
        assert len(codes) == 10
        # forgets the password: reset by email; every session is signed out, and 2FA still protects the account
        c2 = app.test_client()
        c2.post('/forgot-password', data={'email': 'recover@example.com'})
        link = re.search(r'/reset-password/[\w\-]+', sent[-1]).group(0)
        assert c2.post(link, data={'password': 'purple-teapot-77', 'confirm_password': 'purple-teapot-77'}).status_code == 200
        assert c.get('/api/history').status_code == 401
        r = c2.post('/login', data={'username': 'recover', 'password': 'purple-teapot-77'})
        assert '/login/2fa' in r.headers['Location']
        # and the phone is lost too: a recovery code gets them in, once
        assert c2.post('/login/2fa', data={'code': codes[0]}).status_code == 302 and c2.get('/api/history').status_code == 200
        c2.post('/logout')
        c2.post('/login', data={'username': 'recover', 'password': 'purple-teapot-77'})
        assert c2.post('/login/2fa', data={'code': codes[0]}).status_code == 401
        assert any('recovery code was just used' in b or 'recovery code' in b for b in sent)


class TestPrivacyAndRetentionJourney:
    def test_retention_choice_then_sweep_then_deletion_is_physical(self, app):
        from quickaid.core import database as db, history_repo
        c = person(app, 'retain')
        old = c.post('/analyze_symptoms', json={'symptoms': 'zzuniquemarkerzz old complaint'}).get_json()['analysis_id']
        new = c.post('/analyze_symptoms', json={'symptoms': 'recent complaint'}).get_json()['analysis_id']
        with db.get_connection() as conn:
            conn.execute("UPDATE symptom_analyses SET created_at = '2020-01-01T00:00:00+00:00' WHERE id = ?", (old,))
        assert c.put('/api/privacy/retention', json={'days': 90}).get_json()['retention']['effective_days'] == 90
        report = app.extensions['quickaid_services'].retention.sweep(app.config['UPLOAD_FOLDER'])
        assert report['analyses_purged'] == 1 and c.get(f'/api/history/symptom/{old}').status_code == 404 and c.get(f'/api/history/symptom/{new}').status_code == 200
        raw = open(db.DB_PATH, 'rb').read() + (open(db.DB_PATH + '-wal', 'rb').read() if os.path.exists(db.DB_PATH + '-wal') else b'')
        assert b'zzuniquemarkerzz' not in raw


class TestLanguageJourney:
    def test_a_marathi_speaker_from_first_visit_to_a_saved_preference(self, app):
        first = app.test_client()
        page = first.get('/', headers={'Accept-Language': 'mr-IN,mr;q=0.9,en;q=0.4'}).get_data(as_text=True)
        assert '<html lang="mr"' in page and 'class="lang-notice"' in page
        c = person(app, 'marathi')
        c.put('/api/language', json={'language': 'mr'})
        out = c.post('/api/interview/start', json={'symptoms': 'माझ्या वडिलांना छातीत दुखतंय'}).get_json()['analysis']
        assert out['care_level'] == 'emergency' and '112 वर कॉल करा' in out['recommendations'][0]
        elsewhere = app.test_client()
        elsewhere.post('/login', data={'username': 'marathi', 'password': PW})
        assert '<html lang="mr"' in elsewhere.get('/history').get_data(as_text=True)
