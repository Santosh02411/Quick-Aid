"""First-aid help for everyday problems in the built-in mode, plus the time limits that keep results fast."""

import json
import re
from unittest.mock import MagicMock

import pytest

from quickaid.ai import first_aid, model_calls
from quickaid.evaluation.clients import FakeResponse, MockClient
from tests.test_ai_consent import Recorder, photo

PW = 'orange-kettle-42'


def signed_in(app, name='ann'):
    c = app.test_client()
    c.post('/register', data={'username': name, 'email': f'{name}@example.com', 'password': PW, 'confirm_password': PW})
    return c


def check(c, text):
    return c.post('/analyze_symptoms', json={'symptoms': text}).get_json()['analysis']


PHRASES = {
    'cut': ['small cut on my finger', 'I scraped my knee', 'cut myself with a knife while cooking', 'paper cut on my thumb'],
    'nosebleed': ['my nose is bleeding', 'I have a nosebleed', 'bloody nose since this morning'],
    'burn': ['I burned my hand', 'scalded my arm with hot water', 'burn from boiling oil'],
    'sprain': ['I twisted my ankle', 'sprained my wrist', 'pulled a muscle in my back', 'rolled my ankle playing football'],
    'bruise': ['big bruise on my leg', 'I bumped my knee and it is bruising', 'black and blue arm'],
    'insect': ['bee sting on my arm', 'mosquito bites all over my legs', 'a wasp stung me', 'bitten by a spider'],
    'animal_bite': ['a dog bit me', 'cat scratched my hand', 'bitten by a stray dog', 'monkey bite on my arm'],
    'blister': ['blister on my heel', 'my feet are blistering from new shoes'],
    'splinter': ['splinter in my thumb', 'thorn in my foot'],
    'sunburn': ['I got sunburnt', 'bad sunburn on my shoulders', 'too much sun yesterday'],
    'eye': ['dust in my eye', 'something in my eye', 'shampoo got in my eye', 'bleach splashed in my eye'],
    'head_bump': ['I hit my head on the door', 'my child bumped his head', 'head injury from a fall'],
    'broken_bone': ['I think my arm is broken', 'fractured wrist', 'broke my finger', 'my leg might be broken'],
    'choking': ['my baby is choking', 'choking on food', 'something is stuck in my throat', 'child swallowed a coin'],
    'heat': ['heat exhaustion after working outside', 'heatstroke', 'feeling faint in the heat'],
}
NOT_FIRST_AID = ['fever and cough for two days', 'heartburn and burning when I pee', 'itchy skin, scratching a lot', 'headache and nausea', 'I cut down on sugar',
                 'cutting pain in my stomach', 'I feel burnt out', 'sore throat', 'back pain for a week', 'feeling sad and tired', 'diarrhoea and vomiting']


class TestKnowledge:
    @pytest.mark.parametrize('gid', list(PHRASES))
    def test_everyday_phrasings_find_the_right_guide(self, gid):
        for text in PHRASES[gid]:
            assert gid in [g.id for g in first_aid.match(text)], text

    def test_illness_descriptions_and_idioms_are_left_to_the_illness_checker(self):
        assert [(t, [g.id for g in first_aid.match(t)]) for t in NOT_FIRST_AID if first_aid.match(t)] == []

    def test_every_guide_has_the_parts_a_person_needs(self):
        assert {g.id for g in first_aid.GUIDES} == set(PHRASES)
        for g in first_aid.GUIDES:
            assert len(g.steps) >= 3 and g.avoid and g.see_doctor and g.why and g.summary, g.id
            assert all(s[0].isupper() for s in g.steps), g.id

    def test_no_guide_prescribes_a_dose_a_brand_or_a_diagnosis(self):
        for g in first_aid.GUIDES:
            text = ' '.join(g.steps + g.avoid + g.see_doctor)
            assert not re.search(r'\b\d+\s?(?:mg|mcg|ml|g|tablets?|pills?)\b', text, re.I), g.id            # "follow the label", never an amount
            assert not re.search(r'\b(?:Tylenol|Advil|Neosporin|Dettol|Savlon|Crocin|Dolo|Betadine|Calpol)\b', text), g.id
            assert not re.search(r'\byou (?:definitely|certainly) have\b|\bdiagnos', text, re.I), g.id

    def test_the_dangerous_wrong_advice_is_never_given(self):
        text = {g.id: ' '.join(g.steps) for g in first_aid.GUIDES}
        assert 'ice' not in re.sub(r'ice-cold', '', text['burn']).lower().replace('rice', '') or 'Do not use ice' in ' '.join(first_aid.BY_ID['burn'].avoid)
        assert 'tilt' not in text['nosebleed'].lower() or 'Do not tilt' in ' '.join(first_aid.BY_ID['nosebleed'].avoid)
        assert any('Do not tilt your head back' in a for a in first_aid.BY_ID['nosebleed'].avoid)
        assert any('butter' in a and 'toothpaste' in a for a in first_aid.BY_ID['burn'].avoid)
        assert any('chilli' in a or 'turmeric' in a for a in first_aid.BY_ID['animal_bite'].avoid)
        assert any('Never do abdominal thrusts on a baby' in s for s in first_aid.BY_ID['choking'].steps)
        assert 'inhaler' not in text['choking'].lower() and 'sit upright' not in text['choking'].lower()
        assert 'rabies' in text['animal_bite'].lower() and '15 minutes' in text['animal_bite']
        assert any('rinse' in s.lower() and '15' in s for s in first_aid.BY_ID['eye'].steps)

    def test_emergency_numbers_are_placeholders_until_localised(self):
        assert any('{EMERGENCY_NUMBER}' in s for g in first_aid.GUIDES for s in g.steps)


class TestWhatThePersonSees:
    @pytest.mark.parametrize('text,heading', [('small cut on my finger, bleeding stopped', 'Minor cut'), ('I burned my hand with hot water', 'Burn or scald'),
                                              ('twisted my ankle playing football', 'Sprain'), ('bee sting on my arm', 'Insect bite'), ('dust in my eye', 'eye'),
                                              ('big blister on my heel', 'Blister'), ('my nose is bleeding', 'Nosebleed')])
    def test_a_real_explanation_and_real_first_aid_not_just_see_a_doctor(self, app, text, heading):
        c = signed_in(app)
        a = check(c, text)
        assert a['analysis_mode'] == 'basic' and 'error' not in a
        assert heading.lower() in a['summary'].lower() and 'Keyword-based check only' not in a['summary']
        assert heading.lower() in a['possible_explanations'][0]['name'].lower() and a['possible_explanations'][0]['why']
        assert len(a['recommendations']) >= 4 and not a['recommendations'][0].lower().startswith('see a doctor')      # first aid first
        assert a['avoid'] and a['see_doctor_when'] and 'not a diagnosis' in a['disclaimer'].lower()

    def test_the_old_dead_end_is_gone_but_gibberish_still_gets_a_helpful_prompt(self, app):
        c = signed_in(app)
        a = check(c, 'asdf qwer zxcv')
        assert 'error' in a and 'a little more' in a['recommendations'][0] and '112' in ' '.join(a['recommendations'])

    @pytest.mark.parametrize('text,care', [
        ('small cut on my finger', 'self_care'), ('deep cut that will not stop bleeding', 'same_day'), ('cut on my face from glass', 'same_day'),
        ('I burned my hand', 'soon'), ('burn from boiling water on my forearm', 'self_care'), ('a chemical burn on my arm', 'same_day'),
        ('a dog bit me on the leg', 'same_day'), ('my nose is bleeding', 'self_care'), ('nosebleed after I fell and hit my face', 'same_day'),
        ('I twisted my ankle', 'self_care'), ('twisted my ankle and cannot walk on it', 'same_day'), ('bee sting on my arm', 'self_care'),
        ('wasp sting and my lips are swelling', 'emergency'), ('I hit my head on the door', 'soon'), ('hit my head and keep vomiting', 'same_day'),
        ('I think my wrist is broken', 'same_day'), ('chemical splashed in my eye', 'same_day'), ('big blister on my heel', 'self_care'),
        ('my baby is choking', 'emergency'), ('heat exhaustion and now confused', 'emergency'), ('arm broken with bone sticking out', 'emergency')])
    def test_the_care_level_matches_how_serious_the_description_is(self, app, text, care):
        c = signed_in(app)
        assert check(c, text)['care_level'] == care, text

    def test_urgent_guides_put_the_time_critical_step_first_then_the_instruction_to_get_seen(self, app):
        c = signed_in(app)
        bite = check(c, 'a dog bit me on the leg')
        assert 'Wash the wound' in bite['recommendations'][0] and any(r.startswith('Get seen by a doctor today') for r in bite['recommendations'][:5])
        assert bite['timeframe'] == 'Today'

    def test_choking_gets_choking_steps_not_breathing_difficulty_steps_on_every_route(self, app):
        c = signed_in(app)
        one_shot = check(c, 'my baby is choking')
        interview = c.post('/api/interview/start', json={'symptoms': 'my baby is choking'}).get_json()
        cid = c.post('/api/chats', json={}).get_json()['conversation']['id']
        for steps in (' '.join(one_shot['recommendations']), ' '.join(interview['analysis']['recommendations'])):
            assert 'back blows' in steps and 'abdominal thrusts' in steps and 'Never do abdominal thrusts on a baby' in steps
            assert 'inhaler' not in steps.lower() and 'sit upright' not in steps.lower() and '112' in steps
        assert interview['status'] == 'emergency' and one_shot['emergency_alert']['alert']

    def test_the_guides_never_lower_a_danger_the_engine_found(self, app):
        c = signed_in(app)
        a = check(c, 'small cut on my finger and I also have crushing chest pain and cannot breathe')
        assert a['care_level'] == 'emergency' and '112' in a['recommendations'][0] or 'Call 112' in ' '.join(a['recommendations'][:3])

    def test_a_first_aid_problem_together_with_an_illness_gets_both(self, app):
        c = signed_in(app)
        a = check(c, 'I burned my hand and I also have a fever and cough')
        names = [e['name'] for e in a['possible_explanations']]
        assert 'Burn or scald' in names[0] and len(names) > 1 and a['recommendations'][0].startswith('Cool the burn')

    def test_the_emergency_number_follows_the_region(self, app, monkeypatch):
        monkeypatch.setenv('APP_REGION', 'US')
        c = signed_in(app)
        steps = ' '.join(check(c, 'my baby is choking')['recommendations'])
        assert '911' in steps and '{EMERGENCY_NUMBER}' not in steps

    def test_injuries_skip_the_questionnaire_and_illnesses_still_get_questions(self, app):
        c = signed_in(app)
        cut = c.post('/api/interview/start', json={'symptoms': 'I cut my finger while cooking'}).get_json()
        assert cut['status'] == 'complete' and cut['analysis']['care_level'] == 'self_care' and cut['analysis']['recommendations'][0].startswith('Wash your hands')
        ill = c.post('/api/interview/start', json={'symptoms': 'fever and cough for two days'}).get_json()
        assert ill['status'] == 'question' and ill['question']['text']
        both = c.post('/api/interview/start', json={'symptoms': 'I burned my hand and now have a fever'}).get_json()
        assert both['status'] == 'question'                                  # an illness is also described: the questions still help

    def test_the_result_is_saved_and_appears_in_history_and_the_report(self, app):
        c = signed_in(app)
        r = c.post('/api/interview/start', json={'symptoms': 'I cut my finger while cooking'}).get_json()
        page = c.get(f"/history/symptom/{r['analysis_id']}").get_data(as_text=True)
        assert 'Minor cut or scrape' in page and 'Wash your hands' in page


class TestSpeedSettings:
    @pytest.fixture(autouse=True)
    def _clean(self, monkeypatch):
        for k in ('GEMINI_TIMEOUT_SECONDS', 'GEMINI_THINKING_BUDGET', 'GEMINI_TOTAL_BUDGET_SECONDS'):
            monkeypatch.delenv(k, raising=False)
        model_calls._NO_THINKING.clear()
        yield
        model_calls._NO_THINKING.clear()

    def test_every_request_has_a_hard_timeout_and_flash_models_do_not_think(self):
        kw = model_calls.speed_kwargs('gemini-flash-latest')
        assert kw['http_options'].timeout == 25000 and kw['thinking_config'].thinking_budget == 0

    def test_the_limits_are_configurable_and_cannot_be_set_to_nonsense(self, monkeypatch):
        monkeypatch.setenv('GEMINI_TIMEOUT_SECONDS', '12')
        assert model_calls.speed_kwargs('gemini-flash-latest')['http_options'].timeout == 12000
        monkeypatch.setenv('GEMINI_TIMEOUT_SECONDS', 'banana')
        assert model_calls.speed_kwargs('gemini-flash-latest')['http_options'].timeout == 25000
        monkeypatch.setenv('GEMINI_TIMEOUT_SECONDS', '0')
        assert model_calls.speed_kwargs('gemini-flash-latest')['http_options'].timeout >= 2000

    def test_pro_models_are_left_alone_and_thinking_can_be_restored_or_budgeted(self, monkeypatch):
        assert 'thinking_config' not in model_calls.speed_kwargs('gemini-2.5-pro')
        monkeypatch.setenv('GEMINI_THINKING_BUDGET', 'default')
        assert 'thinking_config' not in model_calls.speed_kwargs('gemini-flash-latest')
        monkeypatch.setenv('GEMINI_THINKING_BUDGET', '256')
        assert model_calls.speed_kwargs('gemini-2.5-pro')['thinking_config'].thinking_budget == 256

    def test_a_model_that_rejects_the_thinking_setting_is_remembered(self):
        model_calls.note_error('some-model', Exception('Thinking budget is not supported for this model'))
        assert 'thinking_config' not in model_calls.speed_kwargs('some-model') and 'thinking_config' in model_calls.speed_kwargs('gemini-flash-latest')

    def test_retry_policy(self, monkeypatch):
        b = model_calls.Budget(40)
        assert b.allows_retry(Exception('503 Service Unavailable')) is True and b.allows_retry(Exception('429 rate limited')) is True
        assert b.allows_retry(TimeoutError('x')) is False and b.allows_retry(Exception('Request timed out')) is False and b.allows_retry(Exception('ReadTimeout')) is False
        now = [100.0]
        monkeypatch.setattr(model_calls.time, 'monotonic', lambda: now[0])
        late = model_calls.Budget(40)
        now[0] = 135.0
        assert late.allows_retry(Exception('503')) is False                  # the time budget is spent

    def test_every_model_call_carries_the_limits(self, app, monkeypatch):
        monkeypatch.setenv('INTERVIEW_AI_PLAN', 'true')
        svc = app.extensions['quickaid_services']
        rec = Recorder()
        for obj in (svc.symptom_checker, svc.medical_analyzer, svc.chat_ai):
            obj.use_gemini, obj.client = True, rec
        c = signed_in(app)
        check(c, 'headache and nausea since this morning')
        import io
        c.post('/upload', data={'file': (io.BytesIO(photo()), 'a.png')}, content_type='multipart/form-data')
        cid = c.post('/api/chats', json={}).get_json()['conversation']['id']
        c.post(f'/api/chats/{cid}/messages', json={'message': 'is it serious?'})
        started = c.post('/api/interview/start', json={'symptoms': 'cough for a few days'}).get_json()
        c.post('/api/interview/answer', json={'session_id': started['session_id'], 'question_id': started['question']['id'], 'skip': True})
        kinds = {}
        for k in rec.calls:
            cfg = k['config']
            assert cfg.http_options.timeout >= 8000 and cfg.thinking_config.thinking_budget == 0, k['model']
            kinds[len(kinds)] = cfg.http_options.timeout
        assert len(rec.calls) >= 4 and 35000 in kinds.values() and 8000 in kinds.values()      # photos get a little longer, planning gets a short limit

    def test_a_timeout_goes_straight_to_the_built_in_answer_without_a_pointless_retry(self, app, monkeypatch):
        import quickaid.ai.symptom_checker as sc
        slept = []
        monkeypatch.setattr(sc.time, 'sleep', lambda s: slept.append(s))
        down = MagicMock()
        down.models.generate_content.side_effect = TimeoutError('The read operation timed out')
        svc = app.extensions['quickaid_services']
        svc.symptom_checker.use_gemini, svc.symptom_checker.client = True, down
        c = signed_in(app)
        a = check(c, 'I burned my hand with hot water')
        assert down.models.generate_content.call_count == 1 and slept == []
        assert a['analysis_mode'] == 'basic' and a['recommendations'][0].startswith('Cool the burn')      # and the fallback is genuinely useful

    def test_quick_failures_are_retried_once_then_fall_back(self, app, monkeypatch):
        import quickaid.ai.symptom_checker as sc
        monkeypatch.setattr(sc.time, 'sleep', lambda s: None)
        flaky = MagicMock()
        flaky.models.generate_content.side_effect = Exception('503 Service Unavailable')
        svc = app.extensions['quickaid_services']
        svc.symptom_checker.use_gemini, svc.symptom_checker.client = True, flaky
        a = check(signed_in(app), 'headache and nausea since this morning')
        assert flaky.models.generate_content.call_count == 2 and a['analysis_mode'] == 'basic'

    def test_a_rejected_thinking_setting_is_dropped_on_the_retry_and_the_ai_still_answers(self, app, monkeypatch):
        import quickaid.ai.symptom_checker as sc
        monkeypatch.setattr(sc.time, 'sleep', lambda s: None)

        class Picky(Recorder):
            def generate_content(self, **kwargs):
                if not self.calls and getattr(kwargs['config'], 'thinking_config', None) is not None:
                    self.calls.append(kwargs)
                    raise Exception('400 INVALID_ARGUMENT: thinking budget cannot be set for this model')
                return super().generate_content(**kwargs)
        picky = Picky()
        svc = app.extensions['quickaid_services']
        svc.symptom_checker.use_gemini, svc.symptom_checker.client = True, picky
        a = check(signed_in(app), 'headache and nausea since this morning')
        assert a['analysis_mode'] == 'ai' and len(picky.calls) == 2 and getattr(picky.calls[1]['config'], 'thinking_config', None) is None

    def test_the_question_bank_is_instant_by_default_and_ai_planning_is_opt_in(self, app, monkeypatch):
        svc = app.extensions['quickaid_services']
        rec = Recorder()
        for obj in (svc.symptom_checker, svc.medical_analyzer, svc.chat_ai):
            obj.use_gemini, obj.client = True, rec
        c = signed_in(app)
        first = c.post('/api/interview/start', json={'symptoms': 'cough for a few days'}).get_json()
        assert first['status'] == 'question' and rec.calls == []             # the first question appears with ZERO model calls
        monkeypatch.setenv('INTERVIEW_AI_PLAN', 'true')
        c.post('/api/interview/start', json={'symptoms': 'cough for a few days'})
        assert len(rec.calls) == 1


class TestTheScreen:
    def test_the_script_has_a_request_timeout_a_slow_hint_and_the_ai_off_notice(self, client):
        js = client.get('/static/js/app.js').data.decode('utf-8')
        for needle in ("new AbortController()", "setTimeout(function () { ctl.abort(); }, 75000)", "xhr.timeout = 90000", "slow-hint", "T('slow_hint'", "T('too_slow'", "ai_off_notice", "data-reload"):
            assert needle in js, needle

    def test_the_page_tells_the_script_whether_ai_is_available_and_allowed(self, app, monkeypatch):
        monkeypatch.setenv('REQUIRE_AI_CONSENT', 'true')
        svc = app.extensions['quickaid_services']
        svc.symptom_checker.use_gemini = True
        c = signed_in(app)
        assert '"allowed": false' in c.get('/').get_data(as_text=True).replace('":false', '": false') and '"available": true' in c.get('/').get_data(as_text=True).replace('":true', '": true')
        c.put('/api/privacy/ai-consent', json={'granted': True})
        assert '"allowed":true' in c.get('/').get_data(as_text=True).replace('": true', '":true')
        svc.symptom_checker.use_gemini = False
        assert '"available":false' in c.get('/').get_data(as_text=True).replace('": false', '":false')

    def test_the_new_strings_exist_in_every_language(self):
        from quickaid.i18n.catalog import STRINGS
        for key in ('js.slow_hint', 'js.too_slow', 'js.ai_off_notice', 'js.ai_turn_on'):
            assert set(STRINGS[key]) == {'en', 'hi', 'mr', 'kn'} and all(STRINGS[key][l].strip() for l in STRINGS[key])
