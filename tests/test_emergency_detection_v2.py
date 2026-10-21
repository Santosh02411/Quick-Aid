"""Tests for the emergency-detection improvements made in response to HOLDOUT2_RESULT.md (30.8% -> 100% recall) and
HOLDOUT3_RESULT.md (a fresh set, honestly run once at 37.9% before a second round of fixes brought it to 100% too).

Covers: the combination-signal layer (indirect phrasing with no single strong keyword), the typo-tolerance layer
(curated misspellings + careful fuzzy matching), specific new patterns, context parsing now applying to one-shot
checks (not just interviews), the Hindi/Marathi/Kannada stacked-intensifier regex fix, and the last-resort emergency
check that now runs if the normal analysis pipeline crashes outright, across every place that could crash.
"""

import json
from unittest.mock import MagicMock

import pytest

from quickaid.ai import triage_engine as te
from quickaid.evaluation.clients import FakeResponse
from quickaid.i18n import emergency as em
from quickaid.services import safety


class TestComboSignals:
    """Indirect descriptions with no single strong phrase: a heart attack described without ever saying 'chest'."""

    @pytest.mark.parametrize('text', [
        "my uncle suddenly went pale, clutched his left arm and fell down",
        "my dad went grey in the face, grabbed at his arm and collapsed on the kitchen floor",
        "she went white, was sweating buckets and sank to the floor holding her shoulder",
        "my mother went ghost white, grabbed the table and crumpled to the ground",
    ])
    def test_cardiac_combination_without_the_word_chest_pain(self, text):
        assert 'Chest pain or pressure' in te.scan(text).emergency

    @pytest.mark.parametrize('text', [
        "found him on the floor, completely limp, wouldn't react no matter what we did",
        "he's gone a strange colour and just isn't answering me at all",
        "we couldn't get her to react, she was just lying there limp",
        "newborn's gone floppy and won't latch, her skin looks mottled and blotchy",
    ])
    def test_collapse_combination_without_the_word_unconscious(self, text):
        assert 'Loss of consciousness or seizure' in te.scan(text).emergency

    @pytest.mark.parametrize('text', [
        "she wasn't pale at all, she just tripped and fell over",
        "I was a bit pale this morning but I feel fine now, just tired from the gym",
        "we went to a dog breeding show and I felt a bit faint from the heat",
        "she overdoes her exercise routine most days",
        "I keep smelling gas in the kitchen",
        "he fell asleep on the sofa",
        "my shoulder aches a bit after the gym",
        "I grabbed the table to steady myself while reaching for a book",
        "the paper crumpled in my hand as I threw it in the bin",
        "she went pale when she saw the bill, but she's fine",
        "I can't catch my breath after that run, give me a minute",
        "he dropped to the ground to do push-ups",
        "the ladder in the shed needs fixing, a rung gave way last week",
        "I drank the whole bottle of orange juice this morning",
        "she took the whole packet of biscuits to the park",
        "he seized the opportunity to ask for a raise",
        "the committee seized control of the agenda",
    ])
    def test_a_single_weak_signal_or_an_unrelated_sense_of_the_same_words_never_fires_alone(self, text):
        assert not te.scan(text).any_emergency, text

    def test_a_combo_rule_needs_at_least_two_groups_not_one(self):
        rule = next(r for r in te.COMBO_RULES if r.label == 'Chest pain or pressure')
        assert rule.min_groups == 2 and len(rule.groups) >= 2




class TestTypoTolerance:
    def test_the_curated_dictionary_corrects_common_real_misspellings(self):
        cases = {
            'uncouncious': 'unconscious', 'unconsious': 'unconscious', 'breathng': 'breathing',
            'bleading': 'bleeding', 'siezure': 'seizure', 'poisning': 'poisoning', 'chokeing': 'choking',
            'colapsed': 'collapsed', 'siezed': 'seized', 'weedkiler': 'weedkiller',
        }
        for typo, fixed in cases.items():
            assert te.TYPO_FIXES.get(typo) == fixed or fixed in te.normalize_typos(typo).lower()

    def test_misspelt_emergency_text_is_still_caught(self):
        assert te.scan("he just colapsed and is unconsious, not breething at all").any_emergency
        assert 'Poisoning or overdose' in te.scan("she drank the weedkiler by accident and is vomitting").emergency
        assert 'Loss of consciousness or seizure' in te.scan("he siezed for over a minute and his lips went purple").emergency

    def test_correctly_spelled_words_are_never_altered(self):
        assert te.normalize_typos("the weather is lovely today") == "the weather is lovely today"

    @pytest.mark.parametrize('risky,must_not_become', [
        ("we went to a dog breeding show", "bleeding"),
        ("she overdoes her exercise routine", "overdose"),
        ("I keep smelling gas in the kitchen", "swelling"),
        ("he never had any trouble with it", "fever"),
    ])
    def test_fuzzy_matching_never_corrupts_a_real_everyday_word_into_a_medical_one(self, risky, must_not_become):
        assert must_not_become not in te.normalize_typos(risky).lower()

    def test_the_fuzzy_lane_only_targets_hand_reviewed_long_words(self):
        for term in te._FUZZY_SAFE_TERMS:
            assert len(term) >= 7, term    # short words are exactly where accidental collisions with real words happen

    def test_one_edit_apart_handles_insertion_deletion_substitution_and_transposition(self):
        assert te._one_edit_apart('unconscious', 'unconscious')
        assert te._one_edit_apart('uncoscious', 'unconscious')      # deletion
        assert te._one_edit_apart('unconsciosu', 'unconscious')     # adjacent transposition
        assert not te._one_edit_apart('conscious', 'unconscious')   # two edits (missing 'un')
        assert te._one_edit_apart('breeding', 'bleeding')           # genuinely IS one edit apart - the distance check
        # is honest about that; the actual safety guarantee is that 'bleeding' is deliberately kept OFF the fuzzy-safe
        # list below (handled only by exact misspellings in TYPO_FIXES instead), which the next test checks directly.

    def test_known_collision_risk_words_are_kept_off_the_generic_fuzzy_list(self):
        for risky in ('bleeding', 'overdose', 'swelling', 'fever', 'stroke', 'chest'):
            assert risky not in te._FUZZY_SAFE_TERMS, risky


class TestNewDirectPatterns:
    """Specific gaps found and fixed against HOLDOUT2_RESULT.md and HOLDOUT3_RESULT.md, each checked with wording
    that was never in either dataset (a direct dataset-text match would prove nothing about generalisation)."""

    @pytest.mark.parametrize('text,label', [
        ("I had shellfish and now my throat feels like it's closing up and I'm covered in hives", 'Severe allergic reaction'),
        ("his face is swelling up and he's covered in a rash after the wasp sting", 'Severe allergic reaction'),
        ("she necked a whole bottle of her dad's blood pressure tablets", 'Poisoning or overdose'),
        ("the kids got into the shed, there's weedkiller spilled and an open container", 'Poisoning or overdose'),
        ("he sprayed oven cleaner straight into his own mouth by accident", 'Poisoning or overdose'),
        ("my son is having a bad asthma attack and can barely speak", 'Severe trouble breathing'),
        ("she can't catch her breath at all, it's getting worse by the minute", 'Severe trouble breathing'),
        ("the toddler is grunting with every breath and his ribs are pulling in", 'Severe trouble breathing'),
        ("he's been fitting on the sofa for five minutes and won't stop", 'Loss of consciousness or seizure'),
        ("one side of her face has completely dropped and her speech is jumbled", 'Possible stroke signs'),
        ("36 weeks gone and soaking through more than one pad an hour", 'Pregnancy with heavy bleeding'),
        ("blood's pouring out of the wound and soaking through every bandage we have", 'Severe bleeding'),
        ("I can see tendon through the cut on his palm", 'Deep wound exposing tendon, bone or muscle'),
    ])
    def test_fresh_wording_for_each_category_is_caught(self, text, label):
        sig = te.scan(text)
        assert label in sig.emergency or label in sig.urgent, (text, sig.emergency, sig.urgent)

    @pytest.mark.parametrize('text', [
        "sharp pain behind my right eye that comes and goes",
        "blood when I wipe, on and off for a week",
        "ringing in my ears that won't go away, been like this for 6 days",
        "a lump showed up in my armpit a few days ago",
    ])
    def test_fresh_wording_for_non_emergency_urgent_categories_is_caught(self, text):
        sig = te.scan(text)
        assert sig.soon or sig.urgent, text


class TestOneShotContextWiring:
    """Duration, age and severity parsing used to only apply inside the question-and-answer interview; the exact
    same wording in a single message got none of it. context_floors() now applies to one-shot checks too."""

    def test_a_long_duration_raises_urgency_in_a_single_message(self, app):
        from quickaid.ai.symptom_checker import SymptomChecker
        c = SymptomChecker()
        r = c.analyze_symptoms("I've had a sore throat for 3 weeks now")
        assert r['urgency_level'] in ('medium', 'high')

    def test_a_short_duration_does_not_falsely_raise_urgency(self, app):
        from quickaid.ai.symptom_checker import SymptomChecker
        c = SymptomChecker()
        r = c.analyze_symptoms("I've had a sore throat since yesterday")
        assert r.get('urgency_level') in (None, 'low')

    def test_context_floors_are_reached_through_review_symptom_with_ctx(self):
        r = te.engine.review_symptom({'urgency_level': 'low', 'summary': 's', 'recommendations': [], 'detected_symptoms': ['cough']},
                                     text='a persistent cough', region='IN', mode='basic',
                                     ctx={'duration': 'weeks_2plus', 'age_group': None, 'severity': None})
        assert r['urgency_level'] in ('medium', 'high')


class TestHindiLatinIntensifierFix:
    """The Devanagari-Latin chest-pain pattern could not match two stacked intensifiers ('bahut tez' = 'very
    sharp'); it silently required exactly one or zero. Fixed to allow any number of them in sequence."""

    @pytest.mark.parametrize('text', [
        "seene me bahut tez dard hai aur paseena aa raha hai",
        "seene mein bahut dard hai",
        "seene mein tez dard hai",
        "mujhe bahut zyada tez seene mein dard ho raha hai",
    ])
    def test_stacked_and_single_intensifiers_are_both_recognised(self, text):
        assert 'Chest pain or pressure' in em.detect(text)[0]

    def test_a_fresh_hindi_latin_overdose_phrasing_not_in_any_dataset(self):
        assert 'Poisoning or overdose' in em.detect("maine dawai ki puri bottle pi li hai")[0]


class TestCrashSafetyNet:
    """If the normal analysis pipeline crashes outright, the person must not be left with nothing: a last-resort
    text scan runs first, so a described emergency still gets the full, correct guidance even when everything
    else failed. Covers every place in the pipeline where this was fixed."""

    def test_symptom_checker_crash_still_returns_full_emergency_guidance(self, app):
        from quickaid.ai.symptom_checker import SymptomChecker
        c = SymptomChecker()
        with pytest.MonkeyPatch.context() as m:
            m.setattr(c, '_analyze_basic_symptoms', MagicMock(side_effect=RuntimeError('boom')))
            r = c.analyze_symptoms("crushing chest pain, sweating, can't breathe")
        assert r.get('emergency_alert', {}).get('alert') is True
        assert '112' in (r.get('recommendations') or [''])[0]

    def test_symptom_checker_crash_on_benign_text_still_mentions_the_emergency_number(self, app):
        from quickaid.ai.symptom_checker import SymptomChecker
        c = SymptomChecker()
        with pytest.MonkeyPatch.context() as m:
            m.setattr(c, '_analyze_basic_symptoms', MagicMock(side_effect=RuntimeError('boom')))
            r = c.analyze_symptoms("I have a mild headache")
        assert any('112' in rec for rec in r.get('recommendations', []))

    def test_image_analyzer_crash_still_mentions_the_emergency_number(self, app):
        from quickaid.ai.medical_analyzer import MedicalAnalyzer
        a = MedicalAnalyzer()
        with pytest.MonkeyPatch.context() as m:
            m.setattr(a, '_analyze_offline', MagicMock(side_effect=RuntimeError('boom')), raising=False)
            m.setattr(a, 'client', MagicMock(), raising=False)
            m.setattr(a, 'use_gemini', True, raising=False)
            m.setattr(a.client.models, 'generate_content', MagicMock(side_effect=RuntimeError('boom')), raising=False)
            r = a.analyze_image('/nonexistent/path/that/will/fail.jpg')
        assert 'error' in r and any('112' in rec for rec in r.get('recommendations', []))

    def test_the_facade_is_the_only_door_services_use_to_reach_the_last_resort_check(self):
        assert safety.last_resort_emergency_check(
            "crushing chest pain and can't breathe", 'fallback', 'IN'
        )['emergency_alert']['alert'] is True
        assert safety.last_resort_emergency_check("a mild headache", 'fallback', 'IN') is None

    def test_analysis_service_survives_a_crash_inside_the_analyzer(self, app, client, registered_user):
        services = app.extensions['quickaid_services']
        with pytest.MonkeyPatch.context() as m:
            m.setattr(services.symptom_checker, 'analyze_symptoms', MagicMock(side_effect=RuntimeError('boom')))
            r = client.post('/analyze_symptoms', json={'symptoms': 'crushing chest pain and sweating, can\'t breathe'})
        assert r.status_code == 200
        body = r.get_json()
        assert body['analysis']['emergency_alert']['alert'] is True

    def test_analysis_service_keeps_the_result_even_if_saving_it_fails(self, app, client, registered_user):
        from quickaid.core import database as db
        with pytest.MonkeyPatch.context() as m:
            m.setattr(db, 'save_symptom_analysis', MagicMock(side_effect=RuntimeError('disk full')))
            r = client.post('/analyze_symptoms', json={'symptoms': 'a mild headache since this morning'})
        assert r.status_code == 200
        body = r.get_json()
        assert body['analysis_id'] is None and body['analysis']

    def test_interview_start_crash_message_mentions_the_emergency_number(self, app, client, registered_user):
        services = app.extensions['quickaid_services']
        with pytest.MonkeyPatch.context() as m:
            m.setattr(services.interview_engine, 'start', MagicMock(side_effect=RuntimeError('boom')))
            r = client.post('/api/interview/start', json={'symptoms': 'a mild headache'})
        assert r.status_code == 500
        assert any('112' in h for h in (r.get_json().get('hints') or []))

    def test_chat_crash_message_mentions_the_emergency_number(self, app):
        from quickaid.ai.conversation import ConversationService
        convo = ConversationService()
        convo.use_gemini = True
        convo.client = MagicMock()
        convo.client.models.generate_content = MagicMock(side_effect=RuntimeError('boom'))
        out = convo.reply(None, '', [], 'is a mild headache serious?', region='IN')
        assert '112' in out['answer']


class TestAIPlusTriageWorkflow:
    """Not rules alone, and not a deliberately broken model: a realistic, MOSTLY-competent AI, run through the
    whole pipeline the way it actually works in production. The model does most of the understanding; the safety
    net exists to catch its occasional mistake, not to carry the whole load, and it must not distort a correct
    answer either. Requirement: 'test the complete AI-plus-triage workflow', not just rules alone or a model that
    is wrong about everything.
    """

    FIXTURES = {
        # correct, high-confidence answers from the model - these must pass through essentially unchanged
        'severe difficulty breathing, lips are blue': {
            'summary': 'This describes a respiratory emergency with cyanosis.', 'urgency_level': 'high', 'emergency_alert': True,
            'emergency_reason': 'Severe breathing difficulty with blue lips', 'red_flags_present': ['Severe trouble breathing'],
            'possible_explanations': [{'name': 'severe respiratory distress', 'likelihood': 'more likely', 'why': 'cyanosis and severe breathlessness', 'supported_by': []}],
            'recommendations': ['Call emergency services now'],
        },
        'drank weedkiller by mistake': {
            'summary': 'This describes accidental ingestion of a toxic substance.', 'urgency_level': 'high', 'emergency_alert': True,
            'emergency_reason': 'Poisoning', 'red_flags_present': ['Poisoning or overdose'],
            'possible_explanations': [{'name': 'poisoning', 'likelihood': 'more likely', 'why': 'ingestion of a toxic herbicide', 'supported_by': []}],
            'recommendations': ['Call emergency services or poison control now'],
        },
        'I have a mild headache, been there for an hour': {
            'summary': 'This sounds like a common, mild headache.', 'urgency_level': 'low', 'emergency_alert': False,
            'possible_explanations': [{'name': 'tension headache', 'likelihood': 'more likely', 'why': 'mild, short duration, no red flags', 'supported_by': []}],
            'recommendations': ['Rest, hydrate, and consider an over-the-counter pain reliever if you normally take one'],
        },
        # a realistic MISTAKE: a generally competent model still misses a stroke sign and under-calls the urgency
        'sudden weakness on one side of my face and slurred speech': {
            'summary': 'This could be related to stress or fatigue affecting speech and muscle control.',
            'urgency_level': 'medium', 'emergency_alert': False,
            'possible_explanations': [{'name': 'stress-related symptoms', 'likelihood': 'more likely', 'why': '', 'supported_by': []}],
            'recommendations': ['Rest and monitor; see a doctor if it does not improve'],
        },
    }

    def _competent_model(self):
        from quickaid.evaluation.clients import _flatten
        client = MagicMock()

        def generate_content(**kwargs):
            text = _flatten(kwargs.get('contents'))
            for needle, data in self.FIXTURES.items():
                if needle in text:
                    return FakeResponse(json.dumps(data), 50, 60)
            raise AssertionError(f'no fixture for: {text[:300]}')
        client.models.generate_content = MagicMock(side_effect=generate_content)
        return client

    def _check(self, app, text):
        from quickaid.ai.symptom_checker import SymptomChecker
        c = SymptomChecker()
        c.use_gemini, c.client = True, self._competent_model()
        return c.analyze_symptoms(text)

    def test_a_correct_emergency_call_passes_through_largely_unchanged(self, app):
        r = self._check(app, 'severe difficulty breathing, lips are blue')
        assert r['urgency_level'] == 'high' and r['emergency_alert']['alert'] is True
        assert r['care_level'] == 'emergency' and r['safety_review']['urgency_raised'] is False   # the model was already right; nothing needed raising
        assert 'respiratory' in r['possible_explanations'][0]['name'].lower() or 'distress' in r['possible_explanations'][0]['name'].lower()

    def test_a_correct_low_urgency_call_is_not_falsely_escalated(self, app):
        r = self._check(app, 'I have a mild headache, been there for an hour')
        assert r['urgency_level'] == 'low' and r['emergency_alert']['alert'] is False and r['care_level'] != 'emergency'

    def test_a_correct_poisoning_call_passes_through(self, app):
        r = self._check(app, 'drank weedkiller by mistake')
        assert r['urgency_level'] == 'high' and r['emergency_alert']['alert'] is True

    def test_the_safety_net_catches_the_one_case_the_model_got_wrong(self, app):
        """The model said MEDIUM with no alert for classic stroke signs (face weakness + slurred speech). The text
        scan runs independently of what the model said, and must raise this regardless."""
        r = self._check(app, 'sudden weakness on one side of my face and slurred speech')
        assert r['urgency_level'] == 'high' and r['emergency_alert']['alert'] is True
        assert r['safety_review']['urgency_raised'] is True
        assert any('stroke' in reason.lower() or 'safety rules' in reason.lower() for reason in r['safety_review']['adjustments'])
        assert r['safety_review']['original_urgency'] == 'medium' and r['safety_review']['final_urgency'] == 'high'

    def test_overall_recall_across_a_realistic_mixed_batch_is_complete(self, app):
        """3 of 4 cases are true emergencies; the model alone would only correctly flag 2 of them. The combination
        must still catch all 3, which is the actual point of having both a model AND a safety net."""
        texts_and_expected = [
            ('severe difficulty breathing, lips are blue', True),
            ('drank weedkiller by mistake', True),
            ('sudden weakness on one side of my face and slurred speech', True),
            ('I have a mild headache, been there for an hour', False),
        ]
        results = [self._check(app, t) for t, _ in texts_and_expected]
        model_alone_recall = sum(1 for (t, want), r in zip(texts_and_expected, results) if want and r['safety_review']['original_urgency'] == 'high') / 3
        combined_recall = sum(1 for (t, want), r in zip(texts_and_expected, results) if want and r['emergency_alert']['alert']) / 3
        assert model_alone_recall == pytest.approx(2 / 3)     # the model alone would have missed the stroke case
        assert combined_recall == 1.0                          # the safety net closes that gap
        assert not results[3]['emergency_alert']['alert']      # and the one genuinely benign case is still not a false alarm

    def test_the_workflow_runs_through_the_real_http_endpoint_end_to_end(self, app, client, registered_user):
        """Not just the checker class in isolation: the actual /analyze_symptoms route, with the result saved to history."""
        services = app.extensions['quickaid_services']
        services.symptom_checker.use_gemini, services.symptom_checker.client = True, self._competent_model()
        r = client.post('/analyze_symptoms', json={'symptoms': 'sudden weakness on one side of my face and slurred speech'})
        assert r.status_code == 200
        body = r.get_json()
        assert body['analysis']['emergency_alert']['alert'] is True and body['analysis_id']
        saved = client.get(f"/history/symptom/{body['analysis_id']}")
        assert saved.status_code == 200 and 'emergency' in saved.get_data(as_text=True).lower()


class TestResolvedSymptoms:
    """'The chest pain went away last week' is different from negation ('never had chest pain'): the symptom DID
    happen, but is now over. HOLDOUT2_RESULT.md's known-limitation case (T09) is fixed by this."""

    @pytest.mark.parametrize('text', [
        "the chest pain went away last week, today I just have a sore throat",
        "the chest pain has resolved, I'm fine now",
        "my headache went away, no more chest pain either",
        "had a seizure last month, it has resolved and I'm back to normal now",
    ])
    def test_a_symptoms_own_stated_resolution_is_not_an_alarm(self, text):
        assert not te.scan(text).any_emergency, text

    @pytest.mark.parametrize('text', [
        "my nausea went away but I still have crushing chest pain",
        "I had chest pain yesterday but it's back and worse now",
        "chest pain, severe, started an hour ago",
    ])
    def test_a_different_or_still_current_symptom_is_not_suppressed(self, text):
        assert te.scan(text).any_emergency, text

    def test_the_basic_mode_symptom_extractor_is_resolution_aware_too(self, app):
        from quickaid.ai.symptom_checker import SymptomChecker
        c = SymptomChecker()
        r = c.analyze_symptoms("the chest pain went away last week, today I just have a sore throat")
        assert not r['emergency_alert']['alert'] and r['care_level'] != 'emergency'

    def test_resolution_reaches_across_a_comma_but_not_across_but(self):
        # a comma is often just one continuous thought ("it happened, it has resolved"); "but" is a real topic change
        text = "had a seizure, it has resolved now"
        m_start, m_end = text.index('seizure'), text.index('seizure') + len('seizure')
        assert te.resolved(text, m_start, m_end) is True
        text2 = "nausea went away but chest pain is crushing"
        m2_start, m2_end = text2.index('chest pain'), text2.index('chest pain') + len('chest pain')
        assert te.resolved(text2, m2_start, m2_end) is False
        assert 'Chest pain or pressure' in te.scan(text2).emergency
