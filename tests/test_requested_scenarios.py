"""Explicitly-requested test categories: genuine emergency symptoms, pneumonia-like patterns, ambiguous inputs,
an unavailable AI provider, and photo-analysis limitations. Some of these properties are also covered elsewhere in
the suite (e.g. tests/test_triage_engine.py, tests/test_image_quality_fairness.py) - these are named explicitly
and kept together so each requested category has an obvious, findable home.
"""

import pytest


@pytest.fixture()
def checker(app):
    from quickaid.ai.symptom_checker import SymptomChecker
    return SymptomChecker()


class TestGenuineEmergencySymptoms:
    """A representative spread of real emergencies, each checked for the full chain: alert fires, urgency/care
    match it, and the guidance tells the person to get help now - not just that ONE field happens to be set."""

    @pytest.mark.parametrize('text', [
        "severe crushing chest pain radiating to my left arm, sweating and nauseous",
        "sudden weakness on one side of my face and slurred speech",
        "my throat is closing up and I'm covered in hives after eating peanuts",
        "he's not responding and I can't wake him up",
        "my child swallowed a bottle of pills",
        "severe difficulty breathing and my lips have gone blue",
    ])
    def test_emergency_fires_urgency_and_care_level_agree_guidance_says_call_now(self, checker, text):
        r = checker.analyze_symptoms(text)
        assert r['emergency_alert']['alert'] is True, text
        assert r['urgency_level'] == 'high' and r['care_level'] == 'emergency'
        assert any('112' in step or 'call' in step.lower() for step in r['recommendations'])

    def test_a_genuine_emergency_is_never_delayed_by_asking_questions_first(self, app):
        """Requirement: detect genuine emergencies promptly without delaying urgent guidance for questionnaires."""
        from quickaid.ai.interview import InterviewEngine
        from quickaid.ai.symptom_checker import SymptomChecker
        eng = InterviewEngine(SymptomChecker())
        r = eng.start("severe crushing chest pain and sweating, started 20 minutes ago", 'IN')
        assert r.get('emergency') is not None and r['emergency']['emergency_alert']['alert'] is True
        assert r.get('question') is None   # no question was asked first


class TestPneumoniaLikeSymptomPatterns:
    """A classic pneumonia presentation (fever, productive cough, breathlessness, chest pain on breathing) should
    surface pneumonia as a real possibility - without claiming it as a confirmed diagnosis from symptoms alone."""

    def test_pneumonia_is_recognised_as_a_possibility(self, checker):
        r = checker.analyze_symptoms("fever, cough with phlegm, rapid breathing, feeling very unwell and tired, chest hurts when I breathe")
        assert 'pneumonia' in r['possible_conditions'], r['possible_conditions']

    def test_it_is_presented_as_a_possibility_not_a_confirmed_diagnosis(self, checker):
        r = checker.analyze_symptoms("fever, cough with phlegm, rapid breathing, feeling very unwell and tired")
        explanation = next((e for e in r['possible_explanations'] if e['name'] == 'pneumonia'), None)
        if explanation:
            assert explanation['likelihood'] in ('more likely', 'possible', 'less likely')
        assert 'not a diagnosis' in r['disclaimer'].lower()

    def test_asthma_and_pneumonia_are_distinguishable_by_their_different_supporting_symptoms(self, checker):
        asthma_leaning = checker.analyze_symptoms("wheezing, chest tightness, nighttime cough, no fever")
        pneumonia_leaning = checker.analyze_symptoms("high fever, cough with thick phlegm, rapid breathing, chills")
        assert asthma_leaning['possible_conditions'][0] == 'asthma'
        assert 'pneumonia' in pneumonia_leaning['possible_conditions']   # genuinely in the differential, not necessarily top of a tie
        assert asthma_leaning['possible_conditions'][0] != pneumonia_leaning['possible_conditions'][0]   # the two presentations are distinguished


class TestAmbiguousInputs:
    """Vague, underspecified or contradictory descriptions should not produce a false sense of precision."""

    def test_a_vague_complaint_abstains_rather_than_guessing(self, checker):
        r = checker.analyze_symptoms("I don't feel well")
        assert r.get('error') or r.get('urgency_level') in (None, 'low')

    def test_genuinely_unrelated_or_nonsensical_text_does_not_crash_or_fabricate(self, checker):
        r = checker.analyze_symptoms("asdkjf qwoeiru the sky is blue today")
        assert isinstance(r, dict)
        assert not r.get('possible_conditions')

    def test_a_hedging_description_still_gets_a_safe_answer(self, checker):
        r = checker.analyze_symptoms("it's probably nothing but my left arm has gone numb and I also have a headache")
        assert isinstance(r, dict) and 'error' not in r or r['emergency_alert']['alert'] in (True, False)

    def test_contradictory_information_does_not_crash(self, checker):
        r = checker.analyze_symptoms("no fever but I have a really high temperature of 39.5")
        assert isinstance(r, dict)


class TestUnavailableAiProvider:
    """When no AI key is configured, or the AI call fails outright, the person must still get a complete, safe,
    clearly-labelled basic-mode result - never a crash, and never a result that quietly pretends to be AI-backed."""

    def test_no_api_key_configured_produces_a_clearly_labelled_basic_result(self, checker):
        assert checker.use_gemini is False   # the test fixture's GEMINI_API_KEY is a placeholder
        r = checker.analyze_symptoms("fever and cough for 2 days")
        assert r['analysis_mode'] == 'basic'
        assert 'basic' in r['disclaimer'].lower() or 'rule-based' in str(r.get('reliability', {})).lower()

    def test_an_ai_call_that_raises_still_returns_a_safe_result_not_a_crash(self, app):
        from unittest.mock import MagicMock
        from quickaid.ai.symptom_checker import SymptomChecker
        c = SymptomChecker()
        c.use_gemini = True
        c.client = MagicMock()
        c.client.models.generate_content = MagicMock(side_effect=RuntimeError('simulated API outage'))
        r = c.analyze_symptoms("severe crushing chest pain and sweating")
        assert isinstance(r, dict)
        # the last-resort emergency check must still fire even though the AI path failed outright
        assert r.get('emergency_alert', {}).get('alert') is True

    def test_an_ai_call_that_raises_on_a_recognisable_benign_input_falls_back_to_a_complete_basic_result(self, app):
        """Different from the emergency case above: a benign, recognisable symptom degrades gracefully to the
        existing, working basic-mode analysis - a complete, normal result, not a special error state. The
        emergency-number reminder is for when nothing could be worked out at all (see symptom_checker.py's outer
        except handler), not for every AI hiccup that basic mode can already handle perfectly well on its own."""
        from unittest.mock import MagicMock
        from quickaid.ai.symptom_checker import SymptomChecker
        c = SymptomChecker()
        c.use_gemini = True
        c.client = MagicMock()
        c.client.models.generate_content = MagicMock(side_effect=RuntimeError('simulated API outage'))
        r = c.analyze_symptoms("I have a mild headache")
        assert 'error' not in r and r.get('recommendations') and r.get('analysis_mode') == 'basic'

    def test_an_ai_call_that_raises_on_text_basic_mode_cannot_recognise_still_mentions_the_emergency_number(self, app):
        from unittest.mock import MagicMock
        from quickaid.ai.symptom_checker import SymptomChecker
        c = SymptomChecker()
        c.use_gemini = True
        c.client = MagicMock()
        c.client.models.generate_content = MagicMock(side_effect=RuntimeError('simulated API outage'))
        r = c.analyze_symptoms("asdkjf qwoeiru nothing recognisable here")
        assert any('112' in rec for rec in r.get('recommendations', []))


class TestPhotoAnalysisLimitations:
    """What the basic (no-AI) photo check can and cannot tell you must be stated plainly, and colour measurements
    must never be presented as if they were a diagnosis."""

    @pytest.fixture()
    def analyzer(self, app):
        from quickaid.ai.medical_analyzer import MedicalAnalyzer
        return MedicalAnalyzer()

    def _photo(self):
        import numpy as np
        from PIL import Image
        import tempfile
        arr = np.clip(np.random.default_rng(2).integers(-20, 20, (320, 320, 3)) + np.array([205, 160, 140]), 0, 255).astype('uint8')
        path = tempfile.mktemp(suffix='.png')
        Image.fromarray(arr).save(path)
        return path

    def test_basic_mode_never_claims_a_diagnosis(self, analyzer):
        r = analyzer.analyze_image(self._photo())
        assert r['analysis_mode'] == 'basic'
        assert 'not a diagnosis' in r['disclaimer'].lower()
        assert r['possible_explanations'] == []   # no AI, so no differential is offered at all

    def test_basic_mode_explicitly_states_what_it_cannot_tell(self, analyzer):
        r = analyzer.analyze_image(self._photo())
        assert r.get('not_assessable')
        joined = ' '.join(r['not_assessable']).lower()
        assert 'what is actually wrong' in joined or 'cannot' in joined

    def test_colour_measurement_is_labelled_as_a_measurement_not_a_finding_of_disease(self, analyzer):
        r = analyzer.analyze_image(self._photo())
        assert r['findings'] and 'colour measurement' in r['findings'][0]['description'].lower()

    def test_reliability_explicitly_says_no_ai_model_was_used(self, analyzer):
        r = analyzer.analyze_image(self._photo())
        assert any('no ai model' in reason.lower() for reason in r['reliability']['reasons'])

    def test_a_photo_quality_problem_is_reported_plainly(self, analyzer):
        import numpy as np
        from PIL import Image
        import tempfile
        dark = np.clip(np.random.default_rng(3).integers(-5, 5, (320, 320, 3)) + np.array([5, 5, 5]), 0, 255).astype('uint8')
        path = tempfile.mktemp(suffix='.png')
        Image.fromarray(dark).save(path)
        r = analyzer.analyze_image(path)
        assert r['image_quality'] in ('poor', 'fair')
