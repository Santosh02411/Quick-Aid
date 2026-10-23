"""A real deployment bug: with a Gemini key genuinely configured, a call that times out (504) or otherwise fails
falls back to basic analysis - correct and safe - but the result said "AI is not connected", which is actively
misleading when AI *is* connected and *was* tried. Fixed with an `ai_attempted` flag threaded through every basic-
mode fallback path, so the message can honestly say "the call failed this time" instead of "nothing is configured".
Covers all three surfaces this bug existed in: photo analysis, one-shot symptom analysis, and the interview flow.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image


def _photo():
    import tempfile
    arr = np.clip(np.random.default_rng(1).integers(-20, 20, (320, 320, 3)) + np.array([205, 160, 140]), 0, 255).astype('uint8')
    path = tempfile.mktemp(suffix='.png')
    Image.fromarray(arr).save(path)
    return path


class TestPhotoAnalysis:
    def test_ai_never_configured_says_so_and_invites_connecting_a_key(self, app):
        from quickaid.ai.medical_analyzer import MedicalAnalyzer
        a = MedicalAnalyzer()
        a.use_gemini = False
        r = a.analyze_image(_photo())
        assert r['ai_attempted'] is False
        assert 'connect a gemini api key' in r['disclaimer'].lower()
        assert 'not connect' not in r['disclaimer'].lower()   # never implies a failed call means "not connected"

    def test_a_configured_ai_that_times_out_says_the_call_failed_not_that_nothing_is_connected(self, app):
        from quickaid.ai.medical_analyzer import MedicalAnalyzer
        a = MedicalAnalyzer()
        a.use_gemini = True
        a.client = MagicMock()
        a.client.models.generate_content = MagicMock(side_effect=TimeoutError('simulated 504 DEADLINE_EXCEEDED'))
        r = a.analyze_image(_photo())
        assert r['analysis_mode'] == 'basic'
        assert r['ai_attempted'] is True
        assert 'could not complete this time' in r['disclaimer'].lower()
        assert 'connect a gemini api key' not in r['disclaimer'].lower()   # a key IS already connected


class TestOneShotSymptomAnalysis:
    def test_ai_never_configured_says_so(self, app):
        from quickaid.ai.symptom_checker import SymptomChecker
        c = SymptomChecker()
        assert c.use_gemini is False   # the test fixture's key is a placeholder
        r = c.analyze_symptoms("fever and cough for 2 days")
        assert r.get('ai_attempted', False) is False
        assert 'add a gemini api key' in r['disclaimer'].lower()

    def test_a_configured_ai_that_raises_says_the_call_failed(self, app):
        from quickaid.ai.symptom_checker import SymptomChecker
        c = SymptomChecker()
        c.use_gemini = True
        c.client = MagicMock()
        c.client.models.generate_content = MagicMock(side_effect=TimeoutError('simulated 504 DEADLINE_EXCEEDED'))
        r = c.analyze_symptoms("fever and cough for 2 days")
        assert r['ai_attempted'] is True
        assert 'could not complete this time' in r['disclaimer'].lower()
        assert 'add a gemini api key' not in r['disclaimer'].lower()

    def test_a_configured_ai_that_returns_an_invalid_response_also_says_the_call_failed(self, app):
        """_safe_fallback: the model DID respond, just not usably - also a genuine attempt, not 'not connected'."""
        from quickaid.ai.symptom_checker import SymptomChecker
        c = SymptomChecker()
        c.use_gemini = True
        c.client = MagicMock()
        bad_response = MagicMock()
        bad_response.parsed = None
        bad_response.text = 'not valid json'
        c.client.models.generate_content = MagicMock(return_value=bad_response)
        r = c.analyze_symptoms("fever and cough for 2 days")
        assert r.get('ai_attempted') is True
        assert 'add a gemini api key' not in r['disclaimer'].lower()

    def test_text_with_no_recognisable_symptoms_still_distinguishes_the_two_cases(self, app):
        from quickaid.ai.symptom_checker import SymptomChecker
        gibberish = "asdkjf qwoeiru nothing recognisable"
        c_unconfigured = SymptomChecker()
        r1 = c_unconfigured.analyze_symptoms(gibberish)
        assert r1.get('ai_attempted', False) is False

        c_failed = SymptomChecker()
        c_failed.use_gemini = True
        c_failed.client = MagicMock()
        c_failed.client.models.generate_content = MagicMock(side_effect=TimeoutError('simulated outage'))
        r2 = c_failed.analyze_symptoms(gibberish)
        assert r2.get('ai_attempted') is True
        assert r1['disclaimer'] != r2['disclaimer']


class TestInterviewFlow:
    def test_ai_never_configured_says_so(self, app):
        from quickaid.ai.interview import InterviewEngine
        from quickaid.ai.symptom_checker import SymptomChecker
        checker = SymptomChecker()
        eng = InterviewEngine(checker)
        assert eng.use_gemini is False
        start = eng.start("mild headache", 'IN')   # {'state', 'emergency', 'question'} - finish() takes ['state']
        result = eng.finish(start['state'], 'IN')
        assert result.get('ai_attempted', False) is False

    def test_a_configured_ai_that_fails_says_the_call_failed(self, app):
        # InterviewEngine.use_gemini is a read-only property delegating to checker.use_gemini, and its own
        # _assess_with_gemini calls self.checker.client (not an InterviewEngine.client that does not exist) - both
        # need setting on the wrapped checker, not the engine itself.
        from quickaid.ai.interview import InterviewEngine
        from quickaid.ai.symptom_checker import SymptomChecker
        checker = SymptomChecker()
        checker.use_gemini = True
        checker.client = MagicMock()
        checker.client.models.generate_content = MagicMock(side_effect=TimeoutError('simulated outage'))
        eng = InterviewEngine(checker)
        start = eng.start("mild headache", 'IN')
        result = eng.finish(start['state'], 'IN')
        assert result['ai_attempted'] is True
        assert 'add a gemini api key' not in result['disclaimer'].lower()
