"""
Tests for SymptomChecker: keyword extraction, symptom-combination analysis,
emergency detection, and the Gemini structured-output parsing (including
its fallback path when structured output doesn't come back parsed).
"""

import pytest
from unittest.mock import MagicMock
from symptom_checker import SymptomChecker, SymptomAnalysisResult


@pytest.fixture()
def checker():
    # No GEMINI_API_KEY in the test env -> use_gemini is False, but we can
    # still call the private Gemini-parsing methods directly to test them
    # in isolation without hitting the network.
    return SymptomChecker()


# ---------------------------------------------------------------------------
# Keyword extraction (_extract_symptoms)
# ---------------------------------------------------------------------------

class TestExtractSymptoms:
    def test_single_symptom_simple_phrase(self, checker):
        result = checker._extract_symptoms("i have a headache")
        assert 'headache' in result

    def test_multiple_symptoms(self, checker):
        result = checker._extract_symptoms("i have a fever and a headache")
        assert 'fever' in result
        assert 'headache' in result

    def test_symptom_variation_phrases(self, checker):
        # "high temperature" and "burning up" are both fever variants
        assert 'fever' in checker._extract_symptoms("i have a high temperature")
        assert 'fever' in checker._extract_symptoms("i'm burning up")

    def test_chest_pain_variations(self, checker):
        assert 'chest_pain' in checker._extract_symptoms("my chest hurts")
        assert 'chest_pain' in checker._extract_symptoms("chest tightness all day")

    def test_no_recognizable_symptoms(self, checker):
        result = checker._extract_symptoms("i feel weird about my homework")
        assert result == []

    def test_duplicate_phrases_deduplicated(self, checker):
        # "cough" and "coughing" both map to the same symptom key
        result = checker._extract_symptoms("i have a cough, i keep coughing")
        assert result.count('cough') == 1

    def test_case_and_whitespace_insensitivity(self, checker):
        # caller lowercases before extraction (as analyze_symptoms does)
        result = checker._extract_symptoms("  i am nauseous and dizzy  ")
        assert 'nausea' in result
        assert 'dizziness' in result

    def test_empty_string(self, checker):
        assert checker._extract_symptoms("") == []


# ---------------------------------------------------------------------------
# Symptom-combination analysis and urgency scoring
# ---------------------------------------------------------------------------

class TestAnalyzeSymptomCombination:
    def test_single_low_urgency_symptom(self, checker):
        result = checker._analyze_symptom_combination(['headache'])
        assert result['urgency'] == 'low'
        assert 'tension headache' in result['conditions']

    def test_high_urgency_symptom_dominates(self, checker):
        # chest_pain is 'high' urgency, headache is 'low' - overall should be high
        result = checker._analyze_symptom_combination(['headache', 'chest_pain'])
        assert result['urgency'] == 'high'

    def test_unknown_symptom_key_ignored(self, checker):
        # a symptom not in symptom_database shouldn't crash the analysis
        result = checker._analyze_symptom_combination(['not_a_real_symptom'])
        assert result['urgency'] == 'low'  # default when no scores collected
        assert result['conditions'] == []

    def test_empty_symptom_list(self, checker):
        result = checker._analyze_symptom_combination([])
        assert result['urgency'] == 'low'
        assert result['conditions'] == []


# ---------------------------------------------------------------------------
# Emergency detection
# ---------------------------------------------------------------------------

class TestCheckEmergencySymptoms:
    def test_chest_pain_flags_emergency(self, checker):
        result = checker._check_emergency_symptoms(['chest_pain'])
        assert result['alert'] is True
        assert 'action' in result

    def test_low_urgency_symptom_no_emergency(self, checker):
        result = checker._check_emergency_symptoms(['headache'])
        assert result['alert'] is False

    def test_no_symptoms_no_emergency(self, checker):
        result = checker._check_emergency_symptoms([])
        assert result['alert'] is False


# ---------------------------------------------------------------------------
# Full basic-mode analysis (integration of the above pieces)
# ---------------------------------------------------------------------------

class TestAnalyzeBasicSymptoms:
    def test_recognizable_symptoms_returns_full_structure(self, checker):
        result = checker._analyze_basic_symptoms("I have a headache and nausea")
        assert 'detected_symptoms' in result
        assert 'possible_conditions' in result
        assert 'urgency_level' in result
        assert 'emergency_alert' in result
        assert 'recommendations' in result
        assert result['emergency_alert']['alert'] is False

    def test_emergency_symptoms_flagged_end_to_end(self, checker):
        result = checker._analyze_basic_symptoms("severe chest pain and shortness of breath")
        assert result['urgency_level'] == 'high'
        assert result['emergency_alert']['alert'] is True

    def test_unrecognizable_text_returns_error(self, checker):
        result = checker._analyze_basic_symptoms("qwerty asdf zxcv")
        assert 'error' in result


# ---------------------------------------------------------------------------
# Gemini structured-output parsing + fallback logic
# ---------------------------------------------------------------------------

class TestParseGeminiSymptomResponse:
    def test_parses_structured_response_directly(self, checker):
        """response.parsed populated (the normal, expected path)."""
        fake_response = MagicMock()
        fake_response.parsed = SymptomAnalysisResult(
            detected_symptoms=["chest pain", "shortness of breath"],
            possible_conditions=["heart attack", "angina"],
            urgency_level="high",
            recommendations=["Call 911 immediately"],
            emergency_alert=True,
            safety_tips=["Stay calm"]
        )

        result = checker._parse_gemini_symptom_response(fake_response)

        assert result['detected_symptoms'] == ["chest pain", "shortness of breath"]
        assert result['urgency_level'] == "high"
        assert result['emergency_alert']['alert'] is True
        assert 'SEEK IMMEDIATE MEDICAL ATTENTION' in result['emergency_alert']['message']

    def test_emergency_alert_false_produces_empty_message(self, checker):
        fake_response = MagicMock()
        fake_response.parsed = SymptomAnalysisResult(
            detected_symptoms=["cough"],
            possible_conditions=["cold"],
            urgency_level="low",
            recommendations=["rest"],
            emergency_alert=False,
            safety_tips=[]
        )

        result = checker._parse_gemini_symptom_response(fake_response)

        assert result['emergency_alert']['alert'] is False
        assert result['emergency_alert']['message'] == ''
        assert result['emergency_alert']['action'] == ''

    def test_fallback_to_raw_json_text_when_parsed_is_none(self, checker):
        """response.parsed is None (structured output failed validation) -
        should fall back to parsing response.text as JSON directly."""
        fake_response = MagicMock()
        fake_response.parsed = None
        fake_response.text = (
            '{"detected_symptoms": ["fever"], "possible_conditions": ["flu"], '
            '"urgency_level": "medium", "recommendations": ["rest"], '
            '"emergency_alert": false, "safety_tips": []}'
        )

        result = checker._parse_gemini_symptom_response(fake_response)

        assert result['detected_symptoms'] == ["fever"]
        assert result['urgency_level'] == "medium"
        assert result['emergency_alert']['alert'] is False

    def test_completely_malformed_response_does_not_crash(self, checker):
        """Neither .parsed nor valid JSON text - should degrade gracefully,
        not raise, and use the safe defaults."""
        fake_response = MagicMock()
        fake_response.parsed = None
        fake_response.text = "I'm not JSON at all, sorry!"

        result = checker._parse_gemini_symptom_response(fake_response)

        # Should not raise, and should come back with sane defaults
        assert result['detected_symptoms'] == []
        assert result['urgency_level'] == 'medium'
        assert result['emergency_alert']['alert'] is False
        assert isinstance(result['recommendations'], list)
        assert len(result['recommendations']) > 0  # falls back to default recs

    def test_missing_optional_fields_use_defaults(self, checker):
        """Partial JSON (missing some keys) shouldn't crash the parser."""
        fake_response = MagicMock()
        fake_response.parsed = None
        fake_response.text = '{"detected_symptoms": ["headache"]}'

        result = checker._parse_gemini_symptom_response(fake_response)

        assert result['detected_symptoms'] == ["headache"]
        assert result['urgency_level'] == 'medium'  # default
        assert result['possible_conditions'] == []


# ---------------------------------------------------------------------------
# Gemini call failure -> falls back to basic analysis
# ---------------------------------------------------------------------------

class TestAnalyzeWithGeminiFallback:
    def test_gemini_exception_falls_back_to_basic_mode(self, checker, mocker):
        """If the Gemini client raises (network error, bad API key, etc.),
        analyze_symptoms should still return a usable result via the basic
        rule-based path instead of propagating the exception."""
        checker.use_gemini = True
        checker.client = MagicMock()
        checker.client.models.generate_content.side_effect = RuntimeError("API unavailable")

        result = checker.analyze_symptoms("I have a headache")

        assert 'error' not in result or 'detected_symptoms' in result
        assert 'detected_symptoms' in result
        assert 'headache' in result['detected_symptoms']
