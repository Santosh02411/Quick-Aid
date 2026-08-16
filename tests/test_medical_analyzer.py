"""
Tests for MedicalAnalyzer: basic-mode visual feature analysis and the
Gemini structured-output parsing (including its fallback path).
"""

import numpy as np
import pytest
from unittest.mock import MagicMock
from medical_analyzer import MedicalAnalyzer, ImageAnalysisResult


@pytest.fixture()
def analyzer():
    return MedicalAnalyzer()


# ---------------------------------------------------------------------------
# Basic-mode visual feature analysis (no Gemini)
# ---------------------------------------------------------------------------

class TestAnalyzeVisualFeatures:
    def test_red_image_flags_inflammation(self, analyzer):
        # Solid bright red image
        img = np.full((20, 20, 3), [255, 0, 0], dtype=np.uint8)
        result = analyzer._analyze_visual_features(img)
        assert 'possible_inflammation_or_injury' in result['conditions']

    def test_dark_image_flags_bruising(self, analyzer):
        img = np.full((20, 20, 3), [20, 20, 20], dtype=np.uint8)
        result = analyzer._analyze_visual_features(img)
        assert 'possible_bruising' in result['conditions']

    def test_uniform_bright_image_falls_back_to_general(self, analyzer):
        # Bright, uniform gray with red channel below the inflammation
        # threshold (0.6 * 255 = 153) and mean above the bruising threshold (100)
        img = np.full((20, 20, 3), [140, 140, 140], dtype=np.uint8)
        result = analyzer._analyze_visual_features(img)
        assert 'general_skin_assessment' in result['conditions']

    def test_confidence_is_bounded_0_to_1(self, analyzer):
        img = np.full((20, 20, 3), [255, 0, 0], dtype=np.uint8)
        result = analyzer._analyze_visual_features(img)
        assert 0.0 <= result['confidence'] <= 1.0


class TestGenerateRecommendations:
    def test_inflammation_gets_relevant_recommendations(self, analyzer):
        recs = analyzer._generate_recommendations({'conditions': ['possible_inflammation_or_injury']})
        assert any('cold compress' in r.lower() for r in recs)

    def test_bruising_gets_relevant_recommendations(self, analyzer):
        recs = analyzer._generate_recommendations({'conditions': ['possible_bruising']})
        assert any('ice pack' in r.lower() for r in recs)

    def test_always_includes_general_recommendations(self, analyzer):
        recs = analyzer._generate_recommendations({'conditions': []})
        assert any('hygiene' in r.lower() for r in recs)


# ---------------------------------------------------------------------------
# Gemini structured-output parsing + fallback logic
# ---------------------------------------------------------------------------

class TestParseGeminiResponse:
    def test_parses_structured_response_directly(self, analyzer):
        fake_response = MagicMock()
        fake_response.parsed = ImageAnalysisResult(
            detected_conditions=["minor laceration"],
            confidence="high",
            recommendations=["Clean with water", "Apply bandage"],
            urgency="low",
            safety_tips=["Wash hands first"]
        )

        result = analyzer._parse_gemini_response(fake_response)

        assert result['detected_conditions'] == ["minor laceration"]
        assert result['confidence'] == "high"
        assert result['urgency'] == "low"
        assert 'disclaimer' in result

    def test_fallback_to_raw_json_text_when_parsed_is_none(self, analyzer):
        fake_response = MagicMock()
        fake_response.parsed = None
        fake_response.text = (
            '{"detected_conditions": ["bruise"], "confidence": "medium", '
            '"recommendations": ["ice it"], "urgency": "low", "safety_tips": []}'
        )

        result = analyzer._parse_gemini_response(fake_response)

        assert result['detected_conditions'] == ["bruise"]
        assert result['urgency'] == "low"

    def test_completely_malformed_response_does_not_crash(self, analyzer):
        fake_response = MagicMock()
        fake_response.parsed = None
        fake_response.text = "not json at all"

        result = analyzer._parse_gemini_response(fake_response)

        # Should not raise, should use safe defaults
        assert result['detected_conditions'] == ['Medical condition analysis']
        assert result['confidence'] == 'medium'
        assert result['urgency'] == 'medium'
        assert isinstance(result['recommendations'], list)
        assert len(result['recommendations']) > 0

    def test_missing_optional_fields_use_defaults(self, analyzer):
        fake_response = MagicMock()
        fake_response.parsed = None
        fake_response.text = '{"detected_conditions": ["rash"]}'

        result = analyzer._parse_gemini_response(fake_response)

        assert result['detected_conditions'] == ["rash"]
        assert result['confidence'] == 'medium'  # default
        assert result['urgency'] == 'medium'  # default


# ---------------------------------------------------------------------------
# Gemini call failure -> falls back to basic analysis
# ---------------------------------------------------------------------------

class TestAnalyzeWithGeminiFallback:
    def test_gemini_exception_falls_back_to_basic_mode(self, analyzer, tmp_path):
        from PIL import Image as PILImage

        img_path = tmp_path / "test.png"
        PILImage.new('RGB', (10, 10), color=(255, 0, 0)).save(img_path)

        analyzer.use_gemini = True
        analyzer.client = MagicMock()
        analyzer.client.models.generate_content.side_effect = RuntimeError("API unavailable")

        result = analyzer.analyze_image(str(img_path))

        # Should have fallen back to basic mode instead of propagating the error
        assert 'detected_conditions' in result
        assert 'possible_inflammation_or_injury' in result['detected_conditions']
