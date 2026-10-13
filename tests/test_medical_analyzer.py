"""
Tests for MedicalAnalyzer: basic-mode visual feature analysis and the
Gemini structured-output parsing (including its fallback path).
"""

import numpy as np
import pytest
from unittest.mock import MagicMock
from quickaid.ai.medical_analyzer import MedicalAnalyzer, ImageAnalysisResult


@pytest.fixture()
def analyzer():
    return MedicalAnalyzer()


# ---------------------------------------------------------------------------
# Basic-mode visual feature analysis (no Gemini)
# ---------------------------------------------------------------------------

SKIN = (224, 172, 150)


def skin_image(red_patch=0.0, bruise_patch=0.0, size=60):
    """Synthetic skin-coloured image with optional red / dark-purple patches."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = SKIN
    n = int(size * np.sqrt(red_patch)) if red_patch else 0
    if n:
        img[:n, :n] = (205, 70, 70)
    m = int(size * np.sqrt(bruise_patch)) if bruise_patch else 0
    if m:
        img[-m:, -m:] = (95, 70, 110)
    return img


class TestAnalyzeVisualFeatures:
    def test_plain_skin_has_no_findings(self, analyzer):
        result = analyzer._analyze_visual_features(skin_image())
        assert result['conditions'] == ['no_clear_findings']

    def test_red_patch_on_skin_flags_redness(self, analyzer):
        result = analyzer._analyze_visual_features(skin_image(red_patch=0.2))
        assert 'redness_area' in result['conditions']

    def test_very_large_red_area_raises_urgency(self, analyzer):
        result = analyzer._analyze_visual_features(skin_image(red_patch=0.6))
        assert 'large_redness_area' in result['conditions']
        assert result['urgency'] == 'medium'

    def test_dark_purple_patch_flags_discolouration(self, analyzer):
        result = analyzer._analyze_visual_features(skin_image(bruise_patch=0.2))
        assert 'dark_discolouration' in result['conditions']

    def test_non_skin_image_is_reported_as_unusable(self, analyzer):
        img = np.full((40, 40, 3), [40, 80, 200], dtype=np.uint8)  # solid blue
        result = analyzer._analyze_visual_features(img)
        assert result['conditions'] == ['no_clear_skin_area']

    def test_confidence_is_bounded_0_to_1(self, analyzer):
        for img in (skin_image(red_patch=0.6), skin_image(), np.zeros((20, 20, 3), dtype=np.uint8)):
            assert 0.0 <= analyzer._analyze_visual_features(img)['confidence'] <= 1.0


class TestGenerateRecommendations:
    def test_redness_gets_wound_care_advice(self, analyzer):
        recs = analyzer._generate_recommendations({'conditions': ['redness_area']})
        assert any('cool compress' in r.lower() for r in recs)

    def test_discolouration_gets_cold_pack_advice(self, analyzer):
        recs = analyzer._generate_recommendations({'conditions': ['dark_discolouration']})
        assert any('cold pack' in r.lower() for r in recs)

    def test_unusable_photo_gets_retake_advice(self, analyzer):
        recs = analyzer._generate_recommendations({'conditions': ['no_clear_skin_area']})
        assert any('retake' in r.lower() for r in recs)

    def test_always_includes_general_advice(self, analyzer):
        recs = analyzer._generate_recommendations({'conditions': []})
        assert any('worried' in r.lower() for r in recs)


class TestBasicModeResult:
    def test_basic_result_is_honest_about_limits(self, analyzer, tmp_path):
        from PIL import Image as PILImage
        path = tmp_path / 'skin.png'
        PILImage.fromarray(skin_image(red_patch=0.2)).save(path)
        result = analyzer._analyze_basic(str(path))
        assert result['analysis_mode'] == 'basic'
        assert result['confidence'] == 'low'
        assert 'not a diagnosis' in result['disclaimer'].lower()
        assert result['urgency'] in ('low', 'medium', 'high')

    def test_basic_result_for_non_skin_photo_asks_for_retake(self, analyzer, tmp_path):
        from PIL import Image as PILImage
        path = tmp_path / 'blue.png'
        PILImage.new('RGB', (50, 50), (40, 80, 200)).save(path)
        result = analyzer._analyze_basic(str(path))
        assert result['image_quality'] == 'poor'
        assert result['quality_note']


class TestPrepareImage:
    def test_large_images_are_downscaled(self, analyzer, tmp_path):
        from PIL import Image as PILImage
        path = tmp_path / 'big.png'
        PILImage.new('RGB', (4000, 3000), SKIN).save(path)
        im = analyzer._prepare_image(str(path))
        assert max(im.size) <= 1568

    def test_transparent_png_is_flattened_to_rgb(self, analyzer, tmp_path):
        from PIL import Image as PILImage
        path = tmp_path / 'alpha.png'
        PILImage.new('RGBA', (20, 20), (255, 0, 0, 0)).save(path)
        assert analyzer._prepare_image(str(path)).mode == 'RGB'


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
    def test_gemini_exception_falls_back_to_basic_mode(self, analyzer, tmp_path, monkeypatch):
        from PIL import Image as PILImage
        import quickaid.ai.medical_analyzer as ma
        monkeypatch.setattr(ma.time, 'sleep', lambda *_: None)  # skip the retry delay

        img_path = tmp_path / "test.png"
        PILImage.fromarray(skin_image(red_patch=0.2)).save(img_path)

        analyzer.use_gemini = True
        analyzer.client = MagicMock()
        analyzer.client.models.generate_content.side_effect = RuntimeError("API unavailable")

        result = analyzer.analyze_image(str(img_path))

        # Should have fallen back to basic mode instead of propagating the error
        assert 'detected_conditions' in result
        assert result['analysis_mode'] == 'basic'
        assert analyzer.client.models.generate_content.call_count == 2  # one retry before falling back


# ---------------------------------------------------------------------------
# Safety net applied to model output
# ---------------------------------------------------------------------------

class TestSafetyNet:
    def _parse(self, analyzer, **kw):
        fake = MagicMock()
        base = dict(detected_conditions=['x'], confidence='high', recommendations=['r'], urgency='low')
        base.update(kw)
        fake.parsed = ImageAnalysisResult(**base)
        return analyzer._parse_gemini_response(fake)

    def test_red_flags_raise_low_urgency_to_medium(self, analyzer):
        result = self._parse(analyzer, red_flags=['red streaks spreading from the wound'])
        assert result['urgency'] == 'medium'

    def test_high_urgency_is_never_lowered(self, analyzer):
        assert self._parse(analyzer, urgency='high', red_flags=[])['urgency'] == 'high'

    def test_poor_quality_photo_caps_confidence_at_low(self, analyzer):
        result = self._parse(analyzer, image_quality='poor', confidence='high')
        assert result['confidence'] == 'low'

    def test_non_medical_image_caps_confidence_at_low(self, analyzer):
        result = self._parse(analyzer, image_type='not_medical', confidence='high')
        assert result['confidence'] == 'low'

    def test_good_quality_keeps_model_confidence(self, analyzer):
        assert self._parse(analyzer, image_quality='good', confidence='high')['confidence'] == 'high'

    def test_new_fields_are_passed_through(self, analyzer):
        result = self._parse(analyzer, observations=['2 cm cut'], avoid=['Do not use butter'], summary='Small cut.')
        assert result['observations'] == ['2 cm cut']
        assert result['avoid'] == ['Do not use butter']
        assert result['summary'] == 'Small cut.'
        assert result['analysis_mode'] == 'ai'
