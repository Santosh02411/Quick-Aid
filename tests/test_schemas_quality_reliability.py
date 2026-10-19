"""Tests for response validation, photo-quality checks and reliability scoring."""

import json
from io import BytesIO
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image, ImageFilter
from pydantic import ValidationError

from quickaid.ai import image_quality, reliability
from quickaid.ai.schemas import (ImageAssessment, InterviewPlan, SymptomAssessment, extract_json_object, validate_response)


class Resp:
    def __init__(self, text=None, parsed=None):
        self.text, self.parsed = text, parsed


# ---------------------------------------------------------------------------
# schemas
# ---------------------------------------------------------------------------

class TestExtractJson:
    def test_plain_fenced_and_surrounded(self):
        assert extract_json_object('{"a": 1}') == {'a': 1}
        assert extract_json_object('```json\n{"a": 2}\n```') == {'a': 2}
        assert extract_json_object('Sure! {"a": 3} hope that helps') == {'a': 3}

    @pytest.mark.parametrize('bad', [None, '', 'no json here', '[1, 2]', '{broken', 123])
    def test_unusable_input_returns_none(self, bad):
        assert extract_json_object(bad) is None


class TestSymptomSchema:
    def test_missing_urgency_defaults_to_medium_never_low(self):
        m = SymptomAssessment.model_validate({})
        assert m.urgency_level == 'medium' and m.emergency_alert is False

    @pytest.mark.parametrize('bad', [
        {'urgency_level': 'extreme'},
        {'urgency_level': 'low', 'possible_explanations': [{'name': 'x', 'likelihood': 'certain'}]},
        {'urgency_level': 'low', 'recommendations': [{'nested': 'object'}]},
        {'urgency_level': 'low', 'emergency_alert': 'maybe-ish'},
    ])
    def test_invalid_values_are_rejected(self, bad):
        with pytest.raises((ValidationError, ValueError)):
            SymptomAssessment.model_validate(bad)

    def test_lists_are_cleaned_deduplicated_and_capped(self):
        m = SymptomAssessment.model_validate({'recommendations': ['  rest  ', 'REST', 'x' * 900] + [f'step {i}' for i in range(30)]})
        assert m.recommendations[0] == 'rest' and len(m.recommendations) <= 12
        assert max(len(r) for r in m.recommendations) <= 300

    def test_explanations_are_capped_at_four(self):
        m = SymptomAssessment.model_validate({'possible_explanations': [{'name': f'c{i}'} for i in range(9)]})
        assert len(m.possible_explanations) == 4

    def test_a_single_string_becomes_a_list(self):
        assert SymptomAssessment.model_validate({'recommendations': 'rest'}).recommendations == ['rest']

    def test_all_four_kinds_of_statement_have_their_own_fields(self):
        fields = set(SymptomAssessment.model_fields)
        assert {'reported_symptoms', 'possible_explanations', 'red_flags_present', 'warning_signs', 'recommendations', 'information_gaps'} <= fields


class TestImageSchema:
    def test_findings_accept_plain_strings(self):
        m = ImageAssessment.model_validate({'findings': ['a red patch', {'description': 'swelling', 'certainty': 'clearly visible'}]})
        assert [f.certainty for f in m.findings] == ['possibly visible', 'clearly visible']

    def test_confidence_defaults_to_low_and_is_a_label_not_a_number(self):
        assert ImageAssessment.model_validate({}).confidence == 'low'
        with pytest.raises(ValidationError):
            ImageAssessment.model_validate({'confidence': 0.93})

    def test_observation_and_interpretation_are_separate_fields(self):
        fields = set(ImageAssessment.model_fields)
        assert {'findings', 'possible_explanations', 'not_assessable', 'red_flags_present', 'warning_signs'} <= fields

    def test_plan_requires_an_associated_question(self):
        with pytest.raises(ValidationError):
            InterviewPlan.model_validate({'detected_symptoms': []})


class TestValidateResponse:
    def test_prefers_parsed_object_but_revalidates_it(self):
        good = SymptomAssessment(urgency_level='low', emergency_alert=False)
        assert validate_response(Resp(parsed=good), SymptomAssessment).urgency_level == 'low'

    def test_falls_back_to_text(self):
        assert validate_response(Resp(text='{"urgency_level": "high"}'), SymptomAssessment).urgency_level == 'high'

    def test_returns_none_for_untrustworthy_answers_and_never_raises(self):
        assert validate_response(Resp(text='not json'), SymptomAssessment) is None
        assert validate_response(Resp(text='{"urgency_level": "extreme"}'), SymptomAssessment) is None
        assert validate_response(MagicMock(parsed=None, text=None), ImageAssessment) is None
        assert validate_response(object(), SymptomAssessment) is None

    def test_invalid_parsed_object_falls_through_to_valid_text(self):
        assert validate_response(Resp(text='{"urgency_level": "medium"}', parsed={'urgency_level': 'banana'}), SymptomAssessment).urgency_level == 'medium'


# ---------------------------------------------------------------------------
# image quality
# ---------------------------------------------------------------------------

def textured(n=600, seed=3):
    rng = np.random.default_rng(seed)
    base = np.zeros((n, n, 3), dtype=np.int16)
    base[:] = (215, 165, 145)
    base += rng.integers(-25, 25, (n, n, 1))
    base[150:350, 200:420] = (190, 70, 70)
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


class TestImageQuality:
    def test_sharp_well_lit_photo_is_good(self):
        q = image_quality.assess(textured())
        assert q['rating'] == 'good' and not q['unusable'] and q['issues'] == [] and q['score'] == 100

    def test_blurry_photo_is_poor_with_retake_advice(self):
        q = image_quality.assess(textured().filter(ImageFilter.GaussianBlur(14)))
        assert q['rating'] == 'poor' and any('blurry' in i.lower() for i in q['issues'])
        assert any('steady' in a.lower() or 'focus' in a.lower() for a in q['advice'])

    @pytest.mark.parametrize('img', [
        Image.eval(textured(), lambda v: int(v * 0.1)),
        textured().resize((50, 50)),
        Image.new('RGB', (600, 600), (220, 170, 150)),
        Image.new('RGB', (600, 600), (0, 0, 0)),
        Image.new('RGB', (600, 600), (255, 255, 255)),
    ])
    def test_unusable_photos_are_flagged(self, img):
        q = image_quality.assess(img)
        assert q['unusable'] and q['rating'] == 'poor' and q['score'] <= 15

    def test_low_resolution_and_glare_are_fair(self):
        assert image_quality.assess(textured().resize((200, 200)))['rating'] == 'fair'
        arr = np.asarray(textured()).copy()
        arr[:400, :450] = 255
        assert image_quality.assess(Image.fromarray(arr))['rating'] == 'fair'

    def test_metrics_are_reported_for_transparency(self):
        m = image_quality.assess(textured())['metrics']
        assert {'width', 'height', 'brightness', 'contrast', 'sharpness', 'glare_fraction'} <= set(m)

    def test_worst_of_two_ratings(self):
        assert image_quality.worst('good', 'poor') == 'poor' and image_quality.worst('fair', 'good') == 'fair'

    def test_handles_grayscale_and_rgba(self):
        assert image_quality.assess(textured().convert('L'))['rating'] in ('good', 'fair', 'poor')
        assert image_quality.assess(textured().convert('RGBA'))['rating'] in ('good', 'fair', 'poor')


class TestAnalyzerQualityGate:
    @pytest.fixture()
    def analyzer(self):
        from quickaid.ai.medical_analyzer import MedicalAnalyzer
        a = MedicalAnalyzer()
        a.use_gemini = True
        a.client = MagicMock()
        return a

    def test_unusable_photo_never_reaches_the_model(self, analyzer, tmp_path):
        path = tmp_path / 'dark.png'
        Image.eval(textured(), lambda v: int(v * 0.1)).save(path)
        out = analyzer.analyze_image(str(path))
        analyzer.client.models.generate_content.assert_not_called()
        assert out['assessment_status'] == 'insufficient' and out['possible_explanations'] == []
        assert out['quality_check']['unusable'] and out['recommendations']

    def test_blurry_photo_reaches_the_model_but_confidence_is_capped(self, analyzer, tmp_path):
        path = tmp_path / 'blur.png'
        textured().filter(ImageFilter.GaussianBlur(14)).save(path)
        analyzer.client.models.generate_content.return_value = Resp(
            text=json.dumps({'urgency': 'low', 'confidence': 'high', 'image_quality': 'good',
                             'possible_explanations': [{'name': 'graze', 'likelihood': 'more likely'}], 'recommendations': ['Clean it']}))
        out = analyzer.analyze_image(str(path))
        assert analyzer.client.models.generate_content.called
        assert out['image_quality'] == 'poor' and out['confidence'] == 'low' and out['reliability']['level'] == 'limited'
        assert out['possible_explanations'][0]['likelihood'] == 'possible'

    def test_quality_findings_are_passed_to_the_model(self, analyzer, tmp_path):
        path = tmp_path / 'blur.png'
        textured().filter(ImageFilter.GaussianBlur(14)).save(path)
        analyzer.client.models.generate_content.return_value = Resp(text=json.dumps({'urgency': 'low'}))
        analyzer.analyze_image(str(path))
        sent = analyzer.client.models.generate_content.call_args.kwargs['contents'][1]
        assert 'blurry' in sent.lower()

    def test_unvalidatable_model_answer_falls_back_instead_of_showing_it(self, analyzer, tmp_path, monkeypatch):
        import quickaid.ai.medical_analyzer as ma
        monkeypatch.setattr(ma.time, 'sleep', lambda *_: None)
        path = tmp_path / 'ok.png'
        textured().save(path)
        analyzer.client.models.generate_content.return_value = Resp(text='{"urgency": "extreme", "confidence": 0.99}')
        out = analyzer.analyze_image(str(path))
        assert out['analysis_mode'] == 'basic' and out['confidence'] == 'low'
        assert '0.99' not in json.dumps(out)

    def test_result_carries_all_explainability_fields(self, analyzer, tmp_path):
        path = tmp_path / 'ok.png'
        textured().save(path)
        analyzer.client.models.generate_content.return_value = Resp(text=json.dumps(
            {'urgency': 'low', 'confidence': 'medium', 'findings': [{'description': 'a small graze', 'certainty': 'clearly visible'}],
             'possible_explanations': [{'name': 'superficial graze', 'likelihood': 'more likely', 'supported_by': ['a small graze']}],
             'recommendations': ['Wash gently'], 'not_assessable': ['depth']}))
        out = analyzer.analyze_image(str(path))
        for key in ('findings', 'not_assessable', 'possible_explanations', 'red_flags_present', 'recommendations', 'reliability',
                    'uncertainty_statement', 'safety_review', 'assessment_status', 'quality_check', 'disclaimer'):
            assert key in out, key


# ---------------------------------------------------------------------------
# reliability
# ---------------------------------------------------------------------------

class TestReliability:
    def test_never_returns_a_number(self):
        r = reliability.assess('image', mode='ai', ai_self_rating='high', image_quality='good')
        assert not any(isinstance(v, (int, float)) and not isinstance(v, bool) for v in r.values())

    def test_basic_mode_is_limited(self):
        assert reliability.assess('symptom', mode='basic', interview=True, answered=6, skipped=0)['level'] == 'limited'

    def test_poor_photo_is_limited_and_fair_photo_is_not_better_than_moderate(self):
        assert reliability.assess('image', mode='ai', image_quality='poor')['level'] == 'limited'
        assert reliability.assess('image', mode='ai', ai_self_rating='high', image_quality='fair')['level'] in ('moderate', 'good')

    def test_insufficient_is_always_limited(self):
        assert reliability.assess('symptom', mode='ai', interview=True, answered=8, skipped=0, insufficient=True)['level'] == 'limited'

    def test_ai_low_self_rating_lowers_reliability(self):
        hi = reliability.assess('image', mode='ai', ai_self_rating='high', image_quality='good')
        lo = reliability.assess('image', mode='ai', ai_self_rating='low', image_quality='good')
        order = {'limited': 0, 'moderate': 1, 'good': 2}
        assert order[lo['level']] < order[hi['level']]

    def test_self_rating_is_explained_as_unverified(self):
        r = reliability.assess('image', mode='ai', ai_self_rating='high', image_quality='good')
        assert 'not' in r['ai_self_rating']['note'].lower() and 'probability' in r['ai_self_rating']['note'].lower()
        assert 'not the chance' in r['note'].lower()

    def test_interview_completeness_matters(self):
        full = reliability.assess('symptom', mode='ai', interview=True, answered=6, skipped=0, input_chars=80, symptoms_found=2)
        skipped = reliability.assess('symptom', mode='ai', interview=True, answered=1, skipped=5, input_chars=80, symptoms_found=2)
        order = {'limited': 0, 'moderate': 1, 'good': 2}
        assert order[skipped['level']] < order[full['level']]

    def test_every_level_has_an_honest_statement(self):
        for level in ('limited', 'moderate', 'good'):
            assert 'not a diagnosis' in reliability.uncertainty_statement(level).lower()

    def test_reasons_are_always_given_when_not_good(self):
        r = reliability.assess('symptom', mode='basic', interview=False)
        assert r['reasons']
