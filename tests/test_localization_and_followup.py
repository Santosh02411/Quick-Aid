"""
Tests for localization (region-aware emergency numbers / temperature units)
and multi-turn follow-up conversation support.
"""

import pytest
from unittest.mock import MagicMock
from localization import localize_text, localize_analysis, fever_threshold_text, get_region_info, REGIONS
from conversation import ConversationService, NO_GEMINI_MESSAGE


# ---------------------------------------------------------------------------
# localization.py
# ---------------------------------------------------------------------------

class TestLocalization:
    def test_us_gets_911(self):
        assert localize_text("Call {EMERGENCY_NUMBER} now", "US") == "Call 911 now"

    def test_uk_gets_999(self):
        assert localize_text("Call {EMERGENCY_NUMBER} now", "UK") == "Call 999 now"

    def test_india_gets_112(self):
        assert localize_text("Call {EMERGENCY_NUMBER} now", "IN") == "Call 112 now"

    def test_unknown_region_falls_back_to_intl(self):
        result = localize_text("Call {EMERGENCY_NUMBER} now", "NARNIA")
        assert result == "Call your local emergency number now"

    def test_fever_threshold_us_shows_fahrenheit_first(self):
        text = fever_threshold_text("US")
        assert text.index('°F') < text.index('°C')

    def test_fever_threshold_uk_shows_celsius_first(self):
        text = fever_threshold_text("UK")
        assert text.index('°C') < text.index('°F')

    def test_localize_analysis_walks_recommendations_and_safety_tips(self):
        analysis = {
            'recommendations': ['Call {EMERGENCY_NUMBER} immediately', 'Rest'],
            'safety_tips': ['Emergency: {EMERGENCY_NUMBER}'],
            'emergency_alert': {'alert': True, 'message': 'urgent', 'action': 'Call {EMERGENCY_NUMBER}'}
        }
        result = localize_analysis(analysis, 'AU')
        assert result['recommendations'][0] == 'Call 000 immediately'
        assert result['safety_tips'][0] == 'Emergency: 000'
        assert result['emergency_alert']['action'] == 'Call 000'

    def test_all_regions_have_required_fields(self):
        for code, info in REGIONS.items():
            assert 'label' in info
            assert 'emergency_number' in info
            assert info['temp_unit'] in ('C', 'F')


# ---------------------------------------------------------------------------
# conversation.py
# ---------------------------------------------------------------------------

class TestConversationService:
    def test_no_gemini_returns_clear_message(self):
        cs = ConversationService()
        assert cs.use_gemini is False
        result = cs.ask_follow_up('symptom', '{}', [], 'is this serious?')
        assert result == NO_GEMINI_MESSAGE

    def test_gemini_configured_builds_full_context(self):
        cs = ConversationService()
        cs.use_gemini = True
        cs.client = MagicMock()
        fake_response = MagicMock()
        fake_response.text = "That sounds mild."
        cs.client.models.generate_content.return_value = fake_response

        answer = cs.ask_follow_up(
            'image',
            '{"detected_conditions": ["minor cut"]}',
            [{'role': 'user', 'content': 'q1'}, {'role': 'model', 'content': 'a1'}],
            'q2',
            region='US'
        )
        assert answer == "That sounds mild."

        contents = cs.client.models.generate_content.call_args.kwargs['contents']
        # system context + ack + 2 history turns + new question = 5
        assert len(contents) == 5
        assert contents[-1].parts[0].text == 'q2'

    def test_empty_gemini_response_handled_gracefully(self):
        cs = ConversationService()
        cs.use_gemini = True
        cs.client = MagicMock()
        fake_response = MagicMock()
        fake_response.text = ""
        cs.client.models.generate_content.return_value = fake_response

        answer = cs.ask_follow_up('symptom', '{}', [], 'question')
        assert "wasn't able" in answer.lower() or "rephrase" in answer.lower()

    def test_gemini_exception_returns_graceful_fallback(self):
        cs = ConversationService()
        cs.use_gemini = True
        cs.client = MagicMock()
        cs.client.models.generate_content.side_effect = RuntimeError("down")

        answer = cs.ask_follow_up('symptom', '{}', [], 'question')
        assert "couldn't process" in answer.lower()


# ---------------------------------------------------------------------------
# App-level: /api/follow_up, /api/region, /api/regions
# ---------------------------------------------------------------------------

class TestFollowUpEndpoint:
    def test_follow_up_requires_login(self, client):
        resp = client.post('/api/follow_up', json={
            'analysis_type': 'symptom', 'analysis_id': 1, 'question': 'test'
        })
        assert resp.status_code == 401

    def test_follow_up_on_own_analysis(self, client, registered_user):
        symptom_resp = client.post('/analyze_symptoms', json={'symptoms': 'headache and fever'})
        analysis_id = symptom_resp.get_json()['analysis_id']

        resp = client.post('/api/follow_up', json={
            'analysis_type': 'symptom', 'analysis_id': analysis_id, 'question': 'is this serious?'
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['success'] is True
        assert 'answer' in data
        assert len(data['follow_up_history']) == 2

    def test_follow_up_on_nonexistent_analysis_404s(self, client, registered_user):
        resp = client.post('/api/follow_up', json={
            'analysis_type': 'symptom', 'analysis_id': 99999, 'question': 'test'
        })
        assert resp.status_code == 404

    def test_follow_up_cannot_access_other_users_analysis(self, client, registered_user, app):
        symptom_resp = client.post('/analyze_symptoms', json={'symptoms': 'headache'})
        analysis_id = symptom_resp.get_json()['analysis_id']

        client.post('/logout')
        client.post('/register', data={
            'username': 'otheruser', 'email': 'other@example.com', 'password': 'supersecret123'
        })

        resp = client.post('/api/follow_up', json={
            'analysis_type': 'symptom', 'analysis_id': analysis_id, 'question': 'test'
        })
        assert resp.status_code == 404  # not 403 - don't leak existence

    def test_follow_up_invalid_analysis_type_rejected(self, client, registered_user):
        resp = client.post('/api/follow_up', json={
            'analysis_type': 'bogus', 'analysis_id': 1, 'question': 'test'
        })
        assert resp.status_code == 400

    def test_follow_up_empty_question_rejected(self, client, registered_user):
        symptom_resp = client.post('/analyze_symptoms', json={'symptoms': 'headache'})
        analysis_id = symptom_resp.get_json()['analysis_id']

        resp = client.post('/api/follow_up', json={
            'analysis_type': 'symptom', 'analysis_id': analysis_id, 'question': ''
        })
        assert resp.status_code == 400

    def test_follow_up_too_long_question_rejected(self, client, registered_user):
        symptom_resp = client.post('/analyze_symptoms', json={'symptoms': 'headache'})
        analysis_id = symptom_resp.get_json()['analysis_id']

        resp = client.post('/api/follow_up', json={
            'analysis_type': 'symptom', 'analysis_id': analysis_id, 'question': 'a' * 600
        })
        assert resp.status_code == 400


class TestRegionEndpoints:
    def test_list_regions_is_public(self, client):
        resp = client.get('/api/regions')
        assert resp.status_code == 200
        data = resp.get_json()
        assert 'US' in data
        assert 'INTL' in data

    def test_update_region_requires_login(self, client):
        resp = client.post('/api/region', json={'region': 'UK'})
        assert resp.status_code == 401

    def test_update_region_valid(self, client, registered_user):
        resp = client.post('/api/region', json={'region': 'UK'})
        assert resp.status_code == 200
        assert resp.get_json()['region'] == 'UK'

    def test_update_region_invalid_rejected(self, client, registered_user):
        resp = client.post('/api/region', json={'region': 'ATLANTIS'})
        assert resp.status_code == 400

    def test_register_with_region(self, client):
        resp = client.post('/register', data={
            'username': 'regionuser', 'email': 'regionuser@example.com',
            'password': 'supersecret123', 'region': 'IN'
        })
        assert resp.status_code == 302

    def test_analyze_symptoms_uses_users_region(self, client, registered_user):
        client.post('/api/region', json={'region': 'UK'})
        resp = client.post('/analyze_symptoms', json={'symptoms': 'chest pain'})
        data = resp.get_json()
        assert '999' in data['analysis']['emergency_alert']['action']
        assert '911' not in data['analysis']['emergency_alert']['action']
