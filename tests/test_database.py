"""Tests for database.py: user accounts and history persistence."""

import pytest


class TestUsers:
    def test_create_user_succeeds(self, db_module):
        user = db_module.create_user("alice", "alice@example.com", "password123")
        assert user is not None
        assert user['username'] == 'alice'
        assert user['email'] == 'alice@example.com'
        assert 'password_hash' in user
        assert user['password_hash'] != 'password123'  # never store plaintext

    def test_duplicate_username_rejected(self, db_module):
        db_module.create_user("alice", "alice@example.com", "password123")
        dup = db_module.create_user("alice", "different@example.com", "password456")
        assert dup is None

    def test_duplicate_email_rejected(self, db_module):
        db_module.create_user("alice", "alice@example.com", "password123")
        dup = db_module.create_user("bob", "alice@example.com", "password456")
        assert dup is None

    def test_get_user_by_username_found(self, db_module):
        created = db_module.create_user("alice", "alice@example.com", "password123")
        fetched = db_module.get_user_by_username("alice")
        assert fetched['id'] == created['id']

    def test_get_user_by_username_not_found(self, db_module):
        assert db_module.get_user_by_username("nobody") is None

    def test_get_user_by_id_found(self, db_module):
        created = db_module.create_user("alice", "alice@example.com", "password123")
        fetched = db_module.get_user_by_id(created['id'])
        assert fetched['username'] == 'alice'

    def test_get_user_by_id_not_found(self, db_module):
        assert db_module.get_user_by_id(99999) is None

    def test_verify_password_correct(self, db_module):
        db_module.create_user("alice", "alice@example.com", "correctpassword")
        user = db_module.get_user_by_username("alice")
        assert db_module.verify_password(user, "correctpassword") is True

    def test_verify_password_incorrect(self, db_module):
        db_module.create_user("alice", "alice@example.com", "correctpassword")
        user = db_module.get_user_by_username("alice")
        assert db_module.verify_password(user, "wrongpassword") is False


class TestHistory:
    def _make_user(self, db_module, username="alice"):
        return db_module.create_user(username, f"{username}@example.com", "password123")

    def test_save_and_get_image_analysis(self, db_module):
        user = self._make_user(db_module)
        db_module.save_image_analysis(user['id'], "cut.jpg", {
            "detected_conditions": ["laceration"],
            "confidence": "high",
            "urgency": "medium",
            "recommendations": ["clean it"],
            "safety_tips": ["wash hands"],
            "disclaimer": "test"
        })

        history = db_module.get_history(user['id'])
        assert len(history) == 1
        assert history[0]['type'] == 'image'
        assert history[0]['original_filename'] == 'cut.jpg'
        assert history[0]['detected_conditions'] == ['laceration']

    def test_save_and_get_symptom_analysis(self, db_module):
        user = self._make_user(db_module)
        db_module.save_symptom_analysis(user['id'], "headache", {
            "detected_symptoms": ["headache"],
            "possible_conditions": ["migraine"],
            "urgency_level": "low",
            "emergency_alert": {"alert": False},
            "recommendations": ["rest"],
            "safety_tips": [],
            "disclaimer": "test"
        })

        history = db_module.get_history(user['id'])
        assert len(history) == 1
        assert history[0]['type'] == 'symptom'
        assert history[0]['symptom_text'] == 'headache'
        assert history[0]['emergency_alert'] is False

    def test_emergency_alert_stored_correctly_when_true(self, db_module):
        user = self._make_user(db_module)
        db_module.save_symptom_analysis(user['id'], "chest pain", {
            "detected_symptoms": ["chest_pain"],
            "possible_conditions": ["heart attack"],
            "urgency_level": "high",
            "emergency_alert": {"alert": True},
            "recommendations": ["call 911"],
            "safety_tips": [],
            "disclaimer": "test"
        })
        history = db_module.get_history(user['id'])
        assert history[0]['emergency_alert'] is True

    def test_history_isolated_between_users(self, db_module):
        user_a = self._make_user(db_module, "alice")
        user_b = self._make_user(db_module, "bob")

        db_module.save_symptom_analysis(user_a['id'], "headache", {
            "detected_symptoms": ["headache"], "possible_conditions": [],
            "urgency_level": "low", "emergency_alert": {"alert": False},
            "recommendations": [], "safety_tips": [], "disclaimer": ""
        })

        assert len(db_module.get_history(user_a['id'])) == 1
        assert len(db_module.get_history(user_b['id'])) == 0

    def test_history_combines_and_sorts_both_types(self, db_module):
        user = self._make_user(db_module)
        db_module.save_image_analysis(user['id'], "a.jpg", {
            "detected_conditions": [], "confidence": "low", "urgency": "low",
            "recommendations": [], "safety_tips": [], "disclaimer": ""
        })
        db_module.save_symptom_analysis(user['id'], "cough", {
            "detected_symptoms": [], "possible_conditions": [],
            "urgency_level": "low", "emergency_alert": {"alert": False},
            "recommendations": [], "safety_tips": [], "disclaimer": ""
        })

        history = db_module.get_history(user['id'])
        assert len(history) == 2
        types = {entry['type'] for entry in history}
        assert types == {'image', 'symptom'}
        # newest first
        assert history[0]['created_at'] >= history[1]['created_at']

    def test_delete_history_clears_only_that_user(self, db_module):
        user_a = self._make_user(db_module, "alice")
        user_b = self._make_user(db_module, "bob")

        for user in (user_a, user_b):
            db_module.save_symptom_analysis(user['id'], "cough", {
                "detected_symptoms": [], "possible_conditions": [],
                "urgency_level": "low", "emergency_alert": {"alert": False},
                "recommendations": [], "safety_tips": [], "disclaimer": ""
            })

        db_module.delete_history(user_a['id'])

        assert len(db_module.get_history(user_a['id'])) == 0
        assert len(db_module.get_history(user_b['id'])) == 1

    def test_history_respects_limit(self, db_module):
        user = self._make_user(db_module)
        for i in range(5):
            db_module.save_symptom_analysis(user['id'], f"symptom {i}", {
                "detected_symptoms": [], "possible_conditions": [],
                "urgency_level": "low", "emergency_alert": {"alert": False},
                "recommendations": [], "safety_tips": [], "disclaimer": ""
            })

        history = db_module.get_history(user['id'], limit=3)
        assert len(history) == 3
