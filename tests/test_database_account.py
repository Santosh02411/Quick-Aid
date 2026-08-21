"""Tests for database.py: password reset / verification tokens, profile
updates, OAuth account linking, and 2FA storage."""

import time
import pytest


class TestTokens:
    def _make_user(self, db_module):
        return db_module.create_user("alice", "alice@example.com", "password123")

    def test_create_and_get_valid_token(self, db_module):
        user = self._make_user(db_module)
        raw = db_module.create_token(user['id'], 'password_reset', ttl_seconds=3600)
        token_row = db_module.get_valid_token(raw, 'password_reset')
        assert token_row is not None
        assert token_row['user_id'] == user['id']

    def test_token_hash_not_stored_in_plaintext(self, db_module):
        user = self._make_user(db_module)
        raw = db_module.create_token(user['id'], 'password_reset', ttl_seconds=3600)
        with db_module.get_connection() as conn:
            row = conn.execute("SELECT token_hash FROM user_tokens WHERE user_id = ?", (user['id'],)).fetchone()
        assert row['token_hash'] != raw

    def test_wrong_token_rejected(self, db_module):
        user = self._make_user(db_module)
        db_module.create_token(user['id'], 'password_reset', ttl_seconds=3600)
        assert db_module.get_valid_token('not-the-real-token', 'password_reset') is None

    def test_wrong_purpose_rejected(self, db_module):
        user = self._make_user(db_module)
        raw = db_module.create_token(user['id'], 'password_reset', ttl_seconds=3600)
        assert db_module.get_valid_token(raw, 'email_verify') is None

    def test_expired_token_rejected(self, db_module):
        user = self._make_user(db_module)
        raw = db_module.create_token(user['id'], 'password_reset', ttl_seconds=-1)
        assert db_module.get_valid_token(raw, 'password_reset') is None

    def test_used_token_rejected(self, db_module):
        user = self._make_user(db_module)
        raw = db_module.create_token(user['id'], 'password_reset', ttl_seconds=3600)
        token_row = db_module.get_valid_token(raw, 'password_reset')
        db_module.consume_token(token_row['id'])
        assert db_module.get_valid_token(raw, 'password_reset') is None

    def test_creating_new_token_invalidates_old_one(self, db_module):
        user = self._make_user(db_module)
        first = db_module.create_token(user['id'], 'password_reset', ttl_seconds=3600)
        second = db_module.create_token(user['id'], 'password_reset', ttl_seconds=3600)
        assert db_module.get_valid_token(first, 'password_reset') is None
        assert db_module.get_valid_token(second, 'password_reset') is not None

    def test_token_deleted_when_user_deleted(self, db_module):
        user = self._make_user(db_module)
        raw = db_module.create_token(user['id'], 'password_reset', ttl_seconds=3600)
        db_module.delete_user(user['id'])
        assert db_module.get_valid_token(raw, 'password_reset') is None


class TestProfileUpdates:
    def _make_user(self, db_module):
        return db_module.create_user("alice", "alice@example.com", "password123")

    def test_update_username_succeeds(self, db_module):
        user = self._make_user(db_module)
        assert db_module.update_username(user['id'], 'alice2') is True
        assert db_module.get_user_by_id(user['id'])['username'] == 'alice2'

    def test_update_username_conflict_rejected(self, db_module):
        db_module.create_user("bob", "bob@example.com", "password123")
        user = self._make_user(db_module)
        assert db_module.update_username(user['id'], 'bob') is False

    def test_update_email_resets_verification(self, db_module):
        user = self._make_user(db_module)
        db_module.set_email_verified(user['id'])
        assert db_module.get_user_by_id(user['id'])['email_verified'] == 1

        db_module.update_email(user['id'], 'alice-new@example.com')
        updated = db_module.get_user_by_id(user['id'])
        assert updated['email'] == 'alice-new@example.com'
        assert updated['email_verified'] == 0

    def test_update_email_conflict_rejected(self, db_module):
        db_module.create_user("bob", "bob@example.com", "password123")
        user = self._make_user(db_module)
        assert db_module.update_email(user['id'], 'bob@example.com') is False

    def test_set_password_changes_hash_and_has_password(self, db_module):
        user = self._make_user(db_module)
        old_hash = user['password_hash']
        db_module.set_password(user['id'], 'brandnewpassword')
        updated = db_module.get_user_by_id(user['id'])
        assert updated['password_hash'] != old_hash
        assert updated['has_password'] == 1
        assert db_module.verify_password(updated, 'brandnewpassword') is True

    def test_delete_user_removes_account_and_history(self, db_module):
        user = self._make_user(db_module)
        db_module.save_symptom_analysis(user['id'], "cough", {
            "detected_symptoms": [], "possible_conditions": [],
            "urgency_level": "low", "emergency_alert": {"alert": False},
            "recommendations": [], "safety_tips": [], "disclaimer": ""
        })
        db_module.delete_user(user['id'])
        assert db_module.get_user_by_id(user['id']) is None
        assert db_module.get_history(user['id']) == []


class TestTwoFactor:
    def _make_user(self, db_module):
        return db_module.create_user("alice", "alice@example.com", "password123")

    def test_totp_disabled_by_default(self, db_module):
        user = self._make_user(db_module)
        assert user['totp_enabled'] == 0

    def test_enable_and_disable_totp(self, db_module):
        user = self._make_user(db_module)
        db_module.set_pending_totp_secret(user['id'], 'ABCDEFGH')
        db_module.enable_totp(user['id'])
        enabled = db_module.get_user_by_id(user['id'])
        assert enabled['totp_enabled'] == 1
        assert enabled['totp_secret'] == 'ABCDEFGH'

        db_module.disable_totp(user['id'])
        disabled = db_module.get_user_by_id(user['id'])
        assert disabled['totp_enabled'] == 0
        assert disabled['totp_secret'] is None


class TestOAuth:
    def test_create_oauth_user_has_no_usable_password(self, db_module):
        user = db_module.create_oauth_user('janedoe', 'jane@example.com', 'google', 'sub123')
        assert user['has_password'] == 0
        assert user['email_verified'] == 1
        assert db_module.verify_password(user, 'anything') is False

    def test_get_user_by_oauth_found(self, db_module):
        created = db_module.create_oauth_user('janedoe', 'jane@example.com', 'google', 'sub123')
        found = db_module.get_user_by_oauth('google', 'sub123')
        assert found['id'] == created['id']

    def test_get_user_by_oauth_not_found(self, db_module):
        assert db_module.get_user_by_oauth('google', 'nope') is None

    def test_link_oauth_to_existing_user(self, db_module):
        user = db_module.create_user("alice", "alice@example.com", "password123")
        db_module.link_oauth_to_user(user['id'], 'google', 'sub456')
        found = db_module.get_user_by_oauth('google', 'sub456')
        assert found['id'] == user['id']
        # linking doesn't remove their existing password
        assert found['has_password'] == 1
