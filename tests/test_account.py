"""
Flask test-client tests for the account features: password reset, email
verification, account settings (profile/password/delete), and 2FA.
"""

import re
import pytest
import pyotp


class TestPasswordReset:
    def test_forgot_password_shows_generic_message_for_unknown_email(self, client):
        resp = client.post('/forgot-password', data={'email': 'nobody@example.com'})
        assert resp.status_code == 200
        assert b'password reset link' in resp.data

    def test_forgot_password_shows_same_message_for_known_email(self, client, registered_user):
        resp = client.post('/forgot-password', data={'email': registered_user['email']})
        assert resp.status_code == 200
        assert b'password reset link' in resp.data

    def test_reset_password_with_valid_token_changes_password(self, client, registered_user, app):
        import database as db
        user_row = db.get_user_by_username(registered_user['username'])
        raw_token = db.create_token(user_row['id'], 'password_reset', ttl_seconds=3600)

        resp = client.post(f'/reset-password/{raw_token}', data={
            'password': 'brandnewpassword1', 'confirm_password': 'brandnewpassword1'
        })
        assert resp.status_code == 200
        assert b'Password updated' in resp.data

        client.post('/logout')
        login_resp = client.post('/login', data={
            'username': registered_user['username'], 'password': 'brandnewpassword1'
        })
        assert login_resp.status_code == 302

        client.post('/logout')
        old_login = client.post('/login', data={
            'username': registered_user['username'], 'password': registered_user['password']
        })
        assert old_login.status_code == 401

    def test_reset_password_invalid_token_rejected(self, client):
        resp = client.get('/reset-password/not-a-real-token')
        assert resp.status_code == 400
        assert b'expired or invalid' in resp.data

    def test_reset_password_token_single_use(self, client, registered_user, app):
        import database as db
        user_row = db.get_user_by_username(registered_user['username'])
        raw_token = db.create_token(user_row['id'], 'password_reset', ttl_seconds=3600)

        client.post(f'/reset-password/{raw_token}', data={
            'password': 'firstnewpassword1', 'confirm_password': 'firstnewpassword1'
        })
        second_attempt = client.post(f'/reset-password/{raw_token}', data={
            'password': 'secondnewpassword1', 'confirm_password': 'secondnewpassword1'
        })
        assert second_attempt.status_code == 400

    def test_reset_password_mismatched_confirmation_rejected(self, client, registered_user, app):
        import database as db
        user_row = db.get_user_by_username(registered_user['username'])
        raw_token = db.create_token(user_row['id'], 'password_reset', ttl_seconds=3600)

        resp = client.post(f'/reset-password/{raw_token}', data={
            'password': 'brandnewpassword1', 'confirm_password': 'somethingelse'
        })
        assert resp.status_code == 400


class TestEmailVerification:
    def test_new_account_starts_unverified(self, client, registered_user):
        import database as db
        user_row = db.get_user_by_username(registered_user['username'])
        assert user_row['email_verified'] == 0

    def test_verify_email_with_valid_token(self, client, registered_user):
        import database as db
        user_row = db.get_user_by_username(registered_user['username'])
        raw_token = db.create_token(user_row['id'], 'email_verify', ttl_seconds=3600)

        resp = client.get(f'/verify-email/{raw_token}')
        assert resp.status_code == 200
        assert b'verified' in resp.data.lower()
        assert db.get_user_by_username(registered_user['username'])['email_verified'] == 1

    def test_verify_email_invalid_token_rejected(self, client):
        resp = client.get('/verify-email/not-a-real-token')
        assert resp.status_code == 400

    def test_resend_verification_requires_login(self, client):
        resp = client.post('/account/resend-verification')
        assert resp.status_code in (302, 401)


class TestAccountSettings:
    def test_account_page_requires_login(self, client):
        resp = client.get('/account')
        assert resp.status_code == 302

    def test_account_page_loads_when_logged_in(self, client, registered_user):
        resp = client.get('/account')
        assert resp.status_code == 200
        assert registered_user['username'].encode() in resp.data

    def test_update_profile_requires_correct_current_password(self, client, registered_user):
        resp = client.post('/account/profile', data={
            'username': registered_user['username'],
            'email': 'newemail@example.com',
            'current_password': 'wrongpassword',
        })
        assert resp.status_code == 400
        assert b'incorrect' in resp.data.lower()

    def test_update_profile_succeeds_with_correct_password(self, client, registered_user):
        resp = client.post('/account/profile', data={
            'username': 'renameduser',
            'email': 'renamed@example.com',
            'current_password': registered_user['password'],
        })
        assert resp.status_code == 200
        assert b'Profile updated' in resp.data

        import database as db
        updated = db.get_user_by_username('renameduser')
        assert updated is not None
        assert updated['email'] == 'renamed@example.com'
        assert updated['email_verified'] == 0  # changing email resets verification

    def test_update_profile_duplicate_username_rejected(self, client, registered_user):
        client.post('/logout')
        client.post('/register', data={
            'username': 'someoneelse', 'email': 'someoneelse@example.com', 'password': 'password12345'
        })
        client.post('/logout')
        client.post('/login', data=registered_user)

        resp = client.post('/account/profile', data={
            'username': 'someoneelse',
            'email': registered_user['email'],
            'current_password': registered_user['password'],
        })
        assert resp.status_code == 400
        assert b'already taken' in resp.data

    def test_update_password_requires_correct_current_password(self, client, registered_user):
        resp = client.post('/account/password', data={
            'current_password': 'wrongpassword',
            'new_password': 'brandnewpassword1',
            'confirm_password': 'brandnewpassword1',
        })
        assert resp.status_code == 400

    def test_update_password_succeeds(self, client, registered_user):
        resp = client.post('/account/password', data={
            'current_password': registered_user['password'],
            'new_password': 'brandnewpassword1',
            'confirm_password': 'brandnewpassword1',
        })
        assert resp.status_code == 200
        assert b'Password updated' in resp.data

        client.post('/logout')
        login_resp = client.post('/login', data={
            'username': registered_user['username'], 'password': 'brandnewpassword1'
        })
        assert login_resp.status_code == 302

    def test_update_password_mismatched_confirmation_rejected(self, client, registered_user):
        resp = client.post('/account/password', data={
            'current_password': registered_user['password'],
            'new_password': 'brandnewpassword1',
            'confirm_password': 'somethingdifferent',
        })
        assert resp.status_code == 400


class TestAccountDeletion:
    def test_delete_requires_correct_password(self, client, registered_user):
        resp = client.post('/account/delete', data={
            'current_password': 'wrongpassword', 'confirm_text': 'DELETE'
        })
        assert resp.status_code == 400

    def test_delete_requires_confirm_text(self, client, registered_user):
        resp = client.post('/account/delete', data={
            'current_password': registered_user['password'], 'confirm_text': 'not delete'
        })
        assert resp.status_code == 400

    def test_delete_succeeds_and_removes_account(self, client, registered_user):
        resp = client.post('/account/delete', data={
            'current_password': registered_user['password'], 'confirm_text': 'DELETE'
        })
        assert resp.status_code == 302

        import database as db
        assert db.get_user_by_username(registered_user['username']) is None

        # session should be logged out now
        assert client.get('/api/history').status_code in (302, 401)

    def test_delete_requires_login(self, client):
        resp = client.post('/account/delete', data={'current_password': 'x', 'confirm_text': 'DELETE'})
        assert resp.status_code == 302


class TestTwoFactorAuth:
    def test_setup_page_shows_qr_and_secret(self, client, registered_user):
        resp = client.get('/account/2fa/setup')
        assert resp.status_code == 200
        assert b'data:image/png;base64,' in resp.data

    def test_confirm_with_wrong_code_rejected(self, client, registered_user):
        client.get('/account/2fa/setup')
        resp = client.post('/account/2fa/confirm', data={'code': '000000'})
        assert resp.status_code == 400

    def test_confirm_with_valid_code_enables_2fa(self, client, registered_user):
        setup_resp = client.get('/account/2fa/setup')
        secret_match = re.search(rb'secret-box">([^<]+)<', setup_resp.data)
        secret = secret_match.group(1).decode().strip()

        valid_code = pyotp.TOTP(secret).now()
        resp = client.post('/account/2fa/confirm', data={'code': valid_code})
        assert resp.status_code == 200
        assert b'now enabled' in resp.data

        import database as db
        user_row = db.get_user_by_username(registered_user['username'])
        assert user_row['totp_enabled'] == 1

    def test_login_with_2fa_enabled_requires_code(self, client, registered_user):
        setup_resp = client.get('/account/2fa/setup')
        secret_match = re.search(rb'secret-box">([^<]+)<', setup_resp.data)
        secret = secret_match.group(1).decode().strip()
        client.post('/account/2fa/confirm', data={'code': pyotp.TOTP(secret).now()})

        client.post('/logout')
        login_resp = client.post('/login', data={
            'username': registered_user['username'], 'password': registered_user['password']
        })
        # Should redirect to the 2FA challenge, NOT log in yet.
        assert login_resp.status_code == 302
        assert '/login/2fa' in login_resp.headers['Location']
        assert client.get('/api/history').status_code in (302, 401)

        code_resp = client.post('/login/2fa', data={'code': pyotp.TOTP(secret).now()})
        assert code_resp.status_code == 302
        assert client.get('/api/history').status_code == 200

    def test_login_2fa_page_inaccessible_without_pending_login(self, client):
        resp = client.get('/login/2fa')
        assert resp.status_code == 302  # bounced back to /login

    def test_disable_2fa_requires_password(self, client, registered_user):
        setup_resp = client.get('/account/2fa/setup')
        secret_match = re.search(rb'secret-box">([^<]+)<', setup_resp.data)
        secret = secret_match.group(1).decode().strip()
        client.post('/account/2fa/confirm', data={'code': pyotp.TOTP(secret).now()})

        resp = client.post('/account/2fa/disable', data={'current_password': 'wrongpassword'})
        assert resp.status_code == 400

        resp2 = client.post('/account/2fa/disable', data={'current_password': registered_user['password']})
        assert resp2.status_code == 200

        import database as db
        assert db.get_user_by_username(registered_user['username'])['totp_enabled'] == 0


class TestGoogleOAuthDisabled:
    def test_login_google_returns_503_when_not_configured(self, client):
        resp = client.get('/login/google')
        assert resp.status_code == 503

    def test_google_button_hidden_on_login_page(self, client):
        resp = client.get('/login')
        assert b'Continue with Google' not in resp.data


class TestGoogleOAuthCallbackLogic:
    """
    The actual OAuth handshake (redirect to Google, token exchange) needs a
    live network call to Google and can't run in this test environment -
    Authlib's behavior there is exercised by Authlib's own test suite, not
    ours. What's unique to this app is what happens with the verified
    userinfo Google hands back, so that part is tested directly with the
    network call mocked out.

    GOOGLE_CLIENT_ID/SECRET have to be set *before* oauth.py and app.py are
    imported (they're read at module level), so this rebuilds the app from
    scratch here rather than reusing the standard `app`/`client` fixtures.
    """

    def _build_client(self, monkeypatch, temp_db_path):
        monkeypatch.setenv('SECRET_KEY', 'test-secret-key-for-pytest')
        monkeypatch.setenv('GEMINI_API_KEY', 'your_gemini_api_key_here')
        monkeypatch.setenv('DATABASE_PATH', temp_db_path)
        monkeypatch.setenv('GOOGLE_CLIENT_ID', 'test-client-id')
        monkeypatch.setenv('GOOGLE_CLIENT_SECRET', 'test-client-secret')

        import importlib
        import database
        importlib.reload(database)
        import oauth as oauth_module
        importlib.reload(oauth_module)
        import app as app_module
        importlib.reload(app_module)

        app_module.app.config.update({'TESTING': True, 'WTF_CSRF_ENABLED': False})
        return app_module.app.test_client(), oauth_module, database

    def test_callback_creates_new_account_from_verified_userinfo(self, temp_db_path, monkeypatch):
        client, oauth_module, db = self._build_client(monkeypatch, temp_db_path)
        fake_token = {'userinfo': {
            'sub': 'google-sub-1', 'email': 'newgoogleuser@example.com', 'email_verified': True
        }}
        monkeypatch.setattr(oauth_module.oauth.google, 'authorize_access_token', lambda: fake_token)

        resp = client.get('/login/google/callback')
        assert resp.status_code == 302

        user_row = db.get_user_by_oauth('google', 'google-sub-1')
        assert user_row is not None
        assert user_row['email'] == 'newgoogleuser@example.com'
        assert user_row['has_password'] == 0
        assert user_row['email_verified'] == 1

    def test_callback_links_to_existing_account_with_matching_email(self, temp_db_path, monkeypatch):
        client, oauth_module, db = self._build_client(monkeypatch, temp_db_path)
        client.post('/register', data={
            'username': 'existinguser', 'email': 'existing@example.com', 'password': 'password12345'
        })
        client.post('/logout')

        fake_token = {'userinfo': {
            'sub': 'google-sub-2', 'email': 'existing@example.com', 'email_verified': True
        }}
        monkeypatch.setattr(oauth_module.oauth.google, 'authorize_access_token', lambda: fake_token)

        resp = client.get('/login/google/callback')
        assert resp.status_code == 302

        linked = db.get_user_by_oauth('google', 'google-sub-2')
        assert linked['username'] == 'existinguser'
        assert linked['has_password'] == 1  # existing password not wiped out by linking

    def test_callback_rejects_unverified_email(self, temp_db_path, monkeypatch):
        client, oauth_module, db = self._build_client(monkeypatch, temp_db_path)
        fake_token = {'userinfo': {
            'sub': 'google-sub-3', 'email': 'unverified@example.com', 'email_verified': False
        }}
        monkeypatch.setattr(oauth_module.oauth.google, 'authorize_access_token', lambda: fake_token)

        resp = client.get('/login/google/callback')
        assert resp.status_code == 400
        assert db.get_user_by_oauth('google', 'google-sub-3') is None
