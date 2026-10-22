"""Admin user management: who can reach it, that it never exposes health data, and that every action does exactly
what it says (and is logged and, where applicable, emailed to the affected person).
"""

import pyotp
import pytest

from quickaid.core import database as db
from quickaid.services import admin_accounts

ROUTES = ['/admin/users', '/admin/api/users', '/admin/api/users/someone']   # GET-able routes only; POST-only action routes are checked separately


@pytest.fixture()
def admin(monkeypatch, registered_user):
    monkeypatch.setenv('ADMIN_USERNAMES', registered_user['username'])
    return registered_user


def enable_2fa(client):
    client.get('/account/2fa/setup')
    with client.session_transaction() as s:
        secret = s['pending_totp_secret']
    client.post('/account/2fa/confirm', data={'code': pyotp.TOTP(secret).now()})
    return secret


def make_other_user(app, username='otheruser', email='other@example.com'):
    with app.test_client() as c:
        c.post('/register', data={'username': username, 'email': email, 'password': 'supersecret123', 'confirm_password': 'supersecret123'})
    return username


class TestAccessControl:
    def test_anonymous_users_cannot_reach_any_admin_user_route(self, client):
        for url in ROUTES:
            assert client.get(url).status_code in (302, 401), url

    def test_a_signed_in_non_admin_gets_404_not_a_permission_error(self, client, registered_user):
        # 404, not 403: the page's existence is not revealed to a non-admin, same as the evaluation dashboard
        for url in ROUTES:
            assert client.get(url).status_code == 404, url

    def test_an_admin_can_reach_the_page_and_the_api(self, client, admin):
        assert client.get('/admin/users').status_code == 200
        assert client.get('/admin/api/users').status_code == 200

    def test_action_endpoints_are_also_404_for_a_non_admin(self, client, registered_user, app):
        target = make_other_user(app)
        for suffix in ('reset-2fa', 'force-logout', 'send-password-reset', 'delete'):
            assert client.post(f'/admin/api/users/{target}/{suffix}').status_code == 404

    def test_action_endpoints_require_login_at_all_for_an_anonymous_caller(self, client, app):
        target = make_other_user(app)
        for suffix in ('reset-2fa', 'force-logout', 'send-password-reset', 'delete'):
            assert client.post(f'/admin/api/users/{target}/{suffix}').status_code in (302, 401)


class TestListingNeverExposesHealthData:
    def test_listing_shows_only_account_fields(self, client, admin, app):
        make_other_user(app, 'patient1', 'patient1@example.com')
        with app.test_client() as c:
            c.post('/login', data={'username': 'patient1', 'password': 'supersecret123'})
            c.post('/analyze_symptoms', json={'symptoms': 'NEVERLEAK a very private symptom detail'})
        body = client.get('/admin/api/users').get_json()
        assert 'NEVERLEAK' not in str(body)
        usernames = {u['username'] for u in body['users']}
        assert {'patient1', admin['username']} <= usernames
        for u in body['users']:
            assert set(u.keys()) <= {'id', 'username', 'email', 'email_verified', 'totp_enabled', 'has_password', 'created_at', 'is_admin'}

    def test_detail_view_never_exposes_health_data_either(self, client, admin, app):
        make_other_user(app, 'patient2', 'patient2@example.com')
        with app.test_client() as c:
            c.post('/login', data={'username': 'patient2', 'password': 'supersecret123'})
            c.post('/analyze_symptoms', json={'symptoms': 'NEVERLEAK2 another private detail'})
        body = client.get('/admin/api/users/patient2').get_json()
        assert 'NEVERLEAK2' not in str(body)

    def test_search_filters_by_username_or_email(self, client, admin, app):
        make_other_user(app, 'findme', 'findme@example.com')
        make_other_user(app, 'other2', 'other2@example.com')
        body = client.get('/admin/api/users?q=findme').get_json()
        assert {u['username'] for u in body['users']} == {'findme'}


class TestResetTwoFactor:
    def test_reset_2fa_lets_a_locked_out_user_sign_in_with_just_their_password(self, client, admin, app):
        target = make_other_user(app, 'locked', 'locked@example.com')
        with app.test_client() as c:
            c.post('/login', data={'username': target, 'password': 'supersecret123'})
            enable_2fa(c)
            c.post('/logout')
            still_needs_2fa = c.post('/login', data={'username': target, 'password': 'supersecret123'})
            assert '/login/2fa' in still_needs_2fa.headers['Location']

        r = client.post(f'/admin/api/users/{target}/reset-2fa')
        assert r.status_code == 200 and r.get_json()['reset'] is True

        with app.test_client() as c2:
            now = c2.post('/login', data={'username': target, 'password': 'supersecret123'})
            assert now.status_code == 302 and '/login/2fa' not in now.headers['Location']

    def test_reset_2fa_on_an_account_without_2fa_is_a_clean_error_not_a_crash(self, client, admin, app):
        target = make_other_user(app, 'no2fa', 'no2fa@example.com')
        r = client.post(f'/admin/api/users/{target}/reset-2fa')
        assert r.status_code == 400

    def test_reset_2fa_is_logged_in_the_affected_users_own_security_history(self, client, admin, app):
        target = make_other_user(app, 'logtest', 'logtest@example.com')
        with app.test_client() as c:
            c.post('/login', data={'username': target, 'password': 'supersecret123'})
            enable_2fa(c)
        client.post(f'/admin/api/users/{target}/reset-2fa')
        user = db.get_user_by_username(target)
        events = [e['event'] for e in db.list_security_events(user['id'])]
        assert '2fa_disabled' in events

    def test_reset_2fa_notifies_the_account_owner(self, client, admin, app, monkeypatch):
        sent = []
        import quickaid.services.security as security_module
        monkeypatch.setattr(security_module, 'send_email', lambda to, subject, body, *a, **k: sent.append((to, subject)))
        target = make_other_user(app, 'notifytest', 'notifytest@example.com')
        with app.test_client() as c:
            c.post('/login', data={'username': target, 'password': 'supersecret123'})
            enable_2fa(c)
        client.post(f'/admin/api/users/{target}/reset-2fa')
        assert any('notifytest@example.com' == to and 'two-factor' in subj.lower() for to, subj in sent)

    def test_unknown_identifier_is_a_clean_404(self, client, admin):
        r = client.post('/admin/api/users/does-not-exist/reset-2fa')
        assert r.status_code == 404


class TestForceLogoutAndPasswordReset:
    def test_force_logout_invalidates_an_existing_session(self, client, admin, app):
        # each client must complete its own `with app.test_client() as c:` block before a DIFFERENT client makes a
        # request - nesting one inside another's `with` block confuses which session Flask-Login resolves
        # current_user from. The client object itself stays usable for further requests after its `with` block ends.
        target = make_other_user(app, 'sessiontest', 'sessiontest@example.com')
        with app.test_client() as c:
            c.post('/login', data={'username': target, 'password': 'supersecret123'})
        assert c.get('/api/history').status_code == 200

        resp = client.post(f'/admin/api/users/{target}/force-logout')
        assert resp.status_code == 200

        assert c.get('/api/history').status_code == 401   # the SAME client/session as before, now invalidated

    def test_send_password_reset_emails_a_working_link(self, client, admin, app, monkeypatch):
        sent = []
        import quickaid.blueprints.admin as admin_module
        monkeypatch.setattr(admin_module.admin_accounts, 'send_email', lambda to, subject, body, *a, **k: sent.append({'to': to, 'subject': subject, 'body': body}))
        target = make_other_user(app, 'resettest', 'resettest@example.com')
        r = client.post(f'/admin/api/users/{target}/send-password-reset')
        assert r.status_code == 200 and sent
        import re
        link = re.search(r'/reset-password/[\w\-]+', sent[0]['body']).group(0)
        with app.test_client() as c:
            reset = c.post(link, data={'password': 'Brand-New-Pass-99!', 'confirm_password': 'Brand-New-Pass-99!'})
            assert reset.status_code == 200
            new_login = c.post('/login', data={'username': target, 'password': 'Brand-New-Pass-99!'})
            assert new_login.status_code == 302


class TestDeleteAccount:
    def test_delete_requires_exact_confirmation(self, client, admin, app):
        target = make_other_user(app, 'deleteme', 'deleteme@example.com')
        wrong = client.post(f'/admin/api/users/{target}/delete', json={'confirm': 'not-the-right-name'})
        assert wrong.status_code == 400
        assert db.get_user_by_username(target) is not None   # nothing happened

        right = client.post(f'/admin/api/users/{target}/delete', json={'confirm': target})
        assert right.status_code == 200
        assert db.get_user_by_username(target) is None

    def test_deleted_account_can_no_longer_log_in(self, client, admin, app):
        target = make_other_user(app, 'goneuser', 'gone@example.com')
        client.post(f'/admin/api/users/{target}/delete', json={'confirm': target})
        with app.test_client() as c:
            r = c.post('/login', data={'username': target, 'password': 'supersecret123'})
            assert r.status_code != 302 or '/login' in r.headers.get('Location', '')


class TestServiceLayerDirectly:
    """The service module both the CLI and the web routes call - tested on its own, independent of HTTP."""

    def test_find_account_matches_by_username_or_email(self, app, client, registered_user):
        by_username = admin_accounts.find_account(registered_user['username'])
        by_email = admin_accounts.find_account(registered_user['email'])
        assert by_username and by_email and by_username['id'] == by_email['id']

    def test_find_account_returns_none_for_a_stranger(self, app):
        assert admin_accounts.find_account('nobody-by-this-name') is None

    def test_require_account_raises_a_clear_error(self, app):
        with pytest.raises(admin_accounts.AdminActionError):
            admin_accounts.require_account('nobody-by-this-name')

    def test_reset_two_factor_records_who_did_it(self, app, client, registered_user):
        enable_2fa(client)
        admin_accounts.reset_two_factor(registered_user['username'], by='test-admin')
        user = db.get_user_by_username(registered_user['username'])
        events = db.list_security_events(user['id'])
        assert any('test-admin' in (e.get('detail') or '') for e in events)


class TestCliUsesTheSameService:
    """The CLI is a thin wrapper around the exact same service functions - this just confirms the wiring, since
    the service functions themselves are already covered above and in TestServiceLayerDirectly."""

    def test_cli_reset_2fa_calls_the_service(self, app, client, registered_user, capsys):
        from quickaid.core import accounts_cli
        enable_2fa(client)
        accounts_cli.reset_2fa(registered_user['username'])
        out = capsys.readouterr().out
        assert 'sign in with just their password' in out
        user = db.get_user_by_username(registered_user['username'])
        assert not user['totp_enabled']

    def test_cli_delete_without_confirm_flag_does_nothing(self, app, client, registered_user, capsys):
        from quickaid.core import accounts_cli
        # calling the function directly raises the service-layer error; main() (the actual CLI entry point) is what
        # catches it and turns it into a clean exit code - checked separately below.
        with pytest.raises(admin_accounts.AdminActionError):
            accounts_cli.delete_account(registered_user['username'], confirm=None)
        assert db.get_user_by_username(registered_user['username']) is not None

    def test_cli_main_turns_the_same_error_into_a_clean_exit_code_not_a_traceback(self, app, client, registered_user, capsys):
        from quickaid.core import accounts_cli
        rc = accounts_cli.main(['delete', registered_user['username']])
        assert rc == 1
        assert db.get_user_by_username(registered_user['username']) is not None

    def test_cli_list_shows_the_account(self, app, client, registered_user, capsys):
        from quickaid.core import accounts_cli
        accounts_cli.list_users()
        out = capsys.readouterr().out
        assert registered_user['username'] in out
