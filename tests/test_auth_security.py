"""Authentication, session, 2FA, password-reset and logging security."""

import logging
import re
import time

import pyotp
import pytest

from quickaid import config
from quickaid.services import security

PW = 'orange-kettle-42'


def register(client, name='alice', email='alice@example.com', pw=PW):
    return client.post('/register', data={'username': name, 'email': email, 'password': pw, 'confirm_password': pw})


def login(client, name='alice', pw=PW, **kw):
    return client.post('/login', data={'username': name, 'password': pw}, **kw)


class _Capture(logging.Handler):
    """The app's logger does not propagate to the root logger, so pytest's caplog never sees it. Attach directly to the app's logger.
    Deliberately has NO privacy filter: it shows what the code itself writes (the filter is tested separately)."""

    def __init__(self):
        super().__init__(logging.DEBUG)
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


@pytest.fixture()
def applog():
    handler = _Capture()
    root = logging.getLogger('quickaid')
    root.addHandler(handler)
    yield handler
    root.removeHandler(handler)


@pytest.fixture()
def mails(monkeypatch):
    """Capture every email the app tries to send."""
    sent = []
    fake = lambda to, subject, body, *a, **k: sent.append({'to': to, 'subject': subject, 'body': body})   # noqa: E731
    monkeypatch.setattr('quickaid.blueprints.auth.send_email', fake)
    monkeypatch.setattr('quickaid.services.security.send_email', fake)
    return sent


@pytest.fixture()
def alice(app, client, mails):
    register(client)
    mails.clear()
    return client


def enable_2fa(client):
    client.get('/account/2fa/setup')
    with client.session_transaction() as s:
        secret = s['pending_totp_secret']
    r = client.post('/account/2fa/confirm', data={'code': pyotp.TOTP(secret).now()})
    codes = re.findall(r'\b[A-Z2-9]{5}-[A-Z2-9]{5}\b', r.get_data(as_text=True))
    return secret, codes, r


class TestPasswordPolicy:
    @pytest.mark.parametrize('pw', ['password', 'password123', 'Password1', 'qwerty123', '12345678', 'letmein123', 'welcome1', 'iloveyou1', 'admin@123'])
    def test_common_passwords_are_refused(self, pw):
        assert security.check_password(pw, 'zed', 'zed@example.com')

    @pytest.mark.parametrize('pw,why', [('aaaaaaaaaa', 'repeated'), ('abababababab', 'repeated'), ('abcdefghij', 'sequence'), ('0123456789', 'sequence'),
                                        ('x' * 129, 'at most'), ('short', 'at least')])
    def test_weak_shapes_are_refused(self, pw, why):
        assert why in security.check_password(pw, 'zed', 'z@example.com')

    def test_it_must_not_contain_your_username_or_email_name(self):
        assert 'username' in security.check_password('santosh-is-great-7', 'santosh', 'x@example.com')
        assert 'email' in security.check_password('madannavar-rules-99', 'bob', 'madannavar@gmail.com')

    @pytest.mark.parametrize('pw', ['orange-kettle-42', 'correct horse battery', 'Tr0ub4dor&3-extra', 'ಕನ್ನಡ-ಪಾಸ್ವರ್ಡ್-77', 'हिंदी-पासवर्ड-2026'])
    def test_good_passwords_pass_including_non_english_ones(self, pw):
        assert security.check_password(pw, 'zed', 'zed@example.com') is None

    def test_enforced_at_registration_reset_and_change(self, app, client, mails):
        assert register(client, pw='password123').status_code == 400
        register(client)
        mails.clear()
        r = client.post('/account/password', data={'current_password': PW, 'new_password': 'qwerty123', 'confirm_password': 'qwerty123'})
        assert r.status_code == 400 and 'common' in r.get_data(as_text=True)


class TestLockout:
    def test_five_wrong_passwords_lock_that_account_from_that_address(self, alice, app):
        alice.post('/logout')
        for _ in range(5):
            assert login(alice, pw='wrong-password-1').status_code == 401
        r = login(alice)                                              # even the CORRECT password is refused during the lockout
        assert r.status_code == 429 and 'Too many failed attempts' in r.get_data(as_text=True)

    def test_a_stranger_cannot_lock_the_real_owner_out(self, alice, app):
        alice.post('/logout')
        attacker = app.test_client()
        for _ in range(8):
            login(attacker, pw='guess', environ_overrides={'REMOTE_ADDR': '198.51.100.9'})
        assert login(attacker, environ_overrides={'REMOTE_ADDR': '198.51.100.9'}).status_code == 429       # the attacker's address is locked...
        owner = login(alice, environ_overrides={'REMOTE_ADDR': '203.0.113.5'})                            # ...the owner's is not
        assert owner.status_code == 302

    def test_an_account_attacked_from_many_addresses_is_still_protected(self, alice, app):
        alice.post('/logout')
        for i in range(config.LOGIN_MAX_FAILURES_PER_ACCOUNT):
            login(app.test_client(), pw='guess', environ_overrides={'REMOTE_ADDR': f'192.0.{i // 200}.{i % 200 + 1}'})
        assert login(alice, environ_overrides={'REMOTE_ADDR': '203.0.113.77'}).status_code == 429

    def test_success_clears_the_failure_count(self, alice):
        alice.post('/logout')
        for _ in range(4):
            login(alice, pw='nope-nope-1')
        assert login(alice).status_code == 302
        alice.post('/logout')
        for _ in range(4):
            assert login(alice, pw='nope-nope-1').status_code == 401              # a fresh allowance, not 4 + 4

    def test_unknown_usernames_cost_the_same_work(self, app, client, monkeypatch):
        calls = []
        monkeypatch.setattr('quickaid.core.database.burn_password_check', lambda pw: calls.append(pw))
        assert login(client, name='nobody-here').status_code == 401 and calls == [PW]

    def test_the_message_never_reveals_which_part_was_wrong(self, alice):
        alice.post('/logout')
        a = login(alice, pw='wrong-password-1').get_data(as_text=True)
        b = login(alice, name='nobody-here').get_data(as_text=True)
        assert 'Invalid username or password.' in a and 'Invalid username or password.' in b


class TestSessions:
    def test_login_discards_whatever_the_session_held_before(self, alice):
        alice.post('/logout')
        with alice.session_transaction() as s:
            s['planted_by_attacker'] = 'x'
        login(alice)
        with alice.session_transaction() as s:
            assert 'planted_by_attacker' not in s and s['sv'] == 0 and s['auth_at']

    def test_changing_the_password_signs_out_other_devices_but_not_this_one(self, app, alice):
        other = app.test_client()
        login(other)
        assert other.get('/api/history').status_code == 200
        r = alice.post('/account/password', data={'current_password': PW, 'new_password': 'purple-teapot-77', 'confirm_password': 'purple-teapot-77'})
        assert r.status_code == 200 and 'Other devices were signed out' in r.get_data(as_text=True)
        assert alice.get('/api/history').status_code == 200                       # this device stays in
        assert other.get('/api/history').status_code == 401                       # the other one is out

    def test_sign_out_other_devices(self, app, alice):
        other = app.test_client()
        login(other)
        assert alice.post('/account/sessions/revoke').status_code == 200
        assert other.get('/api/history').status_code == 401 and alice.get('/api/history').status_code == 200

    def test_idle_timeout(self, alice):
        assert alice.get('/api/history').status_code == 200
        with alice.session_transaction() as s:
            s['seen_at'] = time.time() - config.session_idle_minutes() * 60 - 5
        assert alice.get('/api/history').status_code == 401

    def test_absolute_timeout_applies_even_to_an_active_user(self, alice):
        with alice.session_transaction() as s:
            s['auth_at'] = time.time() - config.session_absolute_hours() * 3600 - 5
            s['seen_at'] = time.time()
        assert alice.get('/api/history').status_code == 401

    def test_activity_keeps_a_session_alive_within_the_idle_window(self, alice):
        with alice.session_transaction() as s:
            s['seen_at'] = time.time() - 60 * 30
        assert alice.get('/api/history').status_code == 200
        with alice.session_transaction() as s:
            assert time.time() - s['seen_at'] < 5

    def test_timeouts_are_configurable(self, monkeypatch):
        monkeypatch.setenv('SESSION_IDLE_MINUTES', '15')
        monkeypatch.setenv('SESSION_ABSOLUTE_HOURS', '2')
        assert config.session_idle_minutes() == 15 and config.session_absolute_hours() == 2
        monkeypatch.setenv('SESSION_IDLE_MINUTES', '0')
        assert config.session_idle_minutes() >= 5                               # nobody can accidentally set "never stays signed in"

    def test_cookie_is_httponly_samesite_and_hardened_when_secure(self, monkeypatch):
        monkeypatch.setenv('SESSION_COOKIE_SECURE', 'true')
        from quickaid.app import create_app
        secure_app = create_app()
        c = secure_app.test_client()
        r = c.post('/register', data={'username': 'sec', 'email': 'sec@example.com', 'password': PW, 'confirm_password': PW}, base_url='https://localhost')
        cookie = [h for h in r.headers.getlist('Set-Cookie') if h.startswith('__Host-quickaid=')][0]
        assert 'HttpOnly' in cookie and 'Secure' in cookie and 'SameSite=Lax' in cookie and 'Path=/' in cookie and 'Domain' not in cookie

    def test_logout_ends_the_session(self, alice):
        alice.post('/logout')
        assert alice.get('/api/history').status_code == 401

    def test_security_events_are_recorded_and_shown_to_the_owner(self, alice):
        login(alice, pw='wrong-password-1')
        html = alice.get('/account').get_data(as_text=True)
        assert 'Recent security activity' in html and 'Signed in' in html
        assert '/24' in html and PW not in html                                   # networks are shortened; no secrets

    def test_events_never_store_a_full_address(self, alice):
        from quickaid.core import database as db
        for e in db.list_security_events(1):
            assert not re.search(r'\b\d+\.\d+\.\d+\.\d+\b(?!/)', e['network'] or '') and (not e['network'] or e['network'].endswith(('/24', '/48')))


class TestTwoFactor:
    def test_enabling_shows_ten_single_use_recovery_codes_once(self, alice):
        secret, codes, r = enable_2fa(alice)
        assert len(codes) == 10 and len(set(codes)) == 10 and 'You will not be able to see these again' in r.get_data(as_text=True)
        assert 'RECOVERY' not in alice.get('/account').get_data(as_text=True).upper() or not any(c in alice.get('/account').get_data(as_text=True) for c in codes)

    def test_secret_and_codes_are_not_stored_readable(self, alice):
        from quickaid.core import database as db
        secret, codes, _ = enable_2fa(alice)
        with db.get_connection() as c:
            raw = c.execute('SELECT totp_secret FROM users WHERE id = 1').fetchone()[0]
            hashes = [r[0] for r in c.execute('SELECT code_hash FROM recovery_codes')]
        assert raw.startswith('enc:') and secret not in raw
        assert len(hashes) == 10 and not any(code.replace('-', '') in h or code in h for code in codes for h in hashes)
        assert db.get_user_by_id(1)['totp_secret'] == secret                       # still usable: decrypted on read

    def test_a_recovery_code_signs_you_in_exactly_once(self, app, alice, mails):
        secret, codes, _ = enable_2fa(alice)
        alice.post('/logout')
        login(alice)
        assert alice.post('/login/2fa', data={'code': codes[0]}).status_code == 302
        assert any('recovery code was used' in m['subject'] for m in mails)        # the owner is told
        alice.post('/logout')
        login(alice)
        assert alice.post('/login/2fa', data={'code': codes[0]}).status_code == 401  # same code again: refused
        assert alice.post('/login/2fa', data={'code': codes[1].lower().replace('-', ' ')}).status_code == 302    # formatting is forgiving

    def test_a_totp_code_cannot_be_replayed(self, alice):
        secret, codes, _ = enable_2fa(alice)
        code = pyotp.TOTP(secret).now()
        alice.post('/logout')
        login(alice)
        assert alice.post('/login/2fa', data={'code': code}).status_code == 302
        alice.post('/logout')
        login(alice)
        assert alice.post('/login/2fa', data={'code': code}).status_code == 401          # a shoulder-surfed code is useless a second time

    def test_wrong_codes_lock_the_second_step(self, alice):
        secret, codes, _ = enable_2fa(alice)
        alice.post('/logout')
        login(alice)
        for _ in range(config.TWOFA_MAX_FAILURES):
            assert alice.post('/login/2fa', data={'code': '000000'}).status_code == 401
        assert alice.post('/login/2fa', data={'code': pyotp.TOTP(secret).at(time.time() + 30)}).status_code == 429

    def test_the_second_step_expires(self, alice):
        enable_2fa(alice)
        alice.post('/logout')
        login(alice)
        with alice.session_transaction() as s:
            s['pending_2fa_at'] = time.time() - config.PENDING_2FA_SECONDS - 5
        assert alice.get('/login/2fa').status_code == 302

    def test_regenerating_needs_password_and_a_current_code_and_kills_old_codes(self, alice):
        secret, codes, _ = enable_2fa(alice)
        bad = alice.post('/account/2fa/recovery-codes', data={'current_password': 'wrong-password-1', 'code': pyotp.TOTP(secret).at(time.time() + 30)})
        assert bad.status_code == 400
        ok = alice.post('/account/2fa/recovery-codes', data={'current_password': PW, 'code': pyotp.TOTP(secret).at(time.time() + 30)})
        new = re.findall(r'\b[A-Z2-9]{5}-[A-Z2-9]{5}\b', ok.get_data(as_text=True))
        assert len(new) == 10 and not set(new) & set(codes)
        alice.post('/logout')
        login(alice)
        assert alice.post('/login/2fa', data={'code': codes[0]}).status_code == 401          # old codes are dead
        assert alice.post('/login/2fa', data={'code': new[0]}).status_code == 302

    def test_disabling_removes_the_codes_and_tells_the_owner(self, alice, mails):
        from quickaid.core import database as db
        enable_2fa(alice)
        mails.clear()
        alice.post('/account/2fa/disable', data={'current_password': PW})
        assert db.recovery_codes_remaining(1) == 0 and any('turned off' in m['subject'] for m in mails)

    def test_enabling_2fa_signs_out_other_devices(self, app, alice):
        other = app.test_client()
        login(other)
        enable_2fa(alice)
        assert other.get('/api/history').status_code == 401 and alice.get('/api/history').status_code == 200

    def test_the_account_page_warns_when_codes_run_low(self, alice):
        from quickaid.core import database as db
        enable_2fa(alice)
        with db.get_connection() as c:
            c.execute("UPDATE recovery_codes SET used_at = 'x' WHERE id > 2")
        assert 'only 2 recovery codes left' in alice.get('/account').get_data(as_text=True)


class TestPasswordReset:
    def request_reset(self, client, email='alice@example.com'):
        return client.post('/forgot-password', data={'email': email})

    def link_from(self, mails):
        return re.search(r'/reset-password/[A-Za-z0-9_\-]+', mails[-1]['body']).group(0)

    def test_the_response_is_identical_for_known_and_unknown_addresses(self, alice, mails):
        a = self.request_reset(alice, 'alice@example.com').get_data(as_text=True)
        b = self.request_reset(alice, 'nobody@example.com').get_data(as_text=True)
        strip = lambda t: re.sub(r'nonce="[^"]+"', '', re.sub(r'csrf_token" value="[^"]+"', '', t))   # noqa: E731
        assert strip(a) == strip(b) and len(mails) == 1

    def test_only_a_few_emails_per_address_per_hour(self, alice, mails):
        for _ in range(6):
            self.request_reset(alice)
        assert len(mails) == config.RESET_EMAILS_PER_HOUR

    def test_a_link_works_once_and_a_newer_email_kills_older_links(self, app, alice, mails):
        self.request_reset(alice)
        self.request_reset(alice)
        first, second = [re.search(r'/reset-password/[\w\-]+', m['body']).group(0) for m in mails]
        c = app.test_client()
        assert c.get(first).status_code == 400                                  # only the newest email's link ever works
        assert c.post(second, data={'password': 'purple-teapot-77', 'confirm_password': 'purple-teapot-77'}).status_code == 200
        assert c.get(second).status_code == 400                                 # and only once

    def test_reset_signs_out_every_device_and_tells_the_owner(self, app, alice, mails):
        assert alice.get('/api/history').status_code == 200
        self.request_reset(alice)
        link = self.link_from(mails)
        mails.clear()
        app.test_client().post(link, data={'password': 'purple-teapot-77', 'confirm_password': 'purple-teapot-77'})
        assert alice.get('/api/history').status_code == 401
        assert any('password was changed' in m['subject'] for m in mails)
        assert login(app.test_client(), pw='purple-teapot-77').status_code == 302

    def test_reset_enforces_the_password_policy(self, app, alice, mails):
        self.request_reset(alice)
        r = app.test_client().post(self.link_from(mails), data={'password': 'password123', 'confirm_password': 'password123'})
        assert r.status_code == 400 and 'common' in r.get_data(as_text=True)

    def test_an_expired_or_forged_token_is_refused(self, app, client):
        assert client.get('/reset-password/not-a-real-token').status_code == 400
        assert client.post('/reset-password/' + 'A' * 43, data={'password': PW, 'confirm_password': PW}).status_code == 400

    def test_reset_does_not_remove_two_factor(self, app, alice, mails):
        enable_2fa(alice)
        self.request_reset(alice)
        c = app.test_client()
        c.post(self.link_from(mails), data={'password': 'purple-teapot-77', 'confirm_password': 'purple-teapot-77'})
        r = login(c, pw='purple-teapot-77')
        assert r.status_code == 302 and '/login/2fa' in r.headers['Location']       # a stolen mailbox alone is still not enough


class TestPrivacyInLogs:
    def test_the_capture_really_sees_the_apps_logs(self, app, applog):
        """Control: without this, the tests below could pass forever without checking anything."""
        app.test_client().get('/health')
        assert any('/health' in line for line in applog.lines)

    def test_no_personal_data_reaches_the_logs(self, app, applog, mails):
        c = app.test_client()
        register(c, 'zebulon', 'zebulon.private@example.com', PW)
        c.post('/analyze_symptoms', json={'symptoms': 'my very private shoulder rash'})
        c.get('/api/history/export')
        c.get('/history?q=shoulder')
        login(app.test_client(), 'zebulon', 'wrong-password-9')
        login(app.test_client(), 'typed-my-password-here-by-mistake', 'x')
        c.post('/forgot-password', data={'email': 'zebulon.private@example.com'})
        token = re.search(r'/reset-password/([\w\-]+)', mails[-1]['body']).group(1)
        c.get(f'/reset-password/{token}')
        c.post('/logout')
        text = '\n'.join(applog.lines)
        assert len(applog.lines) > 10                       # we really captured a realistic amount of logging
        for secret in ('zebulon', 'private shoulder', 'shoulder', PW, 'wrong-password-9', 'typed-my-password-here-by-mistake', token):
            assert secret not in text, secret

    def test_the_access_log_records_the_route_not_the_secret_in_it(self, app, applog):
        app.test_client().get('/reset-password/SECRET-TOKEN-VALUE-1234567890')
        text = '\n'.join(applog.lines)
        assert 'SECRET-TOKEN-VALUE' not in text and '/reset-password/<token>' in text

    def test_unmatched_paths_are_not_echoed_either(self, app, applog):
        app.test_client().get('/wp-login.php?password=hunter2hunter2')
        text = '\n'.join(applog.lines)
        assert 'hunter2' not in text and '<unmatched>' in text


class TestLogRedactionFilter:
    """Defence in depth: even if some code DID log personal data, the handler-level filter removes it."""

    def test_redact(self):
        from quickaid.core.logging_config import redact
        out = redact('user jane.doe@example.com from 203.0.113.77 used /reset-password/AbCdEfGhIjKlMnOpQrStUv token=abc123def456 key AIza' + 'SyA' * 11 + 'AB')
        assert 'jane.doe' not in out and '203.0.113.77' not in out and 'AbCdEfGhIj' not in out and 'abc123def456' not in out and 'AIza' not in out
        assert '[email]' in out and '203.0.113.0' in out and '/reset-password/[token]' in out

    def test_the_real_handlers_carry_the_filter(self):
        from quickaid.core.logging_config import PrivacyFilter, get_logger
        get_logger('x')
        handlers = logging.getLogger('quickaid').handlers
        assert handlers and all(any(isinstance(f, PrivacyFilter) for f in h.filters) for h in handlers)

    def test_filter_applies_to_exception_tracebacks_too(self):
        from quickaid.core.logging_config import PrivacyFilter
        try:
            raise ValueError('failed for victim@example.com')
        except ValueError:
            import sys
            rec = logging.LogRecord('quickaid.t', logging.ERROR, __file__, 1, 'boom', (), sys.exc_info())
        PrivacyFilter().filter(rec)
        assert 'victim@' not in (rec.exc_text or '') and '[email]' in rec.exc_text

    def test_json_format_option(self, monkeypatch):
        from quickaid.core.logging_config import JsonFormatter
        import json
        rec = logging.LogRecord('quickaid.t', logging.INFO, __file__, 1, 'hello %s', ('world',), None)
        assert json.loads(JsonFormatter().format(rec))['message'] == 'hello world'


class TestCsrfCoverage:
    def test_every_state_changing_route_requires_a_csrf_token(self, app, client, registered_user):
        app.config['WTF_CSRF_ENABLED'] = True
        checked = 0
        for rule in app.url_map.iter_rules():
            if rule.endpoint == 'static':
                continue
            for method in sorted(rule.methods & {'POST', 'PUT', 'PATCH', 'DELETE'}):
                url = re.sub(r'<any\([^)]*\):[a-z_]+>', 'symptom', rule.rule)
                url = re.sub(r'<int:[a-z_]+>', '1', url)
                url = re.sub(r'<(?:string:)?[a-z_]+>', 'x', url)
                r = client.open(url, method=method)
                assert r.status_code == 400 and ('csrf' in r.get_data(as_text=True).lower() or 'session' in r.get_data(as_text=True).lower()), (method, rule.rule, r.status_code)
                checked += 1
        assert checked >= 30

    def test_no_view_is_exempt_from_csrf(self, app):
        from quickaid.extensions import csrf
        assert not getattr(csrf, '_exempt_views', set()) and not getattr(csrf, '_exempt_blueprints', set())
