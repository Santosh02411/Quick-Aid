"""Accounts: registration, sign-in (password, 2FA, Google), password reset, email verification and account settings."""


import time

from flask import Blueprint, redirect, render_template, request, session, url_for
from flask_login import current_user, login_required, login_user, logout_user

from quickaid.accounts.auth import User
from quickaid.accounts.mailer import send_email
from quickaid.accounts import oauth as oauth_module     # looked up at call time: the module can be reloaded (and is, in tests)
from quickaid.accounts.two_factor import generate_secret, provisioning_uri, qr_code_data_uri, verify_code as verify_totp_code, verify_code_step
from quickaid import config
from quickaid.web import sessions
from quickaid.config import EMAIL_VERIFY_TTL_SECONDS, MIN_PASSWORD_LENGTH, PASSWORD_RESET_TTL_SECONDS, USERNAME_RE
from quickaid.core import database as db
from quickaid.core.localization import REGIONS
from quickaid.core.logging_config import get_logger
from quickaid.extensions import limiter
from quickaid.services import security
from quickaid.services.accounts import build_absolute_url, generate_username_from_email, is_valid_email
from quickaid.services.container import get_services

logger = get_logger('auth')
bp = Blueprint('auth', __name__)


def _send_verification_email(user_row):
    token = db.create_token(user_row['id'], 'email_verify', EMAIL_VERIFY_TTL_SECONDS)
    link = build_absolute_url(url_for('auth.verify_email', token=token))
    send_email(
        user_row['email'],
        'Verify your Quick Aid email address',
        f"Hi {user_row['username']},\n\n"
        f"Please confirm your email address by visiting the link below "
        f"(valid for 24 hours):\n\n{link}\n\n"
        f"If you didn't create a Quick Aid account, you can ignore this email."
    )


def _safe_next_path(next_page):
    """Only follow `next` if it's a safe, local, relative path - never
    redirect off-site based on unvalidated user input."""
    if next_page and next_page.startswith('/') and not next_page.startswith('//'):
        return next_page
    return None


@bp.route('/register', methods=['GET', 'POST'])
@limiter.limit("5 per minute")
def register():
    if current_user.is_authenticated:
        return redirect(url_for('pages.index'))

    if request.method == 'GET':
        return render_template('register.html', google_oauth_enabled=oauth_module.is_google_oauth_configured())

    username = (request.form.get('username') or '').strip()
    email = (request.form.get('email') or '').strip().lower()
    password = request.form.get('password') or ''
    confirm_password = request.form.get('confirm_password') or ''
    ctx = {
        'google_oauth_enabled': oauth_module.is_google_oauth_configured(),
        'form_username': username,
        'form_email': email,
    }

    if not username or not email or not password or not confirm_password:
        return render_template('register.html', error='All fields are required.', **ctx), 400

    if not USERNAME_RE.match(username):
        return render_template(
            'register.html',
            error='Username must be 3-30 characters: letters, numbers, underscores only.', **ctx
        ), 400

    if not is_valid_email(email):
        return render_template('register.html', error='Please enter a valid email address.', **ctx), 400

    problem = security.check_password(password, username, email)
    if problem:
        return render_template('register.html', error=problem, **ctx), 400

    if password != confirm_password:
        return render_template('register.html', error='Passwords do not match.', **ctx), 400

    user_row = db.create_user(username, email, password)
    if user_row is None:
        return render_template('register.html', error='That username or email is already taken.', **ctx), 409

    chosen_region = (request.form.get('region') or '').strip().upper()
    if chosen_region in REGIONS:
        db.set_user_region(user_row['id'], chosen_region)
        user_row['region'] = chosen_region

    _send_verification_email(user_row)
    sessions.start_session(user_row, 'registration')
    logger.info("New user registered and logged in: user_id=%s", user_row['id'])
    return redirect(url_for('pages.index'))


@bp.route('/login', methods=['GET', 'POST'])
@limiter.limit("20 per minute")
def login():
    if current_user.is_authenticated:
        return redirect(url_for('pages.index'))

    google = oauth_module.is_google_oauth_configured()
    if request.method == 'GET':
        return render_template('login.html', google_oauth_enabled=google)

    username = (request.form.get('username') or '').strip()
    password = request.form.get('password') or ''
    net = sessions.client_id()

    if security.login_locked(username, net):
        logger.info("Sign-in blocked by lockout (account tag %s)", security.fingerprint(username))
        return render_template('login.html', error='Too many failed attempts. Please wait about 15 minutes and try again, or reset your password.', google_oauth_enabled=google), 429

    user_row = db.get_user_by_username(username)
    if user_row is None:
        db.burn_password_check(password)             # same work whether or not the username exists
    if user_row and db.verify_password(user_row, password):
        next_page = _safe_next_path(request.args.get('next'))
        security.note_login_success(username, net)

        if user_row['totp_enabled']:
            # Password is correct but a second factor is required - don't log the session in yet.
            session.clear()
            session['pending_2fa_user_id'] = user_row['id']
            session['pending_2fa_next'] = next_page
            session['pending_2fa_at'] = time.time()
            return redirect(url_for('auth.login_2fa'))

        sessions.start_session(user_row, 'password')
        return redirect(next_page or url_for('pages.index'))

    security.note_login_failure(username, net)
    if user_row:
        sessions.event(user_row['id'], 'login_failed')
    # Deliberately NOT logging the username that was typed: people sometimes type their password into that box.
    logger.info("Failed sign-in attempt (account tag %s)", security.fingerprint(username))
    return render_template('login.html', error='Invalid username or password.', google_oauth_enabled=google), 401


@bp.route('/login/2fa', methods=['GET', 'POST'])
@limiter.limit("10 per minute")
def login_2fa():
    """Second step of login for accounts with 2FA enabled - reached only after a correct username/password.
    Accepts the 6-digit authenticator code, or one of the single-use recovery codes."""
    pending_user_id = session.get('pending_2fa_user_id')
    if not pending_user_id or time.time() - session.get('pending_2fa_at', 0) > config.PENDING_2FA_SECONDS:
        session.pop('pending_2fa_user_id', None)
        return redirect(url_for('auth.login'))

    if request.method == 'GET':
        return render_template('login_2fa.html')

    net = sessions.client_id()
    if security.twofa_locked(pending_user_id, net):
        return render_template('login_2fa.html', error="Too many incorrect codes. Please wait about 15 minutes and sign in again."), 429

    code = (request.form.get('code') or '').strip()
    user_row = db.get_user_by_id(pending_user_id)
    method = None
    if user_row and user_row['totp_enabled']:
        step = verify_code_step(user_row['totp_secret'], code)
        if step is not None:
            if db.claim_totp_step(user_row['id'], step):      # a code that was already used cannot be replayed
                method = '2fa'
        elif db.consume_recovery_code(user_row['id'], code):
            method = 'recovery code'

    if method:
        next_page = session.get('pending_2fa_next')
        db.clear_attempts('2fa', security.fingerprint(f"{user_row['id']}|{net}"))
        sessions.start_session(db.get_user_by_id(user_row['id']), method)
        if method == 'recovery code':
            left = db.recovery_codes_remaining(user_row['id'])
            sessions.event(user_row['id'], 'recovery_code_used', f'{left} left')
            security.notify(user_row, 'A Quick Aid recovery code was used',
                            f'A one-time recovery code was just used to sign in to your account. You have {left} left.')
            if left <= 3:
                return redirect(url_for('auth.account') + '#security')
        return redirect(next_page or url_for('pages.index'))

    security.note_twofa_failure(pending_user_id, net)
    if user_row:
        sessions.event(user_row['id'], 'twofa_failed')
    return render_template('login_2fa.html', error='Invalid or expired code. Please try again.'), 401


@bp.route('/logout', methods=['POST'])
@login_required
def logout():
    sessions.event(int(current_user.id), 'logout')
    sessions.end_session()
    return redirect(url_for('pages.index'))


# ---------------------------------------------------------------------------
# Password reset
# ---------------------------------------------------------------------------

@bp.route('/forgot-password', methods=['GET', 'POST'])
@limiter.limit("5 per minute")
def forgot_password():
    if request.method == 'GET':
        return render_template('forgot_password.html')

    email = (request.form.get('email') or '').strip().lower()
    if is_valid_email(email):
        user_row = db.get_user_by_email(email)
        # Same response and the same work whether or not the address exists; a per-address cap stops us being used to flood an inbox.
        if user_row and security.reset_email_allowed(email):
            token = db.create_token(user_row['id'], 'password_reset', PASSWORD_RESET_TTL_SECONDS)
            link = build_absolute_url(url_for('auth.reset_password', token=token))
            sessions.event(user_row['id'], 'password_reset_requested')
            send_email(
                user_row['email'],
                'Reset your Quick Aid password',
                f"Hi {user_row['username']},\n\n"
                f"Someone requested a password reset for your Quick Aid account. "
                f"If this was you, click the link below (valid for 1 hour, works once):\n\n{link}\n\n"
                f"If you didn't request this, you can safely ignore this email - "
                f"your password won't change."
            )
    # Always show the same message whether or not the email exists, so this
    # endpoint can't be used to enumerate registered accounts.
    return render_template(
        'forgot_password.html',
        message="If an account exists for that email, we've sent a password reset link."
    )


@bp.route('/reset-password/<token>', methods=['GET', 'POST'])
@limiter.limit("10 per minute")
def reset_password(token):
    token_row = db.get_valid_token(token, 'password_reset')
    if not token_row:
        return render_template('reset_password.html', invalid=True), 400

    if request.method == 'GET':
        return render_template('reset_password.html', token=token)

    password = request.form.get('password') or ''
    confirm = request.form.get('confirm_password') or ''
    owner = db.get_user_by_id(token_row['user_id'])

    problem = security.check_password(password, owner['username'], owner['email'])
    if problem:
        return render_template('reset_password.html', token=token, error=problem), 400
    if password != confirm:
        return render_template('reset_password.html', token=token, error='Passwords do not match.'), 400

    # Re-check validity right before committing (against a token expiring or being used elsewhere between GET and POST).
    token_row = db.get_valid_token(token, 'password_reset')
    if not token_row:
        return render_template('reset_password.html', invalid=True), 400

    db.set_password(token_row['user_id'], password)           # also signs out every existing session
    db.consume_token(token_row['id'])
    db.invalidate_user_tokens(token_row['user_id'], 'password_reset')     # any other reset email still in the inbox stops working
    sessions.event(token_row['user_id'], 'password_reset_completed')
    security.notify(owner, 'Your Quick Aid password was changed', 'Your password was just reset using a reset link. All devices were signed out.')
    logger.info("Password reset completed for user_id=%s", token_row['user_id'])
    return render_template('reset_password.html', done=True)


@bp.route('/verify-email/<token>')
@limiter.limit("20 per minute")
def verify_email(token):
    token_row = db.get_valid_token(token, 'email_verify')
    if not token_row:
        return render_template('verify_email.html', invalid=True), 400

    db.set_email_verified(token_row['user_id'])
    db.consume_token(token_row['id'])
    logger.info("Email verified for user_id=%s", token_row['user_id'])
    return render_template('verify_email.html', done=True)


# ---------------------------------------------------------------------------
# Google OAuth / SSO
# ---------------------------------------------------------------------------

@bp.route('/login/google')
@limiter.limit("15 per minute")
def login_google():
    if not oauth_module.is_google_oauth_configured():
        return render_template('login.html', error='Google sign-in is not configured on this server.'), 503
    redirect_uri = build_absolute_url(url_for('auth.login_google_callback'))
    return oauth_module.oauth.google.authorize_redirect(redirect_uri)


@bp.route('/login/google/callback')
@limiter.limit("15 per minute")
def login_google_callback():
    if not oauth_module.is_google_oauth_configured():
        return redirect(url_for('auth.login'))

    try:
        token = oauth_module.oauth.google.authorize_access_token()
    except Exception:
        logger.error("Google OAuth callback failed", exc_info=True)
        return render_template('login.html', error='Google sign-in failed. Please try again.'), 400

    userinfo = token.get('userinfo')
    if not userinfo:
        logger.error("Google OAuth callback returned no userinfo/id_token")
        return render_template('login.html', error='Google sign-in failed. Please try again.'), 400

    sub = userinfo.get('sub')
    email = (userinfo.get('email') or '').strip().lower()
    email_verified = bool(userinfo.get('email_verified'))

    if not sub or not email or not email_verified:
        return render_template(
            'login.html',
            error='Your Google account must have a verified email to sign in this way.'
        ), 400

    user_row = db.get_user_by_oauth('google', sub)

    if not user_row:
        # First time seeing this Google account - link it to an existing
        # local account with the same (verified) email, or create a new one.
        existing = db.get_user_by_email(email)
        if existing:
            if not existing['email_verified']:
                # Never linked to this person's mail before: whoever registered it first may be an impostor. Destroy that password.
                db.neutralise_unverified_account(existing['id'])
                sessions.event(existing['id'], 'unverified_account_claimed_via_google')
            db.link_oauth_to_user(existing['id'], 'google', sub)
            user_row = db.get_user_by_id(existing['id'])
        else:
            username = generate_username_from_email(email)
            user_row = db.create_oauth_user(username, email, 'google', sub)
            if user_row is None:
                return render_template('login.html', error='Could not create your account. Please try again.'), 409

    sessions.start_session(db.get_user_by_id(user_row['id']), 'google')
    return redirect(url_for('pages.index'))


# ---------------------------------------------------------------------------
# Account settings
# ---------------------------------------------------------------------------

@bp.route('/account')
@login_required
def account():
    user_row = db.get_user_by_id(int(current_user.id))
    return render_template('account.html', account_user=user_row)


@bp.route('/account/profile', methods=['POST'])
@login_required
@limiter.limit("10 per minute")
def update_profile():
    user_id = int(current_user.id)
    user_row = db.get_user_by_id(user_id)

    username = (request.form.get('username') or '').strip()
    email = (request.form.get('email') or '').strip().lower()
    current_password = request.form.get('current_password') or ''

    if not USERNAME_RE.match(username):
        return _account_error('Username must be 3-30 characters: letters, numbers, underscores only.')
    if not is_valid_email(email):
        return _account_error('Please enter a valid email address.')

    # Accounts that have a real password must confirm it before changing
    # profile details - OAuth-only accounts have none to confirm.
    if user_row['has_password'] and not db.verify_password(user_row, current_password):
        return _account_error('Current password is incorrect.')

    if username != user_row['username'] and db.username_taken(username, exclude_user_id=user_id):
        return _account_error('That username is already taken.')
    if email != user_row['email'] and db.email_taken(email, exclude_user_id=user_id):
        return _account_error('That email is already in use.')

    if username != user_row['username']:
        db.update_username(user_id, username)
    email_changed = email != user_row['email']
    if email_changed:
        db.update_email(user_id, email)

    updated_row = db.get_user_by_id(user_id)
    if email_changed:
        _send_verification_email(updated_row)
        sessions.event(user_id, 'email_changed')
        security.notify(user_row, 'The email on your Quick Aid account was changed', 'The email address on your account was just changed.')

    logger.info("Profile updated for user_id=%s", user_id)
    return render_template('account.html', account_user=updated_row, success='Profile updated.')


@bp.route('/account/password', methods=['POST'])
@login_required
@limiter.limit("5 per minute")
def update_password():
    user_id = int(current_user.id)
    user_row = db.get_user_by_id(user_id)

    current_password = request.form.get('current_password') or ''
    new_password = request.form.get('new_password') or ''
    confirm_password = request.form.get('confirm_password') or ''

    if user_row['has_password'] and not db.verify_password(user_row, current_password):
        return _account_error('Current password is incorrect.')
    problem = security.check_password(new_password, user_row['username'], user_row['email'])
    if problem:
        return _account_error(problem.replace('Password must', 'New password must'))
    if new_password != confirm_password:
        return _account_error('New passwords do not match.')

    db.set_password(user_id, new_password)                    # signs out every OTHER device
    sessions.refresh_after_credential_change(user_id)         # ...but keeps this one signed in
    sessions.event(user_id, 'password_changed')
    security.notify(user_row, 'Your Quick Aid password was changed', 'Your password was just changed. All other devices were signed out.')
    return render_template('account.html', account_user=db.get_user_by_id(user_id), success='Password updated. Other devices were signed out.')


@bp.route('/account/resend-verification', methods=['POST'])
@login_required
@limiter.limit("3 per minute")
def resend_verification():
    user_row = db.get_user_by_id(int(current_user.id))
    if user_row['email_verified']:
        return render_template('account.html', account_user=user_row, success='Your email is already verified.')
    _send_verification_email(user_row)
    logger.info("Verification email resent for user_id=%s", user_row['id'])
    return render_template('account.html', account_user=user_row, success='Verification email sent.')


@bp.route('/account/delete', methods=['POST'])
@login_required
@limiter.limit("5 per minute")
def delete_account():
    user_id = int(current_user.id)
    user_row = db.get_user_by_id(user_id)
    current_password = request.form.get('current_password') or ''
    confirm_text = (request.form.get('confirm_text') or '').strip()

    if user_row['has_password'] and not db.verify_password(user_row, current_password):
        return _account_error('Current password is incorrect.')
    if confirm_text.upper() != 'DELETE':
        return _account_error('Type DELETE to confirm account deletion.')

    logout_user()
    session.clear()
    db.delete_user(user_id)
    logger.info("Account deleted: user_id=%s", user_id)
    return redirect(url_for('pages.index'))


@bp.route('/account/2fa/setup')
@login_required
def setup_2fa():
    user_row = db.get_user_by_id(int(current_user.id))
    if user_row['totp_enabled']:
        return redirect(url_for('auth.account'))

    secret = generate_secret()
    session['pending_totp_secret'] = secret
    uri = provisioning_uri(secret, user_row['email'])
    qr_data_uri = qr_code_data_uri(uri)
    return render_template('setup_2fa.html', secret=secret, qr_data_uri=qr_data_uri)


@bp.route('/account/2fa/confirm', methods=['POST'])
@login_required
@limiter.limit("10 per minute")
def confirm_2fa():
    secret = session.get('pending_totp_secret')
    code = (request.form.get('code') or '').strip()

    if not secret:
        return redirect(url_for('auth.setup_2fa'))

    if not verify_totp_code(secret, code):
        user_row = db.get_user_by_id(int(current_user.id))
        uri = provisioning_uri(secret, user_row['email'])
        return render_template(
            'setup_2fa.html', secret=secret, qr_data_uri=qr_code_data_uri(uri),
            error='Incorrect code. Please try again.'
        ), 400

    user_id = int(current_user.id)
    db.set_pending_totp_secret(user_id, secret)
    db.enable_totp(user_id)                                   # signs out other devices
    session.pop('pending_totp_secret', None)
    sessions.refresh_after_credential_change(user_id)
    codes = db.create_recovery_codes(user_id)                 # shown ONCE, stored only as hashes
    sessions.event(user_id, '2fa_enabled')
    security.notify(db.get_user_by_id(user_id), 'Two-factor authentication was turned on', 'Two-factor authentication was turned on for your account.')
    return render_template('recovery_codes.html', codes=codes, success='Two-factor authentication is now enabled.')


@bp.route('/account/2fa/disable', methods=['POST'])
@login_required
@limiter.limit("5 per minute")
def disable_2fa():
    user_id = int(current_user.id)
    user_row = db.get_user_by_id(user_id)
    current_password = request.form.get('current_password') or ''

    if user_row['has_password'] and not db.verify_password(user_row, current_password):
        return _account_error('Current password is incorrect.')

    db.disable_totp(user_id)                                  # also deletes the recovery codes and signs out other devices
    sessions.refresh_after_credential_change(user_id)
    sessions.event(user_id, '2fa_disabled')
    security.notify(user_row, 'Two-factor authentication was turned off', 'Two-factor authentication was turned off for your account.')
    return render_template(
        'account.html', account_user=db.get_user_by_id(user_id),
        success='Two-factor authentication has been disabled.'
    )


@bp.route('/account/2fa/recovery-codes', methods=['POST'])
@login_required
@limiter.limit("5 per minute")
def regenerate_recovery_codes():
    """Replace the recovery codes with 10 new ones. Needs the password AND a current authenticator code, so a hijacked session cannot do it."""
    user_id = int(current_user.id)
    user_row = db.get_user_by_id(user_id)
    if not user_row['totp_enabled']:
        return _account_error('Turn on two-factor authentication first.')
    if user_row['has_password'] and not db.verify_password(user_row, request.form.get('current_password') or ''):
        return _account_error('Current password is incorrect.')
    step = verify_code_step(user_row['totp_secret'], (request.form.get('code') or '').strip())
    if step is None or not db.claim_totp_step(user_id, step):
        return _account_error('That authenticator code is not valid. Wait for the next one and try again.')
    codes = db.create_recovery_codes(user_id)
    sessions.event(user_id, 'recovery_codes_regenerated')
    security.notify(user_row, 'New Quick Aid recovery codes were created', 'New recovery codes were generated. The old ones no longer work.')
    return render_template('recovery_codes.html', codes=codes, success='New recovery codes created. The old ones no longer work.')


@bp.route('/account/sessions/revoke', methods=['POST'])
@login_required
@limiter.limit("5 per minute")
def revoke_other_sessions():
    """Sign out every other device (and keep this one)."""
    user_id = int(current_user.id)
    db.bump_session_version(user_id)
    sessions.refresh_after_credential_change(user_id)
    sessions.event(user_id, 'signed_out_everywhere')
    return render_template('account.html', account_user=db.get_user_by_id(user_id), success='All other devices have been signed out.')


@bp.context_processor
def _privacy_context():
    """What the account page's 'Your data' panel shows: the retention choice and how much is saved. Available to every view in this blueprint."""
    from quickaid.core import history_repo
    if not current_user.is_authenticated:
        return {}
    uid = int(current_user.id)
    user = db.get_user_by_id(uid)
    return {'retention': get_services().retention.user_setting(uid), 'saved_counts': history_repo.count_entries(uid),
            'security_events': db.list_security_events(uid, 12), 'recovery_remaining': db.recovery_codes_remaining(uid) if user['totp_enabled'] else None,
            'ai_consent_visible': bool(get_services().symptom_checker.use_gemini and config.require_ai_consent()),
            'ai_consent_granted': user.get('ai_consent_version') == config.AI_CONSENT_VERSION,
            'session_idle_minutes': config.session_idle_minutes(), 'session_absolute_hours': config.session_absolute_hours()}


def _account_error(message):
    user_row = db.get_user_by_id(int(current_user.id))
    return render_template('account.html', account_user=user_row, error=message), 400


# ---------------------------------------------------------------------------
# Protected routes (require login)
# ---------------------------------------------------------------------------

