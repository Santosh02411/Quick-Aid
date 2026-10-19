"""Accounts: registration, sign-in (password, 2FA, Google), password reset, email verification and account settings."""


from flask import Blueprint, redirect, render_template, request, session, url_for
from flask_login import current_user, login_required, login_user, logout_user

from quickaid.accounts.auth import User
from quickaid.accounts.mailer import send_email
from quickaid.accounts import oauth as oauth_module     # looked up at call time: the module can be reloaded (and is, in tests)
from quickaid.accounts.two_factor import generate_secret, provisioning_uri, qr_code_data_uri, verify_code as verify_totp_code
from quickaid.config import EMAIL_VERIFY_TTL_SECONDS, MIN_PASSWORD_LENGTH, PASSWORD_RESET_TTL_SECONDS, USERNAME_RE
from quickaid.core import database as db
from quickaid.core.localization import REGIONS
from quickaid.core.logging_config import get_logger
from quickaid.extensions import limiter
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

    if len(password) < MIN_PASSWORD_LENGTH:
        return render_template(
            'register.html',
            error=f'Password must be at least {MIN_PASSWORD_LENGTH} characters.', **ctx
        ), 400

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

    user = User.from_row(user_row)
    login_user(user)
    logger.info("New user registered and logged in: %s", username)
    return redirect(url_for('pages.index'))


@bp.route('/login', methods=['GET', 'POST'])
@limiter.limit("10 per minute")
def login():
    if current_user.is_authenticated:
        return redirect(url_for('pages.index'))

    if request.method == 'GET':
        return render_template('login.html', google_oauth_enabled=oauth_module.is_google_oauth_configured())

    username = (request.form.get('username') or '').strip()
    password = request.form.get('password') or ''

    user_row = db.get_user_by_username(username)
    if user_row and db.verify_password(user_row, password):
        next_page = _safe_next_path(request.args.get('next'))

        if user_row['totp_enabled']:
            # Password is correct but a second factor is required - don't
            # log the session in yet, hand off to the 2FA challenge instead.
            session['pending_2fa_user_id'] = user_row['id']
            session['pending_2fa_next'] = next_page
            logger.info("Password verified for %s, awaiting 2FA code", username)
            return redirect(url_for('auth.login_2fa'))

        user = User.from_row(user_row)
        login_user(user)
        logger.info("User logged in: %s", username)
        return redirect(next_page or url_for('pages.index'))

    logger.info("Failed login attempt for username: %s", username)
    return render_template(
        'login.html', error='Invalid username or password.',
        google_oauth_enabled=oauth_module.is_google_oauth_configured()
    ), 401


@bp.route('/login/2fa', methods=['GET', 'POST'])
@limiter.limit("10 per minute")
def login_2fa():
    """Second step of login for accounts with 2FA enabled - reached only
    after a correct username/password on the /login form."""
    pending_user_id = session.get('pending_2fa_user_id')
    if not pending_user_id:
        return redirect(url_for('auth.login'))

    if request.method == 'GET':
        return render_template('login_2fa.html')

    code = (request.form.get('code') or '').strip()
    user_row = db.get_user_by_id(pending_user_id)

    if user_row and user_row['totp_enabled'] and verify_totp_code(user_row['totp_secret'], code):
        session.pop('pending_2fa_user_id', None)
        next_page = session.pop('pending_2fa_next', None)
        user = User.from_row(user_row)
        login_user(user)
        logger.info("2FA code accepted, user logged in: %s", user_row['username'])
        return redirect(next_page or url_for('pages.index'))

    logger.info("Invalid 2FA code for pending user_id=%s", pending_user_id)
    return render_template('login_2fa.html', error='Invalid or expired code. Please try again.'), 401


@bp.route('/logout', methods=['POST'])
@login_required
def logout():
    logger.info("User logged out: %s", current_user.username)
    logout_user()
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
        if user_row:
            token = db.create_token(user_row['id'], 'password_reset', PASSWORD_RESET_TTL_SECONDS)
            link = build_absolute_url(url_for('auth.reset_password', token=token))
            send_email(
                user_row['email'],
                'Reset your Quick Aid password',
                f"Hi {user_row['username']},\n\n"
                f"Someone requested a password reset for your Quick Aid account. "
                f"If this was you, click the link below (valid for 1 hour):\n\n{link}\n\n"
                f"If you didn't request this, you can safely ignore this email - "
                f"your password won't change."
            )
        else:
            logger.info("Password reset requested for unknown email")
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

    if len(password) < MIN_PASSWORD_LENGTH:
        return render_template(
            'reset_password.html', token=token,
            error=f'Password must be at least {MIN_PASSWORD_LENGTH} characters.'
        ), 400
    if password != confirm:
        return render_template('reset_password.html', token=token, error='Passwords do not match.'), 400

    # Re-check validity right before committing (belt-and-suspenders against
    # a token expiring or being used elsewhere between GET and this POST).
    token_row = db.get_valid_token(token, 'password_reset')
    if not token_row:
        return render_template('reset_password.html', invalid=True), 400

    db.set_password(token_row['user_id'], password)
    db.consume_token(token_row['id'])
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
            db.link_oauth_to_user(existing['id'], 'google', sub)
            user_row = db.get_user_by_id(existing['id'])
        else:
            username = generate_username_from_email(email)
            user_row = db.create_oauth_user(username, email, 'google', sub)
            if user_row is None:
                return render_template('login.html', error='Could not create your account. Please try again.'), 409

    user = User.from_row(user_row)
    login_user(user)
    logger.info("User logged in via Google OAuth: %s", user_row['username'])
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
    if len(new_password) < MIN_PASSWORD_LENGTH:
        return _account_error(f'New password must be at least {MIN_PASSWORD_LENGTH} characters.')
    if new_password != confirm_password:
        return _account_error('New passwords do not match.')

    db.set_password(user_id, new_password)
    logger.info("Password changed for user_id=%s", user_id)
    return render_template(
        'account.html', account_user=db.get_user_by_id(user_id),
        success='Password updated.'
    )


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

    username = user_row['username']
    logout_user()
    session.clear()
    db.delete_user(user_id)
    logger.info("Account deleted: user_id=%s username=%s", user_id, username)
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
    db.enable_totp(user_id)
    session.pop('pending_totp_secret', None)
    logger.info("2FA enabled for user_id=%s", user_id)
    return render_template(
        'account.html', account_user=db.get_user_by_id(user_id),
        success='Two-factor authentication is now enabled.'
    )


@bp.route('/account/2fa/disable', methods=['POST'])
@login_required
@limiter.limit("5 per minute")
def disable_2fa():
    user_id = int(current_user.id)
    user_row = db.get_user_by_id(user_id)
    current_password = request.form.get('current_password') or ''

    if user_row['has_password'] and not db.verify_password(user_row, current_password):
        return _account_error('Current password is incorrect.')

    db.disable_totp(user_id)
    logger.info("2FA disabled for user_id=%s", user_id)
    return render_template(
        'account.html', account_user=db.get_user_by_id(user_id),
        success='Two-factor authentication has been disabled.'
    )


@bp.context_processor
def _privacy_context():
    """What the account page's 'Your data' panel shows: the retention choice and how much is saved. Available to every view in this blueprint."""
    from quickaid.core import history_repo
    if not current_user.is_authenticated:
        return {}
    uid = int(current_user.id)
    return {'retention': get_services().retention.user_setting(uid), 'saved_counts': history_repo.count_entries(uid)}


def _account_error(message):
    user_row = db.get_user_by_id(int(current_user.id))
    return render_template('account.html', account_user=user_row, error=message), 400


# ---------------------------------------------------------------------------
# Protected routes (require login)
# ---------------------------------------------------------------------------

