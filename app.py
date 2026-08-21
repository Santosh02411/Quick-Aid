from flask import Flask, render_template, request, jsonify, redirect, url_for, g, session
from flask_login import (
    LoginManager, login_user, logout_user, login_required, current_user
)
from flask_wtf import CSRFProtect
from flask_wtf.csrf import CSRFError
from werkzeug.exceptions import HTTPException
from werkzeug.middleware.proxy_fix import ProxyFix
import os
import secrets
import time
import numpy as np
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from medical_analyzer import MedicalAnalyzer
from symptom_checker import SymptomChecker
from dotenv import load_dotenv
import database as db
from auth import login_manager, User
from logging_config import get_logger
from mailer import send_email
from two_factor import generate_secret, provisioning_uri, verify_code as verify_totp_code, qr_code_data_uri
from oauth import init_oauth, oauth, is_google_oauth_configured
import json
import re

load_dotenv()

logger = get_logger('app')

app = Flask(__name__)

# SECRET_KEY must come from the environment. If it's missing we generate a
# random one for this process only (sessions won't survive a restart) and
# warn loudly instead of silently using a predictable default.
_secret_key = os.getenv('SECRET_KEY')
if not _secret_key or _secret_key == 'change_this_to_a_random_secret_key':
    _secret_key = secrets.token_hex(32)
    logger.warning(
        "SECRET_KEY not set in .env - using a temporary random key for this run. "
        "Set SECRET_KEY in .env for stable sessions across restarts."
    )
app.config['SECRET_KEY'] = _secret_key

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# ---------------------------------------------------------------------------
# Reverse proxy / TLS trust
#
# In production this app sits behind nginx (see docker-compose.yml), which
# terminates TLS and forwards plain HTTP with X-Forwarded-* headers set.
# ProxyFix is what makes Flask/Werkzeug trust exactly one hop of those
# headers to recover the real client IP (request.remote_addr - used by
# flask-limiter's rate limiting) and the original scheme (request.is_secure
# - used below for secure cookies, and by url_for(..., _external=True) for
# building https:// links in emails).
#
# This is opt-in via BEHIND_PROXY rather than always-on: trusting
# X-Forwarded-* from just anyone lets a client spoof its own IP/scheme,
# which would let it dodge rate limits or fake HTTPS. Only enable this when
# something you control (nginx, a cloud load balancer) is actually the only
# thing that can reach this process - which is exactly the case in the
# provided docker-compose.yml (gunicorn isn't published to the host there).
if os.getenv('BEHIND_PROXY', 'False').lower() in ('1', 'true', 'yes'):
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)
    logger.info("BEHIND_PROXY enabled - trusting one hop of X-Forwarded-* headers.")

# Cookies: HttpOnly (already Flask's default) so JS can't read the session
# cookie, SameSite=Lax as a baseline CSRF/cross-site defense on top of the
# CSRF tokens above, and Secure so the cookie is never sent over plain
# HTTP. Secure defaults on since production should always be behind TLS
# (see docker-compose.yml) - set SESSION_COOKIE_SECURE=False only for
# plain-HTTP local dev (e.g. running `python app.py` directly, no nginx).
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = os.getenv('SESSION_COOKIE_SECURE', 'False').lower() in ('1', 'true', 'yes')
if not app.config['SESSION_COOKIE_SECURE']:
    logger.info(
        "SESSION_COOKIE_SECURE is off - fine for local plain-HTTP dev, but any "
        "real deployment should set it True (docker-compose.yml already does)."
    )

# CSRF protection: every state-changing (POST) request must carry a valid
# token, whether it comes from an HTML <form> (token as a hidden field) or
# JS fetch() (token in the X-CSRFToken header, read from the <meta> tag
# each template sets). Tests disable this via WTF_CSRF_ENABLED=False.
app.config.setdefault('WTF_CSRF_ENABLED', True)
csrf = CSRFProtect(app)


# Endpoints hit by JS fetch() rather than a plain <form> submit - these
# should get a JSON error body back instead of an HTML error page, since
# the calling JS is expecting JSON either way.
_JSON_CSRF_ENDPOINTS = {'/upload', '/analyze_symptoms', '/api/history/clear', '/api/history'}


@app.errorhandler(CSRFError)
def handle_csrf_error(e):
    logger.info("CSRF validation failed on %s %s: %s", request.method, request.path, e.description)
    if request.path in _JSON_CSRF_ENDPOINTS or request.path.startswith('/api/'):
        return jsonify({'error': 'Your session has expired or is invalid. Please refresh the page and try again.'}), 400
    return render_template('csrf_error.html', reason=e.description), 400


# Flask-Login: authentication is required for the actual AI-analysis features
# (upload / symptoms / history) since this is meant to be more than a public
# demo. The landing page and emergency info stay public - safety information
# should never be gated behind a login wall.
login_manager.init_app(app)

# Optional Google OAuth/SSO - only registers if GOOGLE_CLIENT_ID/SECRET are
# set (see oauth.py); templates hide the "Continue with Google" button when
# it isn't, so this is safe to always call.
init_oauth(app)

# Rate limiting: protects the Gemini-backed endpoints from abuse/quota burn,
# and login/register from brute-force/enumeration attempts.
# Storage backend is configurable via RATELIMIT_STORAGE_URI. Defaults to
# in-memory, which is fine for a single dev process but does NOT share
# limit state across multiple gunicorn workers/instances - each process
# gets its own counters, so real limits end up (workers x configured limit).
# For any multi-worker or multi-instance deployment, point this at Redis:
#   RATELIMIT_STORAGE_URI=redis://localhost:6379
# (requires the `redis` package - see requirements.txt)
_ratelimit_storage_uri = os.getenv('RATELIMIT_STORAGE_URI', 'memory://')
if _ratelimit_storage_uri == 'memory://':
    logger.warning(
        "Rate limiter using in-memory storage - limits are per-process only. "
        "Set RATELIMIT_STORAGE_URI (e.g. to a Redis URL) before running with "
        "more than one worker/instance."
    )
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri=_ratelimit_storage_uri
)

# Symptom text constraints
MAX_SYMPTOM_LENGTH = 1000
MIN_SYMPTOM_LENGTH = 3

# Account constraints
USERNAME_RE = re.compile(r'^[A-Za-z0-9_]{3,30}$')
MIN_PASSWORD_LENGTH = 8

# Token lifetimes
PASSWORD_RESET_TTL_SECONDS = 60 * 60          # 1 hour
EMAIL_VERIFY_TTL_SECONDS = 60 * 60 * 24       # 24 hours

# Base URL used to build links in emails (password reset / verification).
# Set APP_BASE_URL in .env for production (e.g. https://quickaid.example.com);
# falls back to the incoming request's own host, which is fine for local dev.
APP_BASE_URL = os.getenv('APP_BASE_URL', '').rstrip('/')

# Signature (magic-byte) checks for the image formats we claim to support.
# Verified against the actual file bytes, not just the filename extension.
IMAGE_SIGNATURES = {
    'png': [b'\x89PNG\r\n\x1a\n'],
    'jpg': [b'\xff\xd8\xff'],
    'jpeg': [b'\xff\xd8\xff'],
    'gif': [b'GIF87a', b'GIF89a'],
    'bmp': [b'BM'],
}

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize medical analyzer and symptom checker
medical_analyzer = MedicalAnalyzer()
symptom_checker = SymptomChecker()

# Initialize database (SQLite file, created on first run)
db.init_db()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}


# ---------------------------------------------------------------------------
# Request logging: every request gets a start time and a completion log line
# with method, path, status, and duration - enough to spot failures and slow
# endpoints in production without needing a separate APM tool.
# ---------------------------------------------------------------------------

@app.before_request
def _start_timer():
    request._start_time = time.monotonic()


@app.after_request
def _log_request(response):
    duration_ms = (time.monotonic() - getattr(request, '_start_time', time.monotonic())) * 1000
    logger.info(
        "%s %s -> %s (%.1fms)",
        request.method, request.path, response.status_code, duration_ms
    )
    return response


# ---------------------------------------------------------------------------
# Security response headers
#
# A per-request CSP nonce is generated once and exposed to templates as
# csp_nonce() so the (few) legitimate inline <script> blocks can be
# allow-listed individually instead of falling back to 'unsafe-inline'
# for scripts. Inline <style> is still allowed via 'unsafe-inline' - the
# templates use plenty of inline style="" attributes and CSP has no
# nonce mechanism for style *attributes* (only <style> elements), so
# locking that down would mean moving every inline style into a
# stylesheet. Given styling can't execute arbitrary JS, that's a much
# lower-value fix than a strict script-src and is left as a follow-up.
# ---------------------------------------------------------------------------

@app.before_request
def _set_csp_nonce():
    g.csp_nonce = secrets.token_urlsafe(16)


app.jinja_env.globals['csp_nonce'] = lambda: g.csp_nonce

_CSP_DIRECTIVES = (
    "default-src 'self'; "
    "script-src 'self' https://unpkg.com 'nonce-{nonce}'; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com "
    "https://cdnjs.cloudflare.com https://unpkg.com; "
    "font-src 'self' https://fonts.gstatic.com https://cdnjs.cloudflare.com; "
    "img-src 'self' data:; "
    "connect-src 'self'; "
    "object-src 'none'; "
    "base-uri 'self'; "
    "form-action 'self'; "
    "frame-ancestors 'none'"
)


@app.after_request
def _set_security_headers(response):
    response.headers['Content-Security-Policy'] = _CSP_DIRECTIVES.format(nonce=g.get('csp_nonce', ''))
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Permissions-Policy'] = (
        'camera=(), microphone=(), geolocation=(), payment=(), usb=()'
    )
    # Browsers only honor HSTS on actual HTTPS responses, so it's safe to
    # always set it - it's simply ignored over plain HTTP (e.g. local dev).
    response.headers['Strict-Transport-Security'] = 'max-age=63072000; includeSubDomains'
    return response


@app.errorhandler(Exception)
def handle_unexpected_error(e):
    # Let Flask/Werkzeug's own HTTP exceptions (404, 405, our jsonify'd 400s,
    # etc.) render normally - only genuinely unhandled Python exceptions
    # should hit this handler and get logged as a real bug.
    if isinstance(e, HTTPException):
        return e
    logger.error("Unhandled exception on %s %s", request.method, request.path, exc_info=True)
    return jsonify({'error': 'An unexpected error occurred. Please try again.'}), 500


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_image_file(filepath, extension):
    """
    Confirm the uploaded file is actually a valid, decodable image whose real
    content matches its extension - not just that it *claims* to be one.
    Returns (is_valid, error_message).
    """
    # 1. Check the file's magic bytes match a known image signature.
    signatures = IMAGE_SIGNATURES.get(extension, [])
    try:
        with open(filepath, 'rb') as f:
            header = f.read(16)
    except OSError as e:
        return False, f'Could not read uploaded file: {e}'

    if signatures and not any(header.startswith(sig) for sig in signatures):
        return False, 'File content does not match a valid image format.'

    # 2. Ask Pillow to verify the file isn't corrupt/truncated.
    try:
        with Image.open(filepath) as img:
            img.verify()
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        return False, 'File is not a valid or readable image.'

    # img.verify() leaves the file unusable for further ops, so re-open to
    # confirm it can still be loaded normally for actual analysis.
    try:
        with Image.open(filepath) as img:
            img.load()
    except (UnidentifiedImageError, OSError, ValueError):
        return False, 'Image could not be decoded.'

    return True, None


def is_valid_email(email):
    return bool(email) and '@' in email and '.' in email.split('@')[-1]


def build_absolute_url(path):
    """Build a link to include in an email. Uses APP_BASE_URL if it's set
    (recommended for production so links are correct behind a proxy/CDN);
    otherwise falls back to the current request's own host."""
    base = APP_BASE_URL or request.host_url.rstrip('/')
    return f"{base}{path}"


def generate_username_from_email(email):
    """Derive a valid, available username for a first-time OAuth sign-in,
    e.g. 'jane.doe+test@x.com' -> 'jane_doe' (with a random suffix appended
    if that's already taken)."""
    local_part = email.split('@')[0]
    base = re.sub(r'[^A-Za-z0-9_]', '_', local_part).strip('_')[:24] or 'user'
    if len(base) < 3:
        base = (base + '_user')[:24]
    candidate = base
    while db.username_taken(candidate):
        candidate = f"{base}_{secrets.token_hex(3)}"[:30]
    return candidate


# ---------------------------------------------------------------------------
# Public routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/emergency')
def emergency():
    return render_template('emergency.html')


@app.route('/health')
def health():
    """Liveness/readiness endpoint for uptime monitors and load balancers."""
    try:
        db.get_user_by_id(-1)  # cheap query just to confirm the DB is reachable
        db_ok = True
    except Exception:
        logger.error("Health check: database is unreachable", exc_info=True)
        db_ok = False

    status = {
        'status': 'ok' if db_ok else 'degraded',
        'database': 'ok' if db_ok else 'unreachable',
        'gemini_configured': medical_analyzer.use_gemini,
    }
    return jsonify(status), (200 if db_ok else 503)


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------

def _send_verification_email(user_row):
    token = db.create_token(user_row['id'], 'email_verify', EMAIL_VERIFY_TTL_SECONDS)
    link = build_absolute_url(url_for('verify_email', token=token))
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


@app.route('/register', methods=['GET', 'POST'])
@limiter.limit("5 per minute")
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'GET':
        return render_template('register.html', google_oauth_enabled=is_google_oauth_configured())

    username = (request.form.get('username') or '').strip()
    email = (request.form.get('email') or '').strip().lower()
    password = request.form.get('password') or ''
    ctx = {'google_oauth_enabled': is_google_oauth_configured()}

    if not username or not email or not password:
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

    user_row = db.create_user(username, email, password)
    if user_row is None:
        return render_template('register.html', error='That username or email is already taken.', **ctx), 409

    _send_verification_email(user_row)

    user = User.from_row(user_row)
    login_user(user)
    logger.info("New user registered and logged in: %s", username)
    return redirect(url_for('index'))


@app.route('/login', methods=['GET', 'POST'])
@limiter.limit("10 per minute")
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'GET':
        return render_template('login.html', google_oauth_enabled=is_google_oauth_configured())

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
            return redirect(url_for('login_2fa'))

        user = User.from_row(user_row)
        login_user(user)
        logger.info("User logged in: %s", username)
        return redirect(next_page or url_for('index'))

    logger.info("Failed login attempt for username: %s", username)
    return render_template(
        'login.html', error='Invalid username or password.',
        google_oauth_enabled=is_google_oauth_configured()
    ), 401


@app.route('/login/2fa', methods=['GET', 'POST'])
@limiter.limit("10 per minute")
def login_2fa():
    """Second step of login for accounts with 2FA enabled - reached only
    after a correct username/password on the /login form."""
    pending_user_id = session.get('pending_2fa_user_id')
    if not pending_user_id:
        return redirect(url_for('login'))

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
        return redirect(next_page or url_for('index'))

    logger.info("Invalid 2FA code for pending user_id=%s", pending_user_id)
    return render_template('login_2fa.html', error='Invalid or expired code. Please try again.'), 401


@app.route('/logout', methods=['POST'])
@login_required
def logout():
    logger.info("User logged out: %s", current_user.username)
    logout_user()
    return redirect(url_for('index'))


# ---------------------------------------------------------------------------
# Password reset
# ---------------------------------------------------------------------------

@app.route('/forgot-password', methods=['GET', 'POST'])
@limiter.limit("5 per minute")
def forgot_password():
    if request.method == 'GET':
        return render_template('forgot_password.html')

    email = (request.form.get('email') or '').strip().lower()
    if is_valid_email(email):
        user_row = db.get_user_by_email(email)
        if user_row:
            token = db.create_token(user_row['id'], 'password_reset', PASSWORD_RESET_TTL_SECONDS)
            link = build_absolute_url(url_for('reset_password', token=token))
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


@app.route('/reset-password/<token>', methods=['GET', 'POST'])
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


@app.route('/verify-email/<token>')
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

@app.route('/login/google')
@limiter.limit("15 per minute")
def login_google():
    if not is_google_oauth_configured():
        return render_template('login.html', error='Google sign-in is not configured on this server.'), 503
    redirect_uri = build_absolute_url(url_for('login_google_callback'))
    return oauth.google.authorize_redirect(redirect_uri)


@app.route('/login/google/callback')
@limiter.limit("15 per minute")
def login_google_callback():
    if not is_google_oauth_configured():
        return redirect(url_for('login'))

    try:
        token = oauth.google.authorize_access_token()
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
    return redirect(url_for('index'))


# ---------------------------------------------------------------------------
# Account settings
# ---------------------------------------------------------------------------

@app.route('/account')
@login_required
def account():
    user_row = db.get_user_by_id(int(current_user.id))
    return render_template('account.html', account_user=user_row)


@app.route('/account/profile', methods=['POST'])
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


@app.route('/account/password', methods=['POST'])
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


@app.route('/account/resend-verification', methods=['POST'])
@login_required
@limiter.limit("3 per minute")
def resend_verification():
    user_row = db.get_user_by_id(int(current_user.id))
    if user_row['email_verified']:
        return render_template('account.html', account_user=user_row, success='Your email is already verified.')
    _send_verification_email(user_row)
    logger.info("Verification email resent for user_id=%s", user_row['id'])
    return render_template('account.html', account_user=user_row, success='Verification email sent.')


@app.route('/account/delete', methods=['POST'])
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
    return redirect(url_for('index'))


@app.route('/account/2fa/setup')
@login_required
def setup_2fa():
    user_row = db.get_user_by_id(int(current_user.id))
    if user_row['totp_enabled']:
        return redirect(url_for('account'))

    secret = generate_secret()
    session['pending_totp_secret'] = secret
    uri = provisioning_uri(secret, user_row['email'])
    qr_data_uri = qr_code_data_uri(uri)
    return render_template('setup_2fa.html', secret=secret, qr_data_uri=qr_data_uri)


@app.route('/account/2fa/confirm', methods=['POST'])
@login_required
@limiter.limit("10 per minute")
def confirm_2fa():
    secret = session.get('pending_totp_secret')
    code = (request.form.get('code') or '').strip()

    if not secret:
        return redirect(url_for('setup_2fa'))

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


@app.route('/account/2fa/disable', methods=['POST'])
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


def _account_error(message):
    user_row = db.get_user_by_id(int(current_user.id))
    return render_template('account.html', account_user=user_row, error=message), 400


# ---------------------------------------------------------------------------
# Protected routes (require login)
# ---------------------------------------------------------------------------

@app.route('/upload', methods=['POST'])
@login_required
@limiter.limit("10 per minute")
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        extension = filename.rsplit('.', 1)[1].lower()
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Reject corrupt/invalid/mismatched files before they ever reach
        # Image.open() in the analyzer or get sent to Gemini.
        is_valid, error_message = validate_image_file(filepath, extension)
        if not is_valid:
            os.remove(filepath)
            logger.info("Rejected invalid image upload from user=%s: %s", current_user.username, error_message)
            return jsonify({'error': f'Invalid image file: {error_message}'}), 400

        try:
            analysis_result = medical_analyzer.analyze_image(filepath)
            os.remove(filepath)

            db.save_image_analysis(int(current_user.id), filename, analysis_result)

            return jsonify({
                'success': True,
                'analysis': analysis_result
            })
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            logger.error("Image analysis failed for user=%s", current_user.username, exc_info=True)
            return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/analyze_symptoms', methods=['POST'])
@login_required
@limiter.limit("15 per minute")
def analyze_symptoms():
    try:
        data = request.get_json(silent=True) or {}
        symptoms = data.get('symptoms', '')

        if not isinstance(symptoms, str):
            return jsonify({'error': 'Symptoms must be provided as text'}), 400

        symptoms = symptoms.strip()

        if not symptoms:
            return jsonify({'error': 'No symptoms provided'}), 400

        if len(symptoms) < MIN_SYMPTOM_LENGTH:
            return jsonify({'error': 'Please describe your symptoms in more detail'}), 400

        if len(symptoms) > MAX_SYMPTOM_LENGTH:
            return jsonify({
                'error': f'Symptom description is too long (max {MAX_SYMPTOM_LENGTH} characters)'
            }), 400

        analysis_result = symptom_checker.analyze_symptoms(symptoms)

        db.save_symptom_analysis(int(current_user.id), symptoms, analysis_result)

        return jsonify({
            'success': True,
            'analysis': analysis_result
        })
    except Exception as e:
        logger.error("Symptom analysis failed for user=%s", current_user.username, exc_info=True)
        return jsonify({'error': f'Symptom analysis failed: {str(e)}'}), 500


@app.route('/history')
@login_required
def history_page():
    entries = db.get_history(int(current_user.id))
    return render_template('history.html', entries=entries)


@app.route('/api/history')
@login_required
def history_api():
    entries = db.get_history(int(current_user.id))
    return jsonify({'success': True, 'history': entries})


@app.route('/api/history/clear', methods=['POST'])
@login_required
@limiter.limit("5 per minute")
def clear_history():
    db.delete_history(int(current_user.id))
    logger.info("History cleared for user=%s", current_user.username)
    return jsonify({'success': True})


@app.errorhandler(429)
def rate_limit_exceeded(e):
    logger.info("Rate limit exceeded on %s %s from %s", request.method, request.path, get_remote_address())
    return jsonify({
        'error': 'Too many requests. Please slow down and try again shortly.'
    }), 429


if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() in ('1', 'true', 'yes')
    logger.info("Starting Quick Aid Medical Assistant...")
    print("Starting Quick Aid Medical Assistant...")
    print("⚠️  DISCLAIMER: This tool is for educational purposes only.")
    print("⚠️  Always consult healthcare professionals for medical advice.")
    if debug_mode:
        logger.warning("Running in DEBUG mode - never do this in production, it exposes an interactive debugger/RCE risk.")
        print("⚠️  Running in DEBUG mode — never do this in production, "
              "it exposes an interactive debugger/RCE risk.")
    else:
        print("Running with debug mode OFF. For production, use a WSGI server "
              "instead of this dev server, e.g.:")
        print("    gunicorn -w 2 -b 0.0.0.0:5000 app:app")
    app.run(debug=debug_mode, host='0.0.0.0', port=5000)
