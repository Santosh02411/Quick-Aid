from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_login import (
    LoginManager, login_user, logout_user, login_required, current_user
)
from werkzeug.exceptions import HTTPException
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

# Flask-Login: authentication is required for the actual AI-analysis features
# (upload / symptoms / history) since this is meant to be more than a public
# demo. The landing page and emergency info stay public - safety information
# should never be gated behind a login wall.
login_manager.init_app(app)

# Rate limiting: protects the Gemini-backed endpoints from abuse/quota burn,
# and login/register from brute-force/enumeration attempts.
# Uses in-memory storage by default (fine for a single dev/small deployment).
# For multi-process production deployments, point storage_uri at Redis, e.g.:
#   Limiter(..., storage_uri="redis://localhost:6379")
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

# Symptom text constraints
MAX_SYMPTOM_LENGTH = 1000
MIN_SYMPTOM_LENGTH = 3

# Account constraints
USERNAME_RE = re.compile(r'^[A-Za-z0-9_]{3,30}$')
MIN_PASSWORD_LENGTH = 8

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

@app.route('/register', methods=['GET', 'POST'])
@limiter.limit("5 per minute")
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'GET':
        return render_template('register.html')

    username = (request.form.get('username') or '').strip()
    email = (request.form.get('email') or '').strip().lower()
    password = request.form.get('password') or ''

    if not username or not email or not password:
        return render_template('register.html', error='All fields are required.'), 400

    if not USERNAME_RE.match(username):
        return render_template(
            'register.html',
            error='Username must be 3-30 characters: letters, numbers, underscores only.'
        ), 400

    if '@' not in email or '.' not in email.split('@')[-1]:
        return render_template('register.html', error='Please enter a valid email address.'), 400

    if len(password) < MIN_PASSWORD_LENGTH:
        return render_template(
            'register.html',
            error=f'Password must be at least {MIN_PASSWORD_LENGTH} characters.'
        ), 400

    user_row = db.create_user(username, email, password)
    if user_row is None:
        return render_template('register.html', error='That username or email is already taken.'), 409

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
        return render_template('login.html')

    username = (request.form.get('username') or '').strip()
    password = request.form.get('password') or ''

    user_row = db.get_user_by_username(username)
    if user_row and db.verify_password(user_row, password):
        user = User.from_row(user_row)
        login_user(user)
        logger.info("User logged in: %s", username)
        next_page = request.args.get('next')
        # Only follow next if it's a safe, local, relative path -
        # never redirect off-site based on unvalidated user input.
        if next_page and next_page.startswith('/') and not next_page.startswith('//'):
            return redirect(next_page)
        return redirect(url_for('index'))

    logger.info("Failed login attempt for username: %s", username)
    return render_template('login.html', error='Invalid username or password.'), 401


@app.route('/logout', methods=['POST'])
@login_required
def logout():
    logger.info("User logged out: %s", current_user.username)
    logout_user()
    return redirect(url_for('index'))


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
