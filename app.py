from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import os
import secrets
import numpy as np
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from medical_analyzer import MedicalAnalyzer
from symptom_checker import SymptomChecker
from dotenv import load_dotenv
import database as db
import json

load_dotenv()

app = Flask(__name__)

# SECRET_KEY must come from the environment. If it's missing we generate a
# random one for this process only (sessions won't survive a restart) and
# warn loudly instead of silently using a predictable default.
_secret_key = os.getenv('SECRET_KEY')
if not _secret_key or _secret_key == 'change_this_to_a_random_secret_key':
    _secret_key = secrets.token_hex(32)
    print("WARNING: SECRET_KEY not set in .env — using a temporary random key "
          "for this run. Set SECRET_KEY in .env for stable sessions across restarts.")
app.config['SECRET_KEY'] = _secret_key

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Rate limiting: protects the Gemini-backed endpoints from abuse/quota burn.
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

# Initialize history database (SQLite file, created on first run)
db.init_db()


def get_session_id() -> str:
    """
    Anonymous per-browser identifier used to scope history entries, since
    this app has no login system. Stored in the signed Flask session cookie.
    """
    if 'session_id' not in session:
        session['session_id'] = secrets.token_hex(16)
    return session['session_id']


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
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
            return jsonify({'error': f'Invalid image file: {error_message}'}), 400

        try:
            # Analyze the uploaded image
            analysis_result = medical_analyzer.analyze_image(filepath)

            # Clean up uploaded file
            os.remove(filepath)

            db.save_image_analysis(get_session_id(), filename, analysis_result)

            return jsonify({
                'success': True,
                'analysis': analysis_result
            })
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/analyze_symptoms', methods=['POST'])
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

        db.save_symptom_analysis(get_session_id(), symptoms, analysis_result)

        return jsonify({
            'success': True,
            'analysis': analysis_result
        })
    except Exception as e:
        return jsonify({'error': f'Symptom analysis failed: {str(e)}'}), 500

@app.route('/emergency')
def emergency():
    return render_template('emergency.html')

@app.route('/history')
def history_page():
    entries = db.get_history(get_session_id())
    return render_template('history.html', entries=entries)

@app.route('/api/history')
def history_api():
    entries = db.get_history(get_session_id())
    return jsonify({'success': True, 'history': entries})

@app.route('/api/history/clear', methods=['POST'])
@limiter.limit("5 per minute")
def clear_history():
    db.delete_history(get_session_id())
    return jsonify({'success': True})

@app.errorhandler(429)
def rate_limit_exceeded(e):
    return jsonify({
        'error': 'Too many requests. Please slow down and try again shortly.'
    }), 429

if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() in ('1', 'true', 'yes')
    print("Starting Quick Aid Medical Assistant...")
    print("⚠️  DISCLAIMER: This tool is for educational purposes only.")
    print("⚠️  Always consult healthcare professionals for medical advice.")
    if debug_mode:
        print("⚠️  Running in DEBUG mode — never do this in production, "
              "it exposes an interactive debugger/RCE risk.")
    else:
        print("Running with debug mode OFF. For production, use a WSGI server "
              "instead of this dev server, e.g.:")
        print("    gunicorn -w 2 -b 0.0.0.0:5000 app:app")
    app.run(debug=debug_mode, host='0.0.0.0', port=5000)
