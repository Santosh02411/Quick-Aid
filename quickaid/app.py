"""Quick Aid: Flask application factory.

This module only WIRES things together. The work lives elsewhere:

    blueprints/   HTTP views, one per area (pages, system, auth, symptoms, images, conversations, history, privacy, admin)
    services/     business logic with no HTTP in it (analysis, chats, history, retention) + the medical-safety facade
    api/          the shared API layer: standard responses, structured errors, request validation, OpenAPI
    web/          cross-cutting request hooks (request ids, security headers, caching rules, maintenance)
    ai/           the models, the triage/safety engine, prompts, telemetry
    core/         database, migrations, backups, encryption, history queries, localisation
    evaluation/   datasets, runs and the evaluation dashboard back end

Everything is built per app instance, so tests (and any future worker) get isolated services and extensions.
"""

import os
import secrets

from flask import Flask
from werkzeug.middleware.proxy_fix import ProxyFix

from quickaid import PROJECT_ROOT
from quickaid.accounts.oauth import init_oauth
from quickaid.ai import telemetry
from quickaid.api.errors import register_error_handlers
from quickaid.blueprints import register_blueprints
from quickaid.config import MAX_UPLOAD_BYTES
from quickaid.core import database as db
from quickaid.core.logging_config import get_logger
from quickaid.extensions import RATELIMIT_STORAGE_URI, csrf, limiter, login_manager
from quickaid.services.container import build_services
from quickaid.web.hooks import register_hooks

logger = get_logger('app')


def create_app() -> Flask:
    app = Flask(__name__)

    # SECRET_KEY must come from the environment. If it is missing we generate a random one for this process only (sessions will not
    # survive a restart) and warn loudly instead of silently using a predictable default.
    secret = os.getenv('SECRET_KEY')
    if not secret or secret == 'change_this_to_a_random_secret_key':
        secret = secrets.token_hex(32)
        logger.warning("SECRET_KEY not set in .env - using a temporary random key for this run. Set SECRET_KEY for stable sessions across restarts.")
    app.config['SECRET_KEY'] = secret
    app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', str(PROJECT_ROOT / 'uploads'))
    app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_BYTES

    # Behind nginx or a load balancer you control, trust exactly one hop of X-Forwarded-* (real client IP for rate limiting, original
    # scheme for secure cookies). Opt-in only: trusting those headers from just anyone would let a client spoof its IP.
    if os.getenv('BEHIND_PROXY', 'False').lower() in ('1', 'true', 'yes'):
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)
        logger.info("BEHIND_PROXY enabled - trusting one hop of X-Forwarded-* headers.")

    app.config['SESSION_COOKIE_HTTPONLY'] = True
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
    app.config['SESSION_COOKIE_SECURE'] = os.getenv('SESSION_COOKIE_SECURE', 'False').lower() in ('1', 'true', 'yes')
    if not app.config['SESSION_COOKIE_SECURE']:
        logger.info("SESSION_COOKIE_SECURE is off - fine for local plain-HTTP dev; any real deployment should set it True.")
    app.config.setdefault('WTF_CSRF_ENABLED', True)

    csrf.init_app(app)               # every state-changing request needs a token (form field or X-CSRFToken header)
    login_manager.init_app(app)
    init_oauth(app)                  # Google sign-in only registers if GOOGLE_CLIENT_ID/SECRET are set
    limiter.init_app(app)
    if RATELIMIT_STORAGE_URI == 'memory://':
        limiter.reset()              # a fresh app starts with fresh counters
        logger.warning("Rate limiter using in-memory storage - limits are per-process only. Set RATELIMIT_STORAGE_URI (e.g. a Redis URL) "
                       "before running more than one worker/instance.")

    app.extensions['quickaid_services'] = build_services()
    register_hooks(app)
    register_error_handlers(app)
    register_blueprints(app)

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    db.init_db()                     # versioned migrations; backs up first when upgrading a database that holds data
    telemetry.install_db_sink()
    return app


app = create_app()

# Names other modules (run.py, wsgi.py, tests) have always imported from here.
_services = app.extensions['quickaid_services']
symptom_checker = _services.symptom_checker
medical_analyzer = _services.medical_analyzer
conversation_service = _services.chat_ai
interview_engine = _services.interview_engine
