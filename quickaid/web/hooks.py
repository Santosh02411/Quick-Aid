"""Cross-cutting request hooks: request ids, access logging, security headers, regional context, caching rules, maintenance."""

import re
import secrets
import time
import uuid

from flask import g, request
from flask_login import current_user

from quickaid import config
from quickaid.core.localization import REGIONS, get_app_region, get_region_info, set_request_region
from quickaid.core.logging_config import get_logger
from quickaid.services.container import get_services

logger = get_logger('app')

_REQUEST_ID_RE = re.compile(r'^[A-Za-z0-9._-]{8,64}$')

# A per-request nonce lets the few legitimate inline <script> blocks run without falling back to 'unsafe-inline' for scripts.
_CSP = (
    "default-src 'self'; script-src 'self' 'nonce-{nonce}'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    "font-src 'self' https://fonts.gstatic.com; img-src 'self' data: blob:; connect-src 'self'; object-src 'none'; "
    "base-uri 'self'; form-action 'self'; frame-ancestors 'none'"
)
# Pages and APIs that carry someone's medical information must never be stored by a browser or proxy cache.
_NO_STORE_PREFIXES = ('/api/', '/admin/', '/history', '/chat', '/account', '/upload', '/analyze_symptoms')


def register_hooks(app) -> None:
    app.jinja_env.globals['csp_nonce'] = lambda: g.csp_nonce

    @app.before_request
    def _begin():
        incoming = request.headers.get('X-Request-ID', '')
        g.request_id = incoming if _REQUEST_ID_RE.match(incoming) else uuid.uuid4().hex[:16]
        g.csp_nonce = secrets.token_urlsafe(16)
        request._start_time = time.monotonic()

    @app.before_request
    def _apply_user_region():
        """Use the signed-in user's chosen region for emergency numbers and AI advice in this request."""
        set_request_region(getattr(current_user, 'region', None) if current_user.is_authenticated else None)

    @app.before_request
    def _maintenance():
        if request.endpoint != 'static':
            get_services().retention.run_if_due(app.config['UPLOAD_FOLDER'])      # throttled; at most once a day across all workers

    @app.after_request
    def _finish(response):
        ms = (time.monotonic() - getattr(request, '_start_time', time.monotonic())) * 1000
        logger.info("%s %s -> %s (%.1fms) [%s]", request.method, request.path, response.status_code, ms, g.get('request_id', '-'))
        response.headers['X-Request-ID'] = g.get('request_id', '')
        response.headers['Content-Security-Policy'] = _CSP.format(nonce=g.get('csp_nonce', ''))
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        response.headers['Permissions-Policy'] = 'camera=(self), microphone=(), geolocation=(), payment=(), usb=()'
        response.headers['Strict-Transport-Security'] = 'max-age=63072000; includeSubDomains'   # ignored over plain HTTP
        if request.endpoint != 'static' and (current_user.is_authenticated or request.path.startswith(_NO_STORE_PREFIXES)):
            response.headers['Cache-Control'] = 'no-store, max-age=0'
            response.headers['Pragma'] = 'no-cache'
        return response

    @app.teardown_request
    def _clear_region(exc=None):
        set_request_region(None)   # never let one request's region leak into the next

    @app.context_processor
    def _inject_context():
        region = get_app_region()
        info = get_region_info(region)
        return {
            'emergency_number': info['emergency_number'],
            'ambulance_number': config.AMBULANCE_NUMBERS.get(region, info['emergency_number']),
            'max_symptom_length': config.MAX_SYMPTOM_LENGTH,
            'ai_connected': bool(get_services().symptom_checker.use_gemini),
            'region_code': region,
            'region_options': [(code, r['label']) for code, r in REGIONS.items()],
        }
