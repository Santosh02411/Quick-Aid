"""Cross-cutting request hooks: request ids, access logging, security headers, regional context, caching rules, maintenance."""

import re
import secrets
import time
import uuid

from flask import g, request
from flask_login import current_user

from quickaid import config, i18n
from quickaid.ai import consent as ai_consent
from quickaid.core.localization import REGIONS, get_app_region, get_region_info, set_request_region
from quickaid.core.logging_config import get_logger
from quickaid.observability import metrics as obs_metrics
from quickaid.services.container import get_services
from quickaid.web import sessions

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
    app.jinja_env.globals['csp_nonce'] = lambda: g.get('csp_nonce', '')

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
    def _apply_language():
        """Pick the language: the person's saved choice, else the cookie, else the browser's preference, else English."""
        pref = getattr(current_user, 'language', None) if current_user.is_authenticated else None
        code = i18n.negotiate(None, pref, request.cookies.get(i18n.COOKIE_NAME), request.headers.get('Accept-Language'))
        g.lang = code
        g._lang_token = i18n.set_language(code)

    @app.before_request
    def _apply_ai_consent():
        """Nothing goes to the external AI provider unless THIS signed-in person has agreed (to the current wording)."""
        from quickaid.core import database as db
        from quickaid.services import consent as consent_service
        allowed = True
        if current_user.is_authenticated and config.require_ai_consent():
            allowed = consent_service.allowed_for(db.get_user_by_id(int(current_user.id)))
        g._ai_consent_token = ai_consent.set_allowed(allowed)
        g.ai_consent_given = allowed

    @app.before_request
    def _session_limits():
        if request.endpoint != 'static':
            sessions.enforce_limits()

    @app.before_request
    def _maintenance():
        if request.endpoint != 'static':
            get_services().retention.run_if_due(app.config['UPLOAD_FOLDER'])      # throttled; at most once a day across all workers

    @app.after_request
    def _finish(response):
        ms = (time.monotonic() - getattr(request, '_start_time', time.monotonic())) * 1000
        # Log the ROUTE PATTERN, never the raw path: /reset-password/<token> must not put a live token in a log file.
        route = request.url_rule.rule if request.url_rule else '<unmatched>'
        logger.info("%s %s -> %s (%.1fms) [%s]", request.method, route, response.status_code, ms, g.get('request_id', '-'))
        obs_metrics.observe_request(request.method, route, response.status_code, ms / 1000)
        response.headers['X-Request-ID'] = g.get('request_id', '')
        response.headers['Content-Security-Policy'] = _CSP.format(nonce=g.get('csp_nonce', ''))
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        response.headers['Permissions-Policy'] = 'camera=(self), microphone=(), geolocation=(), payment=(), usb=()'
        response.headers['Strict-Transport-Security'] = 'max-age=63072000; includeSubDomains'   # ignored over plain HTTP
        response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
        response.headers['Cross-Origin-Resource-Policy'] = 'same-origin'
        if request.endpoint != 'static' and (current_user.is_authenticated or request.path.startswith(_NO_STORE_PREFIXES)):
            response.headers['Cache-Control'] = 'no-store, max-age=0'
            response.headers['Pragma'] = 'no-cache'
        return response

    @app.teardown_request
    def _clear_region(exc=None):
        set_request_region(None)   # never let one request's region leak into the next
        token = g.pop('_ai_consent_token', None)
        if token is not None:
            ai_consent.reset(token)
        lang_token = g.pop('_lang_token', None)
        if lang_token is not None:
            i18n.reset(lang_token)

    @app.context_processor
    def _inject_context():
        region = get_app_region()
        info = get_region_info(region)
        return {
            'emergency_number': info['emergency_number'],
            'ambulance_number': config.AMBULANCE_NUMBERS.get(region, info['emergency_number']),
            'max_symptom_length': config.MAX_SYMPTOM_LENGTH,
            'ai_connected': bool(get_services().symptom_checker.use_gemini),
            'ai_consent_needed': bool(get_services().symptom_checker.use_gemini and current_user.is_authenticated and config.require_ai_consent()
                                      and getattr(current_user, 'ai_consent_version', None) not in (0, config.AI_CONSENT_VERSION)),
            'region_code': region,
            'ai_state': {'available': bool(get_services().symptom_checker.use_gemini), 'allowed': bool(g.get('ai_consent_given', True))},
            'lang': g.get('lang', 'en'), 't': i18n.t, 'languages': i18n.NATIVE_NAMES, 'i18n_js': i18n.js_strings(g.get('lang', 'en')),
            'region_options': [(code, r['label']) for code, r in REGIONS.items()],
        }
