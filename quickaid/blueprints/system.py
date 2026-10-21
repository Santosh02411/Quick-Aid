"""Health, regions and the API documentation."""

from flask import Blueprint, abort, redirect, request
from flask_login import current_user, login_required

import hmac
import os as _os
from flask import Response, current_app
from quickaid.api.openapi import build_spec, doc
from quickaid.observability import metrics as obs_metrics, readiness
from quickaid.api.responses import ApiError, ok
from quickaid import i18n
from quickaid.api.schemas import LanguageBody
from quickaid.api.validation import json_object, parse_model
from quickaid.core import database as db
from quickaid.core.localization import REGIONS
from quickaid.core.logging_config import get_logger
from quickaid.extensions import limiter
from quickaid.services.container import get_services

logger = get_logger('system')
bp = Blueprint('system', __name__)


@bp.route('/health')
@doc('Liveness and readiness', tags=['System'], auth=False, description='Used by uptime monitors and load balancers. 503 when the database is unreachable.',
     responses={200: 'Healthy', 503: 'Database unreachable'})
def health():
    try:
        db.get_user_by_id(-1)       # a cheap query, just to confirm the database is reachable
        db_ok = True
    except Exception:
        logger.error("Health check: database is unreachable", exc_info=True)
        db_ok = False
    body = {'status': 'ok' if db_ok else 'degraded', 'database': 'ok' if db_ok else 'unreachable',
            'gemini_configured': get_services().medical_analyzer.use_gemini}
    resp, _ = ok(body)
    return resp, (200 if db_ok else 503)


@bp.route('/api/regions')
@doc('List supported regions', tags=['System'], auth=False, description='A flat object keyed by region code, each with a label and the local emergency number.')
def list_regions():
    return ok({code: {'label': r['label'], 'emergency_number': r['emergency_number']} for code, r in REGIONS.items()})


@bp.route('/api/region', methods=['POST'])
@login_required
@limiter.limit("20 per minute")
@doc('Set your region (emergency numbers follow it)', tags=['Account'], request='RegionRequest')
def update_region():
    region = str(json_object().get('region') or '').strip().upper()
    if region not in REGIONS:
        raise ApiError('Unknown region', 400, 'unknown_region')
    db.set_user_region(int(current_user.id), region)
    return ok({'region': region, 'emergency_number': REGIONS[region]['emergency_number']})


@bp.route('/api/openapi.json')
@login_required
@doc('OpenAPI 3 description of this API', tags=['System'], admin=True, hidden=True)
def openapi_json():
    _require_docs_access()
    from flask import current_app, jsonify
    return jsonify(build_spec(current_app))


@bp.route('/api/docs')
@login_required
@doc('API documentation page (HTML)', tags=['System'], admin=True, hidden=True)
def api_docs():
    _require_docs_access()
    from flask import render_template
    return render_template('api_docs.html', active_page='docs')


def _require_docs_access():
    """The API description is for developers and admins. API_DOCS_PUBLIC=true opens it to any signed-in user."""
    import os
    from flask import abort
    if os.getenv('API_DOCS_PUBLIC', 'false').lower() not in ('1', 'true', 'yes') and not getattr(current_user, 'is_admin', False):
        abort(404)


def _safe_next(path):
    """Only ever redirect back to a path on THIS site (never //evil.example or a full URL)."""
    if path and path.startswith('/') and not path.startswith('//') and '\\' not in path and '\n' not in path and '\r' not in path:
        return path
    return '/'


def _remember_language(response, code):
    response.set_cookie(i18n.COOKIE_NAME, code, max_age=365 * 24 * 3600, samesite='Lax', secure=request.is_secure, httponly=True)
    if current_user.is_authenticated:
        db.set_user_language(int(current_user.id), code)
    return response


@bp.route('/lang/<code>')
def switch_language(code):
    """Works without JavaScript: a plain link in the language menu. Remembers the choice in a cookie (and on the account when signed in)."""
    norm = i18n.normalize_code(code)
    if norm is None or norm != code.lower():
        abort(404)
    return _remember_language(redirect(_safe_next(request.args.get('next'))), norm)


@bp.route('/api/language', methods=['PUT'])
@limiter.limit("30 per minute")
@doc('Choose the display language', tags=['Account'], auth=False, request='LanguageBody',
     description='Remembers the choice in a cookie, and on your account when signed in. Translated safety text is not yet clinician-reviewed.')
def set_language_api():
    code = parse_model(LanguageBody, json_object()).language
    resp, _ = ok({'language': code})
    return _remember_language(resp, code)


@bp.route('/ready')
@doc('Readiness: can this instance do its job right now?', tags=['System'], auth=False, responses={200: 'Ready', 503: 'Not ready'},
     description='Checks that the database is writable and migrated, photos can be written, and there is disk space. Use for load-balancer readiness.')
def ready():
    ok_all, checks = readiness.check(current_app.config['UPLOAD_FOLDER'], int(_os.getenv('MIN_FREE_DISK_MB', '100') or 100))
    resp, _ = ok({'status': 'ready' if ok_all else 'not_ready', 'checks': checks})
    return resp, (200 if ok_all else 503)


def _metrics_allowed() -> bool:
    """METRICS_TOKEN (bearer) for scrapers, or an admin session. Anyone else gets a plain 404, so the endpoint's existence is not revealed."""
    token = _os.getenv('METRICS_TOKEN', '').strip()
    if token and hmac.compare_digest(request.headers.get('Authorization', ''), f'Bearer {token}'):
        return True
    return bool(current_user.is_authenticated and getattr(current_user, 'is_admin', False))


@bp.route('/metrics')
@doc('Prometheus metrics', tags=['System'], admin=True, hidden=True)
def metrics_endpoint():
    if not _metrics_allowed():
        abort(404)
    from quickaid.core import database, history_repo, migrations
    from datetime import datetime, timezone
    gauges = {'quickaid_uptime_seconds': round(__import__('time').time() - obs_metrics.STARTED, 1), 'quickaid_schema_version': migrations.status(database.DB_PATH)['current']}
    last = history_repo.last_maintenance('sweep')
    if last:
        gauges['quickaid_last_maintenance_age_seconds'] = round((datetime.now(timezone.utc) - datetime.fromisoformat(last)).total_seconds(), 1)
    try:
        gauges['quickaid_database_bytes'] = _os.path.getsize(database.DB_PATH)
    except OSError:
        pass
    return Response(obs_metrics.render(gauges), mimetype=obs_metrics.CONTENT_TYPE)
