"""Health, regions and the API documentation."""

from flask import Blueprint
from flask_login import current_user, login_required

from quickaid.api.openapi import build_spec, doc
from quickaid.api.responses import ApiError, ok
from quickaid.api.validation import json_object
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
