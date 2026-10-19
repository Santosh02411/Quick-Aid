"""Internal AI-evaluation dashboard (admins only). Anyone else gets 404, so its existence is not revealed."""

import functools
import os
import re
from datetime import datetime, timedelta, timezone

from flask import Blueprint, abort, render_template, request
from flask_login import current_user, login_required

from quickaid.ai import prompt_registry, telemetry
from quickaid.ai.triage_engine import ENGINE_VERSION
from quickaid.api.openapi import doc
from quickaid.api.responses import ApiError, ok
from quickaid.api.validation import json_object
from quickaid.core import database as db
from quickaid.core.logging_config import get_logger
from quickaid.evaluation import datasets as eval_datasets, runs as eval_runs, stats as eval_stats
from quickaid.extensions import limiter
from quickaid.services.container import get_services

logger = get_logger('admin')
bp = Blueprint('admin', __name__)
LIVE_NOTE = {400: 'Only offline and mock evaluations can be started from the dashboard; live runs use the command line.'}


def admin_required(view):
    """Only users listed in ADMIN_USERNAMES. Everyone else gets 404."""
    @functools.wraps(view)
    @login_required
    def wrapper(*args, **kwargs):
        if not getattr(current_user, 'is_admin', False):
            abort(404)
        return view(*args, **kwargs)
    return wrapper


@bp.route('/admin/evaluation')
@admin_required
def admin_evaluation():
    return render_template('admin_evaluation.html', active_page='admin')


@bp.route('/admin/api/overview')
@admin_required
@doc('Evaluation dashboard: configuration, datasets and saved runs', tags=['Admin'], admin=True)
def admin_overview():
    svc = get_services()
    return ok({'config': {'model': svc.symptom_checker.MODEL_NAME, 'ai_connected': bool(svc.symptom_checker.use_gemini), 'engine_version': ENGINE_VERSION,
                          'prompts': prompt_registry.fingerprint(), 'price_configured': telemetry.pricing() is not None,
                          'retention_days': int(os.getenv('TELEMETRY_RETENTION_DAYS', '90') or 90)},
               'datasets': eval_datasets.describe(), 'dataset_problems': eval_datasets.verify(), 'runs': eval_runs.list_runs()[:60]})


def _since(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


@bp.route('/admin/api/telemetry')
@admin_required
@doc('Live AI telemetry (latency, validity, failures, cost) over a window', tags=['Admin'], admin=True,
     description='Query parameter `days` (1-90, default 7). Timings, token counts and outcome labels only: never user text.')
def admin_telemetry():
    try:
        days = max(1, min(int(request.args.get('days', 7)), 90))
    except ValueError:
        days = 7
    summary = eval_stats.summarize(db.fetch_ai_calls(_since(days)), db.fetch_ai_analyses(_since(days)))
    summary.update({'window_days': days, 'price_configured': telemetry.pricing() is not None})
    return ok(summary)


@bp.route('/admin/api/runs/<run_id>')
@admin_required
@doc('One saved evaluation run in full', tags=['Admin'], admin=True, responses={404: 'Run not found'})
def admin_run_detail(run_id):
    run = eval_runs.load_run(run_id)
    if run is None:
        raise ApiError('Run not found', 404, 'run_not_found')
    return ok(run)


@bp.route('/admin/api/compare')
@admin_required
@doc('Compare two saved runs (?a=BASELINE&b=CANDIDATE)', tags=['Admin'], admin=True, responses={404: 'Run not found'})
def admin_compare():
    a, b = eval_runs.load_run(request.args.get('a', '')), eval_runs.load_run(request.args.get('b', ''))
    if a is None or b is None:
        raise ApiError('Run not found', 404, 'run_not_found')
    return ok(eval_runs.compare(a, b))


@bp.route('/admin/api/run', methods=['POST'])
@admin_required
@limiter.limit("3 per minute")
@doc('Start an offline or mock evaluation run', tags=['Admin'], admin=True, request='RunRequest', responses=LIVE_NOTE,
     description='Live runs cost real money, so they stay on the command line.')
def admin_start_run():
    from flask import current_app
    from quickaid.evaluation import runner
    data = json_object()
    spec = data.get('model', 'offline')
    if spec not in ('offline', 'mock-rules', 'mock-weak', 'mock-flaky'):
        raise ApiError('Only offline and mock evaluations can be started from the dashboard. Live runs use the command line.', 400, 'run_not_allowed')
    known = eval_datasets.load_manifest().get('datasets', {})
    sets = [s for s in (data.get('sets') or ['symptom_cases']) if s in known]
    if not sets:
        raise ApiError('Choose at least one dataset', 400, 'no_dataset')
    label = re.sub(r'\s+', ' ', str(data.get('label') or '')).strip()[:120]
    try:
        record = runner.build_run(spec, sets=sets, label=label, save=True)
    except Exception as e:
        logger.error("Dashboard evaluation run failed", exc_info=True)
        raise ApiError(f'The evaluation failed: {e}', 500, 'evaluation_failed')
    return ok({'run_id': record['run_id'], 'metrics': record['metrics'], 'threshold_failures': record['threshold_failures']})
