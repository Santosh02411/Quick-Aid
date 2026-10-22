"""Internal admin tooling (AI evaluation + user accounts). Admins only - everyone else gets 404, so its existence
is not revealed. "Admin" means an account listed in ADMIN_USERNAMES (set from the terminal/.env, never from the
sign-up page or any page a logged-in user could reach - see User.is_admin in accounts/auth.py)."""

import functools
import os
import re
from datetime import datetime, timedelta, timezone

from flask import Blueprint, abort, render_template, request, url_for
from flask_login import current_user, login_required

from quickaid.ai import prompt_registry, telemetry
from quickaid.ai.triage_engine import ENGINE_VERSION
from quickaid.api.openapi import doc
from quickaid.api.responses import ApiError, ok
from quickaid.api.schemas import AdminDeleteAccountBody
from quickaid.api.validation import json_object, parse_model
from quickaid.core import database as db
from quickaid.core.logging_config import get_logger
from quickaid.evaluation import datasets as eval_datasets, runs as eval_runs, stats as eval_stats
from quickaid.extensions import limiter
from quickaid.services import admin_accounts
from quickaid.services.accounts import build_absolute_url
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
    return render_template('admin_evaluation.html', active_page='admin_evaluation')


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


# ---------------------------------------------------------------------------
# User management: accounts, not health data. Every action here calls the SAME
# quickaid.services.admin_accounts functions the terminal tool (quickaid.core.accounts_cli) uses, so the two can
# never behave differently, and every action is logged to the affected person's own security history and, where
# there is somewhere to send it, emailed to them - an admin action is never invisible to the person it affects.
# ---------------------------------------------------------------------------

def _admin_label() -> str:
    return getattr(current_user, 'username', '') or 'web admin'


def _run_admin_action(fn, *args, **kwargs):
    try:
        return fn(*args, by=_admin_label(), **kwargs)
    except admin_accounts.AdminActionError as e:
        raise ApiError(str(e), 404 if 'no account found' in str(e).lower() else 400, 'admin_action_failed')


@bp.route('/admin/users')
@admin_required
def admin_users_page():
    return render_template('admin_users.html', active_page='admin_users')


@bp.route('/admin/api/users')
@admin_required
@doc('List accounts (no health data)', tags=['Admin'], admin=True,
     description='Username, email, verification and 2FA status, account creation date. Never symptoms, photos, chats or notes.')
def admin_api_users():
    q = (request.args.get('q') or '').strip()[:120] or None
    rows = admin_accounts.list_accounts(q)
    admin_names = {n.strip().lower() for n in os.getenv('ADMIN_USERNAMES', '').split(',') if n.strip()}
    for r in rows:
        r['is_admin'] = r['username'].lower() in admin_names
    return ok({'users': rows, 'admin_configured': bool(admin_names)})


@bp.route('/admin/api/users/<identifier>')
@admin_required
@doc('One account\'s details and recent security activity', tags=['Admin'], admin=True)
def admin_api_user_detail(identifier):
    try:
        return ok({'user': admin_accounts.account_detail(identifier)})
    except admin_accounts.AdminActionError as e:
        raise ApiError(str(e), 404, 'account_not_found')


@bp.route('/admin/api/users/<identifier>/reset-2fa', methods=['POST'])
@admin_required
@limiter.limit("20 per minute")
@doc('Disable a locked-out account\'s 2FA so they can sign back in with just their password', tags=['Admin'], admin=True,
     description='Also signs the account out everywhere. The person is emailed and it appears in their own security history. '
                'They can set 2FA up again once they are back in.')
def admin_reset_2fa(identifier):
    result = _run_admin_action(admin_accounts.reset_two_factor, identifier)
    return ok({'reset': True, 'username': result['username']})


@bp.route('/admin/api/users/<identifier>/force-logout', methods=['POST'])
@admin_required
@limiter.limit("20 per minute")
@doc('Sign an account out of every device', tags=['Admin'], admin=True)
def admin_force_logout(identifier):
    result = _run_admin_action(admin_accounts.force_logout, identifier)
    return ok({'logged_out': True, 'username': result['username']})


@bp.route('/admin/api/users/<identifier>/send-password-reset', methods=['POST'])
@admin_required
@limiter.limit("10 per minute")
@doc('Email the account a password-reset link', tags=['Admin'], admin=True)
def admin_send_password_reset(identifier):
    build_link = lambda token: build_absolute_url(url_for('auth.reset_password', token=token))  # noqa: E731
    result = _run_admin_action(admin_accounts.start_password_reset, identifier, build_link)
    return ok({'sent': True, 'username': result['username']})


@bp.route('/admin/api/users/<identifier>/delete', methods=['POST'])
@admin_required
@limiter.limit("10 per minute")
@doc('Permanently delete an account and everything it owns', tags=['Admin'], admin=True, request='AdminDeleteAccountBody',
     description='Cannot be undone. The confirm field must exactly repeat the username or email given in the URL, as a typo guard.')
def admin_delete_user(identifier):
    body = parse_model(AdminDeleteAccountBody, json_object())
    result = _run_admin_action(admin_accounts.delete_account, identifier, confirm=body.confirm)
    return ok({'deleted': True, 'username': result['username']})
