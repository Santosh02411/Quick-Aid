"""History: search, detailed reports, personal notes, export and deletion. Everything is scoped to the signed-in user."""

from flask import Blueprint, Response, abort, render_template, request
from flask_login import current_user, login_required

from quickaid.api.openapi import doc
from quickaid.api.responses import ApiError, ok
from quickaid.api.schemas import ExportQuery, HistoryQuery, NotesBody
from quickaid.api.validation import json_object, parse_model, parse_query
from quickaid.extensions import limiter
from quickaid.services.container import get_services

bp = Blueprint('history', __name__)

KIND = "<any(symptom,image):entry_type>"
NOT_YOURS = {404: 'No such entry (or it is not yours)'}


def _uid() -> int:
    return int(current_user.id)


def _download(body: bytes, mimetype: str, filename: str) -> Response:
    return Response(body, mimetype=mimetype, headers={'Content-Disposition': f'attachment; filename="{filename}"'})


# ---------------------------------------------------------------- pages (server-rendered; filters are a plain GET form)

@bp.route('/history')
@login_required
def history_page():
    svc = get_services()
    error = None
    try:
        query = parse_query(HistoryQuery)
    except ApiError as e:
        query, error = HistoryQuery(), e.message
    data = svc.history.search(_uid(), query)
    base_query = {k: (v if len(v) > 1 else v[0]) for k, v in request.args.to_dict(flat=False).items() if k != 'page' and v}
    return render_template('history.html', entries=data['history'], page=data['pagination'], filters=query, error=error, active_page='history',
                           retention=svc.retention.user_setting(_uid()), base_query=base_query if not error else {})


@bp.route(f'/history/{KIND}/<int:entry_id>')
@login_required
def history_report(entry_type, entry_id):
    try:
        entry = get_services().history.detail(_uid(), entry_type, entry_id)
    except ApiError:
        abort(404)
    return render_template('history_detail.html', entry=entry, active_page='history')


# ---------------------------------------------------------------- API

@bp.route('/api/history')
@login_required
@doc('Search, filter and page through your assessments', tags=['History'], query='HistoryQuery',
     description=('Returns `history` (this page), `pagination` and the `filters` that were applied. With no parameters you get the newest 20. '
                  'Text search covers symptoms, conditions, advice, filenames and your own notes.'))
def history_api():
    return ok(get_services().history.search(_uid(), parse_query(HistoryQuery)))


@bp.route('/api/history/clear', methods=['POST'])
@login_required
@limiter.limit("5 per minute")
@doc('Delete ALL of your assessments and the chats about them', tags=['History'], description='Immediate and permanent.')
def clear_history():
    get_services().history.clear(_uid())
    return ok()


@bp.route('/api/history/export')
@login_required
@limiter.limit("10 per hour")
@doc('Download everything you have saved (?format=json or csv)', tags=['History', 'Privacy'], query='ExportQuery', download='application/json',
     description='JSON contains every assessment with its full analysis and your notes, plus all conversations. CSV is a one-row-per-assessment summary.')
def export_all():
    fmt = parse_query(ExportQuery).format
    return _download(*get_services().history.export(_uid(), fmt))


@bp.route(f'/api/history/{KIND}/<int:entry_id>')
@login_required
@doc('One assessment in full: analysis, notes and linked conversations', tags=['History'], responses=NOT_YOURS)
def history_entry(entry_type, entry_id):
    return ok({'entry': get_services().history.detail(_uid(), entry_type, entry_id)})


@bp.route(f'/api/history/{KIND}/<int:entry_id>/export')
@login_required
@limiter.limit("30 per hour")
@doc('Download one assessment as JSON', tags=['History'], download='application/json', responses=NOT_YOURS)
def export_entry(entry_type, entry_id):
    return _download(*get_services().history.export_entry(_uid(), entry_type, entry_id))


@bp.route(f'/api/history/{KIND}/<int:entry_id>/notes', methods=['PATCH'])
@login_required
@limiter.limit("60 per minute")
@doc('Save or clear your own note on an assessment', tags=['History'], request='NotesBody', responses=NOT_YOURS,
     description='Stored encrypted. An empty `notes` clears it. Maximum 2000 characters.')
def update_notes(entry_type, entry_id):
    body = parse_model(NotesBody, json_object())
    return ok(get_services().history.set_notes(_uid(), entry_type, entry_id, body.notes))


@bp.route(f'/api/history/{KIND}/<int:entry_id>', methods=['DELETE'])
@login_required
@limiter.limit("60 per minute")
@doc('Delete one assessment and the chats about it', tags=['History'], responses=NOT_YOURS, description='Immediate and permanent.')
def delete_entry(entry_type, entry_id):
    get_services().history.delete_entry(_uid(), entry_type, entry_id)
    return ok()
