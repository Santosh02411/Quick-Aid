"""Privacy controls: the retention policy and each person's retention choice."""

from flask import Blueprint
from flask_login import current_user, login_required

from quickaid.api.openapi import doc
from quickaid.api.responses import ApiError, ok
from quickaid.api.schemas import RetentionBody
from quickaid.api.validation import json_object, parse_model
from quickaid.extensions import limiter
from quickaid.services.container import get_services

bp = Blueprint('privacy', __name__)


@bp.route('/api/privacy/policy')
@doc('The data-retention policy currently in force', tags=['Privacy'], auth=False)
def policy():
    return ok({'policy': get_services().retention.policy()})


@bp.route('/api/privacy/retention', methods=['GET'])
@login_required
@doc('How long your saved assessments and chats are kept', tags=['Privacy'])
def get_retention():
    return ok({'retention': get_services().retention.user_setting(int(current_user.id))})


@bp.route('/api/privacy/retention', methods=['PUT'])
@login_required
@limiter.limit("20 per minute")
@doc('Choose how long your saved assessments and chats are kept', tags=['Privacy'], request='RetentionBody',
     description='`days`: 30, 90, 365, 0 (keep until I delete) or null (use the default). Older entries are then deleted by the daily sweep.')
def set_retention():
    data = json_object()
    if 'days' not in data:
        raise ApiError('days is required (30, 90, 365, 0 or null).', 400, 'validation_error', details=[{'field': 'days', 'message': 'required'}])
    body = parse_model(RetentionBody, data)
    return ok({'retention': get_services().retention.set_user_days(int(current_user.id), body.days)})
