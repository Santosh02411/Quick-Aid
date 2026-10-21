"""Symptom analysis: the one-shot check and the adaptive interview."""

from flask import Blueprint
from flask_login import current_user, login_required

from quickaid.api.openapi import doc
from quickaid.api.responses import ok
from quickaid.api.validation import json_object
from quickaid.extensions import limiter, user_or_ip
from quickaid.services.container import get_services

bp = Blueprint('symptoms', __name__)


def _uid() -> int:
    return int(current_user.id)


@bp.route('/analyze_symptoms', methods=['POST'])
@login_required
@limiter.limit("15 per minute", key_func=user_or_ip)
@doc('One-shot symptom check', tags=['Analysis'], request='SymptomRequest',
     description='Returns a reviewed assessment. For a better result use the interview endpoints, which ask follow-up questions.')
def analyze_symptoms():
    return ok(get_services().analysis.analyze_symptoms(_uid(), json_object()))


@bp.route('/api/interview/start', methods=['POST'])
@login_required
@limiter.limit("10 per minute", key_func=user_or_ip)
@doc('Start the adaptive symptom interview', tags=['Interview'], request='SymptomRequest',
     description='`status` is `question` (answer it next), `complete`, or `emergency` (guidance returned immediately, no questions).')
def interview_start():
    return ok(get_services().analysis.interview_start(_uid(), json_object()))


@bp.route('/api/interview/answer', methods=['POST'])
@login_required
@limiter.limit("60 per minute", key_func=user_or_ip)
@doc('Answer or skip the current interview question', tags=['Interview'], request='InterviewAnswerRequest',
     responses={409: 'The interview is already finished'})
def interview_answer():
    return ok(get_services().analysis.interview_answer(_uid(), json_object()))


@bp.route('/api/interview/finish', methods=['POST'])
@login_required
@limiter.limit("10 per minute", key_func=user_or_ip)
@doc('Get the assessment now, with whatever has been answered', tags=['Interview'], request='SessionRequest',
     responses={409: 'The interview is already finished'})
def interview_finish():
    return ok(get_services().analysis.interview_finish(_uid(), json_object()))
