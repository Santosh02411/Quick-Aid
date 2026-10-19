"""Request validation. Raises ApiError (turned into a standard 4xx response) so views stay free of if/return-error ladders."""

import re
from typing import Any, Dict, Tuple, Type, TypeVar, get_origin

from flask import request
from pydantic import BaseModel, ValidationError

from quickaid import config
from quickaid.api.responses import ApiError

M = TypeVar('M', bound=BaseModel)


def json_object() -> Dict[str, Any]:
    """The JSON body as a dict. An absent body is {}; a body that is not a JSON object is rejected."""
    data = request.get_json(silent=True)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ApiError('The request body must be a JSON object.', 400, 'invalid_json')
    return data


def parse_model(model: Type[M], data: Dict[str, Any]) -> M:
    """Validate with a Pydantic model. Problems come back as per-field `details`."""
    try:
        return model.model_validate(data)
    except ValidationError as e:
        details = [{'field': '.'.join(str(p) for p in err['loc']) or 'body', 'message': err['msg'].replace('Value error, ', '')}
                   for err in e.errors()]
        raise ApiError(details[0]['message'] if len(details) == 1 else 'Some fields are not valid.', 400, 'validation_error', details=details)


def parse_query(model: Type[M]) -> M:
    """Validate the query string. Repeated or comma-separated values become lists for fields declared as lists."""
    raw: Dict[str, Any] = {}
    for key in request.args:
        info = model.model_fields.get(key)
        if info is not None and get_origin(info.annotation) is list:
            raw[key] = [v for chunk in request.args.getlist(key) for v in chunk.split(',') if v != '']
        else:
            raw[key] = request.args.get(key) or ''
    return parse_model(model, raw)


# ---- the exact rules (and messages) the app has always used, now in one place ----

def symptom_text(data: Dict[str, Any]) -> str:
    symptoms = data.get('symptoms', '')
    if not isinstance(symptoms, str):
        raise ApiError('Symptoms must be provided as text', 400, 'invalid_symptoms')
    symptoms = symptoms.strip()
    if not symptoms:
        raise ApiError('No symptoms provided', 400, 'missing_symptoms')
    if len(symptoms) < config.MIN_SYMPTOM_LENGTH:
        raise ApiError('Please describe your symptoms in more detail', 400, 'symptoms_too_short')
    if len(symptoms) > config.MAX_SYMPTOM_LENGTH:
        raise ApiError(f'Symptom description is too long (max {config.MAX_SYMPTOM_LENGTH} characters)', 400, 'symptoms_too_long')
    return symptoms


def chat_message(raw: Any) -> str:
    if not isinstance(raw, str):
        raise ApiError('Message must be text', 400, 'invalid_message')
    message = re.sub(r'[ \t]+', ' ', raw).strip()
    if not message:
        raise ApiError('Please type a question', 400, 'empty_message')
    if len(message) > config.MAX_CHAT_MESSAGE_LENGTH:
        raise ApiError(f'Message is too long (max {config.MAX_CHAT_MESSAGE_LENGTH} characters)', 400, 'message_too_long')
    return message


def analysis_ref(data: Dict[str, Any]) -> Tuple[str, int]:
    atype = data.get('analysis_type')
    if atype not in ('image', 'symptom'):
        raise ApiError('analysis_type must be "image" or "symptom"', 400, 'invalid_analysis_type')
    try:
        return atype, int(data.get('analysis_id'))
    except (TypeError, ValueError):
        raise ApiError('analysis_id must be a number', 400, 'invalid_analysis_id')
