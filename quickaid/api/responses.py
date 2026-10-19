"""The ONE response shape for every JSON endpoint.

Success:  {"success": true,  ...payload fields..., "request_id": "..."}
Failure:  {"success": false, "error": "<human message>", "code": "<stable machine code>", "request_id": "...",
           "hints": [...optional next steps...], "details": [...optional per-field problems...]}

`error` stays a plain string and payload fields stay at the top level, so existing clients keep working; what changes is that
every failure now carries a stable `code` and a `request_id` that also appears in the server log, so a user's report can be traced.
"""

from typing import Any, Dict, List, Optional

from flask import g, jsonify

DEFAULT_CODES = {400: 'bad_request', 401: 'unauthenticated', 403: 'forbidden', 404: 'not_found', 405: 'method_not_allowed', 409: 'conflict',
                 413: 'payload_too_large', 415: 'unsupported_media_type', 422: 'validation_error', 429: 'rate_limited',
                 500: 'internal_error', 503: 'unavailable'}


class ApiError(Exception):
    """Raise anywhere in a view or service; the error handler turns it into the standard failure response."""

    def __init__(self, message: str, status: int = 400, code: Optional[str] = None, hints: Optional[List[str]] = None,
                 details: Optional[List[Dict]] = None, **extra: Any):
        super().__init__(message)
        self.message, self.status = message, status
        self.code = code or DEFAULT_CODES.get(status, 'error')
        self.hints, self.details, self.extra = hints, details, extra


def current_request_id() -> str:
    return g.get('request_id', '') if g else ''


def ok(payload: Optional[Dict] = None, status: int = 200, **extra: Any):
    body: Dict[str, Any] = {'success': True}
    body.update(payload or {})
    body.update(extra)
    body['request_id'] = current_request_id()
    return jsonify(body), status


def fail(message: str, status: int = 400, code: Optional[str] = None, hints: Optional[List[str]] = None,
         details: Optional[List[Dict]] = None, **extra: Any):
    body: Dict[str, Any] = {'success': False, 'error': message, 'code': code or DEFAULT_CODES.get(status, 'error'),
                            'request_id': current_request_id()}
    if hints:
        body['hints'] = hints
    if details:
        body['details'] = details
    body.update(extra)
    return jsonify(body), status
