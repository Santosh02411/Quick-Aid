"""Structured error handling: every failure becomes a standard response, with a request id, and nothing internal leaks."""

from flask import jsonify, render_template, request
from flask_wtf.csrf import CSRFError
from werkzeug.exceptions import HTTPException

from quickaid.api.responses import ApiError, DEFAULT_CODES, current_request_id, fail
from quickaid.core.logging_config import get_logger

logger = get_logger('errors')

JSON_PREFIXES = ('/api/', '/admin/api/')
JSON_PATHS = {'/upload', '/analyze_symptoms', '/health'}

FRIENDLY = {
    404: 'That was not found.', 405: 'That method is not allowed here.', 413: 'That upload is larger than 16 MB.',
    429: 'Too many requests. Please slow down and try again shortly.',
}


def wants_json() -> bool:
    """JSON for API/AJAX callers, HTML pages for browsers navigating."""
    if request.path.startswith(JSON_PREFIXES) or request.path in JSON_PATHS:
        return True
    return request.is_json or request.accept_mimetypes.best == 'application/json'


def register_error_handlers(app) -> None:
    @app.errorhandler(ApiError)
    def api_error(e: ApiError):
        return fail(e.message, e.status, e.code, e.hints, e.details, **e.extra)

    @app.errorhandler(CSRFError)
    def csrf_error(e):
        logger.info("CSRF validation failed on %s %s: %s", request.method, request.path, e.description)
        if wants_json():
            return fail('Your session has expired or is invalid. Please refresh the page and try again.', 400, 'csrf_failed')
        return render_template('csrf_error.html', reason=e.description), 400

    @app.errorhandler(413)
    def too_large(e):
        return fail(FRIENDLY[413], 413, 'too_large', hints=['Use the Take photo button, which makes a smaller file.',
                                                            'Or choose a lower-resolution copy of the photo.'], retake=True)

    @app.errorhandler(429)
    def rate_limited(e):
        logger.info("Rate limit exceeded on %s %s", request.method, request.path)
        retry = getattr(getattr(e, 'limit', None), 'limit', None)
        return fail(FRIENDLY[429], 429, 'rate_limited', retry_after_hint=str(retry) if retry else None)

    @app.errorhandler(HTTPException)
    def http_error(e: HTTPException):
        if wants_json():
            code = DEFAULT_CODES.get(e.code or 500, 'error')
            return fail(FRIENDLY.get(e.code) or (e.description or 'Request failed.'), e.code or 500, code)
        return e

    @app.errorhandler(Exception)
    def unexpected(e):
        logger.error("Unhandled exception on %s %s [request %s]", request.method, request.path, current_request_id(), exc_info=True)
        if wants_json():
            return fail('An unexpected error occurred. Please try again.', 500, 'internal_error')
        return render_template('error.html', request_id=current_request_id()), 500
