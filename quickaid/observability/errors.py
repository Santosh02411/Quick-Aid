"""Production error reporting (Sentry), configured so that NO personal or health data can leave the server.

Enabled only when SENTRY_DSN is set. Even then: no default PII, no request bodies, cookies, headers, query strings or local variables, no breadcrumbs,
no user object, and every message and exception text is passed through the same redaction as the log files. `scrub_event` is a pure function
and is tested directly, because this is exactly the kind of code that is easy to get subtly wrong.
"""

import os
import re
from typing import Any, Dict, Optional

from quickaid.core.logging_config import get_logger, redact

logger = get_logger('error_reporting')
_PATH_TOKEN = re.compile(r'(/(?:reset-password|verify-email)/)[^/?#\s]+')


def _clean_text(value: Any) -> Any:
    return redact(value) if isinstance(value, str) else value


def scrub_event(event: Dict, hint: Optional[Dict] = None) -> Optional[Dict]:
    """Remove everything that could identify a person or contain their health text. Returns the cleaned event."""
    ev = dict(event)
    request = dict(ev.get('request') or {})
    for key in ('data', 'cookies', 'headers', 'query_string', 'env', 'body'):
        request.pop(key, None)
    if request.get('url'):
        request['url'] = _PATH_TOKEN.sub(r'\1[token]', request['url'].split('?', 1)[0].split('#', 1)[0])
    ev['request'] = request
    for key in ('user', 'server_name', 'extra', 'breadcrumbs'):
        ev.pop(key, None)
    ev['message'] = _clean_text(ev.get('message'))
    if isinstance(ev.get('logentry'), dict):
        ev['logentry'] = {k: _clean_text(v) for k, v in ev['logentry'].items() if k in ('message', 'formatted')}
    for exc in (ev.get('exception') or {}).get('values', []) or []:
        exc['value'] = _clean_text(exc.get('value'))
        for frame in (exc.get('stacktrace') or {}).get('frames', []) or []:
            for k in ('vars', 'pre_context', 'context_line', 'post_context'):      # local variables and source lines can hold anything
                frame.pop(k, None)
    return ev


def init(dsn: Optional[str] = None) -> bool:
    """Start error reporting if (and only if) a DSN is configured. Returns whether it is on."""
    dsn = dsn or os.getenv('SENTRY_DSN', '').strip()
    if not dsn:
        return False
    try:
        import sentry_sdk
    except ImportError:
        logger.warning("SENTRY_DSN is set but sentry-sdk is not installed; error reporting is off")
        return False
    sentry_sdk.init(dsn=dsn, send_default_pii=False, include_local_variables=False, max_breadcrumbs=0, attach_stacktrace=True,
                    traces_sample_rate=float(os.getenv('SENTRY_TRACES_SAMPLE_RATE', '0') or 0), environment=os.getenv('APP_ENV', 'production'),
                    release=os.getenv('APP_VERSION') or None, before_send=scrub_event, before_breadcrumb=lambda crumb, hint: None)
    logger.info("Error reporting enabled (no personal data, bodies, cookies or variables are sent)")
    return True
