"""Per-request switch: may this request's data be sent to the external AI provider?

The web layer sets it from the signed-in person's recorded choice. The AI classes check it at the exact point they would call the provider;
when it is off they use the built-in on-server basic mode instead, so NOTHING the person typed or uploaded leaves the server.

Default is True so the command-line tools, tests and evaluation runs (which have no "person") behave as before.
"""

import contextvars

_allowed: contextvars.ContextVar = contextvars.ContextVar('qa_external_ai_allowed', default=True)


def set_allowed(value: bool):
    return _allowed.set(bool(value))


def reset(token) -> None:
    _allowed.reset(token)


def external_allowed() -> bool:
    return _allowed.get()
