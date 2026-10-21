"""Speed and time limits for calls to the AI model.

Three things made results slow, and this module fixes each:
  1. No timeout: the client waited as long as the server (or network) kept it, so one stuck call looked like "it never answers".
     Every request now has a hard limit (GEMINI_TIMEOUT_SECONDS, default 25 s); when it is hit, the built-in check is shown straight away.
  2. "Thinking": Gemini Flash models spend seconds reasoning internally before answering. Our tasks are structured classification into a fixed JSON
     shape with strict validation and a safety net on top, so that reasoning adds delay, not safety. It is switched off (GEMINI_THINKING_BUDGET=default
     restores the model's own behaviour; a number sets a token budget). If a model rejects the setting, it is remembered and not sent again.
  3. Retrying a timeout: a call that already waited the full limit is not retried (it would just wait again); only quick failures (rate limit, 5xx) are.
     A total budget (GEMINI_TOTAL_BUDGET_SECONDS, default 40 s) caps the worst case.
"""

import os
import time
from typing import Dict, Optional

from google.genai import types

_NO_THINKING = set()          # models that rejected the thinking setting
_TIMEOUT_WORDS = ('timed out', 'timeout', 'deadline', 'readtimeout', 'connecttimeout')


def _float_env(name: str, default: float) -> float:
    try:
        return max(2.0, float(os.getenv(name, '') or default))
    except ValueError:
        return default


def timeout_seconds(override: Optional[float] = None) -> float:
    return override if override else _float_env('GEMINI_TIMEOUT_SECONDS', 25.0)


def speed_kwargs(model: str, timeout: Optional[float] = None) -> Dict:
    """Extra GenerateContentConfig fields: a hard per-request timeout and (for Flash models) no internal 'thinking'."""
    kwargs: Dict = {'http_options': types.HttpOptions(timeout=int(timeout_seconds(timeout) * 1000))}
    setting = os.getenv('GEMINI_THINKING_BUDGET', '').strip().lower()
    name = (model or '').lower()
    if model in _NO_THINKING or setting == 'default':
        return kwargs
    if setting.lstrip('-').isdigit():
        kwargs['thinking_config'] = types.ThinkingConfig(thinking_budget=int(setting))
    elif 'flash' in name and 'pro' not in name:        # Pro models cannot switch thinking off
        kwargs['thinking_config'] = types.ThinkingConfig(thinking_budget=0)
    return kwargs


def note_error(model: str, exc: BaseException) -> None:
    """If the model refused the thinking setting, stop sending it (so the next attempt works)."""
    if 'thinking' in str(exc).lower():
        _NO_THINKING.add(model)


def is_timeout(exc: BaseException) -> bool:
    text = (type(exc).__name__ + ' ' + str(exc)).lower()
    return isinstance(exc, TimeoutError) or any(w in text for w in _TIMEOUT_WORDS)


class Budget:
    """The total time one analysis may spend waiting for the model, across its attempts."""

    def __init__(self, total: Optional[float] = None):
        self.total = total or _float_env('GEMINI_TOTAL_BUDGET_SECONDS', 40.0)
        self.start = time.monotonic()

    def allows_retry(self, exc: BaseException, model: str = '') -> bool:
        note_error(model, exc)
        if is_timeout(exc):
            return False                                   # it already waited the full limit; waiting again only doubles the delay
        return (time.monotonic() - self.start) < self.total - timeout_seconds() * 0.6
