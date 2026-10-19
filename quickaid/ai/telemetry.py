"""Privacy-safe telemetry for every AI model call and every analysis.

What is recorded (never any user text, image or identity):
  per CALL      kind, model, prompt version, API ok?, schema-valid?, latency, tokens, cost, error type
  per ANALYSIS  kind, mode (ai / basic / rules), urgency, emergency?, safety rules raised urgency?,
                fell back?, reliability, total latency, number of calls, tokens, cost

Used by the internal dashboard (live data) and by the evaluation runner (collector).
Recording must NEVER break an analysis: every sink call is wrapped so a failure only logs.

Cost: tokens are always counted. Dollar cost is computed only when prices are configured
(GEMINI_PRICE_INPUT_PER_M and GEMINI_PRICE_OUTPUT_PER_M, USD per million tokens). We do not
guess prices: check your provider's current rates.
"""

import contextvars
import functools
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from quickaid.core.logging_config import get_logger

logger = get_logger('telemetry')

_current: contextvars.ContextVar = contextvars.ContextVar('qa_analysis', default=None)
_sinks: List = []


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

def pricing() -> Optional[Dict[str, float]]:
    """USD per million tokens, or None when not configured."""
    try:
        pin = os.getenv('GEMINI_PRICE_INPUT_PER_M', '').strip()
        pout = os.getenv('GEMINI_PRICE_OUTPUT_PER_M', '').strip()
        if not pin or not pout:
            return None
        return {'input': float(pin), 'output': float(pout)}
    except ValueError:
        return None


def cost_usd(input_tokens: int, output_tokens: int) -> Optional[float]:
    p = pricing()
    if not p:
        return None
    return round(input_tokens * p['input'] / 1e6 + output_tokens * p['output'] / 1e6, 8)


# ---------------------------------------------------------------------------
# Sinks (where records go)
# ---------------------------------------------------------------------------

class Collector:
    """Gathers records in memory. Used by the evaluation runner and by tests.

    `isolate=True` temporarily detaches every other sink (such as the database), so an evaluation run never leaks
    its synthetic or replayed traffic into the live production telemetry.
    """

    def __init__(self, isolate: bool = False):
        self.calls: List[Dict] = []
        self.analyses: List[Dict] = []
        self.isolate = isolate
        self._saved: List = []

    def on_call(self, rec: Dict) -> None:
        self.calls.append(rec)

    def on_analysis(self, rec: Dict) -> None:
        self.analyses.append(rec)

    def __enter__(self):
        if self.isolate:
            self._saved = list(_sinks)
            _sinks[:] = [self]
        else:
            add_sink(self)
        return self

    def __exit__(self, *exc):
        if self.isolate:
            _sinks[:] = self._saved
        else:
            remove_sink(self)
        return False


def add_sink(sink) -> None:
    if sink not in _sinks:
        _sinks.append(sink)


def remove_sink(sink) -> None:
    if sink in _sinks:
        _sinks.remove(sink)


def _emit(method: str, rec: Dict) -> None:
    for sink in list(_sinks):
        try:
            getattr(sink, method)(rec)
        except Exception:   # telemetry must never break an analysis
            logger.warning("Telemetry sink %r failed on %s", sink, method, exc_info=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _num(value) -> int:
    """Token counts can be MagicMocks or None in tests; only trust real numbers."""
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def usage_from(response) -> Dict[str, int]:
    meta = getattr(response, 'usage_metadata', None)
    if meta is None:
        return {'input': 0, 'output': 0}
    out = _num(getattr(meta, 'candidates_token_count', 0)) + _num(getattr(meta, 'thoughts_token_count', 0))
    return {'input': _num(getattr(meta, 'prompt_token_count', 0)), 'output': out}


# ---------------------------------------------------------------------------
# One model call
# ---------------------------------------------------------------------------

class Call:
    """Context manager around ONE model call (each retry is its own Call).

        with telemetry.call('symptom', model, 'symptom') as c:
            response = client.models.generate_content(...)
            c.api_done(response)
            c.valid(validated is not None)
    """

    def __init__(self, kind: str, model: str, prompt: Optional[str] = None):
        self.kind, self.model, self.prompt = kind, model, prompt
        self.api_ok: Optional[bool] = None
        self.is_valid: Optional[bool] = None
        self.error_type: Optional[str] = None
        self.input_tokens = 0
        self.output_tokens = 0
        self._t0 = 0.0
        self.latency_ms = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def api_done(self, response=None) -> None:
        self.latency_ms = (time.perf_counter() - self._t0) * 1000
        recorded = getattr(response, 'recorded_latency_ms', None)     # replayed responses carry their original latency
        if isinstance(recorded, (int, float)) and not isinstance(recorded, bool):
            self.latency_ms = float(recorded)
        self.api_ok = True
        u = usage_from(response)
        self.input_tokens, self.output_tokens = u['input'], u['output']

    def valid(self, ok: Optional[bool]) -> None:
        self.is_valid = ok
        if ok is False and self.error_type is None:
            self.error_type = 'invalid_output'

    def __exit__(self, et, ev, tb):
        if self.api_ok is None:                      # failed before the API answered
            self.latency_ms = (time.perf_counter() - self._t0) * 1000
            self.api_ok = False
            self.error_type = et.__name__ if et else 'no_response'
        elif et is not None and self.is_valid is None:
            self.is_valid = False
            self.error_type = self.error_type or et.__name__
        ctx = _current.get()
        if ctx is not None:
            ctx.calls.append(self)                   # flushed with the analysis, once validity is known
        else:
            _emit('on_call', self.as_record(None))
        return False                                 # never swallow the exception

    def as_record(self, analysis_id: Optional[str]) -> Dict:
        from quickaid.ai import prompt_registry
        try:
            pv = prompt_registry.prompt_version(self.prompt) if self.prompt else None
        except Exception:
            pv = None
        return {'id': uuid.uuid4().hex[:16], 'analysis_id': analysis_id, 'ts': _now(), 'kind': self.kind, 'model': self.model,
                'prompt_version': pv, 'ok': bool(self.api_ok), 'valid': self.is_valid, 'latency_ms': round(self.latency_ms, 1),
                'error_type': self.error_type, 'input_tokens': self.input_tokens, 'output_tokens': self.output_tokens,
                'cost_usd': cost_usd(self.input_tokens, self.output_tokens)}


def call(kind: str, model: str, prompt: Optional[str] = None) -> Call:
    return Call(kind, model, prompt)


def mark_last_valid(ok: Optional[bool]) -> None:
    """Record schema validity on the most recent call of the current analysis (when parsing happens later)."""
    ctx = _current.get()
    if ctx is not None and ctx.calls:
        ctx.calls[-1].valid(ok)


# ---------------------------------------------------------------------------
# One analysis (groups its calls, so cost-per-analysis is meaningful)
# ---------------------------------------------------------------------------

class _Ctx:
    def __init__(self, kind: str):
        self.id = uuid.uuid4().hex[:16]
        self.kind = kind
        self.calls: List[Call] = []


def analysis_scope(kind: str, summarize: Optional[Callable] = None):
    """Decorator: times the function, groups the model calls made inside it, and records one analysis row.

    `summarize(result)` may return {'mode', 'urgency', 'emergency', 'safety_raised', 'status', 'reliability', 'error'}.
    """
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            ctx = _Ctx(kind)
            token = _current.set(ctx)
            t0 = time.perf_counter()
            result, failed = None, False
            try:
                result = fn(*args, **kwargs)
                return result
            except Exception:
                failed = True
                raise
            finally:
                _current.reset(token)
                try:
                    _flush(ctx, result, failed, (time.perf_counter() - t0) * 1000, summarize)
                except Exception:
                    logger.warning("Telemetry flush failed", exc_info=True)
        return wrapper
    return deco


def _default_summary(result) -> Dict:
    if not isinstance(result, dict):
        return {}
    review = result.get('safety_review') or {}
    return {'mode': result.get('analysis_mode'), 'urgency': result.get('urgency_level') or result.get('urgency'),
            'emergency': bool((result.get('emergency_alert') or {}).get('alert')),
            'safety_raised': bool(review.get('urgency_raised')), 'status': result.get('assessment_status'),
            'reliability': (result.get('reliability') or {}).get('level'), 'error': bool(result.get('error'))}


def _flush(ctx: _Ctx, result, failed: bool, latency_ms: float, summarize) -> None:
    info = (summarize or _default_summary)(result) if not failed else {'error': True}
    recs = [c.as_record(ctx.id) for c in ctx.calls]
    for r in recs:
        _emit('on_call', r)
    tin, tout = sum(r['input_tokens'] for r in recs), sum(r['output_tokens'] for r in recs)
    any_bad = any((not r['ok']) or r['valid'] is False for r in recs)
    mode = info.get('mode')
    _emit('on_analysis', {
        'id': ctx.id, 'ts': _now(), 'kind': ctx.kind, 'mode': mode, 'urgency': info.get('urgency'),
        'emergency': bool(info.get('emergency')), 'safety_raised': bool(info.get('safety_raised')),
        'status': info.get('status'), 'reliability': info.get('reliability'), 'error': bool(info.get('error') or failed),
        'fallback': bool(recs and any_bad and mode in ('basic', 'rules')), 'ai_attempted': bool(recs),
        'latency_ms': round(latency_ms, 1), 'calls': len(recs), 'input_tokens': tin, 'output_tokens': tout,
        'cost_usd': cost_usd(tin, tout) if recs else None,
        'model': recs[0]['model'] if recs else None, 'prompt_version': recs[0]['prompt_version'] if recs else None,
    })


# ---------------------------------------------------------------------------
# Persisting to the app database
# ---------------------------------------------------------------------------

class DbSink:
    """Writes records to the app's SQLite database (see core/database.py)."""

    def on_call(self, rec: Dict) -> None:
        from quickaid.core import database as db
        db.record_ai_call(rec)

    def on_analysis(self, rec: Dict) -> None:
        from quickaid.core import database as db
        db.record_ai_analysis(rec)


_db_sink = DbSink()


def install_db_sink(retention_days: Optional[int] = None) -> None:
    """Called once by the web app. Also trims old rows (TELEMETRY_RETENTION_DAYS, default 90)."""
    add_sink(_db_sink)
    try:
        from quickaid.core import database as db
        days = retention_days or int(os.getenv('TELEMETRY_RETENTION_DAYS', '90') or 90)
        removed = db.purge_telemetry(days)
        if removed:
            logger.info("Telemetry retention: removed %d rows older than %d days", removed, days)
    except Exception:
        logger.warning("Telemetry retention cleanup failed", exc_info=True)
