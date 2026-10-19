"""Metrics computed from telemetry records. One implementation, used by the live dashboard and by evaluation runs."""

from collections import defaultdict
from typing import Dict, Iterable, List, Optional

# analyses that represent one finished piece of advice for a person (used for "cost per analysis")
COMPLETED_KINDS = ('symptom', 'image', 'interview_assess', 'chat')


def percentile(values: Iterable[float], p: float) -> Optional[float]:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return round(vals[0], 1)
    k = (len(vals) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(vals) - 1)
    return round(vals[lo] + (vals[hi] - vals[lo]) * (k - lo), 1)


def _rate(num: int, den: int) -> Optional[float]:
    return None if den == 0 else round(num / den, 4)


def _call_block(calls: List[Dict]) -> Dict:
    ok = [c for c in calls if c.get('ok')]
    checked = [c for c in ok if c.get('valid') is not None]
    lat = [c['latency_ms'] for c in ok if c.get('latency_ms') is not None]
    costs = [c['cost_usd'] for c in calls if c.get('cost_usd') is not None]
    return {
        'calls': len(calls),
        'api_failure_rate': _rate(sum(1 for c in calls if not c.get('ok')), len(calls)),
        'structured_output_validity': _rate(sum(1 for c in checked if c.get('valid')), len(checked)),
        'invalid_outputs': sum(1 for c in checked if not c.get('valid')),
        'latency_ms': {'p50': percentile(lat, 50), 'p95': percentile(lat, 95),
                       'mean': round(sum(lat) / len(lat), 1) if lat else None, 'max': round(max(lat), 1) if lat else None},
        'input_tokens': sum(c.get('input_tokens') or 0 for c in calls),
        'output_tokens': sum(c.get('output_tokens') or 0 for c in calls),
        'cost_usd': round(sum(costs), 6) if costs else None,
        'errors': dict(sorted(_count(c.get('error_type') for c in calls if c.get('error_type')).items(), key=lambda kv: -kv[1])),
    }


def _count(items) -> Dict[str, int]:
    out: Dict[str, int] = defaultdict(int)
    for i in items:
        out[i] += 1
    return dict(out)


def summarize(calls: List[Dict], analyses: List[Dict]) -> Dict:
    """Everything the dashboard shows, from raw call + analysis records."""
    overall = _call_block(calls)
    n = len(analyses)
    completed = [a for a in analyses if a.get('kind') in COMPLETED_KINDS]
    attempted = [a for a in analyses if a.get('ai_attempted')]
    total_cost = overall['cost_usd']
    lat_e2e = [a['latency_ms'] for a in analyses if a.get('latency_ms') is not None and a.get('ai_attempted')]

    by_kind = {}
    for kind in sorted({c['kind'] for c in calls}):
        by_kind[kind] = _call_block([c for c in calls if c['kind'] == kind])
    by_model = {m or 'unknown': _call_block([c for c in calls if (c.get('model') or 'unknown') == (m or 'unknown')])
                for m in sorted({c.get('model') for c in calls}, key=lambda x: x or '')}
    by_prompt = {p or 'unknown': _call_block([c for c in calls if (c.get('prompt_version') or 'unknown') == (p or 'unknown')])
                 for p in sorted({c.get('prompt_version') for c in calls}, key=lambda x: x or '')}

    days: Dict[str, Dict] = {}
    for c in calls:
        d = days.setdefault(c['ts'][:10], {'day': c['ts'][:10], 'calls': 0, 'failures': 0, 'invalid': 0})
        d['calls'] += 1
        d['failures'] += 0 if c.get('ok') else 1
        d['invalid'] += 1 if c.get('valid') is False else 0

    return {
        **overall,
        'analyses': n,
        'completed_analyses': len(completed),
        'by_analysis_kind': _count(a['kind'] for a in analyses),
        'by_mode': _count(a.get('mode') or 'unknown' for a in analyses),
        'ai_attempted': len(attempted),
        'fallback_rate': _rate(sum(1 for a in attempted if a.get('fallback')), len(attempted)),
        'error_rate': _rate(sum(1 for a in analyses if a.get('error')), n),
        'safety_intervention_rate': _rate(sum(1 for a in analyses if a.get('safety_raised')), n),
        'emergency_rate': _rate(sum(1 for a in analyses if a.get('emergency')), n),
        'end_to_end_latency_ms': {'p50': percentile(lat_e2e, 50), 'p95': percentile(lat_e2e, 95)},
        'cost_per_analysis_usd': (round(total_cost / len(completed), 6) if (total_cost is not None and completed) else None),
        'cost_configured': total_cost is not None,
        'by_kind': by_kind, 'by_model': by_model, 'by_prompt_version': by_prompt,
        'daily': [days[k] for k in sorted(days)],
    }
