"""Versioned evaluation runs: one JSON file per run in evaluation/runs/.

A run records WHAT was evaluated (model, prompt fingerprint, safety-engine version, dataset versions + hashes, code version)
and WHAT HAPPENED (metrics, telemetry, per-case outcomes), so any two runs can be compared and a regression can be traced to
the change that caused it. Files are plain JSON, so they diff and commit like code.
"""

import json
import re
import secrets
from datetime import datetime, timezone
from typing import Dict, List, Optional

from quickaid.evaluation import datasets

RUNS_DIR = datasets.EVAL_DIR / 'runs'
RUN_ID_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{5,79}$')

# metric -> direction. True = higher is better, False = lower is better, None = informational.
DIRECTIONS = {
    'emergency_recall': True, 'false_negative_rate': False, 'false_positive_rate': False, 'precision': True,
    'under_triage': False, 'severe_under_triage': False, 'over_triage': False, 'abstention': None,
    'structured_output_validity': True, 'api_failure_rate': False, 'fallback_rate': False,
    'latency_p50_ms': False, 'latency_p95_ms': False, 'cost_per_analysis_usd': False, 'safety_intervention_rate': None,
}
METRIC_LABELS = {
    'emergency_recall': 'Emergency recall', 'false_negative_rate': 'False-negative rate (missed emergencies)',
    'false_positive_rate': 'False-positive rate (false alarms)', 'precision': 'Emergency precision',
    'under_triage': 'Under-triage', 'severe_under_triage': 'Severe under-triage', 'over_triage': 'Over-triage',
    'abstention': 'Abstention', 'structured_output_validity': 'Structured-output validity', 'api_failure_rate': 'API failure rate',
    'fallback_rate': 'Fallback to basic mode', 'latency_p50_ms': 'Latency p50 (ms)', 'latency_p95_ms': 'Latency p95 (ms)',
    'cost_per_analysis_usd': 'Cost per analysis (USD)', 'safety_intervention_rate': 'Safety engine raised urgency',
}


def new_run_id(mode: str) -> str:
    safe = re.sub(r'[^A-Za-z0-9]+', '-', mode).strip('-')[:20] or 'run'
    return f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{safe}-{secrets.token_hex(4)}"


def save_run(record: Dict) -> str:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = record['run_id']
    if not RUN_ID_RE.match(run_id):
        raise ValueError('invalid run id')
    (RUNS_DIR / f'{run_id}.json').write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding='utf-8')
    return run_id


def headline(record: Dict) -> Dict:
    """The few numbers shown in lists and charts."""
    m = record.get('metrics', {})
    return {k: m.get(k) for k in DIRECTIONS if k in m}


def list_runs() -> List[Dict]:
    """Run summaries, newest first. Unreadable files are skipped rather than breaking the dashboard."""
    out = []
    if not RUNS_DIR.exists():
        return out
    for f in RUNS_DIR.glob('*.json'):
        try:
            r = json.loads(f.read_text(encoding='utf-8'))
            out.append({'run_id': r['run_id'], 'created': r['created'], 'mode': r.get('mode'), 'label': r.get('label', ''),
                        'model': (r.get('model') or {}).get('name'), 'synthetic_model': (r.get('model') or {}).get('synthetic', False),
                        'prompt_fingerprint': (r.get('prompts') or {}).get('fingerprint'), 'prompt_label': (r.get('prompts') or {}).get('label'),
                        'engine_version': r.get('engine_version'), 'sets': r.get('sets', []),
                        'datasets': {k: v.get('version') for k, v in (r.get('datasets') or {}).items()},
                        'metrics': headline(r), 'threshold_failures': r.get('threshold_failures', []), 'gated': r.get('gated', False),
                        'cases': r.get('cases_evaluated')})
        except (OSError, ValueError, KeyError):
            continue
    return sorted(out, key=lambda x: x['created'], reverse=True)


def load_run(run_id: str) -> Optional[Dict]:
    """Load one run. The id comes from a URL, so it is validated before it touches the filesystem."""
    if not isinstance(run_id, str) or not RUN_ID_RE.match(run_id):
        return None
    path = (RUNS_DIR / f'{run_id}.json').resolve()
    if RUNS_DIR.resolve() not in path.parents or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def _delta(a, b, higher_better):
    if a is None or b is None:
        return {'delta': None, 'verdict': 'n/a'}
    d = round(b - a, 6)
    if d == 0 or higher_better is None:
        return {'delta': d, 'verdict': 'same' if d == 0 else 'info'}
    return {'delta': d, 'verdict': 'better' if (d > 0) == higher_better else 'worse'}


def compare(a: Dict, b: Dict) -> Dict:
    """A (baseline) vs B (candidate): metric deltas, what changed between them, and whether the comparison is fair."""
    from quickaid.ai import prompt_registry
    ma, mb = a.get('metrics', {}), b.get('metrics', {})
    rows = []
    for key, higher in DIRECTIONS.items():
        if key in ma or key in mb:
            rows.append({'metric': key, 'label': METRIC_LABELS.get(key, key), 'a': ma.get(key), 'b': mb.get(key),
                         'higher_is_better': higher, **_delta(ma.get(key), mb.get(key), higher)})

    da, db_ = a.get('datasets') or {}, b.get('datasets') or {}
    dataset_diff = {}
    for name in sorted(set(da) | set(db_)):
        va, vb = (da.get(name) or {}), (db_.get(name) or {})
        if va.get('version') != vb.get('version') or va.get('sha256') != vb.get('sha256'):
            dataset_diff[name] = {'a': va.get('version'), 'b': vb.get('version')}
    comparable = not dataset_diff and sorted(a.get('sets', [])) == sorted(b.get('sets', []))

    oa = {o['id']: o for o in a.get('outcomes', [])}
    ob = {o['id']: o for o in b.get('outcomes', [])}
    regress, improve, changed = [], [], []
    for cid in sorted(set(oa) & set(ob)):
        pa, pb = oa[cid], ob[cid]
        ok_a, ok_b = not pa.get('problems'), not pb.get('problems')
        row = {'id': cid, 'text': pb.get('text') or pa.get('text'), 'category': pb.get('category'),
               'a': {'level': pa.get('level'), 'alert': pa.get('alert'), 'problems': pa.get('problems')},
               'b': {'level': pb.get('level'), 'alert': pb.get('alert'), 'problems': pb.get('problems')}}
        if ok_a and not ok_b:
            regress.append(row)
        elif ok_b and not ok_a:
            improve.append(row)
        elif (pa.get('level'), pa.get('alert')) != (pb.get('level'), pb.get('alert')):
            changed.append(row)

    warnings = []
    if dataset_diff:
        warnings.append('The runs used different dataset versions, so metric differences may come from the data, not the system.')
    elif sorted(a.get('sets', [])) != sorted(b.get('sets', [])):
        warnings.append('The runs evaluated different dataset sets.')
    for r, name in ((a, 'A'), (b, 'B')):
        if (r.get('model') or {}).get('synthetic'):
            warnings.append(f'Run {name} used a synthetic mock model: its numbers describe the safety rules, not a real AI model.')
        if r.get('mode') == 'offline':
            pass
    if (a.get('model') or {}).get('name') != (b.get('model') or {}).get('name') and (a.get('prompts') or {}).get('fingerprint') != (b.get('prompts') or {}).get('fingerprint'):
        warnings.append('Both the model and the prompts changed, so you cannot tell which one caused a difference.')

    return {
        'a': {k: a.get(k) for k in ('run_id', 'created', 'mode', 'label')}, 'b': {k: b.get(k) for k in ('run_id', 'created', 'mode', 'label')},
        'metrics': rows, 'comparable': comparable, 'warnings': warnings, 'dataset_diff': dataset_diff,
        'prompt_diff': prompt_registry.diff(a.get('prompts') or {}, b.get('prompts') or {}),
        'model': {'a': (a.get('model') or {}).get('name'), 'b': (b.get('model') or {}).get('name')},
        'engine_version': {'a': a.get('engine_version'), 'b': b.get('engine_version')},
        'regressions': regress, 'improvements': improve, 'changed': changed,
    }
