"""Versioned evaluation datasets.

`evaluation/datasets/MANIFEST.json` records, for every dataset file: a semantic version, a SHA-256 of the
file, the number of cases, its role (dev / held-out / guardrail / synthetic), who labelled it, whether a
clinician reviewed it, and a changelog. A run stores the versions + hashes it used, so results are only ever
compared like-for-like. If a file changes without a version bump, `verify()` reports it (and a test fails).

Frozen datasets (held-out test sets) cannot be bumped without force: tuning prompts or rules against a test
set quietly turns it into a training set.
"""

import hashlib
import json
import os
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

from quickaid import PROJECT_ROOT

EVAL_DIR = Path(os.getenv('EVAL_DIR') or (PROJECT_ROOT / 'evaluation'))
DATASET_DIR = EVAL_DIR / 'datasets'
MANIFEST_PATH = DATASET_DIR / 'MANIFEST.json'


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def count_cases(path: Path) -> int:
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        return len(data.get('cases', []))
    except (OSError, json.JSONDecodeError):
        return 0


def load_manifest() -> Dict:
    if not MANIFEST_PATH.exists():
        return {'datasets': {}}
    return json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))


def save_manifest(manifest: Dict) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')


def verify() -> List[str]:
    """Problems found: missing files, edited-without-bump files, wrong case counts, unlisted dataset files."""
    problems: List[str] = []
    ds = load_manifest().get('datasets', {})
    for name, meta in ds.items():
        path = DATASET_DIR / meta['file']
        if not path.exists():
            problems.append(f'{name}: file {meta["file"]} is missing')
            continue
        if sha256_of(path) != meta['sha256']:
            problems.append(f'{name}: file changed since version {meta["version"]} was recorded (bump the version with manage_datasets.py)')
        if count_cases(path) != meta['cases']:
            problems.append(f'{name}: manifest says {meta["cases"]} cases but the file has {count_cases(path)}')
    listed = {m['file'] for m in ds.values()}
    for f in sorted(DATASET_DIR.glob('*.json')):
        if f.name != 'MANIFEST.json' and f.name not in listed:
            problems.append(f'{f.name}: not listed in MANIFEST.json')
    return problems


def versions(names: Optional[List[str]] = None) -> Dict[str, Dict]:
    """{name: {version, sha256, cases, role}} for the datasets a run used (all when names is None)."""
    ds = load_manifest().get('datasets', {})
    out = {}
    for name, meta in ds.items():
        if names is None or name in names:
            path = DATASET_DIR / meta['file']
            current = sha256_of(path) if path.exists() else None
            out[name] = {'version': meta['version'], 'sha256': current, 'cases': meta['cases'], 'role': meta.get('role'),
                         'matches_manifest': current == meta['sha256']}
    return out


def describe() -> List[Dict]:
    """Rows for the dashboard's dataset panel."""
    ds = load_manifest().get('datasets', {})
    rows = []
    problems = ' '.join(verify())
    for name, meta in ds.items():
        path = DATASET_DIR / meta['file']
        current = sha256_of(path) if path.exists() else None
        rows.append({**{k: meta.get(k) for k in ('version', 'role', 'cases', 'labelled_by', 'clinician_reviewed', 'frozen', 'purpose')},
                     'name': name, 'file': meta['file'], 'sha256': (meta['sha256'] or '')[:12],
                     'status': 'ok' if current == meta['sha256'] else 'changed-without-bump',
                     'changelog': meta.get('changelog', [])[-5:]})
    return rows


def bump(name: str, note: str, part: str = 'minor', force: bool = False) -> Dict:
    """Record a dataset change: new version, hash, case count and a changelog entry."""
    manifest = load_manifest()
    meta = manifest['datasets'][name]
    if meta.get('frozen') and not force:
        raise PermissionError(f'{name} is frozen (a held-out test set). Add new cases to a NEW dataset instead, or use force.')
    major, minor, patch = (int(x) for x in meta['version'].split('.'))
    major, minor, patch = {'major': (major + 1, 0, 0), 'minor': (major, minor + 1, 0), 'patch': (major, minor, patch + 1)}[part]
    path = DATASET_DIR / meta['file']
    meta.update({'version': f'{major}.{minor}.{patch}', 'sha256': sha256_of(path), 'cases': count_cases(path)})
    meta.setdefault('changelog', []).append({'version': meta['version'], 'date': date.today().isoformat(), 'note': note})
    save_manifest(manifest)
    return meta
