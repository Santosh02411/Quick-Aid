#!/usr/bin/env python
"""Find secrets before they leave your machine. Findings are always MASKED (first 4 characters + length), never printed in full.

    python scripts/secret_scan.py                 # working tree: tracked files + new files git would add (ignored files are skipped)
    python scripts/secret_scan.py --staged        # exactly what is about to be committed (used by the pre-commit hook)
    python scripts/secret_scan.py --history       # every commit on every branch: added lines + forbidden file names
    python scripts/secret_scan.py --archive X.zip # a finished release archive
    python scripts/secret_scan.py --ignored       # also report ignored local secret files (e.g. your real .env): informational

Exit status 1 when anything is found, so CI and the release script can refuse to continue.
Add `# secret-scan: allow` to a line that is deliberately a fake value.
"""

import argparse
import fnmatch
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]

# Specific, low-false-positive shapes of real credentials.
PATTERNS = {
    'Google API key': re.compile(r'AIza[0-9A-Za-z_\-]{35}'),
    'Google OAuth client secret': re.compile(r'GOCSPX-[0-9A-Za-z_\-]{20,}'),
    'Private key block': re.compile(r'-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----'),
    'AWS access key id': re.compile(r'\bAKIA[0-9A-Z]{16}\b'),
    'GitHub token': re.compile(r'\bgh[pousr]_[0-9A-Za-z]{30,}\b'),
    'Slack token': re.compile(r'\bxox[baprs]-[0-9A-Za-z-]{10,}'),
    'OpenAI-style key': re.compile(r'\bsk-[A-Za-z0-9]{32,}\b'),
}
# `NAME=value` in configuration files (never in source code, where tests legitimately use fake values).
ASSIGNMENT = re.compile(r'^\s*([A-Z0-9_]*(?:SECRET|PASSWORD|API_KEY|TOKEN|PRIVATE_KEY|ENCRYPTION_KEY)[A-Z0-9_]*)\s*[=:]\s*[\'"]?([^\s\'"#]+)', re.I)
CONFIG_FILES = ('.env', '.env.*', '*.ini', '*.cfg', '*.toml', 'docker-compose*.yml', '*.env')
PLACEHOLDER = re.compile(r'your_|change_this|example|placeholder|xxxx|<[^>]+>|\$\{|^\$|^none$|^null$|^true$|^false$|^$', re.I)
FORBIDDEN_NAMES = ['.env', '.env.local', '.env.production', '*.db', '*.sqlite', '*.sqlite3', '*.db-wal', '*.db-shm', '.data_key', '*.pem', '*.key',
                   'id_rsa', 'id_ed25519', '*.p12', '*.pfx', '*.jks', 'credentials.json', 'service-account*.json', '*.db.gz', '*.deletions.log']
FORBIDDEN_DIRS = ('.git/', 'backups/', 'node_modules/', '__pycache__/')
ALLOWED_NAMES = ['.env.example']
SKIP_SUFFIXES = ('.png', '.jpg', '.jpeg', '.gif', '.webp', '.ico', '.woff', '.woff2', '.ttf', '.zip', '.gz', '.pyc', '.svg')

Finding = Tuple[str, str, str]      # (where, what, masked)


def looks_fake(value: str) -> bool:
    """`GOCSPX-xxxxxxxxxxxxxxxxxxxx` and friends: a body with almost no variety is documentation, not a credential."""
    body = re.sub(r'^[A-Za-z]{2,8}[-_]?', '', value)
    return len(set(body.lower())) <= 3


def mask(value: str) -> str:
    return f'{value[:4]}… ({len(value)} chars)'


def forbidden_name(path: str) -> bool:
    p = path.replace('\\', '/')
    base = p.rsplit('/', 1)[-1]
    if base in ALLOWED_NAMES:
        return False
    if any(('/' + p).find('/' + d) >= 0 for d in FORBIDDEN_DIRS if d != 'backups/') or any(f'/{d}' in '/' + p for d in ('backups/',)):
        return True
    return any(fnmatch.fnmatch(base, pat) for pat in FORBIDDEN_NAMES)


def scan_text(where: str, text: str) -> List[Finding]:
    out: List[Finding] = []
    is_config = any(fnmatch.fnmatch(where.rsplit('/', 1)[-1], pat) for pat in CONFIG_FILES) and where.rsplit('/', 1)[-1] != '.env.example'
    for n, line in enumerate(text.splitlines(), 1):
        if 'secret-scan: allow' in line:
            continue
        for name, rx in PATTERNS.items():
            for m in rx.finditer(line):
                if looks_fake(m.group(0)):
                    continue
                out.append((f'{where}:{n}', name, mask(m.group(0))))
        if is_config:
            m = ASSIGNMENT.match(line)
            if m and len(m.group(2)) >= 12 and not PLACEHOLDER.search(m.group(2)):
                out.append((f'{where}:{n}', f'{m.group(1)} has a real-looking value', mask(m.group(2))))
    return out


def git(*args: str, check: bool = True) -> str:
    r = subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True, errors='replace')
    if check and r.returncode:
        raise RuntimeError(r.stderr.strip())
    return r.stdout


def scan_files(paths: Iterable[str], reader) -> List[Finding]:
    out: List[Finding] = []
    for rel in paths:
        if forbidden_name(rel):
            out.append((rel, 'forbidden file (must never be committed or shipped)', ''))
            continue
        if rel.lower().endswith(SKIP_SUFFIXES):
            continue
        try:
            data = reader(rel)
        except (OSError, KeyError):
            continue
        if isinstance(data, bytes):
            if b'\x00' in data[:2048]:
                continue
            data = data.decode('utf-8', errors='replace')
        out += scan_text(rel, data)
    return out


_SKIP_DIRS = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', '.pytest_cache', '.ruff_cache', 'data', 'logs', 'uploads', 'backups'}


def _in_git_repo() -> bool:
    try:
        git('rev-parse', '--git-dir')
        return True
    except RuntimeError:
        return False


def _walk_source_files() -> List[str]:
    """For an unzipped copy (no git): the source files only. Local runtime files that git would have ignored (a person's own .env, databases, logs,
    uploads, keys) are skipped, because without git there is no way to tell them from files that were meant to be shared."""
    out = []
    for path in sorted(ROOT.rglob('*')):
        rel = path.relative_to(ROOT)
        if not path.is_file() or any(part in _SKIP_DIRS for part in rel.parts[:-1]):
            continue
        name = rel.name
        if (name.startswith('.env') and name != '.env.example') or name.endswith(('.db', '.db-wal', '.db-shm', '.pem', '.key', '.deletions.log')) or name == '.data_key':
            continue
        out.append(rel.as_posix())
    return out


def scan_worktree(include_ignored: bool = False) -> List[Finding]:
    if not _in_git_repo():
        return scan_files(_walk_source_files(), lambda r: (ROOT / r).read_bytes())
    files = git('ls-files', '-co', '--exclude-standard').splitlines()
    found = scan_files(files, lambda r: (ROOT / r).read_bytes())
    if include_ignored:
        for rel in git('ls-files', '-o', '-i', '--exclude-standard').splitlines():
            if rel.rsplit('/', 1)[-1].startswith('.env') and (ROOT / rel).is_file():
                for where, what, masked in scan_text(rel, (ROOT / rel).read_text(errors='replace')):
                    found.append((where + ' (ignored local file)', what, masked))
    return found


def scan_staged() -> List[Finding]:
    files = git('diff', '--cached', '--name-only', '--diff-filter=ACM').splitlines()
    return scan_files(files, lambda r: subprocess.run(['git', 'show', f':{r}'], cwd=ROOT, capture_output=True).stdout)


def scan_history() -> List[Finding]:
    out: List[Finding] = []
    seen_names = set()
    for line in git('log', '--all', '--name-only', '--format=%h', '--diff-filter=A').splitlines():
        if line and '/' not in line[:1] and len(line) > 12 or (line and not re.fullmatch(r'[0-9a-f]{7,}', line)):
            if forbidden_name(line) and line not in seen_names:
                seen_names.add(line)
                out.append((f'history:{line}', 'forbidden file was committed at some point', ''))
    commit = ''
    current_file = ''
    dupes = set()
    for line in git('log', '--all', '-p', '--format=@@commit %h', '--no-color').splitlines():
        if line.startswith('@@commit '):
            commit = line.split()[1]
        elif line.startswith('+++ b/'):
            current_file = line[6:]
        elif line.startswith('+') and not line.startswith('+++'):
            if current_file.lower().endswith(SKIP_SUFFIXES):
                continue
            for where, what, masked in scan_text(current_file, line[1:]):
                key = (current_file, what, masked)
                if key not in dupes:
                    dupes.add(key)
                    out.append((f'{current_file} @ commit {commit}', what, masked))
    return out


def scan_archive(path: str) -> List[Finding]:
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if not n.endswith('/')]
        return scan_files(names, lambda r: z.read(r))


def report(findings: List[Finding], title: str) -> int:
    if not findings:
        print(f'OK: {title}: no secrets or forbidden files found')
        return 0
    print(f'FOUND {len(findings)} problem(s) in {title}:')
    for where, what, masked in findings[:60]:
        print(f'  - {where}: {what} {masked}')
    if len(findings) > 60:
        print(f'  ... and {len(findings) - 60} more')
    return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--staged', action='store_true')
    ap.add_argument('--history', action='store_true')
    ap.add_argument('--archive')
    ap.add_argument('--ignored', action='store_true', help='also inspect ignored local files such as your real .env (informational)')
    a = ap.parse_args(argv)
    if a.archive:
        return report(scan_archive(a.archive), a.archive)
    if a.staged:
        return report(scan_staged(), 'staged changes')
    if a.history:
        return report(scan_history(), 'git history (all branches)')
    return report(scan_worktree(a.ignored), 'working tree')


if __name__ == '__main__':
    sys.exit(main())
