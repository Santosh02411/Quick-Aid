#!/usr/bin/env python
"""Manage versioned evaluation datasets.

    python evaluation/manage_datasets.py list
    python evaluation/manage_datasets.py check              # exits 1 if any dataset changed without a version bump (used in CI)
    python evaluation/manage_datasets.py bump NAME --note "what changed" [--part patch|minor|major] [--force]

A run records the dataset versions and hashes it used, so results are only ever compared like-for-like.
Frozen held-out test sets refuse to be bumped (that is how test sets quietly turn into training sets).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quickaid.evaluation import datasets  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('list')
    sub.add_parser('check')
    b = sub.add_parser('bump')
    b.add_argument('name')
    b.add_argument('--note', required=True)
    b.add_argument('--part', choices=['patch', 'minor', 'major'], default='minor')
    b.add_argument('--force', action='store_true', help='allow bumping a frozen held-out set (you are changing your test set!)')
    args = ap.parse_args(argv)

    if args.cmd == 'list':
        for r in datasets.describe():
            flag = '' if r['status'] == 'ok' else '  << CHANGED WITHOUT A VERSION BUMP'
            print(f"{r['name']:20s} v{r['version']:7s} {r['cases']:4d} cases  {r['role']}{' [frozen]' if r['frozen'] else ''}{flag}")
        return 0
    if args.cmd == 'check':
        problems = datasets.verify()
        for p in problems:
            print('PROBLEM:', p)
        print('datasets OK' if not problems else f'{len(problems)} problem(s)')
        return 1 if problems else 0
    try:
        meta = datasets.bump(args.name, args.note, args.part, args.force)
    except (KeyError, PermissionError) as e:
        print('error:', e.args[0] if e.args else e)
        return 1
    print(f"{args.name} is now v{meta['version']} ({meta['cases']} cases)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
