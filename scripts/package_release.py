#!/usr/bin/env python
"""Build a release archive that can safely be shared or uploaded.

    python scripts/package_release.py --version 7 [--out /path/to/dir]

The archive contains ONLY files git would publish (tracked + new, honouring .gitignore): never `.env`, never the `.git` history,
never databases, backups or keys. It is then scanned for secrets and the build FAILS (no archive is left behind) if anything is found.
"""

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import secret_scan  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', required=True)
    ap.add_argument('--out', default=str(ROOT.parent))
    ap.add_argument('--name', default='Quick-Aid')
    a = ap.parse_args(argv)
    files = subprocess.run(['git', 'ls-files', '-co', '--exclude-standard'], cwd=ROOT, capture_output=True, text=True).stdout.splitlines()
    files = [f for f in files if (ROOT / f).is_file() and not secret_scan.forbidden_name(f)]
    target = Path(a.out) / f'{a.name}-v{a.version}.zip'
    tmp = target.with_suffix('.zip.partial')
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):
            z.write(ROOT / f, f'{a.name}/{f}')
    if secret_scan.report(secret_scan.scan_archive(str(tmp)), f'{target.name} (before release)'):
        tmp.unlink()
        print('Refusing to release: remove the secrets above first.')
        return 1
    tmp.replace(target)
    print(f'built {target} ({len(files)} files, {target.stat().st_size // 1024} KB), no .env, no .git')
    return 0


if __name__ == '__main__':
    sys.exit(main())
