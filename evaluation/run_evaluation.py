#!/usr/bin/env python
"""Command-line entry point. The implementation lives in quickaid/evaluation/runner.py so the dashboard can use it too.

    python evaluation/run_evaluation.py --help
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quickaid.evaluation.runner import *  # noqa: F401,F403,E402
from quickaid.evaluation.runner import DATA, DOSE_RE, RANK, THRESHOLDS, _Resp, main  # noqa: F401,E402

if __name__ == '__main__':
    sys.exit(main())
