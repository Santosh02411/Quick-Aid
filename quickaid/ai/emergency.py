"""Compatibility shim: emergency guidance now lives in triage_engine (the single source of safety logic)."""

from quickaid.ai.triage_engine import (  # noqa: F401
    DEFAULT_STEPS, EMERGENCY_ACTION, EMERGENCY_MESSAGE, GUIDANCE, build_emergency, scan_red_flags,
)
from quickaid.ai import triage_engine as _t
from typing import Dict, List, Optional


def check_text(text: str) -> Optional[Dict]:
    """Scan free text. Returns None when nothing alarming, otherwise the flags dict."""
    sig = _t.scan(text)
    return sig.as_dict() if sig.any_emergency else None


def as_analysis(reasons: List[str], region: str, crisis: bool, summary: str) -> Dict:
    return _t.emergency_analysis(reasons, region, crisis, summary)
