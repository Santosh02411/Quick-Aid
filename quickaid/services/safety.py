"""Medical-safety façade for the web layer.

The safety logic itself lives in `quickaid.ai.triage_engine` (emergency detection, urgency classification, validation of AI output).
This module is the ONLY door the web layer uses to reach it, so a view or service never has to know how safety is implemented,
and nothing can be shown or saved without passing through it.
"""

from typing import Dict, Optional

from quickaid.ai.triage_engine import engine as triage, immediate_emergency as _immediate_emergency
from quickaid.core.localization import get_app_region


def review_symptom_result(result: Dict, text: str, region: Optional[str] = None) -> Dict:
    return triage.final_gate(result, 'symptom', region or get_app_region(), text=text)


def review_image_result(result: Dict, region: Optional[str] = None) -> Dict:
    return triage.final_gate(result, 'image', region or get_app_region())


def last_resort_emergency_check(text: str, summary: str, region: Optional[str] = None) -> Optional[Dict]:
    """A pure, last-ditch text scan for emergency wording, for callers that only need this one check (typically a
    crash-recovery path that has nothing else left to try). Never raises. See `immediate_emergency` for details."""
    return _immediate_emergency(text, region or get_app_region(), summary)
