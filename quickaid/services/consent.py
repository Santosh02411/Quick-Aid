"""Whether a person has agreed to their text and photos being sent to the external AI provider (Google Gemini)."""

from typing import Dict, Optional

from quickaid import config
from quickaid.core import database as db
from quickaid.services import security
from quickaid.services.container import get_services


def allowed_for(user_row: Optional[Dict]) -> bool:
    """May this person's data go to the external AI? True when consent is not required (operator choice) or they agreed to the CURRENT wording."""
    if not config.require_ai_consent():
        return True
    return bool(user_row) and user_row.get('ai_consent_version') == config.AI_CONSENT_VERSION


def status(user_id: int) -> Dict:
    row = db.get_user_by_id(user_id)
    return {'required': config.require_ai_consent(), 'ai_available': bool(get_services().symptom_checker.use_gemini),
            'granted': allowed_for(row), 'decided': bool(row) and row.get('ai_consent_version') in (0, config.AI_CONSENT_VERSION), 'version': config.AI_CONSENT_VERSION, 'granted_at': row.get('ai_consent_at') if row else None,
            'provider': 'Google Gemini API'}


def set_consent(user_id: int, granted: bool, record_event) -> Dict:
    db.set_ai_consent(user_id, config.AI_CONSENT_VERSION if granted else 0)       # 0 = declined: remembered, not asked again
    record_event(user_id, 'ai_consent_granted' if granted else 'ai_consent_withdrawn')
    return status(user_id)
