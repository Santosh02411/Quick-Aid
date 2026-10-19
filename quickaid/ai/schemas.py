"""Pydantic schemas for everything an AI model returns, plus a safe JSON parser.

Two jobs:
  1. Tell Gemini the exact structure to answer in (used as response_schema).
  2. VALIDATE whatever comes back before anything else touches it: wrong types are
     rejected, lists are cleaned and length-capped, enums are enforced. If a response
     cannot be validated the caller returns a safe fallback instead of guessing.

The schemas deliberately SEPARATE four different kinds of statement so the person can
see what is fact, what is observation and what is inference:
  * reported information   - what the user told us (symptoms, duration, severity...)
  * observable findings    - what is visible in a photo (with how certain that is)
  * possible explanations  - hedged interpretations, each tied to its evidence
  * recommendations        - what to do, in order
"""

import json
import re
from typing import Any, Dict, List, Literal, Optional, Type, TypeVar

from pydantic import BaseModel, Field, ValidationError, field_validator

Level = Literal['low', 'medium', 'high']
Likelihood = Literal['more likely', 'possible', 'less likely']

MAX_ITEMS = 12
MAX_ITEM_CHARS = 300
MAX_TEXT_CHARS = 700

T = TypeVar('T', bound=BaseModel)


def _clean_str(value: Any, limit: int = MAX_TEXT_CHARS) -> str:
    if value is None:
        return ''
    text = re.sub(r'\s+', ' ', str(value)).strip()
    return text[:limit]


def _clean_list(values: Any, limit: int = MAX_ITEMS) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple)):
        raise ValueError('expected a list of strings')
    out, seen = [], set()
    for v in values:
        if isinstance(v, (dict, list)):
            raise ValueError('list items must be plain text')
        s = _clean_str(v, MAX_ITEM_CHARS)
        if s and s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
        if len(out) >= limit:
            break
    return out


class _Base(BaseModel):
    """Shared cleaning for every model-facing schema."""

    @field_validator('*', mode='before')
    @classmethod
    def _tidy(cls, v, info):
        ann = cls.model_fields[info.field_name].annotation
        origin = getattr(ann, '__origin__', None)
        if origin is list and ann.__args__ and ann.__args__[0] is str:
            return _clean_list(v)
        if ann is str:
            return _clean_str(v)
        return v


# ---------------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------------

class Explanation(_Base):
    """One possible explanation. `supported_by` ties it to what the person actually reported or what is visible."""
    name: str
    likelihood: Likelihood = 'possible'
    why: str = ''
    supported_by: List[str] = Field(default_factory=list)


class ImageFinding(_Base):
    """Something that can be SEEN in the photo. Objective description only, never a diagnosis."""
    description: str
    certainty: Literal['clearly visible', 'possibly visible'] = 'possibly visible'


# ---------------------------------------------------------------------------
# Symptom assessment (used by the one-shot checker and the interview)
# ---------------------------------------------------------------------------

class SymptomAssessment(_Base):
    summary: str = ''
    reported_symptoms: List[str] = Field(default_factory=list)
    possible_explanations: List[Explanation] = Field(default_factory=list)
    urgency_level: Level = 'medium'      # if the model omits it, never assume 'low'
    emergency_alert: bool = False
    emergency_reason: str = ''
    red_flags_present: List[str] = Field(default_factory=list)   # warning signs that ARE present now
    warning_signs: List[str] = Field(default_factory=list)       # signs to watch for that would make it an emergency
    timeframe: str = ''
    recommendations: List[str] = Field(default_factory=list)
    avoid: List[str] = Field(default_factory=list)
    see_doctor_when: List[str] = Field(default_factory=list)
    information_gaps: List[str] = Field(default_factory=list)    # what we could not tell, and what would help
    safety_tips: List[str] = Field(default_factory=list)
    # legacy fields, still accepted so older code and tests keep working
    detected_symptoms: List[str] = Field(default_factory=list)
    possible_conditions: List[str] = Field(default_factory=list)

    @field_validator('possible_explanations', mode='before')
    @classmethod
    def _explanations(cls, v):
        return (v or [])[:4]


# ---------------------------------------------------------------------------
# Image assessment
# ---------------------------------------------------------------------------

class ImageAssessment(_Base):
    image_type: Literal['injury_or_wound', 'burn', 'bruise_or_swelling', 'skin_condition',
                        'xray_or_scan', 'other_medical', 'not_medical'] = 'other_medical'
    image_quality: Literal['good', 'fair', 'poor'] = 'fair'
    quality_note: str = ''
    summary: str = ''
    findings: List[ImageFinding] = Field(default_factory=list)       # what is visible
    not_assessable: List[str] = Field(default_factory=list)          # what a photo cannot show
    possible_explanations: List[Explanation] = Field(default_factory=list)
    red_flags_present: List[str] = Field(default_factory=list)       # visible signs needing urgent care NOW
    warning_signs: List[str] = Field(default_factory=list)           # signs to watch for later
    confidence: Literal['low', 'medium', 'high'] = 'low'             # the model's own rough self-rating, NOT a probability
    urgency: Level = 'medium'
    recommendations: List[str] = Field(default_factory=list)
    avoid: List[str] = Field(default_factory=list)
    see_doctor_when: List[str] = Field(default_factory=list)
    safety_tips: List[str] = Field(default_factory=list)
    # legacy names, accepted so older code and tests keep working
    observations: List[str] = Field(default_factory=list)
    detected_conditions: List[str] = Field(default_factory=list)
    red_flags: List[str] = Field(default_factory=list)

    @field_validator('findings', mode='before')
    @classmethod
    def _findings(cls, v):
        out = []
        for item in (v or [])[:MAX_ITEMS]:
            out.append({'description': item} if isinstance(item, str) else item)
        return out

    @field_validator('possible_explanations', mode='before')
    @classmethod
    def _explanations(cls, v):
        return (v or [])[:3]


# ---------------------------------------------------------------------------
# Interview planning (which questions to ask)
# ---------------------------------------------------------------------------

class PlanOption(_Base):
    label: str
    flag: Literal['none', 'urgent', 'emergency'] = 'none'


class PlanQuestion(_Base):
    text: str
    type: Literal['single', 'multi'] = 'single'
    options: List[PlanOption]
    why: str = ''


class InterviewPlan(_Base):
    detected_symptoms: List[str] = Field(default_factory=list)
    associated: PlanQuestion
    clarifying: List[PlanQuestion] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def extract_json_object(text: Any) -> Optional[Dict]:
    """Pull a JSON object out of model text (handles ```json fences and chatter around it)."""
    if not isinstance(text, str) or not text.strip():
        return None
    cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip(), flags=re.I)
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = cleaned.find('{'), cleaned.rfind('}')
    if start != -1 and end > start:
        try:
            data = json.loads(cleaned[start:end + 1])
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def validate_response(response: Any, model: Type[T]) -> Optional[T]:
    """Return a validated model instance from a Gemini response, or None if it cannot be trusted.

    Prefers the SDK's already-parsed object, but re-validates it; falls back to the raw text.
    Never raises.
    """
    candidates: List[Any] = []
    parsed = getattr(response, 'parsed', None)
    if isinstance(parsed, BaseModel):
        candidates.append(parsed.model_dump())
    elif isinstance(parsed, dict):
        candidates.append(parsed)
    raw = extract_json_object(getattr(response, 'text', None))
    if raw is not None:
        candidates.append(raw)
    for data in candidates:
        try:
            return model.model_validate(data)
        except (ValidationError, ValueError, TypeError):
            continue
    return None
