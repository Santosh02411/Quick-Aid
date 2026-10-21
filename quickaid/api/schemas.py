"""Pydantic models for request validation AND the OpenAPI document (one definition, two jobs)."""

from datetime import date
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from quickaid import config


class HistoryQuery(BaseModel):
    """Filters for GET /api/history. All optional; they combine with AND."""
    q: str = Field('', max_length=100, description='Words that must ALL appear in the entry (symptoms, conditions, advice, your notes).')
    type: List[Literal['symptom', 'image']] = Field(default_factory=list, description='symptom and/or image. Comma-separated or repeated.')
    urgency: List[Literal['low', 'medium', 'high']] = Field(default_factory=list)
    emergency: Optional[bool] = Field(None, description='true = only emergencies, false = only non-emergencies')
    date_from: Optional[str] = Field(None, description='YYYY-MM-DD, inclusive')
    date_to: Optional[str] = Field(None, description='YYYY-MM-DD, inclusive')
    sort: Literal['newest', 'oldest', 'urgency'] = 'newest'
    page: int = Field(1, ge=1, le=100000)
    per_page: int = Field(config.DEFAULT_PAGE_SIZE, ge=1, le=config.MAX_PAGE_SIZE)

    @field_validator('emergency', mode='before')
    @classmethod
    def _bool(cls, v):
        if v in ('', None):
            return None
        if isinstance(v, bool):
            return v
        if str(v).lower() in ('1', 'true', 'yes', 'on'):
            return True
        if str(v).lower() in ('0', 'false', 'no', 'off'):
            return False
        raise ValueError('emergency must be true or false')

    @field_validator('date_from', 'date_to', mode='before')
    @classmethod
    def _date(cls, v):
        if v in ('', None):
            return None
        try:
            return date.fromisoformat(str(v)).isoformat()
        except ValueError:
            raise ValueError('dates must look like 2026-10-31')

    @field_validator('q', mode='before')
    @classmethod
    def _q(cls, v):
        return (v or '').strip()


class NotesBody(BaseModel):
    notes: str = Field('', max_length=config.MAX_NOTES_LENGTH, description='Your own note on this entry. Empty clears it.')

    @field_validator('notes', mode='before')
    @classmethod
    def _notes(cls, v):
        if not isinstance(v, str):
            raise ValueError('notes must be text')
        return v.strip()


class RetentionBody(BaseModel):
    days: Optional[Literal[0, 30, 90, 365]] = Field(None, description='0 = keep until I delete. null = use the system default.')


class AiConsentBody(BaseModel):
    granted: bool = Field(..., description='true = I agree my text and photos may be sent to the AI provider; false = withdraw (basic mode only)')


class LanguageBody(BaseModel):
    language: Literal['en', 'hi', 'mr', 'kn'] = Field(..., description='en, hi (Hindi), mr (Marathi) or kn (Kannada)')


class ConversationQuery(BaseModel):
    q: str = Field('', max_length=100)
    page: int = Field(1, ge=1)
    per_page: int = Field(50, ge=1, le=config.MAX_PAGE_SIZE)


class RenameBody(BaseModel):
    title: str = Field(..., min_length=1, max_length=80)

    @field_validator('title', mode='before')
    @classmethod
    def _title(cls, v):
        if not isinstance(v, str):
            raise ValueError('title must be text')
        return ' '.join(v.split())


class ExportQuery(BaseModel):
    format: Literal['json', 'csv'] = 'json'


class EntryExportQuery(BaseModel):
    format: Literal['json'] = 'json'


# ---- request bodies kept for documentation (their validation lives in api.validation with the app's established messages) ----
class SymptomRequest(BaseModel):
    symptoms: str = Field(..., min_length=config.MIN_SYMPTOM_LENGTH, max_length=config.MAX_SYMPTOM_LENGTH)


class InterviewAnswerRequest(BaseModel):
    session_id: str
    question_id: str
    answer: Optional[object] = Field(None, description='string, list of strings or integer, depending on the question type')
    skip: bool = False


class SessionRequest(BaseModel):
    session_id: str


class ChatCreateRequest(BaseModel):
    analysis_type: Optional[Literal['symptom', 'image']] = None
    analysis_id: Optional[int] = None
    new: bool = Field(False, description='start a fresh chat even if one already exists for this analysis')


class ChatMessageRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=config.MAX_CHAT_MESSAGE_LENGTH)


class FollowUpRequest(BaseModel):
    analysis_type: Literal['symptom', 'image']
    analysis_id: int
    question: str = Field(..., min_length=1, max_length=config.MAX_CHAT_MESSAGE_LENGTH)


class RegionRequest(BaseModel):
    region: str = Field(..., description='IN, US, CA, UK, EU, AU or INTL')


class RunRequest(BaseModel):
    model: Literal['offline', 'mock-rules', 'mock-weak', 'mock-flaky'] = 'offline'
    sets: List[str] = Field(default_factory=lambda: ['symptom_cases'])
    label: str = Field('', max_length=120)
