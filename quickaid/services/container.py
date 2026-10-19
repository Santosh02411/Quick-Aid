"""The services one app instance owns, built once per app (so each app/test gets fresh, isolated analyzers)."""

from dataclasses import dataclass

from flask import current_app

from quickaid.ai.conversation import ConversationService as ChatAI
from quickaid.ai.interview import InterviewEngine
from quickaid.ai.medical_analyzer import MedicalAnalyzer
from quickaid.ai.symptom_checker import SymptomChecker
from quickaid.services.analysis import AnalysisService
from quickaid.services.chats import ChatService
from quickaid.services.history import HistoryService
from quickaid.services.retention import RetentionService


@dataclass
class Services:
    symptom_checker: SymptomChecker
    medical_analyzer: MedicalAnalyzer
    chat_ai: ChatAI
    interview_engine: InterviewEngine
    analysis: AnalysisService
    chats: ChatService
    history: HistoryService
    retention: RetentionService


def build_services() -> Services:
    checker, analyzer, chat_ai = SymptomChecker(), MedicalAnalyzer(), ChatAI()
    engine = InterviewEngine(checker)
    return Services(symptom_checker=checker, medical_analyzer=analyzer, chat_ai=chat_ai, interview_engine=engine,
                    analysis=AnalysisService(checker, analyzer, engine), chats=ChatService(chat_ai),
                    history=HistoryService(), retention=RetentionService())


def get_services() -> Services:
    return current_app.extensions['quickaid_services']
