"""
Follow-up conversation support for Quick Aid.

Each image/symptom analysis can be the start of a short conversation - the
user can ask "is this serious?" or "what if it doesn't improve in 2 days?"
and get an answer that's aware of both the original analysis and anything
already asked. This only works when Gemini is configured: the rule-based
basic mode has no way to hold a conversation, so it returns a clear message
saying so instead of pretending to understand.
"""

from typing import Dict, List, Optional
from google import genai
from google.genai import types
from google.genai.types import Content, Part
import os

from logging_config import get_logger
from localization import prompt_context, DEFAULT_REGION

logger = get_logger('conversation')

MODEL_NAME = 'gemini-flash-latest'

NO_GEMINI_MESSAGE = (
    "Follow-up questions need Gemini to be configured (basic mode can only "
    "run a single one-shot analysis, it can't hold a conversation). Ask your "
    "administrator to set GEMINI_API_KEY, or consult a healthcare professional directly."
)


class ConversationService:
    def __init__(self):
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key and api_key != 'your_gemini_api_key_here':
            self.client = genai.Client(api_key=api_key)
            self.use_gemini = True
        else:
            self.client = None
            self.use_gemini = False

    def _build_system_context(self, analysis_type: str, analysis_summary: str, region: str) -> str:
        kind = "an uploaded image" if analysis_type == 'image' else "symptoms the user described"
        return f"""
        You are a medical AI assistant continuing a conversation about a prior analysis.
        The original analysis was based on {kind}. Here is that original analysis, as JSON:

        {analysis_summary}

        The user may now ask follow-up questions about this specific analysis (e.g. "is this
        serious?", "what if it doesn't improve?", "should I be worried about X?"). Answer
        conversationally in plain text (not JSON), staying grounded in the original analysis.
        Keep answers concise (a few sentences to a short paragraph). Always recommend
        professional medical care for anything serious or uncertain, and never provide a
        definitive diagnosis. {prompt_context(region)}
        """.strip()

    def ask_follow_up(
        self,
        analysis_type: str,
        analysis_summary: str,
        history: List[Dict],
        question: str,
        region: str = DEFAULT_REGION,
    ) -> str:
        """
        Answer a follow-up question in the context of a prior analysis and
        any earlier follow-up turns. `history` is a list of
        {'role': 'user'|'model', 'content': str} dicts, oldest first.
        Returns the assistant's plain-text answer.
        """
        if not self.use_gemini:
            return NO_GEMINI_MESSAGE

        try:
            contents: List[Content] = [
                Content(role='user', parts=[Part(text=self._build_system_context(analysis_type, analysis_summary, region))]),
                Content(role='model', parts=[Part(text="Understood - I have the original analysis. What would you like to know?")]),
            ]
            for turn in history:
                role = 'model' if turn.get('role') == 'model' else 'user'
                contents.append(Content(role=role, parts=[Part(text=turn.get('content', ''))]))

            contents.append(Content(role='user', parts=[Part(text=question)]))

            response = self.client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config=types.GenerateContentConfig(
                    max_output_tokens=500,
                ),
            )

            answer = (response.text or '').strip()
            if not answer:
                logger.warning("Gemini follow-up returned an empty response")
                return "I wasn't able to generate a response to that. Could you rephrase your question?"
            return answer

        except Exception:
            logger.error("Gemini follow-up request failed", exc_info=True)
            return (
                "Sorry, I couldn't process that follow-up question right now. "
                "Please try again, or consult a healthcare professional directly."
            )
