"""Symptom, photo and interview analysis: the business logic behind /analyze_symptoms, /upload and /api/interview/*.

Views validate the HTTP request and call these methods; these methods validate the *content*, run the analyzers, pass every result
through the medical-safety review, save it, and clean up. They raise ApiError for anything the caller should see as a 4xx/5xx.
"""

import os
import uuid
from typing import Dict, Optional

from quickaid.ai import telemetry  # noqa: F401  (imported so the DB telemetry sink is installed wherever analysis runs)
from quickaid.ai.interview import InterviewEngine, public_question
from quickaid.ai.medical_analyzer import MedicalAnalyzer
from quickaid.ai.symptom_checker import SymptomChecker
from quickaid.ai.upload_validation import validate_upload
from quickaid.api.responses import ApiError
from quickaid.api.validation import symptom_text
from quickaid.core import database as db, secure_files
from quickaid.core.localization import get_app_region, localize_text
from quickaid.core.logging_config import get_logger
from quickaid.services import safety

logger = get_logger('analysis_service')


class AnalysisService:
    def __init__(self, symptom_checker: SymptomChecker, medical_analyzer: MedicalAnalyzer, interview_engine: InterviewEngine):
        self.symptom_checker = symptom_checker
        self.medical_analyzer = medical_analyzer
        self.interview_engine = interview_engine

    # ---------------------------------------------------------------- one-shot symptom check

    def analyze_symptoms(self, user_id: int, data: Dict) -> Dict:
        text = symptom_text(data)
        region = get_app_region()
        try:
            result = safety.review_symptom_result(self.symptom_checker.analyze_symptoms(text), text)
        except ApiError:
            raise
        except Exception as e:
            # The analyzer itself already has its own last-resort emergency check (see symptom_checker.py), so this
            # is a second, independent layer in case something breaks between there and here instead.
            logger.error("Symptom analysis failed for user_id=%s", user_id, exc_info=True)
            urgent_now = safety.last_resort_emergency_check(text, 'Analysis failed, but your description includes a warning sign.', region)
            if urgent_now:
                return {'analysis': urgent_now, 'analysis_id': None}
            raise ApiError(f'Symptom analysis failed: {e}', 500, 'analysis_failed',
                           hints=[localize_text('If this might be an emergency, call {EMERGENCY_NUMBER} now rather than waiting.', region)])
        # The result is already computed (and correct, including any emergency guidance) at this point: if SAVING it
        # to history fails, the person still gets to see it. It just will not appear in their history afterwards.
        try:
            analysis_id = db.save_symptom_analysis(user_id, text, result)
        except Exception:
            logger.error("Symptom analysis succeeded but could not be saved for user_id=%s", user_id, exc_info=True)
            analysis_id = None
        return {'analysis': result, 'analysis_id': analysis_id}

    # ---------------------------------------------------------------- photo

    def analyze_image(self, user_id: int, data: Optional[bytes], filename: str, upload_folder: str, original_name: str) -> Dict:
        """Validate the real content of the upload, analyse it, save the result. The photo itself is always deleted."""
        image, problem = validate_upload(data, filename)
        if problem:
            logger.info("Rejected photo upload for user_id=%s: %s", user_id, problem.code)
            raise ApiError(problem.message, problem.status, problem.code, problem.hints, retake=True)

        os.makedirs(upload_folder, exist_ok=True)
        # A random name, so two uploads called "photo.jpg" can never overwrite (or delete) each other.
        path = os.path.join(upload_folder, f'{uuid.uuid4().hex}.{image.ext}')
        try:
            with open(path, 'wb') as f:
                f.write(data)
            result = safety.review_image_result(self.medical_analyzer.analyze_image(path))
            analysis_id = db.save_image_analysis(user_id, original_name, result)
            return {'analysis': result, 'analysis_id': analysis_id,
                    'photo': {'width': image.width, 'height': image.height, 'size': image.size, 'format': image.ext}}
        except Exception as e:
            logger.error("Image analysis failed for user_id=%s", user_id, exc_info=True)
            raise ApiError('The analysis failed on our side. Please try again in a moment.', 500, 'analysis_failed',
                           hints=['If it keeps happening, try a different photo.',
                                  localize_text('If this looks serious, do not wait for a photo check: call {EMERGENCY_NUMBER} now or see a doctor.', get_app_region())],
                           detail=str(e)[:200])
        finally:
            if os.path.exists(path):              # the photo is never kept, even if analysis crashed
                secure_files.secure_remove(path)

    # ---------------------------------------------------------------- adaptive interview

    def _complete(self, user_id, session_id, state, analysis, status='completed') -> Dict:
        """Save the finished (or emergency-stopped) interview as a history entry and close the session."""
        analysis = safety.review_symptom_result(analysis, state['text'], state.get('region') or get_app_region())
        analysis_id = db.save_symptom_analysis(user_id, state['text'], analysis)
        if session_id:
            db.update_interview(user_id, session_id, state, status)
        return {'status': 'emergency' if analysis.get('care_level') == 'emergency' else 'complete', 'analysis': analysis, 'analysis_id': analysis_id}

    def _question(self, session_id, state, question) -> Dict:
        return {'status': 'question', 'session_id': session_id, 'question': public_question(question),
                'progress': self.interview_engine.progress(state), 'mode': 'ai' if state.get('plan', {}).get('source') == 'ai' else 'basic'}

    def interview_start(self, user_id: int, data: Dict) -> Dict:
        text = symptom_text(data)
        region = get_app_region()
        try:
            started = self.interview_engine.start(text, region)
            if started['emergency']:                    # a warning sign in the first message: guidance now, no questions
                return self._complete(user_id, None, {'text': text, 'asked': [], 'answers': {}}, started['emergency'])
            state = started['state']
            if started['question'] is None:             # nothing left to ask: assess right away
                return self._complete(user_id, None, state, self.interview_engine.finish(state, region))
            return self._question(db.create_interview(user_id, state), state, started['question'])
        except ApiError:
            raise
        except Exception as e:
            # By this point self.interview_engine.start() has ALREADY run its own immediate_emergency() check as its
            # very first step (see interview.py), so whatever failed here happened after the text was already found
            # not to describe a warning sign on its own. Still worth a reminder, since the person may be mid-interview.
            logger.error("Interview start failed for user_id=%s", user_id, exc_info=True)
            raise ApiError(f'Could not start the check: {e}', 500, 'interview_failed',
                           hints=[localize_text('If this might be an emergency, call {EMERGENCY_NUMBER} now rather than waiting.', region)])

    @staticmethod
    def _load_interview(user_id: int, data: Dict):
        session_id = data.get('session_id')
        if not isinstance(session_id, str) or not session_id or len(session_id) > 64:
            raise ApiError('Missing session', 400, 'missing_session')
        row = db.get_interview(user_id, session_id)
        if row is None:
            raise ApiError('That check was not found. Please start again.', 404, 'session_not_found')
        if row['status'] != 'active':
            raise ApiError('That check is already finished.', 409, 'session_finished')
        return session_id, row

    def interview_answer(self, user_id: int, data: Dict) -> Dict:
        session_id, row = self._load_interview(user_id, data)
        state = row['state']
        region = state.get('region') or get_app_region()
        try:
            out = self.interview_engine.answer(state, str(data.get('question_id') or ''), data.get('answer'), bool(data.get('skip')), region)
            if out.get('error'):
                raise ApiError(out['error'], 400, 'invalid_answer')
            state = out['state']
            if out['emergency']:
                return self._complete(user_id, session_id, state, out['emergency'], status='emergency')
            if out['question'] is None:
                return self._complete(user_id, session_id, state, self.interview_engine.finish(state, region))
            db.update_interview(user_id, session_id, state, 'active')
            return self._question(session_id, state, out['question'])
        except ApiError:
            raise
        except Exception as e:
            logger.error("Interview answer failed for user_id=%s", user_id, exc_info=True)
            raise ApiError(f'Could not process that answer: {e}', 500, 'interview_failed',
                           hints=[localize_text('If this might be an emergency, call {EMERGENCY_NUMBER} now rather than waiting.', region)])

    def interview_finish(self, user_id: int, data: Dict) -> Dict:
        """'Get my assessment now': assess with whatever has been answered so far."""
        session_id, row = self._load_interview(user_id, data)
        state = row['state']
        try:
            return self._complete(user_id, session_id, state, self.interview_engine.finish(state, state.get('region') or get_app_region()))
        except ApiError:
            raise
        except Exception as e:
            logger.error("Interview finish failed for user_id=%s", user_id, exc_info=True)
            raise ApiError(f'Could not create the assessment: {e}', 500, 'interview_failed',
                           hints=[localize_text('If this might be an emergency, call {EMERGENCY_NUMBER} now rather than waiting.', state.get('region') or get_app_region())])
