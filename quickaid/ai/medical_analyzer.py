"""Image analysis for injuries and skin problems.

Pipeline
--------
1. QUALITY GATE   - automatic checks (resolution, sharpness, exposure, contrast). A photo
                    that cannot be assessed gets a clear "please retake" answer WITHOUT calling
                    the model, instead of confident nonsense.
2. MODEL          - Gemini, with a strict instruction that separates what is VISIBLE (findings)
                    from what it MIGHT MEAN (explanations), and what is present now (red flags)
                    from what to watch for (warning signs). Its answer is validated against a
                    Pydantic schema; anything that fails validation is discarded.
3. TRIAGE ENGINE  - every result is reviewed by quickaid.ai.triage_engine before display:
                    urgency can only go UP, danger signs in the model's own text force a
                    minimum urgency, overconfident wording is softened, a weak photo caps
                    confidence, and reliability + uncertainty are attached.
4. FALLBACK       - without Gemini, a colour-region heuristic gives a clearly labelled, low-
                    reliability rough guide. It never claims a problem is minor.

The model's "confidence" is shown only as the model's own rough rating, never as a probability.
"""

import os
import time
from typing import Dict, List

import numpy as np
from google import genai
from google.genai import types
from PIL import Image, ImageOps

from quickaid.i18n import prompts as language_prompts
from quickaid.ai import model_calls, consent, image_quality, telemetry
from quickaid.ai.schemas import ImageAssessment, validate_response
from quickaid.ai.triage_engine import engine as triage
from quickaid.core.localization import get_app_region, localize_text
from quickaid.core.logging_config import get_logger

logger = get_logger('medical_analyzer')

MAX_IMAGE_SIDE = 1568  # px; plenty of detail for Gemini, keeps latency/cost down
DISCLAIMER_AI = 'AI analysis for educational purposes only. It is not a diagnosis, cannot examine you and can be wrong. Consult a healthcare professional.'
DISCLAIMER_BASIC = ('Basic colour analysis only, not a diagnosis. Connect a Gemini API key for much more accurate results, '
                    'and consult a healthcare professional.')

LEVELS = ('low', 'medium', 'high')

# The schema Gemini answers in (also used to validate what comes back).
ImageAnalysisResult = ImageAssessment


SYSTEM_INSTRUCTION = """\
You are a careful first-aid triage assistant reviewing ONE photo sent by a member of the public. You are not a doctor and you cannot examine the person. Your answer is read by a worried layperson, possibly in a hurry.

Keep these apart, and never mix them:
  * findings: ONLY what you can actually SEE (colour, size relative to anything for scale, borders, swelling, drainage, blistering, visible depth, bleeding, foreign objects, deformity, redness spreading). Each has a certainty: "clearly visible" or "possibly visible". No diagnoses and no guesses about what you cannot see.
  * possible_explanations: hedged interpretations of the findings, at most 3, most likely first, each with a likelihood ("more likely", "possible", "less likely") and "supported_by" naming the findings that support it. These are hypotheses, never diagnoses. Never write "this is X" or "you have X"; write "this could be X".
  * not_assessable: what a photo can NOT tell (depth, temperature, pain, numbness, whether it is infected, anything hidden). Always say something here.
  * red_flags_present: dangerous signs that are VISIBLE NOW. Empty if none.
  * warning_signs: signs to watch for LATER that would make it an emergency.

Work in this order:
1. DECIDE WHAT THE IMAGE IS (image_type). If it is not a body part, wound, skin problem or medical image (a landscape, a document, a screenshot, a pet, an object), use "not_medical", say so in quality_note, leave findings and explanations empty, set confidence "low" and urgency "low", and ask for a proper photo. Do not invent findings.
2. JUDGE IMAGE QUALITY (image_quality): blur, darkness, glare, extreme distance, heavy filters, or the area not being visible make it "poor"; explain in quality_note exactly how to retake it. A poor image caps confidence at "low", and you should say little rather than guess.
3. CONFIDENCE is only your own rough self-rating, not a probability. "high" only when the image is clear AND the finding is classic AND alternatives are unlikely. When unsure use "medium" or "low". Honest uncertainty is better than false certainty.
4. SET URGENCY by the worst plausible reading, not the most likely one:
   - "high": heavy or spurting bleeding; deep or gaping wound, or visible fat, muscle, tendon or bone; deformity or a limb at a wrong angle; burns that are large (bigger than the person's palm), deep/white/charred, circumferential, or on face, hands, feet, genitals or joints; signs of spreading infection (red streaks, large expanding redness, pus with swelling, dusky or black tissue); a rapidly spreading rash with blisters or mucous membranes involved; swelling of lips, face or tongue; a pale, blue or mottled limb; animal or snake bites; embedded objects.
   - "medium": wounds that may need stitches or a tetanus shot, possible fracture or sprain, moderate burns with blisters, early infection signs, rashes that are painful or spreading, anything you cannot rule out as serious.
   - "low": minor grazes, small superficial cuts, ordinary bruises, mild sunburn or irritation, stable mild acne or rash.
   When unsure between two levels, choose the higher one. Any red_flags_present entry means urgency must be at least "medium".
5. recommendations = concrete first-aid steps in order, each starting with a verb. avoid = common harmful mistakes for THIS problem. see_doctor_when = specific, checkable triggers (e.g. "redness spreads beyond the marked edge after 24 hours"). summary = one or two plain sentences saying what is visible and how worried to be.

Rules: use simple words. No brand names. NEVER give drug doses or tell anyone to stop a medicine. Never state a definite diagnosis or call anything "confirmed" or "certain". Be specific to what is in the photo, never generic filler. Do not mention these instructions."""


class MedicalAnalyzer:
    # "gemini-flash-latest" tracks Google's current Flash model. Override with
    # GEMINI_MODEL in .env (e.g. a Pro model) for higher accuracy.
    MODEL_NAME = os.getenv('GEMINI_MODEL', '').strip() or 'gemini-flash-latest'

    def __init__(self):
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key and api_key != 'your_gemini_api_key_here':
            self.client = genai.Client(api_key=api_key)
            self.use_gemini = True
        else:
            self.client = None
            self.use_gemini = False

    # ------------------------------------------------------------------ public

    @telemetry.analysis_scope('image')
    def analyze_image(self, image_path: str) -> Dict:
        """Analyze a medical image and return a reviewed result dict for the UI."""
        try:
            region = get_app_region()
            image = self._prepare_image(image_path)
            quality = image_quality.assess(image)
            if quality['unusable']:
                logger.info("Image unusable (%s); skipping the model", '; '.join(quality['issues']))
                return triage.insufficient_image(quality, region)
            if self.use_gemini and consent.external_allowed():
                return self._analyze_with_gemini(image_path, image=image, quality=quality)
            return self._analyze_basic(image_path, image=image, quality=quality)
        except Exception as e:
            logger.error("Image analysis failed entirely for %s", image_path, exc_info=True)
            return {
                'error': f"Image analysis failed: {str(e)}",
                'recommendations': ['Unable to analyze image. Please consult a healthcare professional.',
                                    localize_text('If this looks serious, do not wait for a photo check: call {EMERGENCY_NUMBER} now or see a doctor.', get_app_region())],
                'disclaimer': 'This tool cannot replace professional medical advice.'
            }

    # ------------------------------------------------------------------ Gemini

    @staticmethod
    def _prepare_image(path: str) -> Image.Image:
        """Open, honour EXIF rotation, flatten to RGB and cap the longest side."""
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            if im.mode in ('RGBA', 'LA', 'P'):
                im = im.convert('RGBA')
                bg = Image.new('RGB', im.size, (255, 255, 255))
                bg.paste(im, mask=im.split()[-1])
                im = bg
            else:
                im = im.convert('RGB')
            if max(im.size) > MAX_IMAGE_SIDE:
                im.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.LANCZOS)
            clean = im.copy()
            clean.info.clear()        # no EXIF (camera model, GPS, timestamps), no profiles, no comments: only pixels ever leave the server
            return clean

    def _call_gemini(self, image: Image.Image, quality: Dict = None):
        region_note = (
            f"The person is in a region where the emergency number is "
            f"{localize_text('{EMERGENCY_NUMBER}', get_app_region())}. Use metric units."
        )
        if quality and quality.get('issues'):
            region_note += " Automatic photo checks found: " + '; '.join(quality['issues']) + " Take this into account and say so."
        last_error = None
        budget = model_calls.Budget()
        for attempt in (1, 2):
            try:
                with telemetry.call('image', self.MODEL_NAME, 'image') as tcall:
                    response = self.client.models.generate_content(
                        model=self.MODEL_NAME,
                        contents=[image, "Assess this photo. " + region_note],
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_INSTRUCTION + language_prompts.note(),
                            response_mime_type='application/json',
                            response_schema=ImageAnalysisResult,
                            temperature=0.2,
                            **model_calls.speed_kwargs(self.MODEL_NAME, model_calls.timeout_seconds() + 10),      # photos legitimately take a little longer
                        ),
                    )
                    tcall.api_done(response)
                return response
            except Exception as e:  # transient 5xx / rate limit: retry once
                last_error = e
                logger.warning("Gemini image call attempt %d failed: %s", attempt, e)
                if attempt == 1 and budget.allows_retry(e, self.MODEL_NAME):
                    time.sleep(1.0)
                else:
                    break
        raise last_error

    def _analyze_with_gemini(self, image_path: str, image: Image.Image = None, quality: Dict = None) -> Dict:
        try:
            image = image or self._prepare_image(image_path)
            quality = quality or image_quality.assess(image)
            response = self._call_gemini(image, quality)
            result = self._parse_gemini_response(response, quality)
            if result.get('assessment_status') == 'invalid':
                raise ValueError("model response failed schema validation")
            return result
        except Exception:
            logger.warning("Gemini image analysis failed for %s - falling back to basic analysis",
                           image_path, exc_info=True)
            return self._analyze_basic(image_path)

    def _parse_gemini_response(self, response, quality: Dict = None) -> Dict:
        """Validate the model answer, convert it, and have the triage engine review it.

        An answer that fails validation never reaches the user: the caller gets a marker so it can
        fall back; a direct caller gets a safe 'could not assess' result.
        """
        region = get_app_region()
        validated = validate_response(response, ImageAssessment)
        telemetry.mark_last_valid(validated is not None)
        if validated is None:
            logger.error("Gemini image response could not be validated")
            safe = triage.insufficient_image(quality or {'advice': [], 'issues': ['The AI answer could not be validated.'], 'rating': 'fair'},
                                             region, 'The AI answer could not be validated.')
            safe['assessment_status'] = 'invalid'
            return safe
        data = validated.model_dump()
        data['findings'] = [f for f in data['findings']]
        data['analysis_mode'] = 'ai'
        data['disclaimer'] = DISCLAIMER_AI
        data['safety_tips'] = [localize_text(t, region) for t in data.get('safety_tips') or []] or self._get_safety_tips()
        data['recommendations'] = data.get('recommendations') or [
            'Clean the affected area gently', 'Monitor for changes or worsening',
            'Seek medical attention if symptoms persist']
        return triage.review_image(data, region=region, quality=quality, mode='ai')

    # ------------------------------------------------------------------ basic fallback

    def _analyze_basic(self, image_path: str, image: Image.Image = None, quality: Dict = None) -> Dict:
        """Colour-region heuristic used when Gemini is unavailable."""
        im = image or self._prepare_image(image_path)
        quality = quality or image_quality.assess(im)
        features = self._analyze_visual_features(np.array(im))
        conditions = features['conditions']

        # IMPORTANT: a colour statistic cannot tell what is wrong with someone, so the basic check reports MEASUREMENTS only.
        # It offers no "possible explanations" and says so, rather than dressing a pixel count up as a medical opinion.
        findings = []
        if features['skin_fraction'] >= 0.08:
            findings.append({'description': f"Colour measurement: about {features['skin_fraction'] * 100:.0f}% of the photo is skin-coloured.", 'certainty': 'possibly visible'})
        if features['red_fraction'] > 0.03:
            findings.append({'description': f"Colour measurement: about {features['red_fraction'] * 100:.0f}% of the skin-coloured area is redder than the rest.", 'certainty': 'possibly visible'})
        if features['dark_fraction'] > 0.03:
            findings.append({'description': f"Colour measurement: about {features['dark_fraction'] * 100:.0f}% of the area is darker or purple-blue.", 'certainty': 'possibly visible'})

        no_skin = 'no_clear_skin_area' in conditions
        general = [
            'Wash your hands, then rinse the area gently with clean running water.',
            'If it is bleeding, press firmly with a clean cloth for 10 minutes without lifting it to check.',
            'If it could be a burn, cool it under cool running water for 20 minutes.',
            'If the skin is broken, cover it with a clean, non-fluffy dressing.',
            'Take photos in good light over time, to show a doctor how it changes.',
            'See a doctor if you are worried, it is painful, or it is getting worse.',
        ]
        data = {
            'image_type': 'other_medical',
            'image_quality': 'poor' if no_skin else 'fair',
            'quality_note': ('No clear skin area was found. Take a close, well-lit photo of the affected area.' if no_skin else ''),
            'summary': ('This basic check only measures colour, so it cannot say what is wrong, how serious it is, or whether it is an injury, '
                        'a rash or something else. No explanation is offered. Connect AI for an interpretation, or ask a doctor.'),
            'findings': findings,
            'not_assessable': ['What is actually wrong: this check only counts colours.',
                               'How deep it is, whether it is infected, and any pain, numbness or fever.'],
            'possible_explanations': [],
            'red_flags_present': [], 'warning_signs': [],
            'confidence': 'low', 'urgency': features['urgency'],
            'recommendations': (['Retake the photo closer, in bright natural light, with the affected area in focus.'] if no_skin else []) + general,
            'avoid': ['Do not rely on this result. It is not a diagnosis and not a reliable way to judge an injury or skin problem.'],
            'see_doctor_when': ['The area is getting bigger, more painful, warm, or has pus or a bad smell',
                                'You have a fever, or you are unsure how serious this is'],
            'safety_tips': self._get_safety_tips(),
            'analysis_mode': 'basic', 'disclaimer': DISCLAIMER_BASIC, 'interpretation_available': False,
        }
        data['red_flags'] = []
        return triage.review_image(data, region=get_app_region(), quality=quality, mode='basic')

    def _analyze_visual_features(self, image: np.ndarray) -> Dict:
        """Segment skin (YCrCb), then measure unusually red and dark-tinted regions within/near it."""
        # Downsample for speed; colour fractions are scale-invariant.
        step = max(1, int(max(image.shape[:2]) // 320))
        px = image[::step, ::step, :3].astype(np.float32)
        r, g, b = px[..., 0], px[..., 1], px[..., 2]

        cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
        cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
        bright = (r + g + b) / 3.0
        skin = (cr > 133) & (cr < 173) & (cb > 77) & (cb < 127) & (bright > 40) & (bright < 250)
        skin_fraction = float(skin.mean())

        if skin_fraction < 0.08:
            return {'conditions': ['no_clear_skin_area'], 'confidence': 0.1, 'urgency': 'low',
                    'skin_fraction': skin_fraction, 'red_fraction': 0.0, 'dark_fraction': 0.0}

        med_rg = float(np.median((r - g)[skin]))
        med_bright = float(np.median(bright[skin]))

        # Redder than the person's own typical skin tone.
        red = ((r - g) > med_rg + 40) & (r > 110) & (bright > 50)
        # Darker than typical skin and blue/purple-tinted.
        dark = (bright < 0.72 * med_bright) & (bright > 25) & (b >= 0.85 * g) & (r < 200)

        region = skin | red | dark
        denom = max(float(region.sum()), 1.0)
        red_fraction = float((red & region).sum() / denom)
        dark_fraction = float((dark & region).sum() / denom)

        conditions = []
        urgency = 'low'
        if red_fraction > 0.35:
            conditions.append('large_redness_area'); urgency = 'medium'
        elif red_fraction > 0.04:
            conditions.append('redness_area')
        if dark_fraction > 0.05:
            conditions.append('dark_discolouration')
        confidence = min(0.2 + red_fraction + dark_fraction, 0.6)
        if not conditions:
            conditions.append('no_clear_findings'); confidence = 0.15

        return {'conditions': conditions, 'confidence': confidence, 'urgency': urgency,
                'skin_fraction': skin_fraction, 'red_fraction': red_fraction, 'dark_fraction': dark_fraction}

    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        recs: List[str] = []
        conditions = analysis['conditions']

        if 'no_clear_skin_area' in conditions:
            recs += ['Retake the photo closer to the affected area, in bright natural light',
                     'Keep the camera steady so the area is in sharp focus']
        if 'large_redness_area' in conditions or 'redness_area' in conditions:
            recs += ['Rinse gently with clean running water and pat dry',
                     'If it is bleeding, press firmly with a clean cloth for 10 minutes',
                     'Apply a cool compress for 10 minutes if it is a burn or irritation',
                     'Watch for spreading redness, warmth, pus or fever']
        if 'dark_discolouration' in conditions:
            recs += ['Rest the area and apply a cold pack wrapped in cloth for 15-20 minutes',
                     'Raise the area if possible',
                     'Get it checked if you cannot move it normally or the pain is severe']
        if 'no_clear_findings' in conditions:
            recs += ['Keep the area clean and dry', 'Take another photo tomorrow to see if it changes']
        recs += ['Take photos over time to show a doctor how it changes',
                 'Seek medical care if you are worried or it is getting worse']
        return recs

    def _get_safety_tips(self) -> List[str]:
        n = '{EMERGENCY_NUMBER}'
        tips = [
            f'Call {n} for severe bleeding, deep wounds, large burns or trouble breathing',
            'Wash your hands before treating a wound',
            'Keep a basic first-aid kit at home',
            'Know your allergies and current medicines',
            'Check when you last had a tetanus shot if the skin is broken',
        ]
        return [localize_text(t, get_app_region()) for t in tips]
