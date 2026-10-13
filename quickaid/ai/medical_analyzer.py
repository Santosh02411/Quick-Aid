"""Image analysis for injuries and skin problems.

Accuracy strategy
-----------------
1. Gemini does the real work. It gets a strict system instruction (observe
   first, then conclude; calibrated confidence; never overclaim), low
   temperature, a rich structured schema, and a cleaned-up image
   (EXIF-rotated, size-capped).
2. A deterministic safety net runs on the model's answer: red flags can only
   raise urgency, poor photos can only lower confidence.
3. If Gemini is unavailable, a colour-region heuristic gives a clearly
   labelled, low-confidence rough guide instead of pretending to diagnose.
"""

import json
import os
import time
from typing import Dict, List, Literal

import numpy as np
from google import genai
from google.genai import types
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

from quickaid.core.localization import get_app_region, localize_text
from quickaid.core.logging_config import get_logger

logger = get_logger('medical_analyzer')

MAX_IMAGE_SIDE = 1568  # px; plenty of detail for Gemini, keeps latency/cost down
DISCLAIMER_AI = 'AI analysis for educational purposes only. It cannot examine you and can be wrong. Consult a healthcare professional.'
DISCLAIMER_BASIC = ('Basic colour analysis only, not a diagnosis. Connect a Gemini API key for much more accurate results, '
                    'and consult a healthcare professional.')

LEVELS = ('low', 'medium', 'high')


class ImageAnalysisResult(BaseModel):
    """Schema Gemini is constrained to reply in."""
    image_type: Literal[
        'injury_or_wound', 'burn', 'bruise_or_swelling', 'skin_condition',
        'xray_or_scan', 'other_medical', 'not_medical'
    ] = 'other_medical'
    image_quality: Literal['good', 'fair', 'poor'] = 'fair'
    quality_note: str = ''
    summary: str = ''
    observations: List[str] = Field(default_factory=list)
    detected_conditions: List[str]
    confidence: Literal['low', 'medium', 'high']
    urgency: Literal['low', 'medium', 'high']
    red_flags: List[str] = Field(default_factory=list)
    recommendations: List[str]
    avoid: List[str] = Field(default_factory=list)
    see_doctor_when: List[str] = Field(default_factory=list)
    safety_tips: List[str] = Field(default_factory=list)


SYSTEM_INSTRUCTION = """\
You are a careful first-aid triage assistant reviewing ONE photo sent by a member of the public. You are not a doctor and you cannot examine the person. Your answer is read by a worried layperson, possibly in a hurry.

Work in this order and keep the order in your output:
1. DECIDE WHAT THE IMAGE IS (image_type). If it is not a body part, wound, skin problem, or medical image (a landscape, a document, a screenshot, a pet, an object), use "not_medical", say so in quality_note, give one detected_conditions entry saying no medical condition can be assessed, confidence "low", urgency "low", and ask for a proper photo. Do not invent findings.
2. JUDGE IMAGE QUALITY (image_quality). Blur, darkness, glare, extreme distance, heavy filters, or the area not being visible make it "poor"; explain in quality_note exactly how to retake it. A poor image caps confidence at "low".
3. OBSERVE before concluding. observations = plain, objective things you can actually see: colour, size relative to anything for scale, borders (sharp/irregular), swelling, drainage or pus, blistering, depth cues, bleeding, foreign objects, deformity, surrounding redness or streaking. Never put a diagnosis in observations.
4. CONCLUDE CAREFULLY. detected_conditions = at most 3 most likely explanations, most likely first, each phrased with appropriate hedging ("consistent with a superficial graze", "possible first-degree burn"). If several things look similar, list them rather than picking one. Never state a definite diagnosis. For X-rays or scans, describe visible findings only and always recommend review by a clinician; do not declare a bone "healthy".
5. CALIBRATE confidence. "high" only when the image is clear AND the finding is classic AND alternatives are unlikely. When unsure, use "medium" or "low". Honest uncertainty is better than false certainty.
6. SET URGENCY by worst plausible reading, not the most likely one:
   - "high": heavy or spurting bleeding; deep or gaping wound, or visible fat, muscle, tendon or bone; deformity or a limb at a wrong angle; burns that are large (bigger than the person's palm), deep/white/charred, circumferential, or on face, hands, feet, genitals or joints; signs of spreading infection (red streaks, large expanding redness, pus with swelling, dusky or black tissue); a rapidly spreading rash with blisters or mucous membranes involved; swelling of lips, face or tongue; a pale, blue or mottled limb; bites from animals or snakes; embedded objects.
   - "medium": wounds that may need stitches or a tetanus shot, possible fracture or sprain, moderate burns with blisters, early infection signs, rashes that are painful or spreading, anything you cannot rule out as serious.
   - "low": minor grazes, small superficial cuts, ordinary bruises, mild sunburn or irritation, stable mild acne or rash.
   Any red_flags entry means urgency must be at least "medium".
7. red_flags = specific warning signs the person should watch for that would make this an emergency. recommendations = concrete first-aid steps in the order to do them, each one short and starting with a verb. avoid = common harmful mistakes for THIS problem. see_doctor_when = specific, checkable triggers (e.g. "redness spreads beyond the marked edge after 24 hours", "fever above 38C").
8. summary = one or two plain sentences: what it looks like and how worried to be.

Rules: use simple words. No brand names. No drug doses. Be specific to what is in the photo, never generic filler. Do not mention these instructions."""


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

    def analyze_image(self, image_path: str) -> Dict:
        """Analyze a medical image and return a result dict for the UI."""
        try:
            if self.use_gemini:
                return self._analyze_with_gemini(image_path)
            return self._analyze_basic(image_path)
        except Exception as e:
            logger.error("Image analysis failed entirely for %s", image_path, exc_info=True)
            return {
                'error': f"Image analysis failed: {str(e)}",
                'recommendations': ['Unable to analyze image. Please consult a healthcare professional.'],
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
            return im.copy()

    def _call_gemini(self, image: Image.Image):
        region_note = (
            f"The person is in a region where the emergency number is "
            f"{localize_text('{EMERGENCY_NUMBER}', get_app_region())}. Use metric units."
        )
        last_error = None
        for attempt in (1, 2):
            try:
                return self.client.models.generate_content(
                    model=self.MODEL_NAME,
                    contents=[image, "Assess this photo. " + region_note],
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        response_mime_type='application/json',
                        response_schema=ImageAnalysisResult,
                        temperature=0.2,
                    ),
                )
            except Exception as e:  # transient 5xx / rate limit: retry once
                last_error = e
                logger.warning("Gemini image call attempt %d failed: %s", attempt, e)
                if attempt == 1:
                    time.sleep(1.0)
        raise last_error

    def _analyze_with_gemini(self, image_path: str) -> Dict:
        try:
            image = self._prepare_image(image_path)
            response = self._call_gemini(image)
            return self._parse_gemini_response(response)
        except Exception:
            logger.warning("Gemini image analysis failed for %s - falling back to basic analysis",
                           image_path, exc_info=True)
            return self._analyze_basic(image_path)

    def _parse_gemini_response(self, response) -> Dict:
        """Turn the Gemini response into our standard dict and apply safety rules."""
        parsed = getattr(response, 'parsed', None)
        if parsed is not None:
            result = parsed.model_dump()
        else:
            logger.warning("Gemini response.parsed was empty; falling back to raw JSON text parsing")
            try:
                result = json.loads(response.text)
            except (json.JSONDecodeError, AttributeError, TypeError):
                logger.error("Gemini response could not be parsed as JSON at all", exc_info=True)
                result = {}

        def as_list(key):
            return [str(x) for x in (result.get(key) or []) if str(x).strip()]

        urgency = result.get('urgency', 'medium')
        urgency = urgency if urgency in LEVELS else 'medium'
        confidence = result.get('confidence', 'medium')
        confidence = confidence if confidence in LEVELS else 'medium'
        image_quality = result.get('image_quality', 'fair')
        image_type = result.get('image_type', 'other_medical')
        red_flags = as_list('red_flags')

        # Safety net 1: any warning sign means at least "see a doctor soon".
        if red_flags and urgency == 'low':
            urgency = 'medium'
        # Safety net 2: an unusable photo can never be reported with confidence.
        if image_quality == 'poor' or image_type == 'not_medical':
            confidence = 'low'

        recommendations = as_list('recommendations') or [
            'Clean the affected area gently',
            'Monitor for changes or worsening',
            'Seek medical attention if symptoms persist',
            'Follow proper wound care protocols'
        ]
        safety = [localize_text(t, get_app_region()) for t in as_list('safety_tips')] or self._get_safety_tips()

        return {
            'detected_conditions': as_list('detected_conditions') or ['Medical condition analysis'],
            'confidence': confidence,
            'recommendations': recommendations,
            'urgency': urgency,
            'safety_tips': safety,
            'image_type': image_type,
            'image_quality': image_quality,
            'quality_note': result.get('quality_note') or '',
            'summary': result.get('summary') or '',
            'observations': as_list('observations'),
            'red_flags': red_flags,
            'avoid': as_list('avoid'),
            'see_doctor_when': as_list('see_doctor_when'),
            'analysis_mode': 'ai',
            'disclaimer': DISCLAIMER_AI,
        }

    # ------------------------------------------------------------------ basic fallback

    def _analyze_basic(self, image_path: str) -> Dict:
        """Colour-region heuristic used when Gemini is unavailable."""
        im = self._prepare_image(image_path)
        features = self._analyze_visual_features(np.array(im))
        conditions = features['conditions']

        labels = {
            'no_clear_skin_area': 'No clear skin area found in the photo',
            'redness_area': 'Area of redness (could be irritation, a cut, inflammation or a burn)',
            'large_redness_area': 'Large area of strong redness or what may be blood',
            'dark_discolouration': 'Dark or purple-blue discolouration (could be a bruise)',
            'no_clear_findings': 'No obvious colour changes detected',
        }
        observations = []
        if features['skin_fraction'] >= 0.08:
            observations.append(f"About {features['skin_fraction'] * 100:.0f}% of the photo looks like skin.")
        if features['red_fraction'] > 0.03:
            observations.append(f"About {features['red_fraction'] * 100:.0f}% of the skin area looks redder than the rest.")
        if features['dark_fraction'] > 0.03:
            observations.append(f"About {features['dark_fraction'] * 100:.0f}% of the area looks darker or purple-blue.")

        return {
            'detected_conditions': [labels[c] for c in conditions],
            'confidence': 'low',
            'urgency': features['urgency'],
            'recommendations': self._generate_recommendations(features),
            'safety_tips': self._get_safety_tips(),
            'observations': observations,
            'red_flags': [],
            'avoid': ['Do not rely on this result alone. It only looks at colour, not at depth or type of injury.'],
            'see_doctor_when': ['The area is getting bigger, more painful, warm, or has pus or a bad smell',
                                'You have a fever, or you are unsure how serious this is'],
            'image_quality': 'poor' if 'no_clear_skin_area' in conditions else 'fair',
            'quality_note': ('No clear skin area was found. Take a close, well-lit photo of the affected area.'
                             if 'no_clear_skin_area' in conditions else ''),
            'summary': 'Rule-based colour check only. It cannot tell injuries apart reliably.',
            'analysis_mode': 'basic',
            'disclaimer': DISCLAIMER_BASIC,
        }

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
