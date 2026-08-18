import numpy as np
from PIL import Image
import json
import os
from typing import Dict, List, Literal
from google import genai
from google.genai import types
from pydantic import BaseModel
from dotenv import load_dotenv
from logging_config import get_logger
from localization import localize_analysis, prompt_context, DEFAULT_REGION

load_dotenv()

logger = get_logger('medical_analyzer')


class ImageAnalysisResult(BaseModel):
    """Schema Gemini is constrained to reply in - no more find('{')/rfind('}') guessing."""
    detected_conditions: List[str]
    confidence: Literal['low', 'medium', 'high']
    recommendations: List[str]
    urgency: Literal['low', 'medium', 'high']
    safety_tips: List[str]


class MedicalAnalyzer:
    # "gemini-flash-latest" is Google's stable alias for their current-generation
    # Flash model, so this keeps working as Google ships new versions instead of
    # pointing at a specific release (like the old 'gemini-1.5-flash') that gets
    # deprecated and shut down over time. Pin to an exact version instead if you
    # need reproducible/deterministic behavior across releases.
    MODEL_NAME = 'gemini-flash-latest'

    def __init__(self):
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key and api_key != 'your_gemini_api_key_here':
            self.client = genai.Client(api_key=api_key)
            self.use_gemini = True
        else:
            self.client = None
            self.use_gemini = False

        self.injury_patterns = {
            'cuts_wounds': {
                'keywords': ['red', 'bleeding', 'open', 'laceration'],
                'recommendations': [
                    'Clean hands before treating wound',
                    'Apply gentle pressure to stop bleeding',
                    'Clean wound with clean water',
                    'Apply antibiotic ointment if available',
                    'Cover with sterile bandage',
                    'Seek medical attention if deep or won\'t stop bleeding'
                ]
            },
            'bruises': {
                'keywords': ['purple', 'blue', 'dark', 'discoloration'],
                'recommendations': [
                    'Apply ice pack for 15-20 minutes',
                    'Elevate injured area if possible',
                    'Take over-the-counter pain relief',
                    'Monitor for increased swelling',
                    'Seek medical attention if severe pain persists'
                ]
            },
            'burns': {
                'keywords': ['red', 'blistered', 'peeling', 'charred'],
                'recommendations': [
                    'Cool burn with cool (not cold) water for 10-20 minutes',
                    'Remove jewelry/clothing from burned area',
                    'Do not break blisters',
                    'Apply loose, sterile bandage',
                    'Take over-the-counter pain medication',
                    'Seek immediate medical attention for severe burns'
                ]
            },
            'swelling': {
                'keywords': ['swollen', 'enlarged', 'puffy'],
                'recommendations': [
                    'Apply ice pack to reduce swelling',
                    'Elevate affected area',
                    'Avoid putting weight on swollen area',
                    'Take anti-inflammatory medication if appropriate',
                    'Monitor for increased pain or discoloration'
                ]
            }
        }

        self.skin_conditions = {
            'rash': {
                'keywords': ['red', 'bumpy', 'itchy', 'scattered'],
                'recommendations': [
                    'Keep area clean and dry',
                    'Avoid scratching',
                    'Apply cool compress',
                    'Use gentle, fragrance-free moisturizer',
                    'Consider antihistamine for itching',
                    'Consult doctor if rash spreads or worsens'
                ]
            },
            'acne': {
                'keywords': ['pimples', 'blackheads', 'whiteheads'],
                'recommendations': [
                    'Wash face twice daily with gentle cleanser',
                    'Avoid touching or picking at acne',
                    'Use non-comedogenic products',
                    'Consider over-the-counter acne treatments',
                    'Maintain consistent skincare routine'
                ]
            }
        }

    def analyze_image(self, image_path: str, region: str = DEFAULT_REGION) -> Dict:
        """Analyze medical image and provide recommendations"""
        try:
            if self.use_gemini:
                result = self._analyze_with_gemini(image_path, region)
            else:
                result = self._analyze_basic(image_path)

        except Exception as e:
            logger.error("Image analysis failed entirely for %s", image_path, exc_info=True)
            return {
                'error': f"Image analysis failed: {str(e)}",
                'recommendations': ['Unable to analyze image. Please consult a healthcare professional.'],
                'disclaimer': 'This tool cannot replace professional medical advice.'
            }

        # Applies regardless of which path produced the result: substitutes
        # the {EMERGENCY_NUMBER}/{FEVER_TEMP} placeholders used by our own
        # fallback strings. Real Gemini-generated text won't contain these
        # tokens, so this is a no-op there - the region context passed into
        # the prompt is what steers Gemini's own wording instead.
        return localize_analysis(result, region)

    def _analyze_with_gemini(self, image_path: str, region: str = DEFAULT_REGION) -> Dict:
        """Use Gemini AI for accurate medical image analysis"""
        try:
            with Image.open(image_path) as image:
                # Determine if the image is likely an X-ray (grayscale-like)
                img_rgb = image.convert('RGB')
                arr = np.array(img_rgb)
                channel_std = np.std(arr, axis=(0, 1))
                is_grayscale_like = float(np.mean(channel_std)) < 5.0

                region_note = prompt_context(region)

                if is_grayscale_like:
                    prompt = f"""
                    You are a medical AI assistant specializing in radiography. Analyze this X-ray image and provide a concise, clinically relevant assessment focused on bone and joint findings.

                    Consider: fracture lines, cortical discontinuity, displacement/angulation, joint alignment, visible hardware, soft-tissue swelling.
                    If no clear fracture is seen, state that explicitly and suggest appropriate next steps.

                    detected_conditions should list specific findings (e.g., "distal radius fracture", "no acute fracture detected").
                    recommendations should be specific next steps: immobilization, urgent orthopedic consult, CT/MRI suggestions, follow-up timing.

                    {region_note}
                    """
                else:
                    prompt = f"""
                    You are a medical AI assistant. Analyze this clinical image.

                    Focus on visible features such as wounds, burns, bruises, rashes, swelling, or infection.
                    Be specific in detected_conditions. If uncertain, state uncertainty clearly.
                    recommendations should be specific treatment/care steps.

                    {region_note}
                    """

                response = self.client.models.generate_content(
                    model=self.MODEL_NAME,
                    contents=[prompt, img_rgb],
                    config=types.GenerateContentConfig(
                        response_mime_type='application/json',
                        response_schema=ImageAnalysisResult,
                    ),
                )

                return self._parse_gemini_response(response)

        except Exception:
            logger.warning(
                "Gemini image analysis failed for %s - falling back to basic analysis",
                image_path, exc_info=True
            )
            return self._analyze_basic(image_path)

    def _parse_gemini_response(self, response) -> Dict:
        """
        Turn the Gemini response into our standard dict shape.
        With response_schema set, Gemini is constrained to return valid JSON
        matching ImageAnalysisResult, so `response.parsed` is reliably populated
        instead of needing to hunt for '{' / '}' in free text.
        """
        parsed = response.parsed  # an ImageAnalysisResult instance, or None on failure

        if parsed is not None:
            result = parsed.model_dump()
        else:
            # Structured output failed to validate (rare) - fall back to
            # parsing response.text as JSON directly, since it's still
            # constrained to be JSON by response_mime_type.
            logger.warning("Gemini response.parsed was empty; falling back to raw JSON text parsing")
            try:
                result = json.loads(response.text)
            except (json.JSONDecodeError, AttributeError, TypeError):
                logger.error("Gemini response could not be parsed as JSON at all", exc_info=True)
                result = {}

        return {
            'detected_conditions': result.get('detected_conditions') or ['Medical condition analysis'],
            'confidence': result.get('confidence', 'medium'),
            'recommendations': result.get('recommendations') or [
                'Clean the affected area gently',
                'Monitor for changes or worsening',
                'Seek medical attention if symptoms persist',
                'Follow proper wound care protocols'
            ],
            'urgency': result.get('urgency', 'medium'),
            'safety_tips': result.get('safety_tips') or self._get_safety_tips(),
            'disclaimer': 'AI analysis for educational purposes only. Consult healthcare professionals.'
        }

    def _analyze_basic(self, image_path: str) -> Dict:
        """Fallback basic analysis when Gemini is not available"""
        with Image.open(image_path) as image:
            image_rgb = np.array(image.convert('RGB'))

        analysis_result = self._analyze_visual_features(image_rgb)
        recommendations = self._generate_recommendations(analysis_result)

        return {
            'detected_conditions': analysis_result['conditions'],
            'confidence': analysis_result['confidence'],
            'recommendations': recommendations,
            'safety_tips': self._get_safety_tips(),
            'disclaimer': 'Basic analysis only. For accurate diagnosis, please add Gemini API key and consult healthcare professionals.'
        }

    def _analyze_visual_features(self, image: np.ndarray) -> Dict:
        """Analyze visual features of the image"""
        height, width = image.shape[:2]

        # Color analysis
        avg_color = np.mean(image, axis=(0, 1))
        red_intensity = avg_color[0] / 255.0

        # Detect potential conditions based on color and texture
        conditions = []
        confidence = 0.0

        # Red coloration detection (potential cuts, burns, inflammation)
        if red_intensity > 0.6:
            conditions.append('possible_inflammation_or_injury')
            confidence += 0.3

        # Dark coloration detection (potential bruising)
        if np.mean(avg_color) < 100:
            conditions.append('possible_bruising')
            confidence += 0.2

        # Simple texture analysis using standard deviation
        gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
        texture_variance = np.std(gray)
        edge_density = texture_variance / 255.0

        if edge_density > 0.1:
            conditions.append('textural_changes')
            confidence += 0.2

        # If no specific conditions detected, provide general assessment
        if not conditions:
            conditions.append('general_skin_assessment')
            confidence = 0.1

        return {
            'conditions': conditions,
            'confidence': min(confidence, 1.0),
            'color_analysis': {
                'red_intensity': red_intensity,
                'average_brightness': np.mean(avg_color)
            }
        }

    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate medical recommendations based on analysis"""
        recommendations = []
        conditions = analysis['conditions']

        if 'possible_inflammation_or_injury' in conditions:
            recommendations.extend([
                'Clean the area gently with mild soap and water',
                'Apply a cold compress to reduce inflammation',
                'Monitor for signs of infection (increased redness, warmth, pus)',
                'Seek medical attention if condition worsens'
            ])

        if 'possible_bruising' in conditions:
            recommendations.extend([
                'Apply ice pack for 15-20 minutes several times a day',
                'Elevate the affected area if possible',
                'Avoid further trauma to the area',
                'Monitor for increased swelling or severe pain'
            ])

        if 'textural_changes' in conditions:
            recommendations.extend([
                'Keep the area clean and dry',
                'Avoid harsh scrubbing or irritants',
                'Document changes with photos for medical consultation',
                'Consider scheduling a dermatological examination'
            ])

        # General recommendations
        recommendations.extend([
            'Maintain good hygiene in the affected area',
            'Avoid self-medication without professional guidance',
            'Seek immediate medical attention for severe symptoms',
            'Document symptoms and their progression'
        ])

        return recommendations

    def _get_safety_tips(self) -> List[str]:
        """Get general safety tips"""
        return [
            '🚨 Call emergency services ({EMERGENCY_NUMBER}) for severe injuries',
            '🩹 Keep a well-stocked first aid kit accessible',
            '🧼 Always wash hands before treating wounds',
            '💊 Know your allergies and current medications',
            '📱 Have emergency contacts readily available',
            '🏥 Know the location of nearest hospital/urgent care'
        ]
