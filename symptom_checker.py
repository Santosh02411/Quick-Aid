import json
import os
from typing import Dict, List, Literal
from collections import defaultdict
from google import genai
from google.genai import types
from pydantic import BaseModel
from dotenv import load_dotenv
from logging_config import get_logger
from localization import localize_analysis, prompt_context, DEFAULT_REGION

load_dotenv()

logger = get_logger('symptom_checker')


class SymptomAnalysisResult(BaseModel):
    """Schema Gemini is constrained to reply in - no more find('{')/rfind('}') guessing."""
    detected_symptoms: List[str]
    possible_conditions: List[str]
    urgency_level: Literal['low', 'medium', 'high']
    recommendations: List[str]
    emergency_alert: bool
    safety_tips: List[str]


class SymptomChecker:
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

        self.symptom_database = {
            'fever': {
                'related_symptoms': ['chills', 'sweating', 'headache', 'fatigue'],
                'possible_conditions': ['flu', 'cold', 'infection', 'covid-19'],
                'recommendations': [
                    'Rest and stay hydrated',
                    'Take fever-reducing medication (acetaminophen/ibuprofen)',
                    'Monitor temperature regularly',
                    'Seek medical attention if fever exceeds {FEVER_TEMP}',
                    'Contact doctor if fever persists more than 3 days'
                ],
                'urgency': 'medium'
            },
            'headache': {
                'related_symptoms': ['nausea', 'sensitivity to light', 'neck stiffness'],
                'possible_conditions': ['tension headache', 'migraine', 'sinus infection'],
                'recommendations': [
                    'Rest in a quiet, dark room',
                    'Apply cold or warm compress to head/neck',
                    'Stay hydrated',
                    'Consider over-the-counter pain relievers',
                    'Avoid known triggers'
                ],
                'urgency': 'low'
            },
            'chest_pain': {
                'related_symptoms': ['shortness of breath', 'nausea', 'sweating', 'dizziness'],
                'possible_conditions': ['heart attack', 'angina', 'muscle strain', 'anxiety'],
                'recommendations': [
                    '🚨 SEEK IMMEDIATE MEDICAL ATTENTION',
                    'Call {EMERGENCY_NUMBER} if severe or accompanied by other symptoms',
                    'Do not drive yourself to hospital',
                    'Chew aspirin if not allergic (only if advised by emergency services)'
                ],
                'urgency': 'high'
            },
            'cough': {
                'related_symptoms': ['sore throat', 'runny nose', 'fever', 'fatigue'],
                'possible_conditions': ['cold', 'flu', 'bronchitis', 'allergies'],
                'recommendations': [
                    'Stay hydrated with warm liquids',
                    'Use humidifier or breathe steam',
                    'Honey can help soothe throat (not for children under 1 year)',
                    'Rest and avoid irritants',
                    'See doctor if cough persists over 2 weeks'
                ],
                'urgency': 'low'
            },
            'abdominal_pain': {
                'related_symptoms': ['nausea', 'vomiting', 'fever', 'bloating'],
                'possible_conditions': ['gastritis', 'food poisoning', 'appendicitis', 'gastroenteritis'],
                'recommendations': [
                    'Rest and avoid solid foods initially',
                    'Stay hydrated with clear fluids',
                    'Apply heat pad to abdomen',
                    'Seek immediate care for severe pain or fever',
                    'Monitor for worsening symptoms'
                ],
                'urgency': 'medium'
            },
            'shortness_of_breath': {
                'related_symptoms': ['chest pain', 'wheezing', 'cough', 'fatigue'],
                'possible_conditions': ['asthma', 'pneumonia', 'heart problems', 'anxiety'],
                'recommendations': [
                    '🚨 SEEK IMMEDIATE MEDICAL ATTENTION if severe',
                    'Sit upright and try to stay calm',
                    'Use prescribed inhaler if available',
                    'Loosen tight clothing',
                    'Call {EMERGENCY_NUMBER} if breathing becomes extremely difficult'
                ],
                'urgency': 'high'
            },
            'nausea': {
                'related_symptoms': ['vomiting', 'dizziness', 'abdominal pain', 'headache'],
                'possible_conditions': ['food poisoning', 'gastroenteritis', 'motion sickness', 'pregnancy'],
                'recommendations': [
                    'Sip clear fluids slowly',
                    'Eat bland foods (BRAT diet: bananas, rice, applesauce, toast)',
                    'Rest and avoid strong odors',
                    'Try ginger or peppermint tea',
                    'Seek care if unable to keep fluids down for 24 hours'
                ],
                'urgency': 'low'
            }
        }

        self.emergency_symptoms = [
            'chest pain', 'shortness of breath', 'severe headache', 'loss of consciousness',
            'severe bleeding', 'difficulty breathing', 'severe abdominal pain',
            'signs of stroke', 'severe allergic reaction', 'high fever with stiff neck'
        ]

    def analyze_symptoms(self, symptom_text: str, region: str = DEFAULT_REGION) -> Dict:
        """Analyze symptoms and provide medical recommendations"""
        try:
            if self.use_gemini:
                result = self._analyze_with_gemini(symptom_text, region)
            else:
                result = self._analyze_basic_symptoms(symptom_text)

        except Exception as e:
            logger.error("Symptom analysis failed entirely for input length=%d", len(symptom_text), exc_info=True)
            return {
                'error': f"Symptom analysis failed: {str(e)}",
                'recommendations': ['Unable to analyze symptoms. Please consult a healthcare professional.'],
                'disclaimer': 'This tool cannot replace professional medical advice.'
            }

        # Applies regardless of which path produced the result: substitutes
        # the {EMERGENCY_NUMBER}/{FEVER_TEMP} placeholders used by our own
        # fallback strings. Real Gemini-generated text won't contain these
        # tokens, so this is a no-op there - the region context passed into
        # the prompt is what steers Gemini's own wording instead.
        return localize_analysis(result, region)

    def _analyze_with_gemini(self, symptom_text: str, region: str = DEFAULT_REGION) -> Dict:
        """Use Gemini AI for accurate symptom analysis"""
        region_note = prompt_context(region)
        prompt = f"""
        You are a medical AI assistant. Analyze these symptoms carefully: "{symptom_text}"

        Provide a comprehensive medical analysis considering:
        - Symptom combinations and patterns
        - Severity indicators
        - Duration and progression
        - Age-related factors
        - Emergency warning signs

        Be accurate and specific. If symptoms suggest emergency conditions (chest pain,
        difficulty breathing, severe bleeding, stroke signs), clearly set emergency_alert
        to true and urgency_level to "high".

        {region_note}
        """

        try:
            response = self.client.models.generate_content(
                model=self.MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type='application/json',
                    response_schema=SymptomAnalysisResult,
                ),
            )

            return self._parse_gemini_symptom_response(response)

        except Exception:
            logger.warning(
                "Gemini symptom analysis failed - falling back to basic analysis",
                exc_info=True
            )
            return self._analyze_basic_symptoms(symptom_text)

    def _parse_gemini_symptom_response(self, response) -> Dict:
        """
        Turn the Gemini response into our standard dict shape.
        With response_schema set, Gemini is constrained to return valid JSON
        matching SymptomAnalysisResult, so `response.parsed` is reliably
        populated instead of needing to hunt for '{' / '}' in free text.
        """
        parsed = response.parsed  # a SymptomAnalysisResult instance, or None on failure

        if parsed is not None:
            result = parsed.model_dump()
        else:
            logger.warning("Gemini response.parsed was empty; falling back to raw JSON text parsing")
            try:
                result = json.loads(response.text)
            except (json.JSONDecodeError, AttributeError, TypeError):
                logger.error("Gemini response could not be parsed as JSON at all", exc_info=True)
                result = {}

        is_emergency = bool(result.get('emergency_alert', False))

        return {
            'detected_symptoms': [str(s) for s in result.get('detected_symptoms', [])],
            'possible_conditions': [str(c) for c in result.get('possible_conditions', [])],
            'urgency_level': result.get('urgency_level', 'medium'),
            'recommendations': [str(r) for r in result.get('recommendations', [])] or [
                'Monitor symptoms closely',
                'Rest and stay hydrated',
                'Seek medical attention if symptoms worsen',
                'Follow up with healthcare provider'
            ],
            'emergency_alert': {
                'alert': is_emergency,
                'message': '🚨 EMERGENCY SYMPTOMS DETECTED - SEEK IMMEDIATE MEDICAL ATTENTION' if is_emergency else '',
                'action': 'Call {EMERGENCY_NUMBER} or go to nearest emergency room immediately' if is_emergency else ''
            },
            'safety_tips': [str(t) for t in result.get('safety_tips', [])] or self._get_symptom_safety_tips(),
            'disclaimer': 'AI analysis for educational purposes only. Always consult healthcare professionals.'
        }

    def _analyze_basic_symptoms(self, symptom_text: str) -> Dict:
        """Fallback basic symptom analysis"""
        symptoms = self._extract_symptoms(symptom_text.lower())

        if not symptoms:
            return {
                'error': 'No recognizable symptoms found',
                'recommendations': ['Please describe your symptoms more specifically'],
                'disclaimer': 'Basic analysis only. For accurate diagnosis, please add Gemini API key and consult healthcare professionals.'
            }

        analysis = self._analyze_symptom_combination(symptoms)
        specific_recommendations = self._collect_symptom_specific_recommendations(symptoms)
        generic_recommendations = self._generate_symptom_recommendations(analysis)
        # Specific, symptom-by-symptom guidance (e.g. the exact fever
        # threshold to watch for) is more actionable than the generic
        # urgency-tier text, so it leads; generic text fills in anything
        # not already covered. Dedup while preserving this order.
        recommendations = list(dict.fromkeys(specific_recommendations + generic_recommendations))
        emergency_check = self._check_emergency_symptoms(symptoms)

        return {
            'detected_symptoms': [str(s) for s in symptoms],
            'possible_conditions': [str(c) for c in analysis['conditions']],
            'recommendations': [str(r) for r in recommendations],
            'urgency_level': analysis['urgency'],
            'emergency_alert': emergency_check,
            'safety_tips': [str(t) for t in self._get_symptom_safety_tips()],
            'disclaimer': 'Basic analysis only. For accurate diagnosis, please add Gemini API key and consult healthcare professionals.'
        }

    def _collect_symptom_specific_recommendations(self, symptoms: List[str]) -> List[str]:
        """
        Pull the detailed, symptom-specific recommendations defined in
        self.symptom_database (e.g. the exact fever threshold, chest-pain
        instructions) for each detected symptom, deduplicated and in the
        order symptoms were detected.
        """
        collected: List[str] = []
        for symptom in symptoms:
            if symptom in self.symptom_database:
                collected.extend(self.symptom_database[symptom].get('recommendations', []))
        return list(dict.fromkeys(collected))

    def _extract_symptoms(self, text: str) -> List[str]:
        """Extract symptoms from text input"""
        detected_symptoms = []

        # Define symptom keywords and variations
        symptom_patterns = {
            'fever': ['fever', 'high temperature', 'hot', 'burning up'],
            'headache': ['headache', 'head pain', 'migraine', 'head hurts'],
            'chest_pain': ['chest pain', 'chest hurts', 'heart pain', 'chest tightness'],
            'cough': ['cough', 'coughing', 'hacking'],
            'abdominal_pain': ['stomach pain', 'belly pain', 'abdominal pain', 'stomach ache'],
            'shortness_of_breath': ['shortness of breath', 'hard to breathe', 'breathing difficulty', 'cant breathe'],
            'nausea': ['nausea', 'nauseous', 'sick to stomach', 'queasy'],
            'fatigue': ['tired', 'fatigue', 'exhausted', 'weak', 'no energy'],
            'dizziness': ['dizzy', 'lightheaded', 'spinning', 'vertigo'],
            'sore_throat': ['sore throat', 'throat pain', 'throat hurts']
        }

        for symptom, patterns in symptom_patterns.items():
            for pattern in patterns:
                if pattern in text:
                    detected_symptoms.append(symptom)
                    break

        return list(set(detected_symptoms))  # Remove duplicates

    def _analyze_symptom_combination(self, symptoms: List[str]) -> Dict:
        """Analyze combination of symptoms"""
        conditions = defaultdict(int)
        urgency_scores = []

        for symptom in symptoms:
            if symptom in self.symptom_database:
                symptom_data = self.symptom_database[symptom]

                # Add possible conditions
                for condition in symptom_data['possible_conditions']:
                    conditions[condition] += 1

                # Track urgency
                urgency_map = {'low': 1, 'medium': 2, 'high': 3}
                urgency_scores.append(urgency_map.get(symptom_data['urgency'], 1))

        # Determine overall urgency
        max_urgency = max(urgency_scores) if urgency_scores else 1
        urgency_levels = {1: 'low', 2: 'medium', 3: 'high'}

        # Sort conditions by frequency
        sorted_conditions = sorted(conditions.items(), key=lambda x: x[1], reverse=True)
        top_conditions = [condition for condition, count in sorted_conditions[:5]]

        return {
            'conditions': top_conditions,
            'urgency': urgency_levels[max_urgency]
        }

    def _generate_symptom_recommendations(self, analysis: Dict) -> List[str]:
        """Generate recommendations based on symptom analysis"""
        recommendations = []
        urgency = analysis['urgency']

        if urgency == 'high':
            recommendations.extend([
                '🚨 SEEK IMMEDIATE MEDICAL ATTENTION',
                'Call {EMERGENCY_NUMBER} or go to emergency room',
                'Do not delay medical care',
                'Have someone accompany you if possible'
            ])
        elif urgency == 'medium':
            recommendations.extend([
                'Contact your healthcare provider within 24 hours',
                'Monitor symptoms closely',
                'Seek immediate care if symptoms worsen',
                'Rest and stay hydrated'
            ])
        else:
            recommendations.extend([
                'Monitor symptoms and rest',
                'Stay hydrated and maintain good nutrition',
                'Contact healthcare provider if symptoms persist or worsen',
                'Use over-the-counter remedies as appropriate'
            ])

        # Add general care recommendations
        recommendations.extend([
            'Keep a symptom diary to track changes',
            'Avoid self-medication without professional guidance',
            'Maintain good hygiene to prevent spread of illness'
        ])

        return recommendations

    def _check_emergency_symptoms(self, symptoms: List[str]) -> Dict:
        """Check for emergency symptoms"""
        emergency_found = []

        for symptom in symptoms:
            # Symptom keys are underscore-separated (e.g. 'chest_pain') but
            # self.emergency_symptoms uses space-separated phrases (e.g.
            # 'chest pain'), so normalize before comparing - a plain
            # substring check here would silently never match.
            # Only check emergency-phrase-in-symptom (not the reverse): the
            # reverse would let a plain 'headache' match the emergency
            # phrase 'severe headache' just because it's a substring of it.
            normalized_symptom = symptom.replace('_', ' ')
            if any(emergency in normalized_symptom for emergency in self.emergency_symptoms):
                emergency_found.append(symptom)

        if emergency_found:
            return {
                'alert': True,
                'message': '🚨 EMERGENCY SYMPTOMS DETECTED - SEEK IMMEDIATE MEDICAL ATTENTION',
                'symptoms': emergency_found,
                'action': 'Call {EMERGENCY_NUMBER} or go to nearest emergency room immediately'
            }

        return {'alert': False}

    def _get_symptom_safety_tips(self) -> List[str]:
        """Get safety tips for symptom management"""
        return [
            '📞 Keep emergency contacts easily accessible',
            '💊 Know your current medications and allergies',
            '🌡️ Monitor vital signs (temperature, pulse) when ill',
            '💧 Stay hydrated unless advised otherwise',
            '🏥 Know location of nearest hospital/urgent care',
            '📝 Keep a health diary to track symptoms'
        ]
