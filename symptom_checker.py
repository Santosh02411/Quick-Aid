import json
import re
import os
from typing import Dict, List, Tuple
from collections import defaultdict
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class SymptomChecker:
    def __init__(self):
        # Configure Gemini API
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key and api_key != 'your_gemini_api_key_here':
            genai.configure(api_key=api_key)
            self.use_gemini = True
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.use_gemini = False
        self.symptom_database = {
            'fever': {
                'related_symptoms': ['chills', 'sweating', 'headache', 'fatigue'],
                'possible_conditions': ['flu', 'cold', 'infection', 'covid-19'],
                'recommendations': [
                    'Rest and stay hydrated',
                    'Take fever-reducing medication (acetaminophen/ibuprofen)',
                    'Monitor temperature regularly',
                    'Seek medical attention if fever exceeds 103°F (39.4°C)',
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
                    'Call 911 if severe or accompanied by other symptoms',
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
                    'Call 911 if breathing becomes extremely difficult'
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

    def analyze_symptoms(self, symptom_text: str) -> Dict:
        """Analyze symptoms and provide medical recommendations"""
        try:
            if self.use_gemini:
                return self._analyze_with_gemini(symptom_text)
            else:
                return self._analyze_basic_symptoms(symptom_text)
            
        except Exception as e:
            return {
                'error': f"Symptom analysis failed: {str(e)}",
                'recommendations': ['Unable to analyze symptoms. Please consult a healthcare professional.'],
                'disclaimer': 'This tool cannot replace professional medical advice.'
            }

    def _analyze_with_gemini(self, symptom_text: str) -> Dict:
        """Use Gemini AI for accurate symptom analysis"""
        prompt = f"""
        You are a medical AI assistant. Analyze these symptoms carefully: "{symptom_text}"

        Provide a comprehensive medical analysis with:

        1. DETECTED SYMPTOMS: List all symptoms mentioned
        2. POSSIBLE CONDITIONS: Most likely medical conditions (be specific)
        3. URGENCY LEVEL: low/medium/high based on symptom severity
        4. DETAILED RECOMMENDATIONS: Specific medical advice and next steps
        5. EMERGENCY ALERT: true/false if immediate medical attention needed
        6. SAFETY TIPS: Relevant care instructions

        Consider:
        - Symptom combinations and patterns
        - Severity indicators
        - Duration and progression
        - Age-related factors
        - Emergency warning signs

        Be accurate and specific. If symptoms suggest emergency conditions (chest pain, difficulty breathing, severe bleeding, stroke signs), clearly indicate high urgency.

        Format as JSON with keys: detected_symptoms, possible_conditions, urgency_level, recommendations, emergency_alert, safety_tips
        """

        try:
            response = self.model.generate_content(prompt)
            analysis_text = response.text
            
            return self._parse_gemini_symptom_response(analysis_text)
            
        except Exception as e:
            return self._analyze_basic_symptoms(symptom_text)

    def _parse_gemini_symptom_response(self, response_text: str) -> Dict:
        """Parse Gemini symptom response into structured format"""
        try:
            # Try to extract JSON if present
            if '{' in response_text and '}' in response_text:
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                json_str = response_text[start:end]
                parsed = json.loads(json_str)
                
                emergency_alert = {
                    'alert': parsed.get('emergency_alert', False),
                    'message': '🚨 EMERGENCY SYMPTOMS DETECTED - SEEK IMMEDIATE MEDICAL ATTENTION' if parsed.get('emergency_alert', False) else '',
                    'action': 'Call 911 or go to nearest emergency room immediately' if parsed.get('emergency_alert', False) else ''
                }
                
                return {
                    'detected_symptoms': [str(s) for s in parsed.get('detected_symptoms', [])],
                    'possible_conditions': [str(c) for c in parsed.get('possible_conditions', [])],
                    'urgency_level': parsed.get('urgency_level', 'medium'),
                    'recommendations': [str(r) for r in parsed.get('recommendations', [])],
                    'emergency_alert': emergency_alert,
                    'safety_tips': [str(t) for t in parsed.get('safety_tips', self._get_symptom_safety_tips())],
                    'disclaimer': 'AI analysis for educational purposes only. Always consult healthcare professionals.'
                }
        except:
            pass
        
        # Fallback: parse text response
        lines = response_text.split('\n')
        symptoms = []
        conditions = []
        recommendations = []
        
        for line in lines:
            line = line.strip()
            if any(word in line.lower() for word in ['symptom', 'experiencing', 'reports']):
                symptoms.append(line)
            elif any(word in line.lower() for word in ['condition', 'diagnosis', 'suggests', 'indicates']):
                conditions.append(line)
            elif any(word in line.lower() for word in ['recommend', 'should', 'treatment', 'advice']):
                recommendations.append(line)
        
        # Check for emergency keywords
        emergency_keywords = ['emergency', 'urgent', 'immediate', 'call 911', 'hospital']
        is_emergency = any(keyword in response_text.lower() for keyword in emergency_keywords)
        
        return {
            'detected_symptoms': [str(s) for s in symptoms[:5]] if symptoms else ['Symptom analysis completed'],
            'possible_conditions': [str(c) for c in conditions[:5]] if conditions else ['Medical evaluation needed'],
            'urgency_level': 'high' if is_emergency else 'medium',
            'recommendations': [str(r) for r in recommendations[:8]] if recommendations else [
                'Monitor symptoms closely',
                'Rest and stay hydrated',
                'Seek medical attention if symptoms worsen',
                'Follow up with healthcare provider'
            ],
            'emergency_alert': {
                'alert': is_emergency,
                'message': '🚨 EMERGENCY SYMPTOMS DETECTED - SEEK IMMEDIATE MEDICAL ATTENTION' if is_emergency else '',
                'action': 'Call 911 or go to nearest emergency room immediately' if is_emergency else ''
            },
            'safety_tips': self._get_symptom_safety_tips(),
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
        recommendations = self._generate_symptom_recommendations(analysis)
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
                'Call 911 or go to emergency room',
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
            if any(emergency in symptom for emergency in self.emergency_symptoms):
                emergency_found.append(symptom)
        
        if emergency_found:
            return {
                'alert': True,
                'message': '🚨 EMERGENCY SYMPTOMS DETECTED - SEEK IMMEDIATE MEDICAL ATTENTION',
                'symptoms': emergency_found,
                'action': 'Call 911 or go to nearest emergency room immediately'
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
