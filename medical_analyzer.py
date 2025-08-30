import numpy as np
from PIL import Image
import json
import base64
import io
import os
from typing import Dict, List, Tuple
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class MedicalAnalyzer:
    def __init__(self):
        # Configure Gemini API
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key and api_key != 'your_gemini_api_key_here':
            genai.configure(api_key=api_key)
            self.use_gemini = True
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        else:
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

    def analyze_image(self, image_path: str) -> Dict:
        """Analyze medical image and provide recommendations"""
        try:
            if self.use_gemini:
                return self._analyze_with_gemini(image_path)
            else:
                return self._analyze_basic(image_path)
            
        except Exception as e:
            return {
                'error': f"Image analysis failed: {str(e)}",
                'recommendations': ['Unable to analyze image. Please consult a healthcare professional.'],
                'disclaimer': 'This tool cannot replace professional medical advice.'
            }

    def _analyze_with_gemini(self, image_path: str) -> Dict:
        """Use Gemini AI for accurate medical image analysis"""
        image = Image.open(image_path)
        
        prompt = """
        You are a medical AI assistant. Analyze this medical image carefully and provide:

        1. DETECTED CONDITIONS: List specific conditions, injuries, or abnormalities you can identify
        2. CONFIDENCE LEVEL: Rate your confidence (low/medium/high) 
        3. DETAILED RECOMMENDATIONS: Specific medical advice and treatment steps
        4. URGENCY: Classify as low/medium/high urgency
        5. SAFETY TIPS: Relevant safety and care instructions

        Be specific and accurate. If you cannot identify anything definitive, say so clearly.
        Focus on visible symptoms like:
        - Cuts, wounds, lacerations
        - Burns (1st, 2nd, 3rd degree)
        - Bruises, contusions
        - Rashes, skin conditions
        - Swelling, inflammation
        - Infections, discoloration

        Format as JSON with keys: detected_conditions, confidence, recommendations, urgency, safety_tips
        """

        try:
            response = self.model.generate_content([prompt, image])
            
            # Parse Gemini response
            analysis_text = response.text
            
            # Extract structured data from response
            return self._parse_gemini_response(analysis_text)
            
        except Exception as e:
            return self._analyze_basic(image_path)

    def _parse_gemini_response(self, response_text: str) -> Dict:
        """Parse Gemini response into structured format"""
        try:
            # Try to extract JSON if present
            if '{' in response_text and '}' in response_text:
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                json_str = response_text[start:end]
                parsed = json.loads(json_str)
                
                return {
                    'detected_conditions': parsed.get('detected_conditions', []),
                    'confidence': parsed.get('confidence', 'medium'),
                    'recommendations': parsed.get('recommendations', []),
                    'urgency': parsed.get('urgency', 'medium'),
                    'safety_tips': parsed.get('safety_tips', self._get_safety_tips()),
                    'disclaimer': 'AI analysis for educational purposes only. Consult healthcare professionals.'
                }
        except:
            pass
        
        # Fallback: parse text response
        lines = response_text.split('\n')
        conditions = []
        recommendations = []
        
        for line in lines:
            line = line.strip()
            if any(word in line.lower() for word in ['condition', 'injury', 'appears', 'shows', 'indicates']):
                conditions.append(line)
            elif any(word in line.lower() for word in ['recommend', 'should', 'treatment', 'care']):
                recommendations.append(line)
        
        return {
            'detected_conditions': conditions[:5] if conditions else ['Medical condition analysis'],
            'confidence': 'high',
            'recommendations': recommendations[:8] if recommendations else [
                'Clean the affected area gently',
                'Monitor for changes or worsening',
                'Seek medical attention if symptoms persist',
                'Follow proper wound care protocols'
            ],
            'urgency': 'medium',
            'safety_tips': self._get_safety_tips(),
            'disclaimer': 'AI analysis for educational purposes only. Consult healthcare professionals.'
        }

    def _analyze_basic(self, image_path: str) -> Dict:
        """Fallback basic analysis when Gemini is not available"""
        image = Image.open(image_path)
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
        gray = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
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
            '🚨 Call emergency services (911) for severe injuries',
            '🩹 Keep a well-stocked first aid kit accessible',
            '🧼 Always wash hands before treating wounds',
            '💊 Know your allergies and current medications',
            '📱 Have emergency contacts readily available',
            '🏥 Know the location of nearest hospital/urgent care'
        ]
