from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
from medical_analyzer import MedicalAnalyzer
from symptom_checker import SymptomChecker
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize medical analyzer and symptom checker
medical_analyzer = MedicalAnalyzer()
symptom_checker = SymptomChecker()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Analyze the uploaded image
            analysis_result = medical_analyzer.analyze_image(filepath)
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'analysis': analysis_result
            })
        except Exception as e:
            return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/analyze_symptoms', methods=['POST'])
def analyze_symptoms():
    try:
        data = request.get_json()
        symptoms = data.get('symptoms', '')
        
        if not symptoms:
            return jsonify({'error': 'No symptoms provided'}), 400
        
        analysis_result = symptom_checker.analyze_symptoms(symptoms)
        
        return jsonify({
            'success': True,
            'analysis': analysis_result
        })
    except Exception as e:
        return jsonify({'error': f'Symptom analysis failed: {str(e)}'}), 500

@app.route('/emergency')
def emergency():
    return render_template('emergency.html')

if __name__ == '__main__':
    print("Starting Quick Aid Medical Assistant...")
    print("⚠️  DISCLAIMER: This tool is for educational purposes only.")
    print("⚠️  Always consult healthcare professionals for medical advice.")
    app.run(debug=True, host='0.0.0.0', port=5000)
