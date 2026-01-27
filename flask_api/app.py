from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), '..', 'random_forest_model.pkl')
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Load the label encoder
label_encoder_path = os.path.join(os.path.dirname(__file__), '..', 'label_encoder.pkl')
with open(label_encoder_path, 'rb') as file:
    label_encoder = pickle.load(file)

# Load symptoms from dataset to match model features
dataset_path = os.path.join(os.path.dirname(__file__), 'Disease and symptoms dataset.csv')
df = pd.read_csv(dataset_path, nrows=1)
SYMPTOMS = df.columns.tolist()[1:]  # All 377 symptoms
print(f"Loaded {len(SYMPTOMS)} symptoms from dataset")
print(f"Model expects {model.n_features_in_} features")
print(f"Label encoder has {len(label_encoder.classes_)} disease classes")

@app.route('/api/symptoms', methods=['GET'])
def get_symptoms():
    """Return list of all available symptoms"""
    return jsonify({
        'symptoms': SYMPTOMS,
        'count': len(SYMPTOMS)
    })

@app.route('/api/predict', methods=['POST'])
def predict_disease():
    """Predict disease based on selected symptoms"""
    try:
        data = request.get_json()
        
        if not data or 'symptoms' not in data:
            return jsonify({'error': 'No symptoms provided'}), 400
        
        user_symptoms = data['symptoms']
        
        # Create feature vector
        feature_vector = []
        for symptom in SYMPTOMS:
            feature_vector.append(1 if user_symptoms.get(symptom, False) else 0)
        
        # Convert to numpy array and reshape
        features = np.array(feature_vector).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)
        
        # Decode the prediction to get disease name
        disease_name = label_encoder.inverse_transform(prediction)[0]
        
        # Get prediction probability if available
        try:
            prediction_proba = model.predict_proba(features)
            confidence = float(max(prediction_proba[0]) * 100)
        except:
            confidence = None
        
        response = {
            'disease': str(disease_name),
            'confidence': confidence,
            'symptoms_checked': sum(feature_vector),
            'total_symptoms': len(SYMPTOMS)
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'model_loaded': model is not None,
        'label_encoder_loaded': label_encoder is not None,
        'symptoms_count': len(SYMPTOMS)
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
