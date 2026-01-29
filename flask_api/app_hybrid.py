from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Load the hybrid ensemble models
xgb_model_path = os.path.join(os.path.dirname(__file__), '..', 'xgboost_model.pkl')
knn_model_path = os.path.join(os.path.dirname(__file__), '..', 'knn_model.pkl')
label_encoder_path = os.path.join(os.path.dirname(__file__), '..', 'label_encoder.pkl')
selected_features_path = os.path.join(os.path.dirname(__file__), '..', 'selected_features.pkl')

print("Loading hybrid ensemble models...")
with open(xgb_model_path, 'rb') as file:
    xgb_model = pickle.load(file)
print("✓ XGBoost model loaded")

with open(knn_model_path, 'rb') as file:
    knn_model = pickle.load(file)
print("✓ k-NN model loaded")

with open(label_encoder_path, 'rb') as file:
    label_encoder = pickle.load(file)
print("✓ Label encoder loaded")

with open(selected_features_path, 'rb') as file:
    selected_features = pickle.load(file)
print(f"✓ Selected features loaded: {len(selected_features)} features")

# Load all symptoms from dataset
dataset_path = os.path.join(os.path.dirname(__file__), 'Disease and symptoms dataset.csv')
df = pd.read_csv(dataset_path, nrows=1)
ALL_SYMPTOMS = df.columns.tolist()[1:]  # All 377 symptoms

print(f"✓ Loaded {len(ALL_SYMPTOMS)} symptoms from dataset")
print(f"✓ XGBoost expects {len(selected_features)} features")
print(f"✓ Label encoder has {len(label_encoder.classes_)} disease classes")
print("✓ Hybrid ensemble ready for predictions!")

@app.route('/api/symptoms', methods=['GET'])
def get_symptoms():
    """Return list of all available symptoms"""
    return jsonify({
        'symptoms': ALL_SYMPTOMS,
        'count': len(ALL_SYMPTOMS)
    })

@app.route('/api/predict', methods=['POST'])
def predict_disease():
    """Predict disease using hybrid ensemble (XGBoost 70% + kNN 30%)"""
    try:
        data = request.get_json()
        
        if not data or 'symptoms' not in data:
            return jsonify({'error': 'No symptoms provided'}), 400
        
        user_symptoms = data['symptoms']
        
        # Create feature vector for all symptoms
        feature_vector_all = []
        for symptom in ALL_SYMPTOMS:
            feature_vector_all.append(1 if user_symptoms.get(symptom, False) else 0)
        
        # Select only the features used in training
        feature_vector_selected = [feature_vector_all[ALL_SYMPTOMS.index(feat)] 
                                   for feat in selected_features if feat in ALL_SYMPTOMS]
        
        # Convert to numpy array and reshape
        features = np.array(feature_vector_selected).reshape(1, -1)
        
        # Make predictions with both models
        xgb_proba = xgb_model.predict_proba(features)[0]
        knn_proba = knn_model.predict_proba(features)[0]
        
        # Weighted ensemble (70% XGBoost + 30% kNN)
        ensemble_proba = 0.7 * xgb_proba + 0.3 * knn_proba
        prediction = np.argmax(ensemble_proba)
        
        # Decode the prediction to get disease name
        disease_name = label_encoder.inverse_transform([prediction])[0]
        confidence = float(max(ensemble_proba) * 100)
        
        # Count active symptoms
        symptoms_checked = sum(feature_vector_all)
        
        response = {
            'disease': str(disease_name),
            'confidence': confidence,
            'symptoms_checked': symptoms_checked,
            'total_symptoms': len(ALL_SYMPTOMS),
            'model': 'Hybrid Ensemble (XGBoost 70% + kNN 30%)'
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'model': 'Hybrid Ensemble',
        'xgboost_loaded': xgb_model is not None,
        'knn_loaded': knn_model is not None,
        'label_encoder_loaded': label_encoder is not None,
        'symptoms_count': len(ALL_SYMPTOMS),
        'selected_features_count': len(selected_features)
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
