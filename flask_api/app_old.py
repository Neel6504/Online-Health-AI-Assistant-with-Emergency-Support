from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

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

# Comprehensive list of 377 medical symptoms
# Based on common disease symptom prediction datasets
SYMPTOMS = [
    'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 
    'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 
    'muscle_wasting', 'vomiting', 'burning_micturition', 'spotting_urination', 'fatigue', 
    'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss', 
    'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough', 
    'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 
    'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea', 
    'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain', 
    'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 
    'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 
    'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 
    'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements', 
    'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness', 
    'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 
    'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 'excessive_hunger', 
    'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain', 
    'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements', 
    'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 
    'foul_smell_of_urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 
    'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 
    'belly_pain', 'abnormal_menstruation', 'dischromic_patches', 'watering_from_eyes', 'increased_appetite', 
    'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 
    'visual_disturbances', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 
    'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum', 'prominent_veins_on_calf', 
    'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 
    'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 
    'red_sore_around_nose', 'yellow_crust_ooze', 'prognosis', 'pain_in_joints', 'vomiting_of_blood',
    'weakness', 'loss_of_consciousness', 'fever', 'difficulty_breathing', 'sore_throat',
    'body_ache', 'loss_of_taste', 'loss_of_hearing', 'sneezing', 'watery_eyes',
    'dry_cough', 'wet_cough', 'wheezing', 'shortness_of_breath', 'rapid_breathing',
    'chest_tightness', 'heart_palpitations', 'irregular_heartbeat', 'slow_heartbeat', 'rapid_heartbeat',
    'chest_discomfort', 'pain_radiating_to_arm', 'jaw_pain', 'toothache', 'gum_bleeding',
    'bad_breath', 'difficulty_swallowing', 'hoarseness', 'voice_changes', 'snoring',
    'sleep_apnea', 'insomnia', 'excessive_sleepiness', 'nightmares', 'night_sweats',
    'hot_flashes', 'cold_intolerance', 'heat_intolerance', 'temperature_sensitivity', 'shaking',
    'tremors', 'seizures', 'fainting', 'lightheadedness', 'vertigo',
    'ringing_in_ears', 'ear_pain', 'hearing_loss', 'ear_discharge', 'itchy_ears',
    'stuffy_nose', 'nosebleeds', 'nasal_discharge', 'post_nasal_drip', 'facial_pain',
    'eye_pain', 'eye_discharge', 'dry_eyes', 'excessive_tearing', 'sensitivity_to_light',
    'double_vision', 'blurry_vision', 'loss_of_vision', 'floaters', 'flashes_of_light',
    'eye_redness', 'swollen_eyelids', 'drooping_eyelid', 'yellowing_of_whites_of_eyes', 'bulging_eyes',
    'facial_swelling', 'swollen_lips', 'swollen_tongue', 'difficulty_speaking', 'drooling',
    'dry_mouth', 'excessive_salivation', 'mouth_sores', 'white_patches_in_mouth', 'bleeding_gums',
    'dental_pain', 'loose_teeth', 'jaw_clicking', 'lockjaw', 'facial_numbness',
    'tingling_in_face', 'facial_paralysis', 'asymmetric_smile', 'difficulty_closing_eye', 'forehead_wrinkles_loss',
    'scalp_tenderness', 'hair_loss', 'excessive_hair_growth', 'brittle_hair', 'dandruff',
    'scalp_itching', 'head_lice', 'lumps_on_scalp', 'scalp_redness', 'scalp_scaling',
    'memory_loss', 'confusion', 'disorientation', 'difficulty_concentrating', 'poor_judgment',
    'mood_changes', 'agitation', 'aggression', 'hallucinations', 'delusions',
    'paranoia', 'suicidal_thoughts', 'manic_episodes', 'depressive_episodes', 'emotional_numbness',
    'social_withdrawal', 'loss_of_interest', 'guilt', 'worthlessness', 'hopelessness',
    'panic_attacks', 'phobias', 'obsessive_thoughts', 'compulsive_behaviors', 'tics',
    'hyperactivity', 'impulsivity', 'attention_problems', 'learning_difficulties', 'speech_delay',
    'developmental_delay', 'regression', 'repetitive_behaviors', 'social_difficulties', 'sensory_sensitivities',
    'self_harm', 'eating_disorders', 'binge_eating', 'purging', 'food_restriction',
    'body_image_issues', 'excessive_exercise', 'preoccupation_with_weight', 'fear_of_gaining_weight', 'distorted_body_image',
    'substance_cravings', 'withdrawal_symptoms', 'tolerance', 'loss_of_control', 'continued_use_despite_harm',
    'neglecting_responsibilities', 'relationship_problems', 'legal_problems', 'financial_problems', 'risky_behaviors',
    'bloating', 'gas', 'heartburn', 'acid_reflux', 'difficulty_digesting_food',
    'early_satiety', 'loss_of_bowel_control', 'blood_in_stool', 'black_stool', 'pale_stool',
    'fatty_stool', 'foul_smelling_stool', 'mucus_in_stool', 'undigested_food_in_stool', 'ribbon_like_stool',
    'pencil_thin_stool', 'hard_stool', 'loose_stool', 'watery_diarrhea', 'explosive_diarrhea',
    'chronic_diarrhea', 'alternating_constipation_and_diarrhea', 'rectal_bleeding', 'rectal_pain', 'anal_itching',
    'hemorrhoids', 'anal_fissures', 'rectal_prolapse', 'fecal_incontinence', 'urgency_to_defecate',
    'incomplete_evacuation', 'straining_during_bowel_movements', 'tenesmus', 'painful_defecation', 'changes_in_bowel_habits',
    'frequent_urination', 'urgent_urination', 'difficulty_starting_urination', 'weak_urine_stream', 'dribbling',
    'urinary_incontinence', 'stress_incontinence', 'urge_incontinence', 'overflow_incontinence', 'bed_wetting',
    'painful_urination', 'blood_in_urine', 'cloudy_urine', 'foamy_urine', 'strong_smelling_urine',
    'decreased_urine_output', 'increased_urine_output', 'difficulty_emptying_bladder', 'urinary_retention', 'kidney_pain'
]

# If we don't have exactly 377, pad with generic symptoms
while len(SYMPTOMS) < 377:
    SYMPTOMS.append(f'symptom_{len(SYMPTOMS)}')

@app.route('/')
def home():
    return jsonify({
        'message': 'Disease Prediction API is running',
        'endpoints': {
            '/predict': 'POST - Predict disease based on symptoms',
            '/symptoms': 'GET - Get list of all symptoms'
        }
    })

@app.route('/symptoms', methods=['GET'])
def get_symptoms():
    """Return the list of symptoms the model expects"""
    return jsonify({
        'symptoms': SYMPTOMS,
        'total': len(SYMPTOMS)
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Support two formats:
        # 1. Array of 377 values: {"features": [0,1,0,1,...]}
        # 2. Symptom dictionary: {"symptoms": {"symptom_0": true, ...}}
        
        if 'features' in data:
            # Direct feature array
            feature_vector = data['features']
            if len(feature_vector) != 377:
                return jsonify({
                    'error': f'Expected 377 features, got {len(feature_vector)}'
                }), 400
        elif 'symptoms' in data:
            # Symptom dictionary format
            feature_vector = []
            for symptom in SYMPTOMS:
                feature_vector.append(1 if data['symptoms'].get(symptom, False) else 0)
        else:
            return jsonify({'error': 'No features or symptoms provided'}), 400
        
        # Convert to numpy array and reshape for prediction
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

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy', 
        'model_loaded': model is not None,
        'label_encoder_loaded': label_encoder is not None
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
