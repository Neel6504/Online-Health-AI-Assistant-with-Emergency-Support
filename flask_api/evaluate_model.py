import pickle
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Load the model and label encoder
print("Loading model...")
with open('../random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('../label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

print(f"\n{'='*60}")
print("MODEL EVALUATION SUMMARY")
print(f"{'='*60}")

# Model Information
print(f"\nModel Type: {type(model).__name__}")
print(f"Number of Features: {model.n_features_in_}")
print(f"Number of Disease Classes: {len(label_encoder.classes_)}")
print(f"Number of Trees: {model.n_estimators}")
print(f"Max Depth: {model.max_depth}")

# Check if model has training accuracy stored (from feature_importances_)
if hasattr(model, 'feature_importances_'):
    print("\n✓ Model has feature importances calculated")
    top_features = np.argsort(model.feature_importances_)[-10:][::-1]
    print(f"\nTop 10 Most Important Features (by index):")
    for idx, feat_idx in enumerate(top_features, 1):
        print(f"  {idx}. Feature {feat_idx}: {model.feature_importances_[feat_idx]:.4f}")

# Out-of-bag score (if available)
if hasattr(model, 'oob_score_') and model.oob_score:
    print(f"\n{'='*60}")
    print("OUT-OF-BAG (OOB) ACCURACY")
    print(f"{'='*60}")
    print(f"OOB Score: {model.oob_score_:.4f} ({model.oob_score_*100:.2f}%)")
    print("\nNote: OOB score is an estimate of the model's accuracy")
    print("on unseen data, calculated during training.")
else:
    print(f"\n{'='*60}")
    print("OOB Score not available (model wasn't trained with oob_score=True)")
    print(f"{'='*60}")

# Model parameters
print(f"\n{'='*60}")
print("MODEL PARAMETERS")
print(f"{'='*60}")
print(f"Bootstrap: {model.bootstrap}")
print(f"Criterion: {model.criterion}")
print(f"Min Samples Split: {model.min_samples_split}")
print(f"Min Samples Leaf: {model.min_samples_leaf}")
print(f"Max Features: {model.max_features}")

# Generate sample predictions to show model is working
print(f"\n{'='*60}")
print("SAMPLE PREDICTIONS TEST")
print(f"{'='*60}")

# Test with a few random symptom combinations
np.random.seed(42)
n_tests = 5

print(f"\nGenerating {n_tests} random symptom patterns for testing...\n")
for i in range(n_tests):
    # Create random symptom vector (5-10 random symptoms)
    n_symptoms = np.random.randint(5, 11)
    test_sample = np.zeros((1, model.n_features_in_))
    symptom_indices = np.random.choice(model.n_features_in_, n_symptoms, replace=False)
    test_sample[0, symptom_indices] = 1
    
    # Predict
    prediction = model.predict(test_sample)
    probabilities = model.predict_proba(test_sample)
    confidence = np.max(probabilities) * 100
    
    disease = label_encoder.inverse_transform(prediction)[0]
    
    print(f"Test {i+1}:")
    print(f"  Symptoms selected: {n_symptoms} random symptoms")
    print(f"  Predicted disease: {disease}")
    print(f"  Confidence: {confidence:.2f}%")
    print()

# Performance metrics summary
print(f"{'='*60}")
print("MODEL CAPABILITY SUMMARY")
print(f"{'='*60}")
print(f"✓ Can predict {len(label_encoder.classes_)} different diseases")
print(f"✓ Analyzes {model.n_features_in_} different symptoms")
print(f"✓ Uses ensemble of {model.n_estimators} decision trees")
print(f"✓ Model is functional and making predictions")

print(f"\n{'='*60}")
print("ACCURACY NOTES")
print(f"{'='*60}")
print("""
To calculate actual test accuracy, you need:
1. The original training dataset
2. A separate test dataset with true labels
3. Cross-validation scores from training

Without the original dataset, we can only show:
- Model is loaded and functional ✓
- Model structure and parameters ✓
- Sample predictions are working ✓
- Feature importances (if available) ✓

To get actual accuracy metrics, you would need to:
- Retrain the model with test/train split
- Or save test data during original training
- Or use the training script to show cross-validation scores
""")

print(f"\n{'='*60}")
print("Evaluation Complete!")
print(f"{'='*60}\n")
