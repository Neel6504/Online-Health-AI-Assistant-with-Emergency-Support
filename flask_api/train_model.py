import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("DISEASE PREDICTION MODEL TRAINING")
print("="*60)

# Dataset path
DATASET_PATH = 'Disease and symptoms dataset.csv'

try:
    # Load dataset
    print("\n1. Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    print(f"   ✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Separate features and target
    print("\n2. Preparing features and labels...")
    X = df.drop('diseases', axis=1)  # All columns except 'diseases'
    y = df['diseases']  # Disease names
    
    print(f"   ✓ Features: {X.shape[1]} symptoms")
    print(f"   ✓ Labels: {len(y.unique())} unique diseases")
    
    # Encode disease names to numbers
    print("\n3. Encoding disease labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"   ✓ Encoded {len(label_encoder.classes_)} disease classes")
    
    # Check for classes with too few samples
    print("\n4. Checking data distribution...")
    unique, counts = np.unique(y_encoded, return_counts=True)
    min_samples = counts.min()
    print(f"   Minimum samples per disease: {min_samples}")
    
    # Remove diseases with only 1 sample for stratified split
    if min_samples < 2:
        print(f"   ⚠ Removing diseases with < 2 samples for proper train/test split...")
        valid_mask = np.isin(y_encoded, unique[counts >= 2])
        X = X[valid_mask]
        y_encoded = y_encoded[valid_mask]
        print(f"   ✓ Dataset size after filtering: {len(X)} samples")
    
    # Split dataset
    print("\n5. Splitting dataset (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"   ✓ Training set: {X_train.shape[0]} samples")
    print(f"   ✓ Test set: {X_test.shape[0]} samples")
    
    # Train Random Forest model
    print("\n6. Training Random Forest model...")
    print("   (This may take a few moments...)")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    model.fit(X_train, y_train)
    print("   ✓ Model training completed!")
    
    # Evaluate model
    print("\n7. Evaluating model performance...")
    
    # Evaluate on a sample for memory efficiency
    sample_size = min(10000, len(X_test))
    sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
    X_test_sample = X_test.iloc[sample_indices]
    y_test_sample = y_test[sample_indices]
    
    y_pred_train = model.predict(X_train[:10000])  # Sample training set too
    y_pred_test = model.predict(X_test_sample)
    
    train_accuracy = accuracy_score(y_train[:10000], y_pred_train)
    test_accuracy = accuracy_score(y_test_sample, y_pred_test)
    
    print(f"   ✓ Training Accuracy (sample): {train_accuracy*100:.2f}%")
    print(f"   ✓ Test Accuracy (sample): {test_accuracy*100:.2f}%")
    
    # Save model and label encoder
    print("\n8. Saving model and label encoder...")
    with open('../random_forest_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("   ✓ Model saved: random_forest_model.pkl")
    
    with open('../label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    print("   ✓ Label encoder saved: label_encoder.pkl")
    
    # Display sample predictions
    print("\n9. Testing sample predictions...")
    sample_indices = np.random.choice(len(X_test), min(5, len(X_test)), replace=False)
    for idx in sample_indices:
        sample = X_test.iloc[idx:idx+1]
        true_label = label_encoder.inverse_transform([y_test[idx]])[0]
        pred_label = label_encoder.inverse_transform(model.predict(sample))[0]
        confidence = max(model.predict_proba(sample)[0]) * 100
        
        status = "✓" if true_label == pred_label else "✗"
        print(f"   {status} True: {true_label}")
        print(f"     Predicted: {pred_label} (Confidence: {confidence:.1f}%)")
    
    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\nModel Details:")
    print(f"  - Features: {X.shape[1]} symptoms")
    print(f"  - Classes: {len(label_encoder.classes_)} diseases")
    print(f"  - Trees: {model.n_estimators}")
    print(f"  - Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"\nFiles saved:")
    print(f"  - random_forest_model.pkl")
    print(f"  - label_encoder.pkl")
    print("\nYou can now restart your Flask API to use the new model!")
    print("="*60)
    
except FileNotFoundError:
    print(f"\n✗ ERROR: Dataset file not found!")
    print(f"  Looking for: {DATASET_PATH}")
    print(f"\nPlease:")
    print(f"  1. Place your dataset CSV file in the flask_api folder")
    print(f"  2. Update DATASET_PATH in this script")
    print(f"  3. Ensure CSV has symptom columns (0/1) and 'prognosis' column")
    
except Exception as e:
    print(f"\n✗ ERROR: {str(e)}")
    print(f"\nPlease check:")
    print(f"  1. Dataset format is correct (CSV with symptoms and 'prognosis' column)")
    print(f"  2. All symptom columns contain only 0 or 1 values")
    print(f"  3. 'prognosis' column contains disease names")
