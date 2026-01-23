import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("TRAINING WITH TOP 100 COMMON DISEASES")
print("="*60)

# Load dataset
print("\n1. Loading original dataset...")
df = pd.read_csv('Disease and symptoms dataset.csv')
print(f"   ✓ Original dataset: {df.shape[0]} rows, {len(df['diseases'].unique())} diseases")

# Get top 100 most common diseases
print("\n2. Selecting top 100 most common diseases...")
disease_counts = df['diseases'].value_counts()
top_100_diseases = disease_counts.head(100).index.tolist()
print(f"   ✓ Top 100 diseases selected")
print(f"   Sample range: {disease_counts.head(100).min()} to {disease_counts.head(100).max()} samples per disease")

# Filter dataset
print("\n3. Filtering dataset...")
df_filtered = df[df['diseases'].isin(top_100_diseases)].copy()
print(f"   ✓ Filtered dataset: {df_filtered.shape[0]} rows, {len(df_filtered['diseases'].unique())} diseases")

# Separate features and labels
print("\n4. Preparing features and labels...")
X = df_filtered.drop('diseases', axis=1)
y = df_filtered['diseases']
print(f"   ✓ Features: {X.shape[1]} symptoms")
print(f"   ✓ Labels: {len(y.unique())} diseases")

# Encode labels
print("\n5. Encoding disease labels...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"   ✓ Encoded {len(label_encoder.classes_)} disease classes")

# Split dataset
print("\n6. Splitting dataset (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"   ✓ Training set: {X_train.shape[0]} samples")
print(f"   ✓ Test set: {X_test.shape[0]} samples")

# Train model
print("\n7. Training Random Forest model...")
print("   (Training with 150 trees for better accuracy...)")
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=25,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1,
    verbose=0
)
model.fit(X_train, y_train)
print("   ✓ Model training completed!")

# Evaluate
print("\n8. Evaluating model performance...")
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"   ✓ Training Accuracy: {train_accuracy*100:.2f}%")
print(f"   ✓ Test Accuracy: {test_accuracy*100:.2f}%")

# Save models
print("\n9. Saving model and label encoder...")
with open('../random_forest_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("   ✓ Model saved: random_forest_model.pkl")

with open('../label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
print("   ✓ Label encoder saved: label_encoder.pkl")

# Test predictions
print("\n10. Testing sample predictions...")
sample_indices = np.random.choice(len(X_test), min(10, len(X_test)), replace=False)
correct = 0
confidence_scores = []

for idx in sample_indices:
    sample = X_test.iloc[idx:idx+1]
    true_label = label_encoder.inverse_transform([y_test[idx]])[0]
    pred_label = label_encoder.inverse_transform(model.predict(sample))[0]
    proba = model.predict_proba(sample)
    confidence = max(proba[0]) * 100
    confidence_scores.append(confidence)
    
    status = "✓" if true_label == pred_label else "✗"
    if true_label == pred_label:
        correct += 1
    print(f"   {status} True: {true_label}")
    print(f"     Predicted: {pred_label} (Confidence: {confidence:.1f}%)")

avg_confidence = np.mean(confidence_scores)

print("\n" + "="*60)
print("TRAINING COMPLETED - COMPARISON")
print("="*60)
print(f"\n{'Metric':<30} {'Before (773)':<15} {'After (100)'}")
print("-" * 60)
print(f"{'Number of diseases':<30} {'773':<15} {'100'}")
print(f"{'Test Accuracy':<30} {'65.47%':<15} {f'{test_accuracy*100:.2f}%'}")
print(f"{'Average Confidence':<30} {'~5-10%':<15} {f'{avg_confidence:.1f}%'}")
print(f"{'Sample predictions correct':<30} {'~3/5':<15} {f'{correct}/10'}")

print("\n" + "="*60)
print("✓ Model is now ready to use!")
print("✓ Restart Flask API to load the new model")
print("="*60)
