import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("OPTIMIZING MODEL - REDUCING SYMPTOMS")
print("="*60)

# Load current model to get feature importances
print("\n1. Loading original full model for feature analysis...")
print("   (Loading the 377-symptom model...)")

# We need to train a quick model with all symptoms to get importances
df_temp = pd.read_csv('Disease and symptoms dataset.csv')
disease_counts_temp = df_temp['diseases'].value_counts()
top_100_temp = disease_counts_temp.head(100).index.tolist()
df_temp = df_temp[df_temp['diseases'].isin(top_100_temp)]

X_temp = df_temp.drop('diseases', axis=1)
y_temp = df_temp['diseases']
symptom_names = X_temp.columns.tolist()

print(f"   ✓ Training temporary model with {len(symptom_names)} symptoms...")
le_temp = LabelEncoder()
y_temp_encoded = le_temp.fit_transform(y_temp)

temp_model = RandomForestClassifier(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1)
temp_model.fit(X_temp, y_temp_encoded)
print(f"   ✓ Temporary model trained for feature analysis")

# Load dataset
print("\n2. Loading dataset...")
df = pd.read_csv('Disease and symptoms dataset.csv')

# Filter to top 100 diseases
disease_counts = df['diseases'].value_counts()
top_100_diseases = disease_counts.head(100).index.tolist()
df_filtered = df[df['diseases'].isin(top_100_diseases)].copy()

X = df_filtered.drop('diseases', axis=1)
y = df_filtered['diseases']
symptom_names = X.columns.tolist()

print(f"   ✓ Current symptoms: {len(symptom_names)}")

# Get feature importances
print("\n3. Analyzing feature importance...")
feature_importance = temp_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'symptom': symptom_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(f"   ✓ Feature importances calculated")
print(f"\n   Top 10 most important symptoms:")
for i, row in feature_importance_df.head(10).iterrows():
    print(f"   {row['symptom']}: {row['importance']:.4f}")

# Select top N symptoms (let's try 150 for better balance)
N_SYMPTOMS = 150
top_symptoms = feature_importance_df.head(N_SYMPTOMS)['symptom'].tolist()

print(f"\n4. Selecting top {N_SYMPTOMS} most important symptoms...")
print(f"   ✓ Reduced from {len(symptom_names)} to {N_SYMPTOMS} symptoms")
print(f"   ✓ Reduction: {100 - (N_SYMPTOMS/len(symptom_names)*100):.1f}%")

# Filter dataset to use only top symptoms
print("\n5. Creating optimized dataset...")
X_optimized = X[top_symptoms]
print(f"   ✓ Dataset shape: {X_optimized.shape}")

# Encode labels
print("\n6. Encoding labels...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split dataset
print("\n7. Splitting dataset (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_optimized, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"   ✓ Training set: {X_train.shape[0]} samples")
print(f"   ✓ Test set: {X_test.shape[0]} samples")

# Train optimized model
print("\n8. Training optimized Random Forest model...")
print(f"   (Training with {N_SYMPTOMS} symptoms instead of {len(symptom_names)}...)")
optimized_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=25,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1,
    verbose=0
)
optimized_model.fit(X_train, y_train)
print("   ✓ Model training completed!")

# Evaluate
print("\n9. Evaluating optimized model...")
y_pred_train = optimized_model.predict(X_train)
y_pred_test = optimized_model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"   ✓ Training Accuracy: {train_accuracy*100:.2f}%")
print(f"   ✓ Test Accuracy: {test_accuracy*100:.2f}%")

# Save models
print("\n10. Saving optimized model and label encoder...")
with open('../random_forest_model.pkl', 'wb') as f:
    pickle.dump(optimized_model, f)
print("   ✓ Model saved: random_forest_model.pkl")

with open('../label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
print("   ✓ Label encoder saved: label_encoder.pkl")

# Save symptom list
print("\n11. Saving optimized symptom list...")
with open('symptoms.txt', 'w') as f:
    for symptom in top_symptoms:
        f.write(f"{symptom}\n")
print(f"   ✓ Saved {N_SYMPTOMS} symptoms to symptoms.txt")

# Test predictions
print("\n12. Testing sample predictions...")
sample_indices = np.random.choice(len(X_test), min(10, len(X_test)), replace=False)
correct = 0
confidence_scores = []

for idx in sample_indices:
    sample = X_test.iloc[idx:idx+1]
    true_label = label_encoder.inverse_transform([y_test[idx]])[0]
    pred_label = label_encoder.inverse_transform(optimized_model.predict(sample))[0]
    proba = optimized_model.predict_proba(sample)
    confidence = max(proba[0]) * 100
    confidence_scores.append(confidence)
    
    status = "✓" if true_label == pred_label else "✗"
    if true_label == pred_label:
        correct += 1
    print(f"   {status} True: {true_label}")
    print(f"     Predicted: {pred_label} (Confidence: {confidence:.1f}%)")

avg_confidence = np.mean(confidence_scores)

print("\n" + "="*60)
print("OPTIMIZATION COMPLETE - COMPARISON")
print("="*60)
print(f"\n{'Metric':<30} {'Before':<15} {'After'}")
print("-" * 60)
print(f"{'Number of Symptoms':<30} {'377':<15} {N_SYMPTOMS}")
print(f"{'Symptoms Reduction':<30} {'-':<15} {100 - (N_SYMPTOMS/377*100):.1f}%")
print(f"{'Test Accuracy':<30} {'87.22%':<15} {f'{test_accuracy*100:.2f}%'}")
print(f"{'Average Confidence':<30} {'20.8%':<15} {f'{avg_confidence:.1f}%'}")
print(f"{'Sample predictions correct':<30} {'8/10':<15} {f'{correct}/10'}")
print(f"{'User Experience':<30} {'Long scroll':<15} {'Easy to use!'}")

print("\n📝 Top 20 Most Important Symptoms (for reference):")
print("-" * 60)
for i, row in feature_importance_df.head(20).iterrows():
    print(f"{i+1:2}. {row['symptom']:<40} ({row['importance']:.4f})")

print("\n" + "="*60)
print("✓ Model optimized successfully!")
print("✓ Restart Flask API to load the new model")
print(f"✓ Users now only see {N_SYMPTOMS} relevant symptoms!")
print("="*60)
