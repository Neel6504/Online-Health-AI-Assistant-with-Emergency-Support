import pickle
import sys

# Load the model and inspect it
with open('../random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('../label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

print("Model type:", type(model))
print("Number of features expected:", model.n_features_in_)
print("\nDisease classes:")
print(label_encoder.classes_)
print("\nTotal classes:", len(label_encoder.classes_))
