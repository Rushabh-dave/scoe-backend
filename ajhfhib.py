import json
import joblib

encoder = joblib.load("label_encoder.pkl")

config = {
    "feature_dim": 1434,
    "num_landmarks": 478,
    "num_coords": 3,
    "classes": list(encoder.classes_),
    "feature_order": "interleaved_xyz",
    "feature_order_description": "[x0,y0,z0, x1,y1,z1, ..., x477,y477,z477]"
}

with open("model_config.json", "w") as f:
    json.dump(config, f, indent=2)

print("model_config.json created")