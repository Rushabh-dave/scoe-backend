"""
Classifier service — loads and runs the scikit-learn pipeline.

Expects three pickle files loaded at startup:
  - label_encoder.pkl  → sklearn LabelEncoder (decodes numeric → string label)
  - scaler.pkl         → sklearn StandardScaler (normalises the 1434-dim feature vector)
  - emotion_model.pkl  → trained sklearn classifier (e.g. SVC, RandomForest)

Pipeline per request:
  1. features (1, 1434) → scaler.transform → scaled (1, 1434)
  2. model.predict(scaled) → numeric prediction
  3. model.predict_proba(scaled) → confidence (max probability)
  4. label_encoder.inverse_transform(prediction) → string label
"""

import logging
from typing import Any

import joblib
import numpy as np

logger = logging.getLogger("backend_v2.classifier")


def load_artifact(path: str) -> Any:
    """Load a joblib-serialized artifact file and return the deserialized object."""
    obj = joblib.load(path)
    logger.info("Loaded artifact from %s", path)
    return obj


def load_classifier_artifacts(
    model_path: str, scaler_path: str, encoder_path: str
) -> tuple[Any, Any, Any]:
    """
    Load all three classifier artifacts at startup.

    Returns:
        (model, scaler, label_encoder)
    """
    model = load_artifact(model_path)
    scaler = load_artifact(scaler_path)
    label_encoder = load_artifact(encoder_path)
    logger.info(
        "Classifier pipeline ready — classes: %s",
        list(label_encoder.classes_) if hasattr(label_encoder, "classes_") else "unknown",
    )
    return model, scaler, label_encoder


def predict(
    features: np.ndarray,
    model: Any,
    scaler: Any,
    label_encoder: Any,
) -> tuple[str, float]:
    """
    Run the full classification pipeline on a feature vector.

    Args:
        features:       np.ndarray of shape (1, 1434)
        model:          trained sklearn classifier
        scaler:         fitted sklearn StandardScaler
        label_encoder:  fitted sklearn LabelEncoder

    Returns:
        (label, confidence) — e.g. ("Focused", 0.87)
    """
    # Step 1: Scale
    scaled = scaler.transform(features)

    # Step 2: Predict class
    prediction = model.predict(scaled)

    # Step 3: Get confidence score
    confidence = 0.0
    try:
        probas = model.predict_proba(scaled)
        confidence = float(np.max(probas))
    except AttributeError:
        # Model does not support predict_proba (e.g. some SVM configurations)
        # Fall back to decision_function if available
        try:
            decision = model.decision_function(scaled)
            # For multi-class, take the max decision value as a rough confidence proxy
            if decision.ndim > 1:
                confidence = float(np.max(decision))
            else:
                confidence = float(abs(decision[0]))
        except AttributeError:
            logger.warning("Model supports neither predict_proba nor decision_function")
            confidence = 1.0  # Default confidence when no probability is available

    # Step 4: Decode label
    label = label_encoder.inverse_transform(prediction)[0]

    logger.info("Prediction: %s (confidence=%.4f)", label, confidence)
    return str(label), round(confidence, 4)
