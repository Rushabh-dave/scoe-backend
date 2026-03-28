"""
POST /analyze route — accepts webcam frames + behavioural data,
runs the full inference pipeline, and returns the mental state + LLM nudge.
"""

import json
import logging

from flask import Blueprint, request, jsonify, current_app
from models.schemas import BehaviouralData
from services.mediapipe_service import extract_landmarks_from_frames
from services.classifier_service import predict
from services.groq_service import get_nudge

logger = logging.getLogger("backend_v3.analyze")

analyze_bp = Blueprint('analyze', __name__)

@analyze_bp.route("/analyze", methods=["POST"])
def analyze():
    """
    Analyze webcam frames and behavioural metrics to detect mental state.

    Accepts multipart/form-data with:
      - frames/frame_*: JPEG/PNG image files
      - behavioural_data/context: JSON string with behavioural metrics
    """
    # ── Parse behavioural data (Check both new and legacy keys) ──────────
    raw_behavioural_data = request.form.get("behavioural_data") or request.form.get("context") or "{}"
    try:
        raw_data = json.loads(str(raw_behavioural_data))
        behaviour = BehaviouralData(**raw_data)
    except (json.JSONDecodeError, ValueError) as exc:
        return jsonify({"detail": f"Invalid behavioural_data JSON: {exc}"}), 400

    # ── Read all frames from any key that starts with 'frame' ───────────
    frames_bytes = []
    
    # request.files is a MultiDict
    for field_name in request.files:
        if field_name == "frames" or field_name.startswith("frame"):
            for file_obj in request.files.getlist(field_name):
                data = file_obj.read()
                if data:
                    frames_bytes.append(data)
                    
    print(f"Received {len(frames_bytes)} frames")
    if not frames_bytes:
        return jsonify({
            "mental_state": "unknown",
            "confidence": 0.0,
            "llm_response": None,
            "frame_count_processed": 0,
            "error": "No valid frames received"
        }), 200 # Using 200 because schema allows return with error field

    # ── Retrieve loaded models from app config ────────────────────────────
    detector = current_app.config.get('mp_detector')
    model = current_app.config.get('classifier_model')
    scaler = current_app.config.get('classifier_scaler')
    label_encoder = current_app.config.get('classifier_encoder')
    groq_client = current_app.config.get('groq_client')

    if detector is None:
        return jsonify({"detail": "MediaPipe model not loaded"}), 503

    if model is None or scaler is None or label_encoder is None:
        return jsonify({"detail": "Classifier models not loaded"}), 503

    # ── Stage 1: MediaPipe extraction (synchronous) ──────────────────────
    try:
        features, valid_count, diagnostics = extract_landmarks_from_frames(
            detector, frames_bytes
        )
    except Exception as exc:
        logger.error("MediaPipe extraction failed: %s", exc)
        return jsonify({
            "mental_state": "unknown",
            "confidence": 0.0,
            "llm_response": None,
            "frame_count_processed": 0,
            "error": f"MediaPipe extraction error: {exc}"
        }), 200

    if valid_count == 0:
        return jsonify({
            "mental_state": "unknown",
            "confidence": 0.0,
            "llm_response": None,
            "frame_count_processed": 0,
            "error": "No faces detected in any frame"
        }), 200

    # ── Stage 2: Classification (synchronous) ────────────────────────────
    try:
        label, confidence = predict(
            features, model, scaler, label_encoder
        )
    except Exception as exc:
        logger.error("Classification failed: %s", exc)
        return jsonify({"detail": f"Classifier error: {exc}"}), 500

    # ── Stage 3: LLM nudge (only if not focused) (synchronous) ───────────
    llm_response = None
    print(label)
    if label.lower() != "focused":
        try:
            behaviour_dict = behaviour.model_dump()
            llm_response = get_nudge(
                groq_client, label, behaviour_dict
            )
        except Exception as exc:
            logger.warning("Groq call failed: %s — using fallback", exc)
            llm_response = (
                "It seems like you might be losing focus. "
                "Try taking a short break or removing distractions."
            )

    return jsonify({
        "mental_state": label,
        "confidence": confidence,
        "llm_response": llm_response,
        "frame_count_processed": valid_count,
        "error": None
    }), 200
