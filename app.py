"""
Focus Detection System — Flask Backend (v3)

Clean, modular backend for webcam-based cognitive state detection.

Startup:
  - Loads MediaPipe FaceLandmarker
  - Loads scikit-learn classifier pipeline (scaler, model, label_encoder)
  - Initialises Groq LLM client

Endpoints:
  POST /analyze  — main inference endpoint
  GET  /health   — health check
"""

import os
import sys
import logging
import os
from flask import Flask, jsonify
import dotenv
from flask import Flask, jsonify
from flask_cors import CORS

os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"  # add this too

# ── Ensure backend_v3 package root is on sys.path ────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from routes.analyze import analyze_bp
from services.mediapipe_service import init_face_landmarker
from services.classifier_service import load_classifier_artifacts
from services.groq_service import init_groq_client

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("backend_v3")

# ── Load environment variables ───────────────────────────────────────────────
dotenv.load_dotenv()

# ── Configuration from .env ──────────────────────────────────────────────────
MEDIAPIPE_MODEL_PATH = os.getenv("MEDIAPIPE_MODEL_PATH", "face_landmarker.task")
MODEL_PATH = os.getenv("MODEL_PATH", "emotion_model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "scaler.pkl")
LABEL_ENCODER_PATH = os.getenv("LABEL_ENCODER_PATH", "label_encoder.pkl")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")


def create_app():
    # ── Flask app ──────────────────────────────────────────────────────────────
    app = Flask(__name__)

    # ── CORS — allow all origins for local dev ───────────────────────────────────
    CORS(app, resources={r"/*": {"origins": "*"}})

    # Load models
    try:
        app.config['mp_detector'] = init_face_landmarker(MEDIAPIPE_MODEL_PATH)
    except Exception as exc:
        logger.error("Failed to load MediaPipe model: %s", exc)
        app.config['mp_detector'] = None

    try:
        model, scaler, encoder = load_classifier_artifacts(
            MODEL_PATH, SCALER_PATH, LABEL_ENCODER_PATH
        )
        app.config['classifier_model'] = model
        app.config['classifier_scaler'] = scaler
        app.config['classifier_encoder'] = encoder
    except Exception as exc:
        logger.error("Failed to load classifier artifacts: %s", exc)
        app.config['classifier_model'] = None
        app.config['classifier_scaler'] = None
        app.config['classifier_encoder'] = None

    app.config['groq_client'] = init_groq_client(GROQ_API_KEY)

    logger.info("=== All models loaded. Server ready. ===")

    # ── Register Blueprint ───────────────────────────────────────────────────
    app.register_blueprint(analyze_bp)

    # ── Health check ─────────────────────────────────────────────────────────────
    @app.route("/health", methods=["GET"])
    def health():
        """Simple health check endpoint."""
        models_loaded = all(
            [
                app.config.get('mp_detector') is not None,
                app.config.get('classifier_model') is not None,
                app.config.get('classifier_scaler') is not None,
                app.config.get('classifier_encoder') is not None,
            ]
        )
        return jsonify({
            "status": "ok",
            "models_loaded": models_loaded,
        })
        
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=8000, debug=True, use_reloader=False)
