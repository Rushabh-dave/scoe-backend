"""
Focus Detection System — FastAPI Backend (v2)

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
from contextlib import asynccontextmanager

import dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ── Ensure backend_v2 package root is on sys.path ────────────────────────────
# This allows imports like `from services.mediapipe_service import ...`
# when running with `uvicorn main:app` from inside backend_v2/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from routes.analyze import router as analyze_router
from services.mediapipe_service import init_face_landmarker
from services.classifier_service import load_classifier_artifacts
from services.groq_service import init_groq_client

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("backend_v2")

# ── Load environment variables ───────────────────────────────────────────────
dotenv.load_dotenv()

# ── Configuration from .env ──────────────────────────────────────────────────
MEDIAPIPE_MODEL_PATH = os.getenv("MEDIAPIPE_MODEL_PATH", "face_landmarker.task")
MODEL_PATH = os.getenv("MODEL_PATH", "emotion_model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "scaler.pkl")
LABEL_ENCODER_PATH = os.getenv("LABEL_ENCODER_PATH", "label_encoder.pkl")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")


# ── Lifespan: load all models once at startup ────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models and external clients at startup, clean up on shutdown."""

    # MediaPipe FaceLandmarker
    try:
        app.state.mp_detector = init_face_landmarker(MEDIAPIPE_MODEL_PATH)
    except Exception as exc:
        logger.error("Failed to load MediaPipe model: %s", exc)
        app.state.mp_detector = None

    # Scikit-learn classifier pipeline
    try:
        model, scaler, encoder = load_classifier_artifacts(
            MODEL_PATH, SCALER_PATH, LABEL_ENCODER_PATH
        )
        app.state.classifier_model = model
        app.state.classifier_scaler = scaler
        app.state.classifier_encoder = encoder
    except Exception as exc:
        logger.error("Failed to load classifier artifacts: %s", exc)
        app.state.classifier_model = None
        app.state.classifier_scaler = None
        app.state.classifier_encoder = None

    # Groq LLM client
    app.state.groq_client = init_groq_client(GROQ_API_KEY)

    logger.info("=== All models loaded. Server ready. ===")
    yield
    logger.info("=== Shutting down. ===")


# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Focus Detection System",
    version="2.0.0",
    description="Webcam-based cognitive state detection with LLM-powered nudges",
    lifespan=lifespan,
)

# ── CORS — allow all origins for local dev ───────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Include routers ──────────────────────────────────────────────────────────
app.include_router(analyze_router)


# ── Health check ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """Simple health check endpoint."""
    models_loaded = all(
        [
            app.state.mp_detector is not None,
            app.state.classifier_model is not None,
            app.state.classifier_scaler is not None,
            app.state.classifier_encoder is not None,
        ]
    )
    return {
        "status": "ok",
        "models_loaded": models_loaded,
    }
