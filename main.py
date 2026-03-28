"""
Cognitive State Detection System — FastAPI Backend
Privacy-first, stateless inference pipeline per SRS v1.1
LLM layer: Groq (groq==1.1.2)
"""

import time
import logging
import os
import pickle
import json
from typing import Optional
import dotenv
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from groq import Groq

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

# ─── Logging (operational only — no payload content per SRS §8.2) ────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cognitive_state")

# ─── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="Cognitive State Detection", version="1.1.0")
dotenv.load_dotenv()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["Authorization", "Content-Type"],
)

# ─── Config ──────────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
MEDIAPIPE_MODEL_PATH = os.getenv(
    "MEDIAPIPE_MODEL_PATH", "face_landmarker.task"
)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# Fast, cheap Groq model — swap to llama-3.3-70b-versatile for higher quality
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "5.0"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.55"))
ENTROPY_THRESHOLD = float(os.getenv("ENTROPY_THRESHOLD", "1.2"))
MIN_DETECTABLE_FRAMES = int(os.getenv("MIN_DETECTABLE_FRAMES", "15"))

CLASSES = ["focused", "confused", "fatigued", "distracted"]

# ─── Z-score normalization stats (from training — replace with real values) ──
# Shape: (9,) for [wpm, error_rate, scroll_speed, wpm_avg, error_rate_avg,
#                   scroll_avg, wpm_delta, error_rate_delta, scroll_delta]
BEHAVIOR_MEAN = np.array([60.0, 0.05, 250.0, 60.0, 0.05, 250.0, 0.0, 0.0, 0.0])
BEHAVIOR_STD  = np.array([30.0, 0.04, 150.0, 30.0, 0.04, 150.0, 20.0, 0.03, 100.0])

# ─── Globals loaded at startup ────────────────────────────────────────────────
_model = None           # trained classifier (loaded from pkl)
_mp_detector = None     # MediaPipe FaceLandmarker
_groq_client: Optional[Groq] = None   # Groq SDK client


# ─── Startup ─────────────────────────────────────────────────────────────────
@app.on_event("startup")
def load_models():
    global _model, _mp_detector, _groq_client

    # Load classifier pkl
    if not os.path.exists(MODEL_PATH):
        logger.warning("Model file not found at %s — inference will fail", MODEL_PATH)
    else:
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
        logger.info("Classifier loaded from %s", MODEL_PATH)

    # Load MediaPipe FaceLandmarker
    if not os.path.exists(MEDIAPIPE_MODEL_PATH):
        logger.warning(
            "MediaPipe model not found at %s — visual extraction will fail",
            MEDIAPIPE_MODEL_PATH,
        )
    else:
        base_options = mp_python.BaseOptions(model_asset_path=MEDIAPIPE_MODEL_PATH)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
        )
        _mp_detector = mp_vision.FaceLandmarker.create_from_options(options)
        logger.info("MediaPipe FaceLandmarker loaded")

    # Initialise Groq client (reads GROQ_API_KEY from env automatically)
    if GROQ_API_KEY:
        _groq_client = Groq(api_key=GROQ_API_KEY, timeout=LLM_TIMEOUT)
        logger.info("Groq client initialised (model=%s)", GROQ_MODEL)
    else:
        logger.warning("GROQ_API_KEY not set — LLM reasoning will be disabled")


# ─── Request schema ───────────────────────────────────────────────────────────
class ContextSchema(BaseModel):
    window_index: int
    timestamp_ms: int
    video_absent: bool = False

    wpm: float
    error_rate: float
    scroll_speed: float

    wpm_avg: float
    error_rate_avg: float
    scroll_avg: float

    wpm_delta: float
    error_rate_delta: float
    scroll_delta: float

    @field_validator("wpm", "scroll_speed", "wpm_avg", "scroll_avg")
    @classmethod
    def non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("must be non-negative")
        return v

    @field_validator("error_rate", "error_rate_avg")
    @classmethod
    def proportion(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("must be between 0 and 1")
        return v


# ─── Response schema ──────────────────────────────────────────────────────────
class LatencyBreakdown(BaseModel):
    mediapipe: int
    fusion: int
    classify: int
    llm: Optional[int]


class AnalyzeResponse(BaseModel):
    window_index: int
    state: str
    confidence: float
    class_probs: dict
    high_entropy: bool
    reasoning: Optional[str]
    latency_ms: LatencyBreakdown


# ─── MediaPipe extraction ─────────────────────────────────────────────────────
def extract_landmarks(frames_bytes: list[bytes]) -> tuple[np.ndarray, int]:
    """
    Run MediaPipe on each JPEG frame.

    Returns:
        landmark_matrix: shape [30, 1500]  (30 frames × 468 landmarks × 3 coords)
                         Failed frames are filled with column-wise mean.
        detected_count:  number of frames where a face was found.
    """
    N = 30
    LANDMARKS = 468
    COORDS = 3
    FEATURE_DIM = LANDMARKS * COORDS  # 1404 — padded to 1500 with zeros for pose

    raw: list[Optional[np.ndarray]] = []

    for jpeg_bytes in frames_bytes:
        try:
            nparr = np.frombuffer(jpeg_bytes, np.uint8)
            bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if bgr is None:
                raw.append(None)
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w, _ = rgb.shape

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = _mp_detector.detect(mp_image)

            if result.face_landmarks:
                lm = result.face_landmarks[0]
                coords = np.array(
                    [[p.x, p.y, p.z] for p in lm], dtype=np.float32
                ).flatten()  # 1404
                # Pad to 1500 to accommodate pose keypoints if added later
                padded = np.zeros(1500, dtype=np.float32)
                padded[: len(coords)] = coords
                raw.append(padded)
            else:
                raw.append(None)
        except Exception:
            raw.append(None)

    detected_count = sum(1 for r in raw if r is not None)

    # Column-wise mean of detected frames (used to fill missing frames)
    detected_frames = [r for r in raw if r is not None]
    if detected_frames:
        col_mean = np.mean(detected_frames, axis=0)
    else:
        col_mean = np.zeros(1500, dtype=np.float32)

    matrix = np.stack(
        [r if r is not None else col_mean for r in raw], axis=0
    )  # [30, 1500]

    return matrix, detected_count


# ─── Entropy ──────────────────────────────────────────────────────────────────
def shannon_entropy(probs: np.ndarray) -> float:
    probs = np.clip(probs, 1e-9, 1.0)
    return float(-np.sum(probs * np.log(probs)))


# ─── Behavioral feature vector ────────────────────────────────────────────────
def build_behavior_vector(ctx: ContextSchema) -> np.ndarray:
    raw = np.array(
        [
            ctx.wpm,
            ctx.error_rate,
            ctx.scroll_speed,
            ctx.wpm_avg,
            ctx.error_rate_avg,
            ctx.scroll_avg,
            ctx.wpm_delta,
            ctx.error_rate_delta,
            ctx.scroll_delta,
        ],
        dtype=np.float32,
    )
    # Z-score normalization using training-set statistics (SRS §9.1)
    normalized = (raw - BEHAVIOR_MEAN) / (BEHAVIOR_STD + 1e-8)
    return normalized.astype(np.float32)


# ─── Model inference ──────────────────────────────────────────────────────────
def run_inference(
    landmark_matrix: np.ndarray, behavior_vec: np.ndarray, video_absent: bool
) -> np.ndarray:
    """
    Call the loaded model.  The model is expected to accept a dict with keys
    'landmarks' (shape [30, 1500]) and 'behavior' (shape [9,]) and return
    class probabilities as a numpy array of shape [4].

    Adapt this function to your actual pkl model interface.
    """
    if _model is None:
        raise RuntimeError("Model not loaded")

    if video_absent:
        landmark_matrix = np.zeros_like(landmark_matrix)

    # ── Adjust to your model's actual predict_proba / forward interface ──
    # Option A: sklearn-style (e.g., RandomForest / GradientBoosting)
    #   flat_input = np.concatenate([landmark_matrix.flatten(), behavior_vec])
    #   probs = _model.predict_proba([flat_input])[0]

    # Option B: PyTorch model wrapped in pkl
    #   with torch.no_grad():
    #       lm_tensor = torch.tensor(landmark_matrix).unsqueeze(0)  # [1, 30, 1500]
    #       beh_tensor = torch.tensor(behavior_vec).unsqueeze(0)    # [1, 9]
    #       logits = _model(lm_tensor, beh_tensor)
    #       probs = torch.softmax(logits, dim=-1).squeeze().numpy()

    # ── Placeholder — replace with your real call ──────────────────────────
    flat = np.concatenate([landmark_matrix.flatten(), behavior_vec])
    probs = _model.predict_proba([flat])[0]  # shape [4]

    return probs.astype(np.float32)


# ─── LLM reasoning via Groq ───────────────────────────────────────────────────
def _groq_blocking_call(state: str, ctx: ContextSchema) -> Optional[str]:
    """
    Synchronous Groq call — executed in FastAPI's thread pool via
    asyncio.get_event_loop().run_in_executor so it doesn't block the event loop.
    The Groq SDK (groq==1.1.2) is synchronous; this is the correct pattern.
    """
    if _groq_client is None:
        return None

    system_prompt = (
        "You are an attentive cognitive-state assistant. "
        "Describe the user's cognitive state in 2–3 neutral sentences (40–200 words total). "
        "State what the signals show, what cognitive mechanism this pattern suggests, "
        "and one non-intrusive suggestion. "
        "Never use words like 'failing', 'struggling', or 'poor performance'. "
        "Do not make value judgments about the person's capability. "
        "Reply with plain text only — no markdown, no bullet points."
    )

    user_prompt = (
        f"Predicted cognitive state: {state}\n"
        f"WPM this window : {ctx.wpm:.1f}  (baseline {ctx.wpm_avg:.1f},  delta {ctx.wpm_delta:+.1f})\n"
        f"Error rate       : {ctx.error_rate:.3f} (baseline {ctx.error_rate_avg:.3f}, delta {ctx.error_rate_delta:+.3f})\n"
        f"Scroll speed     : {ctx.scroll_speed:.1f} px/s (baseline {ctx.scroll_avg:.1f}, delta {ctx.scroll_delta:+.1f})\n\n"
        "Provide a brief, neutral explanation."
    )

    try:
        completion = _groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=300,
            temperature=0.4,   # low temp → consistent, factual tone
        )
        text = completion.choices[0].message.content.strip()

        # Validate word count per SRS §7.3 (40–200 words)
        word_count = len(text.split())
        if word_count < 40 or word_count > 200:
            logger.info("Groq response rejected: word count %d out of [40,200]", word_count)
            return None
        return text

    except Exception as exc:
        logger.info("Groq call failed: %s — %s", type(exc).__name__, exc)
        return None


async def call_llm(state: str, ctx: ContextSchema) -> Optional[str]:
    """Non-blocking wrapper: runs the synchronous Groq call in the default thread pool."""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _groq_blocking_call, state, ctx)


# ─── /analyze endpoint ────────────────────────────────────────────────────────
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    request: Request,
    context: str = Form(...),
    frames: list[UploadFile] = File(default=[]),
):
    t_total_start = time.monotonic()

    # ── Parse & validate context ─────────────────────────────────────────
    try:
        ctx_data = json.loads(context)
        ctx = ContextSchema(**ctx_data)
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_context", "detail": str(exc)},
        )

    # ── Validate frame count ─────────────────────────────────────────────
    if not ctx.video_absent and len(frames) != 30:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_frames",
                "detail": f"expected 30 frame parts, got {len(frames)}",
            },
        )

    # ── Read frames into memory (never touch disk) ───────────────────────
    frames_bytes: list[bytes] = []
    for f in frames:
        frames_bytes.append(await f.read())

    video_absent = ctx.video_absent

    # ── Stage 1: MediaPipe extraction ────────────────────────────────────
    t_mp_start = time.monotonic()
    if video_absent or not frames_bytes:
        landmark_matrix = np.zeros((30, 1500), dtype=np.float32)
        detected_count = 0
    else:
        landmark_matrix, detected_count = extract_landmarks(frames_bytes)
        if detected_count < MIN_DETECTABLE_FRAMES:
            logger.info(
                "Only %d/%d frames detected — applying video_absent fallback",
                detected_count,
                30,
            )
            video_absent = True
            landmark_matrix = np.zeros((30, 1500), dtype=np.float32)
    t_mp_ms = int((time.monotonic() - t_mp_start) * 1000)

    # ── Stage 2 & 3: Fusion + Classification ────────────────────────────
    t_infer_start = time.monotonic()
    behavior_vec = build_behavior_vector(ctx)

    try:
        probs = run_inference(landmark_matrix, behavior_vec, video_absent)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail={"error": "model_unavailable", "detail": str(exc)},
        )

    t_infer_ms = int((time.monotonic() - t_infer_start) * 1000)
    t_fusion_ms = t_infer_ms // 2          # approximate split; adjust if stages are separate
    t_classify_ms = t_infer_ms - t_fusion_ms

    # ── Entropy & confidence ─────────────────────────────────────────────
    top_idx = int(np.argmax(probs))
    top_state = CLASSES[top_idx]
    confidence = float(probs[top_idx])
    entropy = shannon_entropy(probs)
    high_entropy = entropy > ENTROPY_THRESHOLD

    class_probs = {cls: float(probs[i]) for i, cls in enumerate(CLASSES)}

    # ── Stage 4: LLM reasoning (conditional) ────────────────────────────
    t_llm_ms: Optional[int] = None
    reasoning: Optional[str] = None

    llm_triggered = (
        top_state != "focused"
        and confidence > CONFIDENCE_THRESHOLD
        and not high_entropy
    )

    if llm_triggered:
        t_llm_start = time.monotonic()
        reasoning = await call_llm(top_state, ctx)
        t_llm_ms = int((time.monotonic() - t_llm_start) * 1000)

    # ── Operational log (no payload content per SRS §8.2) ────────────────
    t_total_ms = int((time.monotonic() - t_total_start) * 1000)
    logger.info(
        "status=200 latency_ms=%d mediapipe_ms=%d infer_ms=%d llm_called=%s model=v1.1",
        t_total_ms,
        t_mp_ms,
        t_infer_ms,
        llm_triggered,
    )

    return AnalyzeResponse(
        window_index=ctx.window_index,
        state=top_state,
        confidence=round(confidence, 4),
        class_probs={k: round(v, 4) for k, v in class_probs.items()},
        high_entropy=high_entropy,
        reasoning=reasoning,
        latency_ms=LatencyBreakdown(
            mediapipe=t_mp_ms,
            fusion=t_fusion_ms,
            classify=t_classify_ms,
            llm=t_llm_ms,
        ),
    )


# ─── Error handlers ───────────────────────────────────────────────────────────
@app.exception_handler(413)
async def payload_too_large(_req: Request, _exc):
    return JSONResponse(
        status_code=413,
        content={"error": "payload_too_large", "max_bytes": 819200},
    )


@app.exception_handler(429)
async def rate_limited(_req: Request, _exc):
    return JSONResponse(
        status_code=429,
        content={"error": "rate_limited", "retry_after_ms": 8000},
    )