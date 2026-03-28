# Focus Detection System — Backend v2

A modular FastAPI backend for webcam-based cognitive state detection. Processes video frames via MediaPipe Face Landmarker, classifies mental state with a pre-trained scikit-learn pipeline, and optionally generates LLM-powered nudges via Groq.

## Quick Start

### 1. Install dependencies

```bash
cd backend_v2
pip install -r requirements.txt
```

### 2. Place model files

Copy these files into the `backend_v2/` directory (or configure paths in `.env`):

| File | Source | Description |
|---|---|---|
| `face_landmarker.task` | [Download](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task) or copy from `scoe-backend/` | MediaPipe FaceLandmarker model |
| `emotion_model.pkl` | `scoe-backend/emotion_model.pkl` | Trained scikit-learn classifier |
| `scaler.pkl` | `scoe-backend/scaler.pkl` | Fitted StandardScaler |
| `label_encoder.pkl` | `scoe-backend/label_encoder.pkl` | Fitted LabelEncoder |

### 3. Set up `.env`

```bash
cp .env.example .env
```

Edit `.env` and add your Groq API key:

```
GROQ_API_KEY=gsk_your_actual_key_here
```

### 4. Run the server

```bash
uvicorn main:app --reload
```

The server starts at `http://127.0.0.1:8000`.

## API Endpoints

### `GET /health`

Health check — returns model loading status.

```json
{"status": "ok", "models_loaded": true}
```

### `POST /analyze`

Main inference endpoint. Accepts `multipart/form-data`:

**Request fields:**
- `frames` — list of JPEG/PNG image files (up to 30)
- `behavioural_data` — JSON string:

```json
{
  "wpm": 45,
  "error_rate": 0.12,
  "scroll_rate": 3.2,
  "idle_time": 8.5,
  "mouse_jitter": 0.4,
  "tab_switches": 2
}
```

**Response:**

```json
{
  "mental_state": "Distracted",
  "confidence": 0.87,
  "llm_response": "You seem distracted. Try closing extra tabs and...",
  "frame_count_processed": 28,
  "error": null
}
```

## Architecture

```
backend_v2/
├── main.py                      # FastAPI app, lifespan, CORS, health
├── routes/
│   └── analyze.py               # POST /analyze endpoint
├── services/
│   ├── mediapipe_service.py     # Face landmark extraction (Tasks API)
│   ├── classifier_service.py    # sklearn pipeline (scaler → model → encoder)
│   └── groq_service.py          # LLM nudge generation via Groq
├── models/
│   └── schemas.py               # Pydantic request/response schemas
├── utils/
│   └── frame_utils.py           # Frame decode/conversion helpers
├── requirements.txt
├── .env.example
└── README.md
```

## Notes

- **MediaPipe**: Uses the new Tasks API (`mp.tasks.python.vision`), NOT the deprecated `mp.solutions`
- **Feature vector**: 1434 dimensions (478 landmarks × 3 coordinates) matching the trained model
- **Models are loaded once** at startup via FastAPI's `lifespan` context manager
- **Blocking operations** (MediaPipe, sklearn, Groq) run in `asyncio` executor threads
- **LLM** is only called when the detected state is not "Focused"
