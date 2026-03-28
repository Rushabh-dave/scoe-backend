import cv2
import time
import joblib
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles


# =========================
# FILE PATHS
# =========================
EMOTION_MODEL_PATH = "emotion_model.pkl"
SCALER_PATH = "scaler.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"
FACE_LANDMARKER_TASK_PATH = "face_landmarker.task"

# From your training pipeline
NUM_LANDMARKS = 478
NUM_COORDS = 3
FEATURE_DIM = NUM_LANDMARKS * NUM_COORDS  # 1434


# =========================
# LOAD CLASSIFIER ARTIFACTS
# =========================
model = joblib.load(EMOTION_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)


# =========================
# GOOGLE FILE STYLE:
# Create FaceLandmarker object
# =========================
base_options = python.BaseOptions(model_asset_path=FACE_LANDMARKER_TASK_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)


# =========================
# GOOGLE FILE STYLE:
# draw_landmarks_on_image
# =========================
def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style()
        )
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style()
        )
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style()
        )
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style()
        )

    return annotated_image


# =========================
# FEATURE EXTRACTION
# =========================
def detection_result_to_feature_vector(detection_result):
    if not detection_result.face_landmarks:
        return None

    face_landmarks = detection_result.face_landmarks[0]

    feature_vector = []
    for lm in face_landmarks:
        feature_vector.extend([lm.x, lm.y, lm.z])

    feature_vector = np.array(feature_vector, dtype=np.float32)

    if feature_vector.shape[0] != FEATURE_DIM:
        print(f"[WARN] Expected {FEATURE_DIM} features, got {feature_vector.shape[0]}")
        return None

    return feature_vector


# =========================
# MODEL PREDICTION
# =========================
def predict_emotion(feature_vector, confidence_threshold=0.5):
    x = feature_vector.reshape(1, -1)
    x_scaled = scaler.transform(x)

    pred_idx = model.predict(x_scaled)[0]
    probs = model.predict_proba(x_scaled)[0]

    emotion = label_encoder.inverse_transform([pred_idx])[0]
    confidence = float(probs[pred_idx])

    if confidence < confidence_threshold:
        emotion = "Uncertain"

    return {
        "emotion": emotion,
        "confidence": confidence,
        "probabilities": {
            cls: float(prob) for cls, prob in zip(label_encoder.classes_, probs)
        }
    }


def get_color(label):
    if label == "Focused":
        return (0, 255, 0)
    if label == "Confused":
        return (0, 165, 255)
    if label == "Distracted":
        return (0, 0, 255)
    if label == "Uncertain":
        return (180, 180, 180)
    return (255, 255, 255)


# =========================
# MAIN WEBCAM LOOP
# =========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Could not open camera.")
    raise SystemExit

prev_time = time.time()

print("Press q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read frame from camera.")
        break

    frame = cv2.flip(frame, 1)

    # Google file style: create mp.Image from RGB image data
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # Google file style: detect(image)
    detection_result = detector.detect(mp_image)

    # Draw landmarks
    annotated_rgb = draw_landmarks_on_image(frame_rgb, detection_result)
    output_frame = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

    # Extract classifier features
    feature_vector = detection_result_to_feature_vector(detection_result)

    # FPS
    current_time = time.time()
    fps = 1.0 / max(current_time - prev_time, 1e-6)
    prev_time = current_time

    if feature_vector is None:
        lines = [
            "Emotion: No face detected",
            f"FPS: {fps:.1f}"
        ]
        color = (0, 255, 255)
    else:
        result = predict_emotion(feature_vector, confidence_threshold=0.5)
        color = get_color(result["emotion"])

        lines = [
            f"Emotion: {result['emotion']}",
            f"Confidence: {result['confidence']:.2%}",
            f"Focused: {result['probabilities'].get('Focused', 0.0):.2%}",
            f"Confused: {result['probabilities'].get('Confused', 0.0):.2%}",
            f"Distracted: {result['probabilities'].get('Distracted', 0.0):.2%}",
            f"FPS: {fps:.1f}"
        ]

    y = 30
    for line in lines:
        cv2.putText(
            output_frame,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA
        )
        y += 30

    cv2.imshow("Emotion Detection - MediaPipe Tasks", output_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()