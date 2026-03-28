"""
MediaPipe Face Landmarker service.

Uses the new MediaPipe Tasks API (mp.tasks.python.vision) — NOT the deprecated
mp.solutions API.

Feature extraction produces a 1434-dimensional vector per frame:
  478 landmarks × 3 coordinates (x, y, z) = 1434 features
  Layout: [x0, y0, z0, x1, y1, z1, ..., x477, y477, z477]

This matches the feature format used to train the scaler.pkl and emotion_model.pkl
classifiers, as confirmed by model_config.json:
  {
    "feature_dim": 1434,
    "num_landmarks": 478,
    "num_coords": 3,
    "feature_order": "interleaved_xyz"
  }

Landmark index reference (MediaPipe FaceLandmarker — 478 landmarks total):
  - Landmarks 0–467:   468 core face mesh landmarks
  - Landmarks 468–472: Left iris (5 points)
  - Landmarks 473–477: Right iris (5 points)

Eye Aspect Ratio (EAR) landmarks (computed for diagnostics, NOT fed to model):
  Left eye:  [362, 385, 387, 263, 373, 380]
  Right eye: [33, 160, 158, 133, 153, 144]

Mouth Aspect Ratio (MAR) landmarks (computed for diagnostics):
  Upper lip top:    [13]
  Lower lip bottom: [14]
  Left corner:      [78]
  Right corner:     [308]

Head Pose estimation landmarks (for cv2.solvePnP, diagnostics only):
  Nose tip:         [1]
  Chin:             [152]
  Left eye corner:  [263]
  Right eye corner: [33]
  Left mouth corner:[287]
  Right mouth corner:[57]
"""

import logging
import math
from typing import Optional

import cv2
import numpy as np
import os
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"  # set before importing mediapipe

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from utils.frame_utils import decode_frame, bgr_to_rgb

logger = logging.getLogger("backend_v2.mediapipe")

# ─── Constants ────────────────────────────────────────────────────────────────
NUM_LANDMARKS = 478
NUM_COORDS = 3
FEATURE_DIM = NUM_LANDMARKS * NUM_COORDS  # 1434


def init_face_landmarker(model_path: str) -> mp_vision.FaceLandmarker:
    """
    Create and return a MediaPipe FaceLandmarker using the Tasks API.
    Uses VisionRunningMode.IMAGE since frames are processed individually.
    """
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )
    detector = mp_vision.FaceLandmarker.create_from_options(options)
    logger.info("FaceLandmarker initialised from %s", model_path)
    return detector


# ─── Diagnostic helpers (EAR / MAR / Head Pose) ──────────────────────────────
# These are computed for logging / future use but are NOT fed into the classifier,
# because the trained model expects raw 1434 landmark coordinates.

# Eye landmark indices for EAR calculation
# Left eye: P1=362, P2=385, P3=387, P4=263, P5=373, P6=380
# Right eye: P1=33, P2=160, P3=158, P4=133, P5=153, P6=144
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]

# Mouth landmark indices for MAR calculation
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308

# Head pose landmark indices for solvePnP
POSE_LANDMARK_IDX = [1, 152, 263, 33, 287, 57]

# 3D model points for head pose (generic face model, in arbitrary units)
MODEL_POINTS_3D = np.array(
    [
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corner
        (-150.0, -150.0, -125.0),  # Left mouth corner
        (150.0, -150.0, -125.0),  # Right mouth corner
    ],
    dtype=np.float64,
)


def _euclidean(p1, p2) -> float:
    """2D Euclidean distance between two landmark points."""
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def _compute_ear(landmarks, eye_indices: list[int]) -> float:
    """
    Eye Aspect Ratio (EAR) = (|P2-P6| + |P3-P5|) / (2 * |P1-P4|)
    A low EAR indicates the eye is closing (blink / drowsiness).
    """
    p1 = landmarks[eye_indices[0]]
    p2 = landmarks[eye_indices[1]]
    p3 = landmarks[eye_indices[2]]
    p4 = landmarks[eye_indices[3]]
    p5 = landmarks[eye_indices[4]]
    p6 = landmarks[eye_indices[5]]

    vertical_1 = _euclidean(p2, p6)
    vertical_2 = _euclidean(p3, p5)
    horizontal = _euclidean(p1, p4)

    if horizontal < 1e-6:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def _compute_mar(landmarks) -> float:
    """
    Mouth Aspect Ratio (MAR) = |top-bottom| / |left-right|
    A high MAR indicates the mouth is open (yawning / drowsiness).
    """
    top = landmarks[MOUTH_TOP]
    bottom = landmarks[MOUTH_BOTTOM]
    left = landmarks[MOUTH_LEFT]
    right = landmarks[MOUTH_RIGHT]

    vertical = _euclidean(top, bottom)
    horizontal = _euclidean(left, right)

    if horizontal < 1e-6:
        return 0.0
    return vertical / horizontal


def _compute_head_pose(landmarks, img_w: int, img_h: int) -> tuple[float, float, float]:
    """
    Estimate head pose (yaw, pitch, roll) via cv2.solvePnP.
    Returns angles in degrees.
    """
    image_points = np.array(
        [
            (landmarks[idx].x * img_w, landmarks[idx].y * img_h)
            for idx in POSE_LANDMARK_IDX
        ],
        dtype=np.float64,
    )

    focal_length = img_w
    center = (img_w / 2, img_h / 2)
    camera_matrix = np.array(
        [
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vec, _ = cv2.solvePnP(
        MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs
    )
    if not success:
        return 0.0, 0.0, 0.0

    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_mat)
    yaw, pitch, roll = float(angles[1]), float(angles[0]), float(angles[2])
    return yaw, pitch, roll


# ─── Main extraction function ────────────────────────────────────────────────


def extract_landmarks_from_frames(
    detector: mp_vision.FaceLandmarker,
    frames_bytes: list[bytes],
) -> tuple[np.ndarray, int, dict]:
    """
    Process a list of raw frame byte arrays through the FaceLandmarker.

    For each frame:
      1. Decode JPEG/PNG bytes → BGR → RGB
      2. Run detector.detect() to get face landmarks
      3. Flatten 478 landmarks × 3 coords into a 1434-dim vector

    Aggregation:
      - Compute the MEAN feature vector across all valid frames → shape (1, 1434)
      - If no frames have a face, returns a zero vector

    Returns:
        features:      np.ndarray of shape (1, 1434) — mean landmark vector
        valid_count:    int — number of frames where a face was detected
        diagnostics:    dict — average EAR, MAR, head pose across valid frames
    """
    valid_vectors: list[np.ndarray] = []
    ear_values: list[float] = []
    mar_values: list[float] = []
    pose_values: list[tuple[float, float, float]] = []

    for i, raw in enumerate(frames_bytes):
        try:
            bgr = decode_frame(raw)
            if bgr is None:
                logger.debug("Frame %d: decode failed, skipping", i)
                continue

            rgb = bgr_to_rgb(bgr)
            h, w, _ = rgb.shape

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)

            if not result.face_landmarks:
                logger.debug("Frame %d: no face detected, skipping", i)
                continue

            landmarks = result.face_landmarks[0]

            # ── Primary feature vector: raw 1434 landmark coordinates ──
            feature_vector = []
            for lm in landmarks:
                feature_vector.extend([lm.x, lm.y, lm.z])
            feature_arr = np.array(feature_vector, dtype=np.float32)

            if feature_arr.shape[0] != FEATURE_DIM:
                logger.warning(
                    "Frame %d: expected %d features, got %d — skipping",
                    i,
                    FEATURE_DIM,
                    feature_arr.shape[0],
                )
                continue

            valid_vectors.append(feature_arr)

            # ── Diagnostic features (logged, not fed to model) ──
            left_ear = _compute_ear(landmarks, LEFT_EYE_IDX)
            right_ear = _compute_ear(landmarks, RIGHT_EYE_IDX)
            avg_ear = (left_ear + right_ear) / 2.0
            ear_values.append(avg_ear)

            mar = _compute_mar(landmarks)
            mar_values.append(mar)

            yaw, pitch, roll = _compute_head_pose(landmarks, w, h)
            pose_values.append((yaw, pitch, roll))

        except Exception as exc:
            logger.warning("Frame %d: MediaPipe error — %s: %s", i, type(exc).__name__, exc)
            continue

    valid_count = len(valid_vectors)

    # ── Aggregate: mean across valid frames ──
    if valid_count > 0:
        stacked = np.stack(valid_vectors, axis=0)  # (N, 1434)
        mean_vector = np.mean(stacked, axis=0)  # (1434,)
        features = mean_vector.reshape(1, -1)  # (1, 1434)
    else:
        features = np.zeros((1, FEATURE_DIM), dtype=np.float32)

    # ── Diagnostic summary ──
    diagnostics = {
        "avg_ear": float(np.mean(ear_values)) if ear_values else 0.0,
        "avg_mar": float(np.mean(mar_values)) if mar_values else 0.0,
        "avg_yaw": float(np.mean([p[0] for p in pose_values])) if pose_values else 0.0,
        "avg_pitch": float(np.mean([p[1] for p in pose_values])) if pose_values else 0.0,
        "avg_roll": float(np.mean([p[2] for p in pose_values])) if pose_values else 0.0,
        "blink_rate_proxy": float(np.mean(ear_values)) if ear_values else 0.0,
    }

    logger.info(
        "Processed %d/%d frames | EAR=%.3f MAR=%.3f yaw=%.1f pitch=%.1f",
        valid_count,
        len(frames_bytes),
        diagnostics["avg_ear"],
        diagnostics["avg_mar"],
        diagnostics["avg_yaw"],
        diagnostics["avg_pitch"],
    )

    return features, valid_count, diagnostics
