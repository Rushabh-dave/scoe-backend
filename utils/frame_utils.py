"""
Utilities for decoding and converting webcam frames.
"""

import numpy as np
import cv2


def decode_frame(raw_bytes: bytes) -> np.ndarray | None:
    """
    Decode raw JPEG/PNG bytes into a BGR OpenCV image.
    Returns None if decoding fails.
    """
    nparr = np.frombuffer(raw_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return bgr


def bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR OpenCV image to RGB."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
