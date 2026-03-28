"""
Pydantic schemas for the focus-detection API.
"""

from typing import Optional
from pydantic import BaseModel, Field


class BehaviouralData(BaseModel):
    """Behavioural metrics captured by the browser frontend."""

    wpm: float = Field(0.0, description="Words per minute typing speed")
    error_rate: float = Field(0.0, description="Ratio of errors to total keystrokes")
    scroll_rate: float = Field(0.0, description="Scroll events per second")
    idle_time: float = Field(0.0, description="Seconds of inactivity in the window")
    mouse_jitter: float = Field(0.0, description="Mouse movement variance / jitter factor")
    tab_switches: int = Field(0, description="Number of tab switches in the window")


class AnalyzeResponse(BaseModel):
    """Response returned by POST /analyze."""

    mental_state: str = Field(..., description="Predicted mental state label")
    confidence: float = Field(..., description="Confidence score for the prediction")
    llm_response: Optional[str] = Field(
        None, description="LLM-generated nudge (only if not focused)"
    )
    frame_count_processed: int = Field(
        ..., description="Number of frames where a face was successfully detected"
    )
    error: Optional[str] = Field(None, description="Error message, if any")
