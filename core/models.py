from datetime import datetime
from typing import List

from pydantic import BaseModel, Field, field_validator


class DataPoint(BaseModel):
    """Sensor reading with validated fields for wildfire detection."""

    timestamp: str
    temperature: float = Field(..., ge=-50, le=100)
    smoke: float = Field(..., ge=0.0, le=1.0)
    wind: float = Field(..., ge=0.0)

    @field_validator('timestamp')
    @classmethod
    def validate_timestamp(cls, v: str) -> str:
        """Validate ISO-8601 format."""
        try:
            ts_str = v.replace('Z', '+00:00') if v.endswith('Z') else v
            datetime.fromisoformat(ts_str)
        except (ValueError, AttributeError):
            raise ValueError('timestamp must be a valid ISO-8601 format string (e.g., "2025-08-01T10:00:00Z")')
        return v


class Event(BaseModel):
    timestamp: str
    score: float


class EventsSummary(BaseModel):
    events: List[Event]
    event_count: int
    max_score: float
