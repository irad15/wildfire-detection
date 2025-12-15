from typing import List
from fastapi import HTTPException

from core.models import DataPoint


def validate_detection_input(data_points: List[DataPoint]) -> None:
    """Guardrail around the core logic: reject empty payloads early."""
    if not data_points:
        raise HTTPException(
            status_code=422,
            detail="Input data cannot be empty. Please provide at least one data point."
        )
