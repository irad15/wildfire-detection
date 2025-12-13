from typing import List
from fastapi import APIRouter

from core.detection_service import DetectionService
from core.models import DataPoint
from .validation import validate_detection_input

router = APIRouter()


@router.post("/detect")
def detect(request: List[DataPoint]):
    # Validate input data
    validate_detection_input(request)

    # Process the validated data
    return DetectionService.run_detection(request)


@router.get("/health")
def health():
    return {"status": "ok"}
