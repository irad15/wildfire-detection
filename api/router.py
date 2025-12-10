from typing import List

from fastapi import APIRouter

from core.data_processor import DataProcessor
from core.event_detector import EventDetector

router = APIRouter()


@router.post("/detect")
def detect(request: List[dict]):
    # Step 1: Process and smooth the data
    processor = DataProcessor(request)
    processed = processor.process()

    # Step 2: Run detection and get the full summary
    detector = EventDetector(processed)
    summary = detector.detect()  # Returns AlertsSummary directly

    # Return it — FastAPI + Pydantic will serialize it correctly
    return summary


@router.get("/health")
def health():
    return {"status": "ok"}