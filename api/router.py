from typing import List
from fastapi import APIRouter

from core.detection_service import DetectionService
from core.models import DataPoint

router = APIRouter()


@router.post("/detect")
def detect(request: List[DataPoint]):
    return DetectionService.run_detection(request)


@router.get("/health")
def health():
    return {"status": "ok"}
