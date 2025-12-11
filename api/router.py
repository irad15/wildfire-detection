from fastapi import APIRouter

from core.detection_service import DetectionService
from core.models import SensorDataList

router = APIRouter()


@router.post("/detect")
def detect(request: SensorDataList):
    request_dicts = [item.model_dump() for item in request.root]
    return DetectionService.run_detection(request_dicts)


@router.get("/health")
def health():
    return {"status": "ok"}
