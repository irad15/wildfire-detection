from typing import List

from core.data_processor import DataProcessor
from core.event_detector import EventDetector
from core.models import DataPoint


class DetectionService:
    """High-level service coordinating data processing + event detection."""

    @staticmethod
    def run_detection(raw_records: List[DataPoint]):
        """
        Full detection pipeline:
        1. Smooth/process incoming records
        2. Run event detection
        3. Return EventsSummary (Pydantic model)
        """
        processed = DataProcessor.process(raw_records)
        summary = EventDetector.detect(processed)
        return summary