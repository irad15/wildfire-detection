from typing import List

from core.data_processor import DataProcessor
from core.event_detector import EventDetector


class DetectionService:
    """High-level service coordinating data processing + event detection."""

    @staticmethod
    def run_detection(raw_records: List[dict]):
        """
        Full detection pipeline:
        1. Smooth/process incoming records
        2. Run event detection
        3. Return AlertsSummary (Pydantic model)
        """
        processor = DataProcessor(raw_records)
        processed = processor.process()

        detector = EventDetector(processed)
        summary = detector.detect()

        return summary
