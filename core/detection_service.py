from typing import List

from core.data_processor import DataProcessor
from core.event_detector import EventDetector
from core.models import DataPoint


class DetectionService:
    """High-level service coordinating data processing + event detection."""

    @staticmethod
    def run_detection(raw_records: List[DataPoint]):
        """
        Full detection pipeline used by the API.

        Currently wired to the V1 algorithms (smoothing + scoring). To experiment
        with V2 (spike suppression + hysteresis), switch the calls below.
        """
        # V1 path: baseline smoothing + scoring
        processed = DataProcessor.process(raw_records)
        summary = EventDetector.detect(processed)

        # V2 path (optional, for local experiments):
        # processed = DataProcessor.process_v2(raw_records)
        # summary = EventDetector.detect_v2(processed)
        
        return summary