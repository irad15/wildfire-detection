# tests/core/test_data_processor.py
import numpy as np
from datetime import datetime
from core.data_processor import DataProcessor
from core.models import DataPoint


def test_smoothing_applied_on_long_data():
    """Verify Savitzky-Golay smoothing actually changes values when there's noise."""
    raw = [
        DataPoint(timestamp="2025-08-01T00:00:00Z", temperature=25.0, smoke=0.01, wind=2.0),
        DataPoint(timestamp="2025-08-01T00:01:00Z", temperature=30.0, smoke=0.05, wind=2.1),
        DataPoint(timestamp="2025-08-01T00:02:00Z", temperature=60.0, smoke=0.80, wind=8.0),  # spike
        DataPoint(timestamp="2025-08-01T00:03:00Z", temperature=58.0, smoke=0.75, wind=7.5),
        DataPoint(timestamp="2025-08-01T00:04:00Z", temperature=35.0, smoke=0.20, wind=5.0),
    ] * 5  # make it long enough

    result = DataProcessor.process(raw)

    # Extract smoothed values
    smoothed_temps = [p.smoothed_temp for p in result]
    original_temps = [p.temperature for p in result]

    # The spike should be pulled down significantly
    peak_original = max(original_temps)
    peak_smoothed = max(smoothed_temps)
    assert peak_smoothed < peak_original - 5.0  # big reduction
    assert len(result) == len(raw)  # length preserved
    assert all(p.smoothed_temp >= 0 for p in result)
    assert all(0.0 <= p.smoothed_smoke <= 1.0 for p in result)


def test_short_data_no_smoothing():
    """Very short data should return original values unchanged."""
    raw = [
        DataPoint(timestamp="2025-08-01T00:00:00Z", temperature=25.0, smoke=0.01, wind=2.0),
        DataPoint(timestamp="2025-08-01T00:01:00Z", temperature=26.0, smoke=0.02, wind=2.1),
    ]

    result = DataProcessor.process(raw)

    assert len(result) == 2
    assert result[0].smoothed_temp == 25.0
    assert result[1].smoothed_smoke == 0.02