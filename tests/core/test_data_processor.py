# tests/core/test_data_processor.py

from core.data_processor import DataProcessor
from core.models import DataPoint
from core.config import SAVITZKY_GOLAY_WINDOW


def test_smoothing_reduces_extreme_spike():
    """A huge single spike should be significantly reduced after smoothing."""
    raw = [
        DataPoint(timestamp=f"2025-08-01T00:{i:02d}:00Z", temperature=25.0, smoke=0.01, wind=2.0)
        for i in range(30)
    ]

    # Inject massive spike
    raw[15].temperature = 99.9
    raw[15].smoke = 0.95

    result = DataProcessor.process(raw)

    processed_temps = [dp.temperature for dp in result]
    processed_smokes = [dp.smoke for dp in result]

    # Spike must be heavily suppressed
    assert max(processed_temps) < 60.0
    assert max(processed_smokes) < 0.6

    # Length preserved
    assert len(result) == len(raw)

    # Physical bounds respected
    assert all(0.0 <= dp.smoke <= 1.0 for dp in result)


def test_flat_data_remains_almost_unchanged():
    """Flat data with minor variations should not be significantly altered by smoothing."""
    raw = [
        DataPoint(
            timestamp=f"2025-08-01T10:{i:02d}:00Z",
            temperature=25.0 + 0.1 * i,
            smoke=0.01 + 0.001 * i,
            wind=2.0
        )
        for i in range(30)
    ]

    result = DataProcessor.process(raw)

    for original, processed in zip(raw, result):
        assert abs(processed.temperature - original.temperature) < 0.2
        assert abs(processed.smoke - original.smoke) < 0.005

    assert len(result) == len(raw)


def test_unsorted_data_is_sorted_by_timestamp():
    """Data should be sorted by timestamp before processing."""
    raw = [
        DataPoint(timestamp="2025-08-01T10:05:00Z", temperature=24.0, smoke=0.02, wind=2.1),
        DataPoint(timestamp="2025-08-01T10:01:00Z", temperature=23.0, smoke=0.01, wind=2.0),
        DataPoint(timestamp="2025-08-01T10:03:00Z", temperature=23.5, smoke=0.015, wind=2.05),
        DataPoint(timestamp="2025-08-01T10:00:00Z", temperature=22.0, smoke=0.009, wind=2.0),
    ]

    result = DataProcessor.process(raw)

    sorted_timestamps = sorted(dp.timestamp for dp in raw)
    result_timestamps = [dp.timestamp for dp in result]

    assert result_timestamps == sorted_timestamps
    assert len(result) == len(raw)


def test_short_signal_is_not_smoothed():
    """Signals shorter than the Savitzky-Golay window should remain unchanged."""
    size = SAVITZKY_GOLAY_WINDOW - 2  # ensure strictly smaller than window
    raw = [
        DataPoint(
            timestamp=f"2025-08-01T11:{i:02d}:00Z",
            temperature=20.0 + i,
            smoke=0.01 + i * 0.001,
            wind=1.5
        )
        for i in range(size)
    ]

    result = DataProcessor.process(raw)

    for original, processed in zip(raw, result):
        assert processed.temperature == round(original.temperature, 2)
        assert processed.smoke == round(original.smoke, 4)

    assert len(result) == len(raw)


def test_empty_input_returns_empty_list():
    """Empty input should return an empty result without error."""
    assert DataProcessor.process([]) == []
