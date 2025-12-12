# tests/core/test_data_processor.py
from core.data_processor import DataProcessor
from core.models import DataPoint


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

    # Extract original vs processed values
    original_temps = [dp.temperature for dp in raw]
    processed_temps = [dp.temperature for dp in result]  # now smoothed values are in same fields
    original_smokes = [dp.smoke for dp in raw]
    processed_smokes = [dp.smoke for dp in result]

    # Spike must be heavily suppressed
    assert max(processed_temps) < 60.0
    assert max(processed_smokes) < 0.6

    # Length preserved
    assert len(result) == len(raw)

    # Physical bounds respected
    assert all(dp.temperature >= 0 for dp in result)
    assert all(dp.smoke >= 0.0 and dp.smoke <= 1.0 for dp in result))


def test_short_data_returns_unchanged():
    """When data is shorter than window → return original values."""
    raw = [
        DataPoint(timestamp="2025-08-01T10:00:00Z", temperature=30.0, smoke=0.05, wind=3.0),
        DataPoint(timestamp="2025-08-01T10:01:00Z", temperature=31.0, smoke=0.06, wind=3.1),
    ]

    result = DataProcessor.process(raw)

    assert len(result) == 2
    assert result[0].temperature == 30.0
    assert result[1].smoke == 0.06