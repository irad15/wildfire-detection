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
    processed_temps = [dp.temperature for dp in result]  # now smoothed values are in same fields
    processed_smokes = [dp.smoke for dp in result]

    # Spike must be heavily suppressed
    assert max(processed_temps) < 60.0
    assert max(processed_smokes) < 0.6

    # Length preserved
    assert len(result) == len(raw)

    # Physical bounds respected
    assert all(dp.smoke >= 0.0 and dp.smoke <= 1.0 for dp in result)


def test_flat_data_remains_almost_unchanged():
    """Flat data with minor variations should not be significantly altered by smoothing."""
    # Create mostly flat data with tiny random noise
    raw = [
        DataPoint(
            timestamp=f"2025-08-01T10:{i:02d}:00Z",
            temperature=25.0 + 0.1 * i,  # tiny rise
            smoke=0.01 + 0.001 * i,
            wind=2.0
        )
        for i in range(30)
    ]

    result = DataProcessor.process(raw)

    # Values should remain close to original (difference < 0.2 for temp, < 0.005 for smoke)
    for original, processed in zip(raw, result):
        assert abs(processed.temperature - original.temperature) < 0.2
        assert abs(processed.smoke - original.smoke) < 0.005

    # Length preserved
    assert len(result) == len(raw)
