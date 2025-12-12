# tests/core/test_event_detector.py
from core.event_detector import EventDetector
from core.models import ProcessedDataPoint, EventsSummary


def test_no_events_on_flat_data():
    """Zero variance day → zero events"""
    data = [
        ProcessedDataPoint(
            timestamp="2025-08-01T00:00:00Z",
            temperature=25.1,
            smoke=0.010,
            wind=2.1,
            smoothed_temp=25.1,
            smoothed_smoke=0.010,
        )
    ] * 60

    result = EventDetector.detect(data)
    assert result.event_count == 0
    assert result.max_score < 30.0


def test_fire_with_smoke_and_temp_spike():
    """Real fire scenario → multiple high-score events"""
    data = []
    for i in range(20):
        t = 25.0 + (i if i < 10 else 60 - i * 2)
        s = 0.01 + i * 0.07
        data.append(ProcessedDataPoint(
            timestamp=f"2025-08-01T01:{i:02d}:00Z",
            temperature=t,
            smoke=s,
            wind=3.0 + i * 0.5,
            smoothed_temp=t,
            smoothed_smoke=s,
        ))

    result = EventDetector.detect(data)
    assert result.event_count >= 5
    assert result.max_score >= 90.0


def test_only_temperature_spike_no_smoke():
    """Temperature spike without smoke → lower score, maybe no alert"""
    data = [
        ProcessedDataPoint(
            timestamp=f"2025-08-01T02:00:{i:02d}Z",
            temperature=25.0 if i < 10 else 55.0,
            smoke=0.01,
            wind=2.0,
            smoothed_temp=25.0 if i < 10 else 55.0,
            smoothed_smoke=0.01,
        ) for i in range(20)
    ]

    result = EventDetector.detect(data)
    # Should still trigger some alert due to temp, but lower score
    assert result.event_count > 0
    assert result.max_score > 60.0