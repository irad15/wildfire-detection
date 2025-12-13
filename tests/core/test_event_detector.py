# tests/core/test_event_detector.py
from core.event_detector import EventDetector
from core.models import DataPoint


def test_real_fire_triggers_multiple_alerts():
    """Sudden temperature + smoke rise → high scores and multiple events."""
    data = []
    base_temp = 25.0
    base_smoke = 0.01
    for i in range(25):
        temp = base_temp + (i if i < 12 else 60 - (i - 12) * 2)
        smoke = base_smoke + i * 0.04
        wind = 2.0 + i * 0.3
        data.append(DataPoint(
            timestamp=f"2025-08-01T14:{i:02d}:00Z",
            temperature=temp,
            smoke=smoke,
            wind=wind
        ))

    result = EventDetector.detect(data)
    assert result.event_count >= 4
    assert result.max_score >= 95.0


def test_calm_day_no_alerts():
    """Perfectly flat data → zero events"""
    data = [
        DataPoint(timestamp="2025-08-01T12:00:00Z", temperature=25.0, smoke=0.010, wind=2.1)
        for _ in range(100)
    ]

    result = EventDetector.detect(data)
    assert result.event_count == 0
    assert result.max_score <= 20.0


def test_temperature_spike_without_smoke_limited_alert():
    """Only temperature spike → some score, but not full fire"""
    data = [
        DataPoint(timestamp=f"2025-08-01T15:{i:02d}:00Z",
                 temperature=25.0 if i < 10 else 60.0,
                 smoke=0.02,
                 wind=4.0)
        for i in range(20)
    ]

    result = EventDetector.detect(data)
    assert result.event_count > 0
    assert result.max_score > 50.0
    assert result.max_score < 90.0  # Not a full fire without smoke