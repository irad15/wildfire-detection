# tests/core/test_event_detector.py

from core.config import ALERT_THRESHOLD
from core.event_detector import EventDetector
from core.models import DataPoint


def test_real_fire_triggers_multiple_alerts():
    """Combined temperature + smoke anomalies should trigger alerts."""
    data = []

    for i in range(25):
        data.append(DataPoint(
            timestamp=f"2025-08-01T14:{i:02d}:00Z",
            temperature=25.0 + i * 2,
            smoke=0.01 + i * 0.04,
            wind=2.0 + i * 0.2
        ))

    result = EventDetector.detect(data)

    assert result.event_count > 0
    assert result.max_score > ALERT_THRESHOLD


def test_calm_day_no_alerts():
    """Stable signals must not generate alerts."""
    data = [
        DataPoint(
            timestamp=f"2025-08-01T12:{i:02d}:00Z",
            temperature=25.0,
            smoke=0.01,
            wind=2.0
        )
        for i in range(50)
    ]

    result = EventDetector.detect(data)

    assert result.event_count == 0


def test_temperature_spike_without_smoke_no_alert():
    """Temperature anomaly alone must not trigger an alert."""
    data = [
        DataPoint(
            timestamp=f"2025-08-01T15:{i:02d}:00Z",
            temperature=25.0 if i < 10 else 65.0,
            smoke=0.02,
            wind=3.0
        )
        for i in range(20)
    ]

    result = EventDetector.detect(data)

    assert result.event_count == 0

    # Even with a significant temperature spike, the absence of smoke should yield a moderately elevated score,
    # but it should not reach the alert threshold (no false positives).
    assert 30.0 < result.max_score < 60.0


def test_empty_input_returns_empty_summary():
    """Detector must safely handle empty input."""
    result = EventDetector.detect([])

    assert result.events == []
    assert result.event_count == 0
    assert result.max_score == 0.0


def test_scores_are_bounded():
    """Whenever events are produced, their scores must be within [0, 100]."""
    data = []

    for i in range(30):
        data.append(DataPoint(
            timestamp=f"2025-08-01T16:{i:02d}:00Z",
            temperature=25.0 + i * 2,   # rising temp → variance
            smoke=0.01 + i * 0.03,      # rising smoke → variance
            wind=5.0 + i * 0.5          # amplifying wind
        ))

    result = EventDetector.detect(data)

    # Sanity: ensure the test actually exercises the scoring path
    assert result.event_count > 0

    for event in result.events:
        assert 0.0 <= event.score <= 100.0

    assert 0.0 <= result.max_score <= 100.0