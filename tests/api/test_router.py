# tests/api/test_router.py
from fastapi.testclient import TestClient
from main import app  # or wherever your FastAPI app instance is

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_detect_valid_payload():
    """Payload with valid data points should succeed (200)."""
    payload = [
        {"timestamp": "2025-08-01T10:00:00Z", "temperature": 25.0, "smoke": 0.01, "wind": 2.0},
        {"timestamp": "2025-08-01T10:01:00Z", "temperature": 25.5, "smoke": 0.02, "wind": 2.1},
        {"timestamp": "2025-08-01T10:02:00Z", "temperature": 26.0, "smoke": 0.01, "wind": 2.2},
    ]
    response = client.post("/detect", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "event_count" in data
    assert "max_score" in data
    assert isinstance(data["event_count"], int)
    assert isinstance(data["max_score"], float)


def test_detect_empty_payload():
    response = client.post("/detect", json=[])
    assert response.status_code == 422


def test_detect_missing_field():
    payload = [
        {
            "timestamp": "2025-08-01T10:00:00Z",
            "temperature": 40.0,
            # missing "smoke" and "wind"
        }
    ]
    response = client.post("/detect", json=payload)
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any("smoke" in err["loc"] for err in errors)
    assert any("wind" in err["loc"] for err in errors)


def test_detect_invalid_timestamp():
    payload = [
        {
            "timestamp": "not-a-timestamp",
            "temperature": 30.0,
            "smoke": 0.05,
            "wind": 2.0
        }
    ]
    response = client.post("/detect", json=payload)
    assert response.status_code == 422
    assert "timestamp" in str(response.json())


def test_detect_out_of_range_values():
    payload = [
        {
            "timestamp": "2025-08-01T10:00:00Z",
            "temperature": -100.0,   # invalid
            "smoke": 1.5,             # invalid
            "wind": -5.0              # invalid
        }
    ]
    response = client.post("/detect", json=payload)
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any("temperature" in str(err) for err in errors)
    assert any("smoke" in str(err) for err in errors)
    assert any("wind" in str(err) for err in errors)