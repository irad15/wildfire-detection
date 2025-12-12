# tests/api/test_router.py
from fastapi.testclient import TestClient
from main import app  # or wherever your FastAPI app instance is

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_detect_success():
    # Minimal valid payload — should work
    payload = [
        {
            "timestamp": "2025-08-01T10:00:00Z",
            "temperature": 35.0,
            "smoke": 0.1,
            "wind": 3.0
        }
    ]
    response = client.post("/detect", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "events" in data
    assert "event_count" in data
    assert "max_score" in data


def test_detect_empty_payload():
    response = client.post("/detect", json=[])
    assert response.status_code == 422
    assert "detail" in response.json()


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