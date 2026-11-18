
import sys
from pathlib import Path

# Add the main FlySmart project directory to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import app as app_module
from fastapi.testclient import TestClient


client = TestClient(app_module.app)


# Valid flight data for testing
VALID_FLIGHT = {
    "airline": "AirAsia",
    "source_city": "Bangalore",
    "departure_time": "Afternoon",
    "stops": "zero",
    "arrival_time": "Evening",
    "destination_city": "Delhi",
    "class": "Economy",
    "duration": 2.5,
    "days_left": 15
}


# TEST 1: Health check
def test_health_check():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["application"] == "FlySmart"
    assert data["model_loaded"] is True
    assert data["fare_dataset_loaded"] is True
    assert data["available_routes"] > 0
    assert data["available_cities"] > 0


# TEST 2: Homepage
def test_homepage():

    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


# TEST 3: Valid ML prediction (domain logic)
def test_valid_flight_prediction(tmp_path, monkeypatch):

    # Keep test prediction logs separate from actual project logs
    monkeypatch.setattr(
        app_module,
        "LOG_PATH",
        tmp_path / "test_predictions.jsonl"
    )

    response = client.post(
        "/predict",
        json=VALID_FLIGHT
    )

    assert response.status_code == 200

    data = response.json()

    assert data["predicted_category"] in ["Low", "Medium", "High"]
    assert data["category_price_range"] is not None
    assert isinstance(data["recommendations"], list)
    assert len(data["recommendations"]) <= 3


# TEST 4: Invalid flight input
def test_invalid_flight_input():

    invalid_flight = VALID_FLIGHT.copy()

    # Duration must be greater than zero
    invalid_flight["duration"] = 0

    response = client.post(
        "/predict",
        json=invalid_flight
    )

    assert response.status_code == 422
