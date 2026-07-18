import pytest

from app import app as flask_app


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as client:
        yield client


VALID_PAYLOAD = {
    "PM2.5": 50,
    "PM10": 100,
    "NO2": 20,
    "SO2": 10,
    "CO": 1.2,
    "O3": 30,
}


def test_home(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "running"


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "healthy"


def test_predict_success(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.get_json()
    assert "Predicted_AQI" in data
    assert isinstance(data["Predicted_AQI"], float)


def test_predict_missing_field(client):
    payload = dict(VALID_PAYLOAD)
    del payload["O3"]

    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert data["missing_fields"] == ["O3"]


def test_predict_invalid_type(client):
    payload = dict(VALID_PAYLOAD)
    payload["PM2.5"] = "not-a-number"

    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert "PM2.5" in data["invalid_fields"]


def test_predict_non_json(client):
    response = client.post("/predict", data="not json", content_type="text/plain")
    assert response.status_code == 415


def test_404(client):
    response = client.get("/no-such-route")
    assert response.status_code == 404
