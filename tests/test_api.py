from fastapi.testclient import TestClient
from src.main import app
import pytest

client = TestClient(app)

def test_health_check():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "version" in data

def test_predict_invalid_input():
    response = client.post("/api/v1/predict", json={"features": []})
    assert response.status_code == 422  # Validation error