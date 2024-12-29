import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_predict_endpoint_without_api_key():
    response = client.post("/api/v1/predict", json={
        "features": [{"feature1": 1.0, "feature2": 2.0}]
    })
    assert response.status_code == 403

def test_predict_endpoint_invalid_input():
    response = client.post("/api/v1/predict", json={
        "features": []
    })
    assert response.status_code == 422

def test_predict_endpoint_valid_input(mocker):
    # Mock the model prediction
    mocker.patch(
        "src.models.xgboost_model.XGBoostModel.predict",
        return_value=[0.8, 0.2]
    )
    
    response = client.post("/api/v1/predict", 
        json={"features": [{"feature1": 1.0, "feature2": 2.0}]},
        headers={"X-API-Key": "test-key"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2