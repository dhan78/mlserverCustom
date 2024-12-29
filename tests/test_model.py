import pytest
from src.models.xgboost_model import XGBoostModel
import numpy as np

@pytest.fixture
def model():
    return XGBoostModel()

def test_model_initialization(model):
    assert model is not None
    assert model.model is None

def test_feature_preparation(model):
    model.feature_names = ["feature1", "feature2"]
    features = [{"feature1": 1.0, "feature2": 2.0}]
    
    result = model._prepare_features(features)
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 2)
    assert np.array_equal(result, np.array([[1.0, 2.0]]))