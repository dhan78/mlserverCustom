import pytest
from src.models.xgboost_model import XGBoostModel
from src.config.settings import Settings

@pytest.fixture
def test_settings():
    return Settings(
        MODEL_PATH="tests/fixtures/test_model.json",
        API_KEY="test-key",
        LOG_LEVEL="INFO"
    )

@pytest.fixture
def model():
    return XGBoostModel()