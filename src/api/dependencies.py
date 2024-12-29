from fastapi import Security, HTTPException, Depends
from fastapi.security.api_key import APIKeyHeader
from src.config.settings import get_settings
from src.models.xgboost_model import XGBoostModel
from typing import Generator

settings = get_settings()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    """Verify API key if configured."""
    if settings.API_KEY and (not api_key or api_key != settings.API_KEY):
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    return api_key

def get_model() -> Generator[XGBoostModel, None, None]:
    """Dependency that provides a loaded model instance."""
    model = XGBoostModel()
    try:
        model.load()
        yield model
    finally:
        # Cleanup if needed
        pass