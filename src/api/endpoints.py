from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from src.api.models import PredictionRequest, PredictionResponse, HealthResponse
from src.models.xgboost_model import XGBoostModel
from src.config.settings import get_settings
from typing import List

router = APIRouter()
settings = get_settings()

# Security
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    """Verify API key if configured."""
    if settings.API_KEY and (not api_key or api_key != settings.API_KEY):
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    return api_key

@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    model: XGBoostModel = Depends(),
    api_key: str = Security(verify_api_key)
):
    """Make predictions using the XGBoost model."""
    try:
        predictions = model.predict(request.features)
        return PredictionResponse(
            predictions=predictions,
            model_version=settings.MODEL_VERSION
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@router.get("/health", response_model=HealthResponse)
async def health_check(model: XGBoostModel = Depends()):
    """Check the health status of the service."""
    return HealthResponse(
        status="healthy",
        model_loaded=model.model is not None,
        version=settings.MODEL_VERSION
    )