from fastapi import APIRouter, Depends, HTTPException
from src.api.schemas import PredictionRequest, PredictionResponse
from src.api.dependencies import verify_api_key, get_model
from src.models.xgboost_model import XGBoostModel
from src.config.settings import get_settings
from src.processing.pipeline import get_processor

router = APIRouter()
settings = get_settings()

@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    model: XGBoostModel = Depends(get_model),
    api_key: str = Depends(verify_api_key)
):
    """Make predictions with optional processing steps."""
    try:
        preprocessors = [
            get_processor(step, "pre") 
            for step in request.preprocessing_steps
        ]
        postprocessors = [
            get_processor(step, "post") 
            for step in request.postprocessing_steps
        ]
        
        predictions = model.predict(
            features=request.features,
            preprocessors=preprocessors,
            postprocessors=postprocessors
        )
        
        return PredictionResponse(
            predictions=predictions,
            model_version=settings.MODEL_VERSION,
            processing_info={
                "preprocessing": request.preprocessing_steps,
                "postprocessing": request.postprocessing_steps
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@router.get("/health")
async def health_check(model: XGBoostModel = Depends(get_model)):
    """Check health status."""
    return {
        "status": "healthy",
        "model_loaded": model.model is not None,
        "version": settings.MODEL_VERSION
    }