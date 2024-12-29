from pydantic import BaseModel
from typing import List

class PredictionResponse(BaseModel):
    """Model for prediction response data."""
    predictions: List[float]
    model_version: str

class HealthResponse(BaseModel):
    """Model for health check response."""
    status: str
    model_loaded: bool
    version: str