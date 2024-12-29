from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional

class Feature(BaseModel):
    """Single feature schema with validation."""
    value: float = Field(..., description="Feature value")
    
    @validator('value')
    def validate_value(cls, v):
        if not isinstance(v, (int, float)):
            raise ValueError('Feature value must be numeric')
        return float(v)

class PredictionRequest(BaseModel):
    """Validated prediction request schema."""
    features: List[Dict[str, float]] = Field(
        ...,
        description="List of feature dictionaries",
        example=[{"feature1": 1.0, "feature2": 2.0}]
    )
    
    preprocessing_steps: Optional[List[str]] = Field(
        default=[],
        description="List of preprocessing steps to apply"
    )
    
    postprocessing_steps: Optional[List[str]] = Field(
        default=[],
        description="List of postprocessing steps to apply"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "features": [
                    {"feature1": 0.5, "feature2": 1.0},
                    {"feature1": -1.0, "feature2": 0.0}
                ],
                "preprocessing_steps": ["standardize", "clip"],
                "postprocessing_steps": ["softmax"]
            }
        }

class PredictionResponse(BaseModel):
    """Validated prediction response schema."""
    predictions: List[float] = Field(..., description="Model predictions")
    model_version: str = Field(..., description="Model version used for prediction")
    processing_info: Optional[Dict[str, List[str]]] = Field(
        default_factory=lambda: {"preprocessing": [], "postprocessing": []},
        description="Information about applied processing steps"
    )