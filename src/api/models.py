from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class PredictionRequest(BaseModel):
    """Model for prediction request data."""
    features: List[Dict[str, float]] = Field(
        ...,
        description="List of feature dictionaries",
        example=[{"feature1": 1.0, "feature2": 2.0}]
    )
    
    class Config:
        schema_extra = {
            "example": {
                "features": [
                    {"feature1": 0.5, "feature2": 1.0},
                    {"feature1": -1.0, "feature2": 0.0}
                ]
            }
        }