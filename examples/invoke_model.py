import requests
import json
from typing import List, Dict, Optional

def invoke_prediction(
    features: List[Dict[str, float]],
    preprocessing_steps: Optional[List[str]] = None,
    postprocessing_steps: Optional[List[str]] = None,
    url: str = "http://localhost:8000/api/v1/predict",
    api_key: str = None
) -> Dict:
    """Invoke model prediction endpoint with processing steps."""
    headers = {
        "Content-Type": "application/json"
    }
    if api_key:
        headers["X-API-Key"] = api_key
        
    data = {
        "features": features,
        "preprocessing_steps": preprocessing_steps or [],
        "postprocessing_steps": postprocessing_steps or []
    }
        
    response = requests.post(
        url,
        headers=headers,
        json=data
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Prediction failed: {response.text}")

if __name__ == "__main__":
    # Example 1: Basic prediction
    features = [
        {"feature1": 0.5, "feature2": 1.0},
        {"feature1": -1.0, "feature2": 0.0}
    ]
    
    result = invoke_prediction(
        features=features,
        api_key="your-api-key"
    )
    print("Basic prediction:", result)
    
    # Example 2: With preprocessing and postprocessing
    result = invoke_prediction(
        features=features,
        preprocessing_steps=["standardize", "clip"],
        postprocessing_steps=["softmax"],
        api_key="your-api-key"
    )
    print("Prediction with processing:", result)