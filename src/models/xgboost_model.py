"""XGBoost model wrapper for inference."""
import xgboost as xgb
import numpy as np
from typing import List, Dict, Any, Callable
from src.config.settings import get_settings
from src.processing.pipeline import ProcessingPipeline
from src.models.loader import load_model
import logging

logger = logging.getLogger(__name__)

class XGBoostModel:
    """XGBoost model wrapper for inference."""
    
    def __init__(self):
        self.settings = get_settings()
        self.model: Optional[xgb.Booster] = None
        self.feature_names: List[str] = []
        self.pipeline = ProcessingPipeline()
        
    def load(self) -> None:
        """Load the XGBoost model from configured path."""
        try:
            self.model = load_model(self.settings.MODEL_PATH)
            self.feature_names = self.model.feature_names
            logger.info(f"Model loaded with features: {self.feature_names}")
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise
    
    def predict(self, 
                features: List[Dict[str, float]], 
                preprocessors: List[Callable] = None,
                postprocessors: List[Callable] = None) -> List[float]:
        """Make predictions using the loaded model with optional processing."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Create pipeline for this prediction
        pipeline = ProcessingPipeline(preprocessors, postprocessors)
        
        # Preprocess
        processed_features = pipeline.preprocess(features)
        
        # Convert to DMatrix and predict
        data = self._prepare_features(processed_features)
        dmatrix = xgb.DMatrix(data)
        predictions = self.model.predict(dmatrix)
        
        # Postprocess
        return pipeline.postprocess(predictions.tolist())
    
    def _prepare_features(self, features: List[Dict[str, float]]) -> np.ndarray:
        """Convert feature dictionaries to numpy array in correct order."""
        if not self.feature_names:
            raise ValueError("Feature names not available")
            
        feature_matrix = np.zeros((len(features), len(self.feature_names)))
        for i, feature_dict in enumerate(features):
            for j, feature_name in enumerate(self.feature_names):
                feature_matrix[i, j] = feature_dict.get(feature_name, 0.0)
                
        return feature_matrix