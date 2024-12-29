"""Model loading utilities."""
from pathlib import Path
from typing import Optional
import xgboost as xgb
import logging

logger = logging.getLogger(__name__)

def load_model(model_path: str) -> Optional[xgb.Booster]:
    """Load XGBoost model from path."""
    path = Path(model_path)
    
    if not path.exists():
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    try:
        model = xgb.Booster()
        model.load_model(str(path))
        logger.info(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise RuntimeError(f"Failed to load model: {str(e)}")