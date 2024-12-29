import numpy as np
from typing import List, Dict, Any

def standardize_features(features: List[Dict[str, float]], 
                        mean: Dict[str, float] = None,
                        std: Dict[str, float] = None) -> List[Dict[str, float]]:
    """Standardize features using provided or computed mean and std."""
    if mean is None or std is None:
        # Compute statistics if not provided
        all_values = {k: [] for k in features[0].keys()}
        for f in features:
            for k, v in f.items():
                all_values[k].append(v)
        
        mean = {k: np.mean(v) for k, v in all_values.items()}
        std = {k: np.std(v) if np.std(v) > 0 else 1.0 
               for k, v in all_values.items()}
    
    return [{k: (v - mean[k]) / std[k] 
             for k, v in f.items()} 
            for f in features]

def clip_values(features: List[Dict[str, float]], 
                min_val: float = -5.0,
                max_val: float = 5.0) -> List[Dict[str, float]]:
    """Clip feature values to specified range."""
    return [{k: max(min_val, min(v, max_val)) 
             for k, v in f.items()} 
            for f in features]

def softmax(predictions: List[float]) -> List[float]:
    """Apply softmax to predictions."""
    exp_preds = np.exp(predictions - np.max(predictions))
    return (exp_preds / exp_preds.sum()).tolist()

def threshold_predictions(predictions: List[float], 
                        threshold: float = 0.5) -> List[float]:
    """Apply threshold to predictions."""
    return [1.0 if p >= threshold else 0.0 for p in predictions]