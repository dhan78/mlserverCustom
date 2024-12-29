from typing import List, Callable, Dict, Any, Optional
from functools import reduce
from src.processing.common import (
    standardize_features,
    clip_values,
    softmax,
    threshold_predictions
)

# Registry of available processors
PROCESSORS = {
    "pre": {
        "standardize": standardize_features,
        "clip": clip_values
    },
    "post": {
        "softmax": softmax,
        "threshold": threshold_predictions
    }
}

def get_processor(name: str, step_type: str) -> Optional[Callable]:
    """Get processing function by name and type."""
    return PROCESSORS.get(step_type, {}).get(name)

class ProcessingPipeline:
    """Pipeline for preprocessing and postprocessing data."""
    
    def __init__(self, 
                 preprocessors: List[Callable] = None,
                 postprocessors: List[Callable] = None):
        self.preprocessors = preprocessors or []
        self.postprocessors = postprocessors or []
    
    def preprocess(self, data: Any) -> Any:
        """Apply all preprocessing functions in order."""
        return reduce(lambda d, proc: proc(d), self.preprocessors, data)
    
    def postprocess(self, data: Any) -> Any:
        """Apply all postprocessing functions in order."""
        return reduce(lambda d, proc: proc(d), self.postprocessors, data)