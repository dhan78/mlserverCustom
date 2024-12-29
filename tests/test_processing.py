import pytest
import numpy as np
from src.processing.common import (
    standardize_features,
    clip_values,
    softmax,
    threshold_predictions
)
from src.processing.pipeline import ProcessingPipeline

def test_standardize_features():
    features = [
        {"a": 1.0, "b": 2.0},
        {"a": 3.0, "b": 4.0}
    ]
    result = standardize_features(features)
    assert len(result) == 2
    assert all(k in result[0] for k in ["a", "b"])

def test_clip_values():
    features = [
        {"a": 10.0, "b": -10.0},
        {"a": 3.0, "b": 4.0}
    ]
    result = clip_values(features, -5, 5)
    assert result[0]["a"] == 5.0
    assert result[0]["b"] == -5.0

def test_softmax():
    predictions = [1.0, 2.0, 3.0]
    result = softmax(predictions)
    assert len(result) == 3
    assert abs(sum(result) - 1.0) < 1e-6
    assert all(0 <= x <= 1 for x in result)

def test_threshold_predictions():
    predictions = [0.3, 0.7, 0.5]
    result = threshold_predictions(predictions)
    assert result == [0.0, 1.0, 0.0]

def test_processing_pipeline():
    pipeline = ProcessingPipeline(
        preprocessors=[
            lambda x: [i * 2 for i in x],
            lambda x: [i + 1 for i in x]
        ],
        postprocessors=[
            lambda x: [i / 2 for i in x]
        ]
    )
    
    input_data = [1, 2, 3]
    processed = pipeline.preprocess(input_data)
    assert processed == [3, 5, 7]
    
    final = pipeline.postprocess(processed)
    assert final == [1.5, 2.5, 3.5]