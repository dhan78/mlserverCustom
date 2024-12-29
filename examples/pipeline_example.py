from src.processing.common import (
    standardize_features,
    clip_values,
    softmax,
    threshold_predictions
)
from src.models.xgboost_model import XGBoostModel

# Initialize model
model = XGBoostModel()
model.load()

# Sample features
features = [
    {"feature1": 0.5, "feature2": 1.0, "feature3": -0.5},
    {"feature1": -1.0, "feature2": 0.0, "feature3": 2.0}
]

# Example 1: Binary classification with preprocessing
predictions = model.predict(
    features=features,
    preprocessors=[
        lambda f: standardize_features(f),
        lambda f: clip_values(f, -3, 3)
    ],
    postprocessors=[
        lambda p: threshold_predictions(p, 0.5)
    ]
)
print("Binary Classification Results:", predictions)

# Example 2: Multi-class classification
predictions = model.predict(
    features=features,
    preprocessors=[
        lambda f: standardize_features(f)
    ],
    postprocessors=[
        softmax
    ]
)
print("Multi-class Classification Results:", predictions)