from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass
class CustomModel:
    """
    Represents one machine learning experiment.
    """

    model_name: str
    model_parameters: str
    model: Any

    y_train_pred: np.ndarray | None = None
    y_test_pred: np.ndarray | None = None

    train_accuracy: float | None = None
    test_accuracy: float |None = None