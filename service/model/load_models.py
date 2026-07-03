import joblib
from pathlib import Path


def load_models():

    model_path = Path("data/models")

    models = {

        "svc": joblib.load(model_path / "svc.joblib"),

        "random_forest": joblib.load(model_path / "random_forest.joblib"),

        "logistic_regression": joblib.load(model_path / "logistic_regression.joblib"),

        "gradient_boosting": joblib.load(model_path / "gradient_boosting.joblib"),
    }

    return models