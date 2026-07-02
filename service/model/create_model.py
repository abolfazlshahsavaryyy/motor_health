from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from service.model.CustomModel import CustomModel

def create_models() -> list[CustomModel]:
    """
    Create all models with different hyperparameters.

    Returns
    -------
    list[CustomModel]
    """

    models = []

    # ---------------- Logistic Regression ----------------
    for c in [0.01, 0.1, 1.0, 10.0]:
        model = LogisticRegression(
            C=c,
            max_iter=1000,
            random_state=42,
        )

        models.append(
            CustomModel(
                model_name="Logistic Regression",
                model_parameters=f"C={c}, max_iter=1000",
                model=model,
            )
        )

    # ---------------- SVC ----------------
    for c in [0.1, 1, 10, 100]:
        model = SVC(
            kernel="rbf",
            C=c,
            gamma="scale",
            probability=True,
        )

        models.append(
            CustomModel(
                model_name="SVC",
                model_parameters=f"kernel=rbf, C={c}, gamma=scale",
                model=model,
            )
        )

    # ---------------- Random Forest ----------------
    rf_configs = [
        (100, 5),
        (100, 10),
        (200, 10),
        (300, None),
    ]

    for n_estimators, max_depth in rf_configs:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
        )

        models.append(
            CustomModel(
                model_name="Random Forest",
                model_parameters=(
                    f"n_estimators={n_estimators}, "
                    f"max_depth={max_depth}"
                ),
                model=model,
            )
        )

    # ---------------- Gradient Boosting ----------------
    gb_configs = [
        (100, 0.01),
        (100, 0.1),
        (200, 0.1),
        (300, 0.05),
    ]

    for n_estimators, learning_rate in gb_configs:
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=42,
        )

        models.append(
            CustomModel(
                model_name="Gradient Boosting",
                model_parameters=(
                    f"n_estimators={n_estimators}, "
                    f"learning_rate={learning_rate}"
                ),
                model=model,
            )
        )

    return models