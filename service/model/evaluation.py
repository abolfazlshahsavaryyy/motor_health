from sklearn.metrics import accuracy_score, classification_report


def evaluate_models(models, y_train, y_test):
    """
    Evaluate all trained models on both the training and test sets.

    Parameters
    ----------
    models : list[CustomModel]
        Trained models.
    y_train : array-like
    y_test : array-like

    Returns
    -------
    list[CustomModel]
    """

    print("\n" + "=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)

    for index, custom_model in enumerate(models, start=1):

        custom_model.train_accuracy = accuracy_score(
            y_train,
            custom_model.y_train_pred
        )

        custom_model.test_accuracy = accuracy_score(
            y_test,
            custom_model.y_test_pred
        )

        print(f"\n[{index}/{len(models)}]")
        print(f"Model      : {custom_model.model_name}")
        print(f"Parameters : {custom_model.model_parameters}")

        print(f"Train Accuracy : {custom_model.train_accuracy:.4f}")
        print(f"Test Accuracy  : {custom_model.test_accuracy:.4f}")

        gap = custom_model.train_accuracy - custom_model.test_accuracy
        print(f"Gap            : {gap:.4f}")

        if gap > 0.05:
            print("Status         : Possible overfitting")
        else:
            print("Status         : Generalization looks good")

        print("\nTest Classification Report")
        print(classification_report(y_test, custom_model.y_test_pred))

        print("-" * 80)

    return models