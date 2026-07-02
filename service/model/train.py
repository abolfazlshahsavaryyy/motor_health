def train_models(models, X_train, y_train, X_test):
    """
    Train all models and store their predictions.

    Parameters
    ----------
    models : list[CustomModel]
        List returned by create_models().
    X_train : array-like
    y_train : array-like
    X_test : array-like

    Returns
    -------
    list[CustomModel]
        The same list with trained models and predictions.
    """

    print("=" * 80)
    print(f"Training {len(models)} models...")
    print("=" * 80)

    for index, custom_model in enumerate(models, start=1):
        print(f"\n[{index}/{len(models)}]")
        print(f"Model      : {custom_model.model_name}")
        print(f"Parameters : {custom_model.model_parameters}")

        # Train
        custom_model.model.fit(X_train, y_train)

        # Save predictions
        custom_model.y_train_pred = custom_model.model.predict(X_train)
        custom_model.y_test_pred = custom_model.model.predict(X_test)

        print("Status     : ✓ Training completed")

    print("\n" + "=" * 80)
    print("All models have been trained successfully.")
    print("=" * 80)

    return models