import joblib
from pathlib import Path


def select_and_save_models(models):
    """
    Let the user select the best model for each algorithm and save it.

    Parameters
    ----------
    models : list[CustomModel]

    Returns
    -------
    dict
        Dictionary containing the selected models.
    """

    save_path = Path("data/models")
    save_path.mkdir(parents=True, exist_ok=True)

    # Group models by algorithm name
    grouped_models = {}

    for model in models:
        grouped_models.setdefault(model.model_name, []).append(model)

    selected_models = {}

    print("\n" + "=" * 80)
    print("MODEL SELECTION")
    print("=" * 80)

    for model_name, model_list in grouped_models.items():

        print(f"\n{model_name}")
        print("-" * 80)

        for index, model in enumerate(model_list, start=1):

            print(
                f"{index}. "
                f"Train Accuracy = {model.train_accuracy:.4f} | "
                f"Test Accuracy = {model.test_accuracy:.4f}"
            )

            print(f"   Parameters: {model.model_parameters}")

        # Get valid user input
        while True:
            try:
                choice = int(
                    input(
                        f"\nSelect the best {model_name} model "
                        f"(1-{len(model_list)}): "
                    )
                )

                if 1 <= choice <= len(model_list):
                    break

                print("Invalid selection.")

            except ValueError:
                print("Please enter a number.")

        selected_model = model_list[choice - 1]

        filename = (
            save_path /
            f"{model_name.lower().replace(' ', '_')}.joblib"
        )

        joblib.dump(selected_model.model, filename)

        selected_models[model_name] = selected_model

        print(f"Saved to {filename}")

    return selected_models