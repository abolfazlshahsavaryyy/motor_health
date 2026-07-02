from service.data.process_data import *
from service.data.data import *
from service.model.create_model import *
from service.model.train import *
from service.model.evaluation import *
from service.model.select_and_save_models import *
DATASET_PATH = "data/engine_dataset.csv"

def main():
    # Uncomment this if you need to rebuild the dataset.
    # create_dataset(DATASET_PATH)

    df = load_dataset(DATASET_PATH)

    X, y = split_features_and_labels(df)

    X_scaled, scaler = scale_features(X)

    X_train, X_test, y_train, y_test = split_dataset(
        X_scaled,
        y
    )

    print_dataset_info(
        df,
        X,
        y,
        X_train,
        X_test
    )

    models = create_models()

    trained_models = train_models(
        models=models,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
    )

    trained_models = evaluate_models(
        trained_models,
        y_train=y_train,
        y_test=y_test,
    )
    selected_models = select_and_save_models(trained_models)

if __name__ == "__main__":
    main()