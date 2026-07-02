from service.data.build_data import build_dataset
from service.data.save_dataset import save_dataset_to_csv
import pandas as pd
def create_dataset(DATASET_PATH):
    
    dataset = build_dataset()
    save_dataset_to_csv(dataset, DATASET_PATH)


def load_dataset(DATASET_PATH):
    """
    Load the dataset from a CSV file.
    """
    return pd.read_csv(DATASET_PATH)


def print_dataset_info(df, X, y, X_train, X_test):
    """
    Print dataset statistics.
    """
    print("Dataset shape:", df.shape)
    print("Features shape:", X.shape)
    print("Labels shape:", y.shape)
    print("Training set:", X_train.shape)
    print("Testing set:", X_test.shape)