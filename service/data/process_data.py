from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def split_features_and_labels(df):
    """
    Separate features and labels.
    """
    X = df.drop("label", axis=1)
    y = df["label"]

    return X, y


def scale_features(X):
    """
    Scale the feature matrix.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler


def split_dataset(X, y):
    """
    Split the dataset into training and testing sets.
    """
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
