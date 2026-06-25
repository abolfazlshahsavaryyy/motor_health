import pandas as pd
from service.data.build_data import build_dataset
from service.data.save_dataset import save_dataset_to_csv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
# dataset = build_dataset()

# save_dataset_to_csv(
#     dataset,
#     "data/engine_dataset.csv"
# )

df = pd.read_csv("data/engine_dataset.csv")

print(df.shape)
# print(df.head())
# print(df["label"].value_counts())
X = df.drop("label", axis=1)
y = df["label"]

print(X.shape)
print(y.shape)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(X_train.shape)
print(X_test.shape)

model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale"
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(
    y_test,
    y_pred
)

print("Accuracy:", acc)
print(
    classification_report(
        y_test,
        y_pred
    )
)