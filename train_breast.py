import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def train(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def test(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, f1


def run(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=["class"]).astype("float64")
    y_train = train_df["class"]
    X_test = test_df.drop(columns=["class"]).astype("float64")
    y_test = test_df["class"]

    model = LogisticRegression(max_iter=1000)
    model = train(model, X_train, y_train)
    accuracy, f1 = test(model, X_test, y_test)
    return accuracy, f1


if __name__ == "__main__":
    accuracy, f1 = run("data/breast_train.csv", "data/breast_test.csv")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
