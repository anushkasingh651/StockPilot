from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def create_baseline_model():

    model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000
            )
        )
    ])

    return model


def train_baseline_model(X_train, y_train):

    model = create_baseline_model()

    model.fit(
        X_train,
        y_train
    )

    return model


def make_predictions(model, X_test):

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    return predictions, probabilities