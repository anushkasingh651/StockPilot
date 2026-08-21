from sklearn.ensemble import GradientBoostingClassifier


def create_gradient_boosting_model():

    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )

    return model


def train_gradient_boosting(X_train, y_train):

    model = create_gradient_boosting_model()

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