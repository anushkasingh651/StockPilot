from sklearn.ensemble import RandomForestClassifier


def create_random_forest_model():

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced"
    )

    return model


def train_random_forest(X_train, y_train):

    model = create_random_forest_model()

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