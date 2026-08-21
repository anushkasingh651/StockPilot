import pickle
from pathlib import Path


def save_model(model, filename="stockpilot_random_forest.pkl"):

    model_directory = Path("models")

    model_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = model_directory / filename

    with open(
        model_path,
        "wb"
    ) as file:

        pickle.dump(
            model,
            file
        )

    print(
        f"\nModel saved successfully:"
        f"\n{model_path}"
    )

    return model_path


def load_model(filename="stockpilot_random_forest.pkl"):

    model_path = (
        Path("models") / filename
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    with open(
        model_path,
        "rb"
    ) as file:

        model = pickle.load(
            file
        )

    return model