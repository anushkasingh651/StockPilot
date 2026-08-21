from pathlib import Path
import pickle

import pandas as pd


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "stockpilot_random_forest.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    """
    Load the trained Random Forest model.
    """

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}"
        )


    with open(
        MODEL_PATH,
        "rb"
    ) as file:

        model = pickle.load(file)


    return model


# ============================================================
# GET FEATURE IMPORTANCE
# ============================================================

def get_feature_importance():

    """
    Extract feature importance values
    from the trained Random Forest model.
    """

    model = load_model()


    # --------------------------------------------------------
    # Check whether model supports feature importance
    # --------------------------------------------------------

    if not hasattr(
        model,
        "feature_importances_"
    ):

        raise ValueError(
            "This model does not provide "
            "feature importance."
        )


    # --------------------------------------------------------
    # Get feature names
    # --------------------------------------------------------

    if not hasattr(
        model,
        "feature_names_in_"
    ):

        raise ValueError(
            "The trained model does not contain "
            "feature names."
        )


    feature_names = list(
        model.feature_names_in_
    )


    # --------------------------------------------------------
    # Get importance values
    # --------------------------------------------------------

    importance_values = (
        model.feature_importances_
    )


    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    importance_df = pd.DataFrame({

        "Feature":
            feature_names,

        "Importance":
            importance_values

    })


    # --------------------------------------------------------
    # Sort from most important to least important
    # --------------------------------------------------------

    importance_df = (
        importance_df
        .sort_values(
            by="Importance",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


    return importance_df


# ============================================================
# GET TOP FEATURES
# ============================================================

def get_top_features(
    number_of_features=10
):

    """
    Return the most important features.
    """

    importance_df = (
        get_feature_importance()
    )


    return importance_df.head(
        number_of_features
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("          STOCKPILOT MODEL EXPLAINABILITY")
    print("=" * 60)


    importance_df = (
        get_feature_importance()
    )


    print("\nFeature Importance:\n")


    for index, row in (
        importance_df.iterrows()
    ):

        print(
            f"{index + 1:2}. "
            f"{row['Feature']:<30} "
            f"{row['Importance']:.4f}"
        )


    print("\n")
    print("=" * 60)
    print("          TOP 10 FEATURES")
    print("=" * 60)


    top_features = (
        get_top_features(10)
    )


    print(
        top_features.to_string(
            index=False
        )
    )


    print("\n")