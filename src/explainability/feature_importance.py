from pathlib import Path
import pickle

import pandas as pd


# ==========================================
# MODEL PATH
# ==========================================

MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "stockpilot_random_forest.pkl"
)


# ==========================================
# LOAD MODEL
# ==========================================

def load_model():

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


# ==========================================
# GET FEATURE IMPORTANCE
# ==========================================

def get_feature_importance():

    model = load_model()


    # --------------------------------------
    # Check if model supports importance
    # --------------------------------------

    if not hasattr(
        model,
        "feature_importances_"
    ):

        raise ValueError(
            "This model does not support "
            "feature importance."
        )


    # --------------------------------------
    # Get feature names
    # --------------------------------------

    if not hasattr(
        model,
        "feature_names_in_"
    ):

        raise ValueError(
            "Model does not contain "
            "feature names."
        )


    feature_names = list(
        model.feature_names_in_
    )


    # --------------------------------------
    # Get importance values
    # --------------------------------------

    importance_values = (
        model.feature_importances_
    )


    # --------------------------------------
    # Create DataFrame
    # --------------------------------------

    importance_df = pd.DataFrame({

        "Feature": feature_names,

        "Importance": importance_values
    })


    # --------------------------------------
    # Sort descending
    # --------------------------------------

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