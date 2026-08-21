from pathlib import Path
import pickle

import pandas as pd

from src.data.stock_data import get_stock_data
from src.features.feature_pipeline import build_features


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
            f"Model not found: {MODEL_PATH}"
        )

    with open(
        MODEL_PATH,
        "rb"
    ) as file:

        model = pickle.load(file)

    return model


# ==========================================
# INTERNAL LOOKBACK PERIOD
# ==========================================

def get_model_period(
    user_period
):

    """
    The user-selected period is used for
    displaying the chart.

    The model needs additional historical
    data to calculate technical indicators.
    """

    period_mapping = {

        "1mo": "6mo",

        "3mo": "6mo",

        "6mo": "1y",

        "1y": "2y",

        "2y": "3y",

        "5y": "10y",

        "10y": "10y"
    }

    return period_mapping.get(
        user_period,
        "1y"
    )


# ==========================================
# PREPARE MODEL FEATURES
# ==========================================

def prepare_prediction_data(
    ticker,
    user_period
):

    # --------------------------------------
    # Get enough historical data
    # --------------------------------------

    model_period = get_model_period(
        user_period
    )


    data = get_stock_data(
        ticker,
        model_period
    )


    if data is None or data.empty:

        raise ValueError(
            f"Unable to fetch data for {ticker}"
        )


    # --------------------------------------
    # Build technical features
    # --------------------------------------

    data = build_features(
        data.copy()
    )


    # --------------------------------------
    # Remove invalid rows
    # --------------------------------------

    data = data.dropna().copy()


    if data.empty:

        raise ValueError(
            "No valid feature row available "
            "for prediction."
        )


    return data


# ==========================================
# PREDICT STOCK
# ==========================================

def predict_stock(
    ticker,
    period="1y"
):

    ticker = (
        ticker
        .upper()
        .strip()
    )


    # ======================================
    # LOAD MODEL
    # ======================================

    model = load_model()


    # ======================================
    # PREPARE FEATURES
    # ======================================

    data = prepare_prediction_data(
        ticker,
        period
    )


    # ======================================
    # GET MODEL FEATURES
    # ======================================

    if not hasattr(
        model,
        "feature_names_in_"
    ):

        raise ValueError(
            "The trained model does not contain "
            "feature names."
        )


    feature_columns = list(
        model.feature_names_in_
    )


    # ======================================
    # CHECK FEATURES
    # ======================================

    missing_features = [

        feature

        for feature in feature_columns

        if feature not in data.columns

    ]


    if missing_features:

        raise ValueError(
            "Missing model features: "
            f"{missing_features}"
        )


    # ======================================
    # LATEST VALID ROW
    # ======================================

    latest_data = (
        data[
            feature_columns
        ]
        .iloc[-1:]
        .copy()
    )


    if latest_data.empty:

        raise ValueError(
            "No valid feature row available "
            "for prediction."
        )


    # ======================================
    # PREDICTION
    # ======================================

    prediction = model.predict(
        latest_data
    )[0]


    probabilities = model.predict_proba(
        latest_data
    )[0]


    # ======================================
    # PROBABILITIES
    # ======================================

    down_probability = (
        probabilities[0] * 100
    )

    up_probability = (
        probabilities[1] * 100
    )


    # ======================================
    # PREDICTION LABEL
    # ======================================

    if prediction == 1:

        prediction_label = "UP"

        confidence = up_probability

    else:

        prediction_label = "DOWN"

        confidence = down_probability


    # ======================================
    # LATEST PRICE
    # ======================================

    latest_price = float(
        data["Close"].iloc[-1]
    )


    # ======================================
    # RESULT
    # ======================================

    result = {

        "ticker":
            ticker,

        "latest_price":
            latest_price,

        "prediction":
            prediction_label,

        "confidence":
            confidence,

        "up_probability":
            up_probability,

        "down_probability":
            down_probability,

        "features_used":
            len(feature_columns),

        "model_period":
            get_model_period(period)
    }


    return result


# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    result = predict_stock(
        "AAPL",
        "1mo"
    )


    print("\n")
    print("=" * 50)
    print("        STOCKPILOT PREDICTION")
    print("=" * 50)


    print(
        f"\nTicker: "
        f"{result['ticker']}"
    )


    print(
        f"Latest Price: "
        f"${result['latest_price']:.2f}"
    )


    print(
        f"Prediction: "
        f"{result['prediction']}"
    )


    print(
        f"Confidence: "
        f"{result['confidence']:.2f}%"
    )


    print(
        f"UP Probability: "
        f"{result['up_probability']:.2f}%"
    )


    print(
        f"DOWN Probability: "
        f"{result['down_probability']:.2f}%"
    )


    print(
        f"Features Used: "
        f"{result['features_used']}"
    )


    print(
        f"Model Lookback: "
        f"{result['model_period']}"
    )


    print("\n")