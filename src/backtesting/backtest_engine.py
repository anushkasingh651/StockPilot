from pathlib import Path
import pickle

import pandas as pd

from src.data.stock_data import get_stock_data
from src.features.feature_pipeline import build_features
from src.targets import create_target


# ==========================================
# CONFIGURATION
# ==========================================

MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "stockpilot_random_forest.pkl"
)

TEST_SIZE = 0.20


# ==========================================
# LOAD TRAINED MODEL
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
# LOAD AND PREPARE DATA
# ==========================================

def prepare_data(
    ticker="AAPL",
    period="10y"
):

    print("Fetching stock data...")

    data = get_stock_data(
        ticker,
        period
    )

    if data is None or data.empty:

        raise ValueError(
            f"Unable to fetch data for {ticker}"
        )


    print("Building technical features...")

    data = build_features(
        data
    )


    print("Creating prediction target...")

    data = create_target(
        data
    )


    return data


# ==========================================
# CREATE SAME TIME-BASED TEST SET
# ==========================================

def create_test_data(
    data
):

    # --------------------------------------
    # Remove rows containing NaN
    # --------------------------------------

    data = data.dropna().copy()


    # --------------------------------------
    # Calculate split index
    # --------------------------------------

    split_index = int(
        len(data)
        * (1 - TEST_SIZE)
    )


    # --------------------------------------
    # Use ONLY final 20%
    # --------------------------------------

    test_data = data.iloc[
        split_index:
    ].copy()


    return test_data


# ==========================================
# GENERATE TEST PREDICTIONS
# ==========================================

def generate_predictions(
    test_data,
    model
):

    if not hasattr(
        model,
        "feature_names_in_"
    ):

        raise ValueError(
            "Model does not contain "
            "feature names."
        )


    feature_columns = list(
        model.feature_names_in_
    )


    # --------------------------------------
    # Verify features
    # --------------------------------------

    missing_features = [

        feature

        for feature in feature_columns

        if feature not in test_data.columns

    ]


    if missing_features:

        raise ValueError(
            f"Missing features: "
            f"{missing_features}"
        )


    # --------------------------------------
    # Select EXACT model features
    # --------------------------------------

    X_test = test_data[
        feature_columns
    ].copy()


    # --------------------------------------
    # Generate predictions
    # --------------------------------------

    predictions = model.predict(
        X_test
    )


    probabilities = model.predict_proba(
        X_test
    )[:, 1]


    # --------------------------------------
    # Store predictions
    # --------------------------------------

    test_data[
        "Prediction"
    ] = predictions


    test_data[
        "Prediction_Probability"
    ] = probabilities


    return test_data


# ==========================================
# CALCULATE RETURNS
# ==========================================

def calculate_returns(
    data
):

    data = data.copy()


    # --------------------------------------
    # Daily stock return
    # --------------------------------------

    data[
        "Market_Return"
    ] = (
        data["Close"]
        .pct_change()
    )


    # --------------------------------------
    # Position
    # --------------------------------------
    #
    # Prediction:
    #
    # 1 = UP
    # 0 = DOWN
    #
    # We hold the stock only when
    # the model predicts UP.
    #

    data[
        "Position"
    ] = data[
        "Prediction"
    ].shift(1)


    # --------------------------------------
    # Strategy return
    # --------------------------------------

    data[
        "Strategy_Return"
    ] = (
        data["Position"]
        * data["Market_Return"]
    )


    # --------------------------------------
    # Remove first invalid row
    # --------------------------------------

    data = data.dropna(
        subset=[
            "Market_Return",
            "Strategy_Return"
        ]
    )


    # --------------------------------------
    # Equity curves
    # --------------------------------------

    data[
        "Strategy_Equity"
    ] = (
        1
        + data["Strategy_Return"]
    ).cumprod()


    data[
        "Buy_Hold_Equity"
    ] = (
        1
        + data["Market_Return"]
    ).cumprod()


    return data


# ==========================================
# PERFORMANCE METRICS
# ==========================================

def calculate_metrics(
    data
):

    # --------------------------------------
    # Strategy return
    # --------------------------------------

    strategy_return = (
        data["Strategy_Equity"].iloc[-1]
        - 1
    )


    # --------------------------------------
    # Buy & Hold
    # --------------------------------------

    buy_hold_return = (
        data["Buy_Hold_Equity"].iloc[-1]
        - 1
    )


    # --------------------------------------
    # Maximum drawdown
    # --------------------------------------

    running_max = (
        data["Strategy_Equity"]
        .cummax()
    )

    drawdown = (
        data["Strategy_Equity"]
        / running_max
        - 1
    )

    maximum_drawdown = (
        drawdown.min()
    )


    # --------------------------------------
    # Win rate
    # --------------------------------------

    profitable_days = (
        data["Strategy_Return"] > 0
    ).sum()

    total_days = len(data)

    win_rate = (
        profitable_days / total_days
        if total_days > 0
        else 0
    )


    # --------------------------------------
    # Predictions
    # --------------------------------------

    up_predictions = (
        data["Prediction"] == 1
    ).sum()

    down_predictions = (
        data["Prediction"] == 0
    ).sum()


    return {

        "strategy_return":
            float(strategy_return),

        "buy_hold_return":
            float(buy_hold_return),

        "maximum_drawdown":
            float(maximum_drawdown),

        "win_rate":
            float(win_rate),

        "up_predictions":
            int(up_predictions),

        "down_predictions":
            int(down_predictions),

        "total_days":
            int(total_days)
    }


# ==========================================
# RUN BACKTEST
# ==========================================

def run_backtest(
    ticker="AAPL",
    period="10y"
):

    print("\n")
    print("=" * 60)
    print("        STOCKPILOT OUT-OF-SAMPLE BACKTEST")
    print("=" * 60)


    # --------------------------------------
    # Load model
    # --------------------------------------

    print("\nLoading trained model...")

    model = load_model()


    # --------------------------------------
    # Prepare complete dataset
    # --------------------------------------

    data = prepare_data(
        ticker,
        period
    )


    # --------------------------------------
    # Create unseen test period
    # --------------------------------------

    print(
        "Creating chronological test set..."
    )

    test_data = create_test_data(
        data
    )


    print(
        f"Test samples: {len(test_data)}"
    )


    # --------------------------------------
    # Generate predictions
    # --------------------------------------

    print(
        "Generating predictions "
        "on unseen test data..."
    )

    test_data = generate_predictions(
        test_data,
        model
    )


    # --------------------------------------
    # Calculate returns
    # --------------------------------------

    print(
        "Calculating strategy returns..."
    )

    test_data = calculate_returns(
        test_data
    )


    # --------------------------------------
    # Metrics
    # --------------------------------------

    metrics = calculate_metrics(
        test_data
    )


    return test_data, metrics


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    data, metrics = run_backtest(
        ticker="AAPL",
        period="10y"
    )


    print("\n")
    print("=" * 60)
    print("              BACKTEST RESULTS")
    print("=" * 60)


    print(
        f"\nTest Samples: "
        f"{metrics['total_days']}"
    )


    print(
        f"Strategy Return: "
        f"{metrics['strategy_return']:.2%}"
    )


    print(
        f"Buy & Hold Return: "
        f"{metrics['buy_hold_return']:.2%}"
    )


    print(
        f"Maximum Drawdown: "
        f"{metrics['maximum_drawdown']:.2%}"
    )


    print(
        f"Win Rate: "
        f"{metrics['win_rate']:.2%}"
    )


    print(
        f"UP Predictions: "
        f"{metrics['up_predictions']}"
    )


    print(
        f"DOWN Predictions: "
        f"{metrics['down_predictions']}"
    )


    print("\n")
    print("=" * 60)
    print("          OUT-OF-SAMPLE BACKTEST COMPLETE")
    print("=" * 60)