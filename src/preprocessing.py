import pandas as pd


FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",

    "Daily_Return",
    "Volatility",

    "SMA_20",
    "SMA_50",
    "EMA_20",

    "RSI_14",

    "MACD",
    "MACD_Signal",
    "MACD_Histogram",

    "BB_Middle",
    "BB_Upper",
    "BB_Lower",

    "Price_to_SMA20",
    "Price_to_SMA50",

    "BB_Width",
    "BB_Position"
]


def prepare_dataset(data):

    data = data.copy()

    # Remove rows where feature calculations
    # have not produced valid values.
    data = data.dropna(
        subset=FEATURE_COLUMNS + ["Target"]
    )

    X = data[FEATURE_COLUMNS]

    y = data["Target"]

    return X, y, data


def time_series_split(X, y, train_size=0.8):

    split_index = int(
        len(X) * train_size
    )

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )