import pandas as pd


def add_moving_averages(data):
    """
    Add Simple Moving Average (SMA) and
    Exponential Moving Average (EMA).
    """

    data = data.copy()

    data["SMA_20"] = (
        data["Close"]
        .rolling(window=20)
        .mean()
    )

    data["SMA_50"] = (
        data["Close"]
        .rolling(window=50)
        .mean()
    )

    data["EMA_20"] = (
        data["Close"]
        .ewm(span=20, adjust=False)
        .mean()
    )

    return data


def add_rsi(data, window=14):
    """
    Calculate Relative Strength Index (RSI).
    """

    data = data.copy()

    price_change = data["Close"].diff()

    gains = price_change.clip(lower=0)
    losses = -price_change.clip(upper=0)

    average_gain = (
        gains
        .rolling(window=window)
        .mean()
    )

    average_loss = (
        losses
        .rolling(window=window)
        .mean()
    )

    relative_strength = (
        average_gain / average_loss
    )

    data["RSI_14"] = (
        100 - (
            100 /
            (1 + relative_strength)
        )
    )

    return data


def add_macd(data):
    """
    Calculate MACD and Signal Line.
    """

    data = data.copy()

    ema_12 = (
        data["Close"]
        .ewm(span=12, adjust=False)
        .mean()
    )

    ema_26 = (
        data["Close"]
        .ewm(span=26, adjust=False)
        .mean()
    )

    data["MACD"] = ema_12 - ema_26

    data["MACD_Signal"] = (
        data["MACD"]
        .ewm(span=9, adjust=False)
        .mean()
    )

    data["MACD_Histogram"] = (
        data["MACD"] -
        data["MACD_Signal"]
    )

    return data


def add_bollinger_bands(data, window=20):
    """
    Calculate Bollinger Bands.
    """

    data = data.copy()

    middle_band = (
        data["Close"]
        .rolling(window=window)
        .mean()
    )

    standard_deviation = (
        data["Close"]
        .rolling(window=window)
        .std()
    )

    data["BB_Middle"] = middle_band

    data["BB_Upper"] = (
        middle_band +
        (2 * standard_deviation)
    )

    data["BB_Lower"] = (
        middle_band -
        (2 * standard_deviation)
    )

    return data


def add_all_indicators(data):
    """
    Add all technical indicators.
    """

    data = add_moving_averages(data)
    data = add_rsi(data)
    data = add_macd(data)
    data = add_bollinger_bands(data)

    return data