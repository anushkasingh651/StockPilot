import pandas as pd
import numpy as np


def calculate_returns(data):
    """
    Calculate daily percentage returns.
    """

    data = data.copy()

    data["Daily_Return"] = (
        data["Close"].pct_change() * 100
    )

    return data


def calculate_statistics(data):
    """
    Calculate basic statistical measures.
    """

    close_prices = data["Close"]

    statistics = {
        "Mean Price": close_prices.mean(),
        "Median Price": close_prices.median(),
        "Minimum Price": close_prices.min(),
        "Maximum Price": close_prices.max(),
        "Standard Deviation": close_prices.std(),
    }

    return statistics


def calculate_volatility(data, window=20):
    """
    Calculate rolling volatility using daily returns.
    """

    data = data.copy()

    data["Daily_Return"] = (
        data["Close"].pct_change()
    )

    data["Volatility"] = (
        data["Daily_Return"]
        .rolling(window)
        .std()
        * np.sqrt(252)
    )

    return data