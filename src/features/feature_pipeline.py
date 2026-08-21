from src.features.eda import (
    calculate_returns,
    calculate_volatility
)

from src.features.technical_indicators import (
    add_all_indicators
)


def build_features(data):
    """
    Build the complete feature dataset
    from raw stock market data.
    """

    data = data.copy()

    # -----------------------------
    # EDA FEATURES
    # -----------------------------

    data = calculate_returns(data)

    data = calculate_volatility(data)


    # -----------------------------
    # TECHNICAL INDICATORS
    # -----------------------------

    data = add_all_indicators(data)


    # -----------------------------
    # ADDITIONAL FEATURES
    # -----------------------------

    data["Price_to_SMA20"] = (
        data["Close"] / data["SMA_20"]
    )

    data["Price_to_SMA50"] = (
        data["Close"] / data["SMA_50"]
    )

    data["BB_Width"] = (
        (
            data["BB_Upper"] -
            data["BB_Lower"]
        )
        / data["BB_Middle"]
    )

    data["BB_Position"] = (
        (
            data["Close"] -
            data["BB_Lower"]
        )
        /
        (
            data["BB_Upper"] -
            data["BB_Lower"]
        )
    )


    return data