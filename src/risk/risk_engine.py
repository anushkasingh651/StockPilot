import numpy as np
import pandas as pd


# ============================================================
# RISK ENGINE
# ============================================================

def calculate_returns(price_data):
    """
    Calculate daily percentage returns.
    """

    if price_data is None or price_data.empty:
        raise ValueError("Price data is empty.")

    if "Close" not in price_data.columns:
        raise ValueError(
            "Price data must contain a 'Close' column."
        )

    data = price_data.copy()

    data["Daily_Return"] = (
        data["Close"]
        .pct_change()
    )

    data = data.dropna()

    return data


# ============================================================
# VOLATILITY
# ============================================================

def calculate_volatility(price_data):
    """
    Calculate annualized volatility.

    Formula:
        Daily volatility × sqrt(252)
    """

    data = calculate_returns(
        price_data
    )

    daily_volatility = (
        data["Daily_Return"]
        .std()
    )

    annualized_volatility = (
        daily_volatility * np.sqrt(252)
    )

    return annualized_volatility


# ============================================================
# SHARPE RATIO
# ============================================================

def calculate_sharpe_ratio(
    price_data,
    risk_free_rate=0.0
):
    """
    Calculate annualized Sharpe Ratio.

    Sharpe Ratio measures return relative
    to the amount of risk taken.
    """

    data = calculate_returns(
        price_data
    )

    daily_return = (
        data["Daily_Return"]
        .mean()
    )

    daily_std = (
        data["Daily_Return"]
        .std()
    )

    if daily_std == 0:

        return 0.0

    daily_risk_free = (
        risk_free_rate / 252
    )

    sharpe = (
        (daily_return - daily_risk_free)
        / daily_std
    ) * np.sqrt(252)

    return sharpe


# ============================================================
# MAXIMUM DRAWDOWN
# ============================================================

def calculate_max_drawdown(price_data):
    """
    Calculate maximum drawdown.

    Drawdown measures the largest decline
    from a previous peak.
    """

    if price_data is None or price_data.empty:
        raise ValueError("Price data is empty.")

    if "Close" not in price_data.columns:
        raise ValueError(
            "Price data must contain a 'Close' column."
        )

    prices = (
        price_data["Close"]
        .dropna()
    )

    running_max = (
        prices
        .cummax()
    )

    drawdown = (
        prices / running_max
    ) - 1

    max_drawdown = (
        drawdown
        .min()
    )

    return max_drawdown


# ============================================================
# TOTAL RETURN
# ============================================================

def calculate_total_return(price_data):
    """
    Calculate total buy-and-hold return.
    """

    if price_data is None or price_data.empty:
        raise ValueError("Price data is empty.")

    prices = (
        price_data["Close"]
        .dropna()
    )

    if len(prices) < 2:
        return 0.0

    total_return = (
        prices.iloc[-1]
        / prices.iloc[0]
    ) - 1

    return total_return


# ============================================================
# WIN RATE
# ============================================================

def calculate_win_rate(price_data):
    """
    Calculate percentage of positive-return days.
    """

    data = calculate_returns(
        price_data
    )

    winning_days = (
        data["Daily_Return"] > 0
    ).sum()

    total_days = len(data)

    if total_days == 0:
        return 0.0

    win_rate = (
        winning_days
        / total_days
    )

    return win_rate


# ============================================================
# COMPLETE RISK REPORT
# ============================================================

def generate_risk_report(price_data):
    """
    Generate complete risk analytics.
    """

    volatility = calculate_volatility(
        price_data
    )

    sharpe_ratio = calculate_sharpe_ratio(
        price_data
    )

    max_drawdown = calculate_max_drawdown(
        price_data
    )

    total_return = calculate_total_return(
        price_data
    )

    win_rate = calculate_win_rate(
        price_data
    )


    report = {

        "volatility": volatility,

        "sharpe_ratio": sharpe_ratio,

        "max_drawdown": max_drawdown,

        "total_return": total_return,

        "win_rate": win_rate

    }


    return report


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)

    print(
        "           STOCKPILOT RISK ENGINE"
    )

    print("=" * 60)


    # Example test data

    prices = pd.DataFrame({

        "Close": [
            100,
            102,
            101,
            105,
            103,
            108,
            110,
            107,
            112,
            115
        ]

    })


    report = generate_risk_report(
        prices
    )


    print(
        f"\nVolatility: "
        f"{report['volatility'] * 100:.2f}%"
    )

    print(
        f"Sharpe Ratio: "
        f"{report['sharpe_ratio']:.2f}"
    )

    print(
        f"Maximum Drawdown: "
        f"{report['max_drawdown'] * 100:.2f}%"
    )

    print(
        f"Total Return: "
        f"{report['total_return'] * 100:.2f}%"
    )

    print(
        f"Win Rate: "
        f"{report['win_rate'] * 100:.2f}%"
    )


    print("\n" + "=" * 60)

    print(
        "          RISK ENGINE TEST COMPLETE"
    )

    print("=" * 60)