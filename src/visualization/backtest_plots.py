import matplotlib.pyplot as plt


# ==========================================
# EQUITY CURVE
# ==========================================

def plot_equity_curve(data):
    """
    Compare StockPilot strategy with Buy & Hold.
    """

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.plot(
        data.index,
        data["Strategy_Equity"],
        label="StockPilot Strategy",
        linewidth=2
    )

    ax.plot(
        data.index,
        data["Buy_Hold_Equity"],
        label="Buy & Hold",
        linewidth=2
    )

    ax.set_title(
        "StockPilot vs Buy & Hold"
    )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        "Portfolio Value"
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )

    fig.tight_layout()

    return fig


# ==========================================
# DRAWDOWN
# ==========================================

def plot_drawdown(data):
    """
    Show the decline of the StockPilot
    strategy from its previous peak.
    """

    running_max = (
        data["Strategy_Equity"]
        .cummax()
    )

    drawdown = (
        data["Strategy_Equity"]
        / running_max
        - 1
    )

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        data.index,
        drawdown * 100,
        linewidth=2
    )

    ax.set_title(
        "StockPilot Strategy Drawdown"
    )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        "Drawdown (%)"
    )

    ax.grid(
        alpha=0.3
    )

    fig.tight_layout()

    return fig


# ==========================================
# PREDICTION DISTRIBUTION
# ==========================================

def plot_prediction_distribution(data):
    """
    Show the number of UP and DOWN predictions.
    """

    up_count = (
        data["Prediction"] == 1
    ).sum()

    down_count = (
        data["Prediction"] == 0
    ).sum()

    labels = [
        "DOWN",
        "UP"
    ]

    values = [
        down_count,
        up_count
    ]

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        labels,
        values
    )

    ax.set_title(
        "StockPilot Prediction Distribution"
    )

    ax.set_xlabel(
        "Prediction"
    )

    ax.set_ylabel(
        "Number of Days"
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    fig.tight_layout()

    return fig


# ==========================================
# SAVE BACKTEST PLOTS
# ==========================================

def save_backtest_plots(
    data,
    output_directory="reports"
):

    from pathlib import Path


    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------
    # Equity curve
    # --------------------------------------

    equity_figure = (
        plot_equity_curve(
            data
        )
    )

    equity_path = (
        output_directory
        / "equity_curve.png"
    )

    equity_figure.savefig(
        equity_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(
        equity_figure
    )


    # --------------------------------------
    # Drawdown
    # --------------------------------------

    drawdown_figure = (
        plot_drawdown(
            data
        )
    )

    drawdown_path = (
        output_directory
        / "drawdown.png"
    )

    drawdown_figure.savefig(
        drawdown_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(
        drawdown_figure
    )


    # --------------------------------------
    # Prediction distribution
    # --------------------------------------

    prediction_figure = (
        plot_prediction_distribution(
            data
        )
    )

    prediction_path = (
        output_directory
        / "prediction_distribution.png"
    )

    prediction_figure.savefig(
        prediction_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(
        prediction_figure
    )


    return {

        "equity_curve":
            str(equity_path),

        "drawdown":
            str(drawdown_path),

        "prediction_distribution":
            str(prediction_path)
    }