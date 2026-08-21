from src.backtesting.backtest_engine import run_backtest

from src.visualization.backtest_plots import (
    save_backtest_plots
)


print("\nRunning backtest...")

data, metrics = run_backtest(
    ticker="AAPL",
    period="10y"
)


print("\nGenerating visualizations...")

plots = save_backtest_plots(
    data,
    output_directory="reports"
)


print("\n")
print("=" * 60)
print("       VISUALIZATION COMPLETE")
print("=" * 60)

print(
    f"\nEquity Curve:"
    f"\n{plots['equity_curve']}"
)

print(
    f"\nDrawdown Chart:"
    f"\n{plots['drawdown']}"
)

print(
    f"\nPrediction Distribution:"
    f"\n{plots['prediction_distribution']}"
)

print("\n")
print("=" * 60)