from src.data.stock_data import get_stock_data

from src.prediction.predict import predict_stock

from src.risk.risk_engine import generate_risk_report

from src.ai.llm_engine import generate_stock_analysis

from src.backtesting.backtest_engine import run_backtest


# ============================================================
# STOCKPILOT AI AGENT
# ============================================================

class StockPilotAgent:
    """
    StockPilot AI orchestration agent.

    Tools:

    1. Market Data
    2. ML Prediction
    3. Risk Engine
    4. Backtesting Engine
    5. LLM Intelligence
    """

    def __init__(self):

        self.name = (
            "StockPilot Intelligence Agent"
        )


    # ========================================================
    # TOOL 1 — MARKET DATA
    # ========================================================

    def get_market_data(
        self,
        ticker,
        period
    ):

        print(
            f"[Agent] Fetching {ticker} data..."
        )

        data = get_stock_data(
            ticker,
            period
        )

        if data is None or data.empty:

            raise ValueError(
                f"No market data available for {ticker}."
            )

        return data


    # ========================================================
    # TOOL 2 — ML PREDICTION
    # ========================================================

    def run_prediction(
        self,
        ticker,
        period
    ):

        print(
            "[Agent] Running ML prediction..."
        )

        result = predict_stock(
            ticker,
            period
        )

        if result is None:

            raise ValueError(
                "Prediction engine returned no result."
            )

        return result


    # ========================================================
    # TOOL 3 — RISK ENGINE
    # ========================================================

    def run_risk_analysis(
        self,
        data
    ):

        print(
            "[Agent] Calculating risk metrics..."
        )

        risk_report = (
            generate_risk_report(
                data
            )
        )

        if risk_report is None:

            raise ValueError(
                "Risk engine returned no result."
            )

        return risk_report


    # ========================================================
    # TOOL 4 — BACKTESTING ENGINE
    # ========================================================

    def run_historical_backtest(
        self,
        ticker,
        period
    ):

        print(
            "[Agent] Running out-of-sample backtest..."
        )

        backtest_data, metrics = (
            run_backtest(
                ticker=ticker,
                period=period
            )
        )

        if metrics is None:

            raise ValueError(
                "Backtesting engine returned "
                "no metrics."
            )

        return {
            "data": backtest_data,
            "metrics": metrics
        }


    # ========================================================
    # TOOL 5 — LLM INTELLIGENCE
    # ========================================================

    def generate_intelligence(
        self,
        ticker,
        prediction_result,
        risk_report,
        backtest_metrics
    ):

        print(
            "[Agent] Generating AI intelligence..."
        )


        # ----------------------------------------------------
        # Backtest metrics
        # ----------------------------------------------------

        strategy_return = (
            backtest_metrics[
                "strategy_return"
            ]
        )

        buy_hold_return = (
            backtest_metrics[
                "buy_hold_return"
            ]
        )

        maximum_drawdown = (
            backtest_metrics[
                "maximum_drawdown"
            ]
        )

        backtest_win_rate = (
            backtest_metrics[
                "win_rate"
            ]
        )


        # ----------------------------------------------------
        # Generate LLM analysis
        # ----------------------------------------------------

        analysis = generate_stock_analysis(

            ticker=ticker,

            latest_price=(
                prediction_result[
                    "latest_price"
                ]
            ),

            prediction=(
                prediction_result[
                    "prediction"
                ]
            ),

            confidence=(
                prediction_result[
                    "confidence"
                ]
            ),

            up_probability=(
                prediction_result[
                    "up_probability"
                ]
            ),

            down_probability=(
                prediction_result[
                    "down_probability"
                ]
            ),

            volatility=(
                risk_report[
                    "volatility"
                ]
            ),

            sharpe_ratio=(
                risk_report[
                    "sharpe_ratio"
                ]
            ),

            max_drawdown=(
                risk_report[
                    "max_drawdown"
                ]
            ),

            total_return=(
                risk_report[
                    "total_return"
                ]
            ),

            win_rate=(
                risk_report[
                    "win_rate"
                ]
            )
        )


        # ----------------------------------------------------
        # Return LLM analysis
        # ----------------------------------------------------

        return analysis


    # ========================================================
    # MAIN AGENT WORKFLOW
    # ========================================================

    def analyze_stock(
        self,
        ticker,
        period
    ):

        print("\n")

        print(
            "=" * 60
        )

        print(
            "       STOCKPILOT AI AGENT"
        )

        print(
            "=" * 60
        )


        # ====================================================
        # STEP 1 — MARKET DATA
        # ====================================================

        data = (
            self.get_market_data(
                ticker,
                period
            )
        )


        # ====================================================
        # STEP 2 — ML PREDICTION
        # ====================================================

        prediction_result = (
            self.run_prediction(
                ticker,
                period
            )
        )


        # ====================================================
        # STEP 3 — RISK ANALYSIS
        # ====================================================

        risk_report = (
            self.run_risk_analysis(
                data
            )
        )


        # ====================================================
        # STEP 4 — BACKTESTING
        # ====================================================

        backtest_result = (
            self.run_historical_backtest(
                ticker,
                period
            )
        )


        backtest_data = (
            backtest_result[
                "data"
            ]
        )

        backtest_metrics = (
            backtest_result[
                "metrics"
            ]
        )


        # ====================================================
        # STEP 5 — LLM INTELLIGENCE
        # ====================================================

        ai_analysis = (
            self.generate_intelligence(

                ticker,

                prediction_result,

                risk_report,

                backtest_metrics

            )
        )


        # ====================================================
        # FINAL AGENT RESULT
        # ====================================================

        result = {

            "ticker": ticker,

            "period": period,

            "market_data": data,

            "prediction": (
                prediction_result
            ),

            "risk": (
                risk_report
            ),

            "backtest": {

                "data": backtest_data,

                "metrics": backtest_metrics

            },

            "ai_analysis": (
                ai_analysis
            )

        }


        print("\n")

        print(
            "[Agent] Analysis complete."
        )


        return result


# ============================================================
# TERMINAL TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\nStarting StockPilot Agent..."
    )


    agent = StockPilotAgent()


    result = agent.analyze_stock(

        ticker="AAPL",

        period="10y"

    )


    # ========================================================
    # AGENT RESULT
    # ========================================================

    print("\n")

    print(
        "=" * 60
    )

    print(
        "             AGENT RESULT"
    )

    print(
        "=" * 60
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    prediction = (
        result[
            "prediction"
        ]
    )


    print(
        f"\nTicker: "
        f"{result['ticker']}"
    )

    print(
        f"Period: "
        f"{result['period']}"
    )

    print(
        f"Latest Price: "
        f"${prediction['latest_price']:.2f}"
    )

    print(
        f"Prediction: "
        f"{prediction['prediction']}"
    )

    print(
        f"Confidence: "
        f"{prediction['confidence']:.2f}%"
    )

    print(
        f"UP Probability: "
        f"{prediction['up_probability']:.2f}%"
    )

    print(
        f"DOWN Probability: "
        f"{prediction['down_probability']:.2f}%"
    )


    # ========================================================
    # RISK
    # ========================================================

    risk = (
        result[
            "risk"
        ]
    )


    print("\n")

    print(
        "Risk Metrics:"
    )


    print(
        f"Volatility: "
        f"{risk['volatility'] * 100:.2f}%"
    )

    print(
        f"Sharpe Ratio: "
        f"{risk['sharpe_ratio']:.2f}"
    )

    print(
        f"Maximum Drawdown: "
        f"{risk['max_drawdown'] * 100:.2f}%"
    )

    print(
        f"Total Return: "
        f"{risk['total_return'] * 100:.2f}%"
    )

    print(
        f"Win Rate: "
        f"{risk['win_rate'] * 100:.2f}%"
    )


    # ========================================================
    # BACKTEST
    # ========================================================

    backtest = (
        result[
            "backtest"
        ]
    )


    metrics = (
        backtest[
            "metrics"
        ]
    )


    print("\n")

    print(
        "Backtesting Metrics:"
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

    print(
        f"Total Trading Days: "
        f"{metrics['total_days']}"
    )


    # ========================================================
    # LLM
    # ========================================================

    print("\n")

    print(
        "🤖 AI MARKET INTELLIGENCE"
    )

    print(
        "-" * 60
    )

    print(
        result[
            "ai_analysis"
        ]
    )


    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n")

    print(
        "=" * 60
    )

    print(
        "       STOCKPILOT AGENT COMPLETE"
    )

    print(
        "=" * 60
    )