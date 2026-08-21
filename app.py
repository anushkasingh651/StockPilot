import streamlit as st
import pandas as pd

from src.data.stock_data import get_stock_data
from src.prediction.predict import predict_stock
from src.features.feature_pipeline import build_features

from src.explainability.model_explainability import (
    get_feature_importance
)

from src.risk.risk_engine import (
    generate_risk_report
)

from src.ai.llm_engine import (
    generate_stock_analysis
)

from src.backtesting.backtest_engine import (
    run_backtest
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="StockPilot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# HIDE STREAMLIT DEFAULT TOP BAR
# ============================================================

st.markdown(
    """
    <style>

    header[data-testid="stHeader"] {
        display: none;
    }

    div[data-testid="stToolbar"] {
        display: none;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "stock_data" not in st.session_state:
    st.session_state.stock_data = None

if "feature_data" not in st.session_state:
    st.session_state.feature_data = None

if "risk_report" not in st.session_state:
    st.session_state.risk_report = None

if "backtest_result" not in st.session_state:
    st.session_state.backtest_result = None


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🔍 Stock Analysis")

ticker = st.sidebar.text_input(
    "Stock Symbol",
    value="AAPL",
    help="Example: AAPL, MSFT, NVDA, GOOGL"
).upper().strip()


period = st.sidebar.selectbox(
    "Historical Period",
    [
        "1mo",
        "3mo",
        "6mo",
        "1y",
        "2y",
        "5y",
        "10y"
    ],
    index=3
)


analyze_button = st.sidebar.button(
    "🚀 Analyze Stock",
    width="stretch"
)




# ============================================================
# STOCKPILOT HEADER
# ============================================================

st.title("📈 StockPilot")

st.subheader(
    "AI-Powered Stock Intelligence Platform"
)


st.divider()


# ============================================================
# ANALYZE STOCK
# ============================================================

if analyze_button:

    if not ticker:

        st.error(
            "Please enter a valid stock symbol."
        )

        st.stop()


    try:

        # ====================================================
        # MARKET DATA
        # ====================================================

        with st.spinner(
            f"Fetching {ticker} market data..."
        ):

            data = get_stock_data(
                ticker,
                period
            )


        if data is None or data.empty:

            st.error(
                f"No market data found for {ticker}."
            )

            st.stop()


        # ====================================================
        # ML PREDICTION
        # ====================================================

        with st.spinner(
            "Running machine-learning prediction..."
        ):

            prediction_result = predict_stock(
                ticker,
                period
            )


        # ====================================================
        # TECHNICAL FEATURES
        # ====================================================

        with st.spinner(
            "Calculating technical indicators..."
        ):

            feature_data = build_features(
                data.copy()
            )

            feature_data = (
                feature_data
                .dropna()
            )


        # ====================================================
        # RISK ANALYSIS
        # ====================================================

        with st.spinner(
            "Calculating risk analytics..."
        ):

            risk_report = generate_risk_report(
                data
            )


        # ====================================================
        # OUT-OF-SAMPLE BACKTEST
        # ====================================================

        with st.spinner(
            "Running out-of-sample backtest..."
        ):

            backtest_data, backtest_metrics = (
                run_backtest(
                    ticker=ticker,
                    period="10y"
                )
            )


        # ====================================================
        # SAVE RESULTS
        # ====================================================

        st.session_state.analysis_done = True

        st.session_state.analysis_result = (
            prediction_result
        )

        st.session_state.stock_data = data

        st.session_state.feature_data = (
            feature_data
        )

        st.session_state.risk_report = (
            risk_report
        )

        st.session_state.backtest_result = {
            "data": backtest_data,
            "metrics": backtest_metrics
        }


        st.success(
            f"Successfully analyzed {ticker}"
        )


    except Exception as error:

        st.session_state.analysis_done = False

        st.error(
            "An error occurred while analyzing the stock."
        )

        st.exception(error)

        st.stop()


# ============================================================
# MAIN DASHBOARD
# ============================================================

if st.session_state.analysis_done:

    result = (
        st.session_state.analysis_result
    )

    data = (
        st.session_state.stock_data
    )

    feature_data = (
        st.session_state.feature_data
    )

    risk_report = (
        st.session_state.risk_report
    )

    backtest = (
        st.session_state.backtest_result
    )


    # ========================================================
    # PREDICTION VALUES
    # ========================================================

    latest_price = result[
        "latest_price"
    ]

    prediction = result[
        "prediction"
    ]

    confidence = result[
        "confidence"
    ]

    up_probability = result[
        "up_probability"
    ]

    down_probability = result[
        "down_probability"
    ]

    features_used = result[
        "features_used"
    ]


    # ========================================================
    # DASHBOARD HEADER
    # ========================================================

    st.header(
        f"📊 {ticker} Market Dashboard"
    )

    st.caption(
        f"Analysis period: {period}"
    )


    # ========================================================
    # TOP METRICS
    # ========================================================

    col1, col2, col3, col4 = (
        st.columns(4)
    )


    with col1:

        st.metric(
            label="Latest Price",
            value=f"${latest_price:.2f}"
        )


    with col2:

        if prediction == "UP":

            st.metric(
                label="ML Prediction",
                value="📈 UP"
            )

        else:

            st.metric(
                label="ML Prediction",
                value="📉 DOWN"
            )


    with col3:

        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )


    with col4:

        st.metric(
            label="Model",
            value="Random Forest"
        )


    # ========================================================
    # PROBABILITIES
    # ========================================================

    st.subheader(
        "🎯 Prediction Probabilities"
    )


    probability_col1, probability_col2 = (
        st.columns(2)
    )


    with probability_col1:

        st.metric(
            label="📈 UP Probability",
            value=f"{up_probability:.2f}%"
        )

        st.progress(
            min(
                max(
                    up_probability / 100,
                    0.0
                ),
                1.0
            )
        )


    with probability_col2:

        st.metric(
            label="📉 DOWN Probability",
            value=f"{down_probability:.2f}%"
        )

        st.progress(
            min(
                max(
                    down_probability / 100,
                    0.0
                ),
                1.0
            )
        )


    st.info(
        f"StockPilot uses a Random Forest classification "
        f"model with {features_used} technical features "
        f"to estimate the next market direction."
    )


    # ========================================================
    # FINANCIAL CHARTS
    # ========================================================

    st.divider()

    st.header(
        "📈 Interactive Financial Charts"
    )


    # ========================================================
    # PRICE + MOVING AVERAGES
    # ========================================================

    st.subheader(
        "Price & Moving Averages"
    )


    price_columns = []


    if "Close" in feature_data.columns:

        price_columns.append(
            "Close"
        )


    if "SMA20" in feature_data.columns:

        price_columns.append(
            "SMA20"
        )


    if "EMA20" in feature_data.columns:

        price_columns.append(
            "EMA20"
        )


    if price_columns:

        st.line_chart(
            feature_data[
                price_columns
            ],
            width="stretch"
        )

    else:

        st.warning(
            "Price or moving-average data unavailable."
        )


    # ========================================================
    # RSI + MACD
    # ========================================================

    indicator_col1, indicator_col2 = (
        st.columns(2)
    )


    with indicator_col1:

        st.subheader(
            "📊 RSI"
        )


        if "RSI" in feature_data.columns:

            st.line_chart(
                feature_data[
                    ["RSI"]
                ],
                width="stretch"
            )

        else:

            st.warning(
                "RSI data unavailable."
            )


    with indicator_col2:

        st.subheader(
            "📉 MACD"
        )


        macd_columns = []


        if "MACD" in feature_data.columns:

            macd_columns.append(
                "MACD"
            )


        if "MACD_Signal" in feature_data.columns:

            macd_columns.append(
                "MACD_Signal"
            )


        if macd_columns:

            st.line_chart(
                feature_data[
                    macd_columns
                ],
                width="stretch"
            )

        else:

            st.warning(
                "MACD data unavailable."
            )


    # ========================================================
    # BOLLINGER BANDS
    # ========================================================

    st.subheader(
        "📊 Bollinger Bands"
    )


    bollinger_columns = []


    if "Close" in feature_data.columns:

        bollinger_columns.append(
            "Close"
        )


    if "BB_Upper" in feature_data.columns:

        bollinger_columns.append(
            "BB_Upper"
        )


    if "BB_Lower" in feature_data.columns:

        bollinger_columns.append(
            "BB_Lower"
        )


    if len(bollinger_columns) >= 2:

        st.line_chart(
            feature_data[
                bollinger_columns
            ],
            width="stretch"
        )

    else:

        st.warning(
            "Bollinger Band data unavailable."
        )


    # ========================================================
    # RISK ANALYTICS
    # ========================================================

    st.divider()

    st.header(
        "🛡️ Risk Analytics"
    )


    risk_col1, risk_col2, risk_col3 = (
        st.columns(3)
    )


    with risk_col1:

        st.metric(
            label="Volatility",
            value=(
                f"{risk_report['volatility'] * 100:.2f}%"
            )
        )


    with risk_col2:

        st.metric(
            label="Sharpe Ratio",
            value=(
                f"{risk_report['sharpe_ratio']:.2f}"
            )
        )


    with risk_col3:

        st.metric(
            label="Maximum Drawdown",
            value=(
                f"{risk_report['max_drawdown'] * 100:.2f}%"
            )
        )


    risk_col4, risk_col5 = (
        st.columns(2)
    )


    with risk_col4:

        st.metric(
            label="Total Return",
            value=(
                f"{risk_report['total_return'] * 100:.2f}%"
            )
        )


    with risk_col5:

        st.metric(
            label="Win Rate",
            value=(
                f"{risk_report['win_rate'] * 100:.2f}%"
            )
        )


    # ========================================================
    # BACKTESTING
    # ========================================================

    st.divider()

    st.header(
        "📊 Out-of-Sample Backtesting"
    )


    st.caption(
        "The strategy is evaluated on an unseen "
        "historical test period."
    )


    backtest_data = backtest[
        "data"
    ]

    backtest_metrics = backtest[
        "metrics"
    ]


    # ========================================================
    # BACKTEST METRICS
    # ========================================================

    bt_col1, bt_col2, bt_col3, bt_col4 = (
        st.columns(4)
    )


    with bt_col1:

        st.metric(
            label="Strategy Return",
            value=(
                f"{backtest_metrics['strategy_return']:.2%}"
            )
        )


    with bt_col2:

        st.metric(
            label="Buy & Hold",
            value=(
                f"{backtest_metrics['buy_hold_return']:.2%}"
            )
        )


    with bt_col3:

        st.metric(
            label="Maximum Drawdown",
            value=(
                f"{backtest_metrics['maximum_drawdown']:.2%}"
            )
        )


    with bt_col4:

        st.metric(
            label="Win Rate",
            value=(
                f"{backtest_metrics['win_rate']:.2%}"
            )
        )


    # ========================================================
    # EQUITY CURVE
    # ========================================================

    st.subheader(
        "📈 Strategy vs Buy & Hold"
    )


    equity_columns = []


    if "Strategy_Equity" in backtest_data.columns:

        equity_columns.append(
            "Strategy_Equity"
        )


    if "Buy_Hold_Equity" in backtest_data.columns:

        equity_columns.append(
            "Buy_Hold_Equity"
        )


    if equity_columns:

        st.line_chart(
            backtest_data[
                equity_columns
            ],
            width="stretch"
        )

    else:

        st.warning(
            "Backtest equity data unavailable."
        )


    # ========================================================
    # DAILY STRATEGY RETURNS
    # ========================================================

    st.subheader(
        "📊 Daily Strategy Returns"
    )


    if "Strategy_Return" in backtest_data.columns:

        st.bar_chart(
            backtest_data[
                ["Strategy_Return"]
            ],
            width="stretch"
        )


    # ========================================================
    # BACKTEST SUMMARY
    # ========================================================

    st.subheader(
        "📋 Backtest Summary"
    )


    summary_col1, summary_col2 = (
        st.columns(2)
    )


    with summary_col1:

        st.write(
            f"**Test Samples:** "
            f"{backtest_metrics['total_days']}"
        )

        st.write(
            f"**UP Predictions:** "
            f"{backtest_metrics['up_predictions']}"
        )

        st.write(
            f"**DOWN Predictions:** "
            f"{backtest_metrics['down_predictions']}"
        )


    with summary_col2:

        st.write(
            f"**Strategy Return:** "
            f"{backtest_metrics['strategy_return']:.2%}"
        )

        st.write(
            f"**Buy & Hold Return:** "
            f"{backtest_metrics['buy_hold_return']:.2%}"
        )

        st.write(
            f"**Maximum Drawdown:** "
            f"{backtest_metrics['maximum_drawdown']:.2%}"
        )


    # ========================================================
    # MODEL EXPLAINABILITY
    # ========================================================

    st.divider()

    st.header(
        "🧠 Model Explainability"
    )


    try:

        importance_df = (
            get_feature_importance()
        )


        top_features = (
            importance_df
            .head(10)
            .copy()
        )


        chart_data = (
            top_features
            .set_index("Feature")
            [["Importance"]]
        )


        st.bar_chart(
            chart_data,
            width="stretch"
        )


        with st.expander(
            "View complete feature importance"
        ):

            st.dataframe(
                importance_df,
                width="stretch"
            )


    except Exception as error:

        st.warning(
            "Feature importance could not be loaded."
        )

        st.exception(error)


    # ========================================================
    # AI MARKET INTELLIGENCE
    # ========================================================

    st.divider()

    st.header(
        "🤖 AI Market Intelligence"
    )


    st.caption(
        "AI-generated interpretation of StockPilot's "
        "prediction and risk metrics."
    )


    try:

        with st.spinner(
            "Generating AI market intelligence..."
        ):

            ai_analysis = generate_stock_analysis(

                ticker=ticker,

                latest_price=latest_price,

                prediction=prediction,

                confidence=confidence,

                up_probability=up_probability,

                down_probability=down_probability,

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


        st.markdown(
            ai_analysis
        )


    except Exception as error:

        st.warning(
            "AI market intelligence could not be generated."
        )

        st.exception(error)


    # ========================================================
    # HISTORICAL DATA
    # ========================================================

    st.divider()

    st.header(
        "📋 Historical Market Data"
    )


    with st.expander(
        "View historical stock data"
    ):

        st.dataframe(
            data,
            width="stretch"
        )


    # ========================================================
    # ANALYSIS SUMMARY
    # ========================================================

    st.header(
        "🧾 Analysis Summary"
    )


    summary1, summary2, summary3 = (
        st.columns(3)
    )


    with summary1:

        st.write(
            f"**Ticker:** {ticker}"
        )

        st.write(
            f"**Analysis Period:** {period}"
        )


    with summary2:

        st.write(
            f"**Prediction:** {prediction}"
        )

        st.write(
            f"**Confidence:** "
            f"{confidence:.2f}%"
        )


    with summary3:

        st.write(
            "**Model:** Random Forest"
        )

        st.write(
            f"**Features:** {features_used}"
        )


# ============================================================
# INITIAL SCREEN
# ============================================================

else:

    st.header(
        "Welcome to StockPilot 🚀"
    )


    st.write(
        "Enter a stock symbol in the sidebar and "
        "click **Analyze Stock** to begin."
    )


    st.divider()


    feature_col1, feature_col2, feature_col3 = (
        st.columns(3)
    )


    with feature_col1:

        st.subheader(
            "🤖 Machine Learning"
        )

        st.write(
            "Random Forest predicts market direction "
            "using technical features."
        )


    with feature_col2:

        st.subheader(
            "📊 Financial Analytics"
        )

        st.write(
            "Analyze price trends, RSI, MACD, "
            "Bollinger Bands and risk."
        )


    with feature_col3:

        st.subheader(
            "📈 Backtesting"
        )

        st.write(
            "Evaluate the strategy on an unseen "
            "historical test period."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "📈 StockPilot • AI-Powered Stock Intelligence Platform"
)
