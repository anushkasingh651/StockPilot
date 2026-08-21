import os

from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# CREATE OPENAI CLIENT
# ============================================================

def get_client():

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    if not api_key:

        raise ValueError(
            "OPENAI_API_KEY is not configured. "
            "Please add it to the .env file."
        )

    return OpenAI(
        api_key=api_key
    )


# ============================================================
# GENERATE STOCK ANALYSIS
# ============================================================

def generate_stock_analysis(
    ticker,
    latest_price,
    prediction,
    confidence,
    up_probability,
    down_probability,
    volatility,
    sharpe_ratio,
    max_drawdown,
    total_return,
    win_rate
):

    client = get_client()


    # ========================================================
    # PROMPT
    # ========================================================

    prompt = f"""
You are StockPilot's AI financial intelligence assistant.

Analyze the following quantitative stock-analysis results.

IMPORTANT RULES:

- Do not invent financial data.
- Do not change the supplied ML prediction.
- Do not claim certainty.
- Clearly distinguish model prediction from factual data.
- Do not provide personalized financial advice.
- Explain the numbers in simple professional language.

============================================================
STOCK
============================================================

Ticker:
{ticker}

Latest Price:
${latest_price:.2f}


============================================================
MACHINE LEARNING MODEL
============================================================

Prediction:
{prediction}

Confidence:
{confidence:.2f}%

UP Probability:
{up_probability:.2f}%

DOWN Probability:
{down_probability:.2f}%


============================================================
RISK ANALYTICS
============================================================

Volatility:
{volatility * 100:.2f}%

Sharpe Ratio:
{sharpe_ratio:.2f}

Maximum Drawdown:
{max_drawdown * 100:.2f}%

Total Return:
{total_return * 100:.2f}%

Win Rate:
{win_rate * 100:.2f}%


============================================================
TASK
============================================================

Provide a concise StockPilot intelligence report.

Use these sections:

1. Market Signal
2. Model Confidence
3. Risk Interpretation
4. Key Observation
5. Important Caution
6. Overall Takeaway

Keep the response below 250 words.

Do not tell the user to buy or sell the stock.
"""


    # ========================================================
    # CALL OPENAI
    # ========================================================

    response = client.responses.create(

        model="gpt-5-mini",

        input=prompt

    )


    # ========================================================
    # RETURN RESPONSE
    # ========================================================

    return response.output_text


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)

    print(
        "        STOCKPILOT LLM ENGINE TEST"
    )

    print("=" * 60)


    try:

        analysis = generate_stock_analysis(

            ticker="AAPL",

            latest_price=316.67,

            prediction="DOWN",

            confidence=52.05,

            up_probability=47.95,

            down_probability=52.05,

            volatility=0.4513,

            sharpe_ratio=1.25,

            max_drawdown=-0.0273,

            total_return=0.15,

            win_rate=0.6667

        )


        print("\n🤖 AI ANALYSIS\n")

        print(
            analysis
        )


        print("\n" + "=" * 60)

        print(
            "        LLM ENGINE TEST COMPLETE"
        )

        print("=" * 60)


    except Exception as error:

        print("\n❌ LLM ERROR")

        print(
            error
        )