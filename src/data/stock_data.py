import yfinance as yf
import pandas as pd


def get_stock_data(ticker, period):

    try:

        data = yf.download(
            ticker,
            period=period,
            auto_adjust=False,
            progress=False
        )

        if data.empty:
            return None

        # Handle MultiIndex columns returned by yfinance
        if isinstance(data.columns, pd.MultiIndex):

            data.columns = data.columns.get_level_values(0)

        data = data.reset_index()

        return data

    except Exception as e:

        print(f"Error fetching stock data: {e}")

        return None