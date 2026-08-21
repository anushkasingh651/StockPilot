def create_target(data):
    """
    Create a binary target based on
    the next trading day's return.
    """

    data = data.copy()

    data["Future_Return"] = (
        data["Close"].shift(-1) /
        data["Close"]
        - 1
    )

    data["Target"] = (
        data["Future_Return"] > 0
    ).astype(int)

    return data