from src.prediction.predict import predict_stock


result = predict_stock(
    "AAPL",
    "1y"
)


print("\n")
print("=" * 50)
print("          STOCKPILOT PREDICTION")
print("=" * 50)

print(
    f"\nTicker: {result['ticker']}"
)

print(
    f"Latest Price: "
    f"${result['latest_price']:.2f}"
)

print(
    f"Prediction: "
    f"{result['direction']}"
)

print(
    f"Confidence: "
    f"{result['confidence']:.2%}"
)

print(
    f"UP Probability: "
    f"{result['up_probability']:.2%}"
)

print(
    f"DOWN Probability: "
    f"{result['down_probability']:.2%}"
)

print(
    f"Features Used: "
    f"{result['feature_count']}"
)