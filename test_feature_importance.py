from src.explainability.feature_importance import (
    get_feature_importance
)


importance = get_feature_importance()


print("\n")
print("=" * 60)
print("             STOCKPILOT FEATURE IMPORTANCE")
print("=" * 60)


print("\n")

print(
    importance.to_string(
        index=False
    )
)


print("\n")
print("=" * 60)
print("                 TOP 5 FEATURES")
print("=" * 60)


print(
    importance.head(5).to_string(
        index=False
    )
)