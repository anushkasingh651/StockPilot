from src.data.stock_data import get_stock_data

from src.features.feature_pipeline import build_features

from src.targets import create_target

from src.preprocessing import (
    prepare_dataset,
    time_series_split
)

from src.models.baseline_model import (
    train_baseline_model,
    make_predictions
)

from src.models.random_forest_model import (
    train_random_forest,
    make_predictions as make_rf_predictions
)

from src.models.gradient_boosting_model import (
    train_gradient_boosting,
    make_predictions as make_gb_predictions
)

from src.evaluation import evaluate_model

from src.models.save_model import save_model


def print_metrics(model_name, metrics):

    print("\n")
    print("=" * 60)
    print(f"              {model_name}")
    print("=" * 60)

    print(
        f"\nAccuracy:  "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall:    "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1 Score:  "
        f"{metrics['f1_score']:.4f}"
    )

    print(
        f"ROC-AUC:   "
        f"{metrics['roc_auc']:.4f}"
    )

    print("\nConfusion Matrix:")

    print(
        metrics["confusion_matrix"]
    )


def main():

    # ==========================================
    # 1. CONFIGURATION
    # ==========================================

    ticker = "AAPL"
    period = "10y"


    # ==========================================
    # 2. FETCH STOCK DATA
    # ==========================================

    print("\nFetching stock data...")

    data = get_stock_data(
        ticker,
        period
    )

    if data is None or data.empty:

        print(
            "Unable to fetch stock data."
        )

        return

    print(
        f"Successfully loaded {ticker} data."
    )


    # ==========================================
    # 3. FEATURE ENGINEERING
    # ==========================================

    print("\nBuilding technical features...")

    data = build_features(
        data
    )


    # ==========================================
    # 4. CREATE TARGET
    # ==========================================

    print("Creating prediction target...")

    data = create_target(
        data
    )


    # ==========================================
    # 5. PREPARE DATASET
    # ==========================================

    print("Preparing ML dataset...")

    X, y, data = prepare_dataset(
        data
    )


    # ==========================================
    # 6. TIME-BASED TRAIN / TEST SPLIT
    # ==========================================

    print("Creating time-based train/test split...")

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = time_series_split(
        X,
        y
    )


    # ==========================================
    # DATASET INFORMATION
    # ==========================================

    print("\n")
    print("=" * 60)
    print("             STOCKPILOT DATA")
    print("=" * 60)

    print(
        f"\nTicker: {ticker}"
    )

    print(
        f"Historical Period: {period}"
    )

    print(
        f"Total Samples: {len(X)}"
    )

    print(
        f"Training Samples: {len(X_train)}"
    )

    print(
        f"Testing Samples: {len(X_test)}"
    )

    print(
        f"Number of Features: {X.shape[1]}"
    )


    # ==========================================
    # 7. LOGISTIC REGRESSION
    # ==========================================

    print("\nTraining Logistic Regression...")

    baseline_model = train_baseline_model(
        X_train,
        y_train
    )


    baseline_predictions, baseline_probabilities = (
        make_predictions(
            baseline_model,
            X_test
        )
    )


    baseline_metrics = evaluate_model(
        y_test,
        baseline_predictions,
        baseline_probabilities
    )


    print_metrics(
        "LOGISTIC REGRESSION BASELINE",
        baseline_metrics
    )


    # ==========================================
    # 8. RANDOM FOREST
    # ==========================================

    print("\nTraining Random Forest...")

    rf_model = train_random_forest(
        X_train,
        y_train
    )


    rf_predictions, rf_probabilities = (
        make_rf_predictions(
            rf_model,
            X_test
        )
    )


    rf_metrics = evaluate_model(
        y_test,
        rf_predictions,
        rf_probabilities
    )


    print_metrics(
        "RANDOM FOREST",
        rf_metrics
    )


    # ==========================================
    # 9. GRADIENT BOOSTING
    # ==========================================

    print("\nTraining Gradient Boosting...")

    gb_model = train_gradient_boosting(
        X_train,
        y_train
    )


    gb_predictions, gb_probabilities = (
        make_gb_predictions(
            gb_model,
            X_test
        )
    )


    gb_metrics = evaluate_model(
        y_test,
        gb_predictions,
        gb_probabilities
    )


    print_metrics(
        "GRADIENT BOOSTING",
        gb_metrics
    )


    # ==========================================
    # 10. MODEL COMPARISON
    # ==========================================

    print("\n")
    print("=" * 80)
    print("                         MODEL COMPARISON")
    print("=" * 80)

    print(
        f"\n{'Metric':<15}"
        f"{'Logistic Regression':<23}"
        f"{'Random Forest':<20}"
        f"{'Gradient Boosting'}"
    )

    print("-" * 80)

    print(
        f"{'Accuracy':<15}"
        f"{baseline_metrics['accuracy']:<23.4f}"
        f"{rf_metrics['accuracy']:<20.4f}"
        f"{gb_metrics['accuracy']:.4f}"
    )

    print(
        f"{'Precision':<15}"
        f"{baseline_metrics['precision']:<23.4f}"
        f"{rf_metrics['precision']:<20.4f}"
        f"{gb_metrics['precision']:.4f}"
    )

    print(
        f"{'Recall':<15}"
        f"{baseline_metrics['recall']:<23.4f}"
        f"{rf_metrics['recall']:<20.4f}"
        f"{gb_metrics['recall']:.4f}"
    )

    print(
        f"{'F1 Score':<15}"
        f"{baseline_metrics['f1_score']:<23.4f}"
        f"{rf_metrics['f1_score']:<20.4f}"
        f"{gb_metrics['f1_score']:.4f}"
    )

    print(
        f"{'ROC-AUC':<15}"
        f"{baseline_metrics['roc_auc']:<23.4f}"
        f"{rf_metrics['roc_auc']:<20.4f}"
        f"{gb_metrics['roc_auc']:.4f}"
    )


    # ==========================================
    # 11. SELECT BEST MODEL
    # ==========================================

    model_scores = {

        "Logistic Regression":
            baseline_metrics["roc_auc"],

        "Random Forest":
            rf_metrics["roc_auc"],

        "Gradient Boosting":
            gb_metrics["roc_auc"]
    }


    best_model_name = max(
        model_scores,
        key=model_scores.get
    )


    best_score = model_scores[
        best_model_name
    ]


    # ==========================================
    # 12. GET BEST MODEL OBJECT
    # ==========================================

    if best_model_name == "Logistic Regression":

        best_model = baseline_model

    elif best_model_name == "Random Forest":

        best_model = rf_model

    else:

        best_model = gb_model


    # ==========================================
    # 13. DISPLAY BEST MODEL
    # ==========================================

    print("\n")
    print("=" * 60)
    print("                   BEST MODEL")
    print("=" * 60)

    print(
        f"\nModel: {best_model_name}"
    )

    print(
        f"ROC-AUC: {best_score:.4f}"
    )


    # ==========================================
    # 14. SAVE BEST MODEL
    # ==========================================

    print("\nSaving best model...")

    save_model(
        best_model
    )


    # ==========================================
    # 15. SAMPLE PREDICTIONS
    # ==========================================

    print("\n")
    print("=" * 60)
    print("              SAMPLE PREDICTIONS")
    print("=" * 60)


    print("\nLogistic Regression:")

    print(
        "Predictions:",
        baseline_predictions[:10]
    )

    print(
        "Probabilities:",
        baseline_probabilities[:10]
    )


    print("\nRandom Forest:")

    print(
        "Predictions:",
        rf_predictions[:10]
    )

    print(
        "Probabilities:",
        rf_probabilities[:10]
    )


    print("\nGradient Boosting:")

    print(
        "Predictions:",
        gb_predictions[:10]
    )

    print(
        "Probabilities:",
        gb_probabilities[:10]
    )


    # ==========================================
    # 16. COMPLETE
    # ==========================================

    print("\n")
    print("=" * 60)
    print("              TRAINING COMPLETE")
    print("=" * 60)

    print(
        "\nStockPilot successfully trained, "
        "evaluated and saved the best model."
    )


if __name__ == "__main__":

    main()