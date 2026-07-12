

import sys

import config
from utils.metrics import (
    compute_metrics,
    get_predictions,
    plot_confusion_matrix,
    plot_roc_curve,
    save_evaluation_report,
)
from utils.preprocessing import preprocess_pipeline
from utils.train_utils import load_model


def main() -> None:
    """Run full model evaluation on the test set."""
    print("=" * 60)
    print("Predictive Maintenance - Model Evaluation")
    print("=" * 60)
    print(f"Device: {config.DEVICE}")
    print()

    try:
        # Load trained model
        print("Loading trained model...")
        model, input_size = load_model()
        print(f"Model loaded (input_size={input_size}).")
        print()

        # Preprocess data (uses same split/seed for consistency)
        print("Loading test data...")
        data = preprocess_pipeline()
        print()

        # Generate predictions
        print("Generating predictions on test set...")
        y_true, y_pred, y_prob = get_predictions(model, data.test_loader)

        # Compute metrics
        results = compute_metrics(y_true, y_pred, y_prob)

        # Display results
        print("-" * 40)
        print("Evaluation Results")
        print("-" * 40)
        print(f"Accuracy  : {results.accuracy:.4f}")
        print(f"Precision : {results.precision:.4f}")
        print(f"Recall    : {results.recall:.4f}")
        print(f"F1 Score  : {results.f1:.4f}")
        print(f"ROC-AUC   : {results.roc_auc:.4f}")
        print()
        print("Confusion Matrix:")
        print(results.confusion_matrix)
        print()
        print("Classification Report:")
        print(results.classification_report)

        # Save reports and plots
        print("\nSaving reports and plots...")
        save_evaluation_report(results)
        plot_confusion_matrix(results.confusion_matrix)
        plot_roc_curve(results.y_true, results.y_prob, results.roc_auc)

        print("\nEvaluation completed successfully!")
        print(f"Reports saved to: {config.REPORTS_DIR}")
        print(f"Plots saved to: {config.PLOTS_DIR}")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error during evaluation: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
