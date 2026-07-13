
import sys

import numpy as np
import torch

import config
from utils.preprocessing import load_preprocessing_artifacts
from utils.train_utils import load_model


def get_sensor_input(
    type_encoder,
    feature_names: list[str],
) -> np.ndarray:
    
    print("\nEnter sensor readings for the machine:")
    print("-" * 50)

    # Typical value ranges for numeric sensors (guidance for the user)
    typical_ranges = {
        "Air temperature [K]": (298.0, 310.0),
        "Process temperature [K]": (308.0, 320.0),
        "Rotational speed [rpm]": (1100, 2900),
        "Torque [Nm]": (3.0, 77.0),
        "Tool wear [min]": (0, 253),
    }

    feature_values: dict[str, float] = {}

    for feature_name in feature_names:
        if feature_name == config.TYPE_COLUMN:
            while True:
                product_type = input(
                    "Product Type (L=Low, M=Medium, H=High): "
                ).strip().upper()
                if product_type in type_encoder.classes_:
                    feature_values[feature_name] = float(
                        type_encoder.transform([product_type])[0]
                    )
                    break
                print(f"Invalid type. Choose from: {list(type_encoder.classes_)}")
        else:
            low, high = typical_ranges.get(feature_name, (0.0, 1000.0))
            while True:
                try:
                    raw = input(
                        f"{feature_name} (typical range {low}-{high}): "
                    ).strip()
                    feature_values[feature_name] = float(raw)
                    break
                except ValueError:
                    print("Please enter a valid numeric value.")

    ordered_values = [feature_values[name] for name in feature_names]
    feature_vector = np.array(ordered_values, dtype=np.float32).reshape(1, -1)
    return feature_vector


def predict_failure(
    model: torch.nn.Module,
    feature_vector: np.ndarray,
    scaler,
) -> tuple[str, float]:
    
    
    scaled = scaler.transform(feature_vector)
    tensor = torch.tensor(scaled, dtype=torch.float32).to(config.DEVICE)

    model.eval()
    with torch.no_grad():
        probability = model(tensor).item()

    label = "Machine Failure" if probability >= 0.5 else "No Machine Failure"
    return label, probability


def main() -> None:
    """Run interactive prediction from terminal input."""
    print("=" * 60)
    print("Predictive Maintenance - Failure Prediction")
    print("=" * 60)
    print(f"Device: {config.DEVICE}")
    print()

    try:
        # Load model and preprocessing artifacts
        print("Loading model and preprocessing artifacts...")
        model, input_size = load_model()
        scaler, type_encoder, feature_names = load_preprocessing_artifacts()
        print(f"Model ready (input_size={input_size}).")
        print(f"Expected features: {feature_names}")

        while True:
           
            raw_features = get_sensor_input(type_encoder, feature_names)

            
            label, probability = predict_failure(
                model, raw_features, scaler
            )

            
            print("\n" + "=" * 50)
            print("PREDICTION RESULT")
            print("=" * 50)
            print(f"Prediction        : {label}")
            print(f"Failure Probability: {probability:.4f} ({probability * 100:.2f}%)")
            print(f"No Failure Probability: {1 - probability:.4f} ({(1 - probability) * 100:.2f}%)")
            print("=" * 50)

            
            again = input("\nPredict another reading? (y/n): ").strip().lower()
            if again != "y":
                print("Exiting prediction script. Goodbye!")
                break

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nPrediction cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error during prediction: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
