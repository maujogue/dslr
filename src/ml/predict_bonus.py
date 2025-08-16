import numpy as np
import os
from sklearn.metrics import accuracy_score
from data_handling.constants import HOUSES, GREEN
from ml.train import forward
from ml.predict import load_weights


def load_all_weights():
    weights_dir = "weights"
    weights = {}

    weight_files = [
        ("Batch GD", f"{weights_dir}/weights_batch.txt"),
        ("SGD", f"{weights_dir}/weights_sgd.txt"),
        ("Mini-batch", f"{weights_dir}/weights_mini_batch.txt")
    ]

    for name, filepath in weight_files:
        try:
            model_weights = load_weights(filepath)
            weights[name] = model_weights
            print(f"{GREEN}✓ Loaded {name} model")
        except Exception as e:
            print(f"✗ Failed to load {name} model: {e}")

    return weights


def predict_with_model(X, model_weights):
    probabilities = [forward(X, model_weights[house]) for house in HOUSES]
    probabilities = np.hstack(probabilities)
    best_house_indices = np.argmax(probabilities, axis=1)
    return [HOUSES[idx] for idx in best_house_indices]


def calculate_prediction_agreement(predictions1, predictions2):
    if len(predictions1) != len(predictions2):
        return 0.0

    agreement = sum(1 for p1, p2 in zip(predictions1, predictions2) if p1 == p2)
    return (agreement / len(predictions1)) * 100


def compare_predictions(all_predictions):
    print(f"\n{GREEN}PREDICTION COMPARISON")
    print(f"{GREEN}=========================")

    model_names = list(all_predictions.keys())

    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i < j:
                agreement = calculate_prediction_agreement(
                    all_predictions[model1],
                    all_predictions[model2]
                )
                print(f"{model1:<12} vs {model2:<12}: {agreement:.1f}% agreement")


def compare_accuracy(all_predictions, labels):
    if labels is None:
        print(f"\n{GREEN}No labels available for accuracy comparison")
        return

    print(f"\n{GREEN}ACCURACY COMPARISON")
    print(f"{GREEN}=========================")

    for model_name, predictions in all_predictions.items():
        try:
            accuracy = accuracy_score(labels, predictions)
            print(f"{model_name:<12}: {accuracy:.1%}")
        except ValueError:
            print(f"{model_name:<12}: Error calculating accuracy")


def save_all_predictions(all_predictions):
    predictions_dir = "datasets"
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)

    for model_name, predictions in all_predictions.items():
        filename = f"{predictions_dir}/houses_{model_name.lower().replace(' ', '_')}.csv"
        with open(filename, "w") as f:
            f.write("Index,Hogwarts House\n")
            for i, pred in enumerate(predictions):
                f.write(f"{i},{pred}\n")
        print(f"{GREEN}  - {filename} ({model_name} predictions)")

    print(f"\n{GREEN}All predictions saved to {predictions_dir}/")


def predict_all_bonus(X, labels):
    print(f"\n{GREEN}BONUS MODE: Predicting with multiple models!")
    print(f"{GREEN}Comparing Batch GD, SGD, and Mini-batch GD predictions\n")

    all_weights = load_all_weights()
    if not all_weights:
        print("Error: No models loaded. Make sure to run training with --bonus first.")
        return

    all_predictions = {}
    for model_name, weights in all_weights.items():
        print(f"Making predictions with {model_name}...")
        predictions = predict_with_model(X, weights)
        all_predictions[model_name] = predictions
        print(f"  {len(predictions)} predictions completed")

    compare_predictions(all_predictions)
    compare_accuracy(all_predictions, labels)

    save_all_predictions(all_predictions)

    print(f"\n{GREEN}BONUS PREDICTION COMPLETED!")
    print(f"{GREEN}Use the saved files to analyze model differences.")
