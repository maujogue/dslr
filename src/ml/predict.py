import numpy as np
from sklearn.metrics import accuracy_score
from data_handling.constants import HOUSES, GREEN, BLUE, RESET
from ml.train import forward


def load_weights():
    weights = {}
    with open("weights.txt", "r") as f:
        for line in f:
            house, weight_str = line.strip().split(":")
            weight_list = eval(weight_str)
            weights[house] = np.array(weight_list).reshape(-1, 1)
    return weights


def predict(X, weights):
    # Compute probabilities for all houses, shape: (n_samples, n_houses)
    probabilities = [forward(X, weights[house]) for house in HOUSES]
    probabilities = np.hstack(probabilities)
    # For each sample, pick the house with the highest probability
    best_house_indices = np.argmax(probabilities, axis=1)
    return [HOUSES[idx] for idx in best_house_indices]


def save_predictions(predictions):
    with open("datasets/houses.csv", "w") as f:
        f.write("Index,Hogwarts House\n")
        for i, pred in enumerate(predictions):
            f.write(f"{i},{pred}\n")
    print(f"\n{GREEN}Predictions saved to datasets/houses.csv{RESET}")


def print_precision(predictions, labels):
    try:
        precision = accuracy_score(labels, predictions)
        print(f"\nAccuracy Score from Scikit-Learn: {BLUE}{precision}{RESET}\n")
        if precision > 0.98:
            print(f"{GREEN}PASSED! ✅{RESET}\n")
    except ValueError:
        print(
            "Error: labels and predictions must ",
            "have the same length and no NaN values",
        )
