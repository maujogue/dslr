import numpy as np
import os
from data_handling.constants import BLUE, GREEN, HOUSES, RESET


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forward(X, w):
    return sigmoid(np.matmul(X, w))


# We don't use mse because it's not convex when used with sigmoid
# There will be multiple local minima.
# We use log loss because it's convex.
def loss(X, Y, w):
    y_hat = forward(X, w)
    first_term = Y * np.log(y_hat)
    second_term = (1 - Y) * np.log(1 - y_hat)
    return -np.average(first_term + second_term)


def gradient(X, Y, w):
    return np.matmul(X.T, (forward(X, w) - Y)) / X.shape[0]


def train(X, Y, iterations, lr):
    w = np.zeros((X.shape[1], 1))
    for i in range(iterations):
        if i % 500 == 0:
            print("Iteration %4d => Loss: %.20f" % (i, loss(X, Y, w)))
        w_gradient = gradient(X, Y, w) * lr
        w -= w_gradient
    return w


def save_weights(weights, filename, description):
    with open(filename, "w") as f:
        for house in HOUSES:
            f.write(f"{house}:{weights[house].flatten().tolist()}\n")
    print(f"  - {filename} ({description})")
    print("Weights saved successfully!")


def create_weights_directory():
    weights_dir = "weights"
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
        print(f"Created weights directory: {weights_dir}/")
    return weights_dir


def train_all(X, Y_dict, iterations, lr):
    weights = {}

    for house in HOUSES:
        print(f"{BLUE}Training {house}...{RESET}")
        weights[house] = train(X, Y_dict[house], iterations, lr)

    weights_dir = create_weights_directory()
    save_weights(
        weights, f"{weights_dir}/weights_batch.txt", "regular training"
    )
    print(f"\n{GREEN}Weights saved in {weights_dir}/:")

    print(f"{GREEN}You can now use the model to make predictions{RESET}")
