import numpy as np
import time
from data_handling.constants import GREEN, HOUSES
from .train import (train, loss, save_weights, create_weights_directory,
                    gradient)


def train_sgd(X, Y, iterations, lr):
    w = np.zeros((X.shape[1], 1))
    n_samples = X.shape[0]

    for i in range(iterations):
        if i % 500 == 0:
            print("Iteration %4d => Loss: %.20f" % (i, loss(X, Y, w)))

        random_idx = np.random.randint(0, n_samples)
        X_sample = X[random_idx:random_idx+1]
        Y_sample = Y[random_idx:random_idx+1]

        w_gradient = gradient(X_sample, Y_sample, w) * lr
        w -= w_gradient
    return w


def train_mini_batch(X, Y, iterations, lr, batch_size=32):
    w = np.zeros((X.shape[1], 1))
    n_samples = X.shape[0]

    for i in range(iterations):
        if i % 500 == 0:
            print("Iteration %4d => Loss: %.20f" % (i, loss(X, Y, w)))

        indices = np.random.choice(n_samples, batch_size, replace=False)
        X_batch = X[indices]
        Y_batch = Y[indices]

        w_gradient = gradient(X_batch, Y_batch, w) * lr
        w -= w_gradient
    return w


def train_optimizer(X, Y_dict, iterations, lr, train_func, kwargs):
    start_time = time.time()
    weights = {}

    for house in HOUSES:
        print(f"  Training {house}...")
        weights[house] = train_func(X, Y_dict[house], iterations, lr, **kwargs)

    training_time = time.time() - start_time
    return weights, training_time


def calculate_average_loss(X, Y_dict, weights):
    return (sum(loss(X, Y_dict[house], weights[house]) for house in HOUSES)
            / len(HOUSES))


def train_all_optimizers(X, Y_dict, iterations, lr, optimizers):
    results = {}

    for name, train_func, kwargs in optimizers:
        print(f"{GREEN}Training with {name}...")
        weights, training_time = train_optimizer(X, Y_dict, iterations, lr,
                                                 train_func, kwargs)
        final_loss = calculate_average_loss(X, Y_dict, weights)

        results[name] = {
            "weights": weights,
            "time": training_time,
            "loss": final_loss
        }

    return results


def format_speedup(optimizer_name, batch_time, optimizer_time):
    if optimizer_name == "Batch GD":
        return ""

    speedup = batch_time / optimizer_time
    if speedup > 1:
        return f"{speedup:.1f}x faster!"
    else:
        return f"{1/speedup:.1f}x slower"


def display_performance_comparison(results):
    batch_time = results["Batch GD"]["time"]

    print(f"\n{GREEN}PERFORMANCE COMPARISON")
    print(f"{GREEN}=========================")

    for name, result in results.items():
        speedup_text = format_speedup(name, batch_time, result["time"])
        print(f"{name:<12} Final Loss: {result['loss']:.6f}, "
              f"Training Time: {result['time']:.1f}s  {speedup_text}")

    fastest_optimizer = min(results.items(), key=lambda x: x[1]["time"])[0]
    best_loss_optimizer = min(results.items(), key=lambda x: x[1]["loss"])[0]

    print(f"\n{GREEN}RECOMMENDATION: {fastest_optimizer} is fastest, "
          f"{best_loss_optimizer} has best loss!\n")


def save_all_weights_bonus(weights_batch, weights_sgd, weights_mini):
    weights_dir = create_weights_directory()
    save_weights(weights_batch, f"{weights_dir}/weights_batch.txt", "baseline")
    save_weights(weights_sgd, f"{weights_dir}/weights_sgd.txt", "stochastic")
    save_weights(weights_mini, f"{weights_dir}/weights_mini_batch.txt",
                 "mini-batch")

    print(f"\n{GREEN}All weights saved in {weights_dir}/:")
    print(f"{GREEN}You can now use --bonus with logreg_predict.py "
          f"to compare predictions!")


def train_all_bonus(X, Y_dict, iterations, lr):
    print(f"\n{GREEN}BONUS MODE: Training with multiple optimizers!")
    print(f"{GREEN}Comparing Batch GD, SGD, and Mini-batch GD\n")

    optimizers = [
        ("Batch GD", train, {}),
        ("SGD", train_sgd, {}),
        ("Mini-batch", train_mini_batch, {"batch_size": 32})
    ]

    results = train_all_optimizers(X, Y_dict, iterations, lr, optimizers)
    display_performance_comparison(results)
    save_all_weights_bonus(
        results["Batch GD"]["weights"],
        results["SGD"]["weights"],
        results["Mini-batch"]["weights"]
    )
