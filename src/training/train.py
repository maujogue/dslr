import numpy as np
from tools.constants import GREEN


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


def train_all(X, Y_dict, iterations, lr):
    w_gryffindor = train(X, Y_dict["Gryffindor"], iterations, lr)
    w_hufflepuff = train(X, Y_dict["Hufflepuff"], iterations, lr)
    w_ravenclaw = train(X, Y_dict["Ravenclaw"], iterations, lr)
    w_slytherin = train(X, Y_dict["Slytherin"], iterations, lr)

    with open("weights.txt", "w") as f:
        f.write(f"Gryffindor:{w_gryffindor.flatten().tolist()}\n")
        f.write(f"Hufflepuff:{w_hufflepuff.flatten().tolist()}\n")
        f.write(f"Ravenclaw:{w_ravenclaw.flatten().tolist()}\n")
        f.write(f"Slytherin:{w_slytherin.flatten().tolist()}")
    print(f"\n{GREEN}Weights saved to weights.txt")
    print(f"{GREEN}You can now use the model to make predictions")
