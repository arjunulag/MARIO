import numpy as np
import matplotlib.pyplot as plt
from functions import Parameter_init
from mse_vector import mse_vector
from gradient_descent import gradient_descent


def generate_data(n_samples=200, n_features=1, noise=0.3, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features)
    true_W = np.random.randn(n_features, 1) * 2
    true_b = np.array([1.5])
    y = X @ true_W + true_b + np.random.randn(n_samples, 1) * noise
    return X, y, true_W, true_b


def build_model(n_features):
    config = [
        {"type": "linear", "in": n_features, "out": 1},
    ]
    return Parameter_init(config)


def train(X, y, lr=0.01, epochs=500):
    n_features = X.shape[1]
    model = build_model(n_features)
    losses = gradient_descent(model, X, y, lr=lr, epochs=epochs)
    return model, losses


def predict(model, X):
    return model.forward(X)


def evaluate(model, X, y):
    y_hat = predict(model, X)
    return mse_vector(y_hat, y)


def plot_results(model, X, y, losses):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(losses)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True, alpha=0.3)

    if X.shape[1] == 1:
        sort_idx = X[:, 0].argsort()
        X_sorted = X[sort_idx]
        y_sorted = y[sort_idx]
        y_hat_sorted = predict(model, X_sorted)

        axes[1].scatter(X_sorted, y_sorted, alpha=0.4, s=15, label="Data")
        axes[1].plot(X_sorted, y_hat_sorted, color="red", linewidth=2, label="Fit")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        axes[1].legend()
    else:
        y_hat = predict(model, X)
        axes[1].scatter(y, y_hat, alpha=0.4, s=15)
        axes[1].plot(
            [y.min(), y.max()], [y.min(), y.max()],
            color="red", linewidth=2, linestyle="--", label="Perfect"
        )
        axes[1].set_xlabel("True y")
        axes[1].set_ylabel("Predicted y")
        axes[1].legend()

    axes[1].set_title("Predictions")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    X, y, true_W, true_b = generate_data(n_samples=200, n_features=1, noise=0.3)

    print(f"True W: {true_W.flatten()}")
    print(f"True b: {true_b.flatten()}")

    model, losses = train(X, y, lr=0.05, epochs=300)

    learned_W = model.layers[0]["W"]
    learned_b = model.layers[0]["b"]
    print(f"Learned W: {learned_W.flatten()}")
    print(f"Learned b: {learned_b.flatten()}")

    final_loss = evaluate(model, X, y)
    print(f"Final MSE: {final_loss:.6f}")

    plot_results(model, X, y, losses)
