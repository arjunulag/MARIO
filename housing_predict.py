import numpy as np
import matplotlib.pyplot as plt
from functions import Parameter_init
from adam import adam
from linear_regression import predict, evaluate


def load_housing_data(path="Housing.csv"):
    data = np.genfromtxt(path, delimiter=",", skip_header=1, usecols=(0, 1, 2, 3))
    y = data[:, 0:1]   # price
    X = data[:, 1:4]    # area, bedrooms, bathrooms
    return X, y


def normalize(arr):
    mu = arr.mean(axis=0)
    sigma = arr.std(axis=0)
    sigma[sigma == 0] = 1.0
    return (arr - mu) / sigma, mu, sigma


if __name__ == "__main__":
    X_raw, y_raw = load_housing_data()
    print(f"Loaded {X_raw.shape[0]} samples, {X_raw.shape[1]} features")

    X, x_mu, x_sig = normalize(X_raw)
    y, y_mu, y_sig = normalize(y_raw)

    config = [
        {"type": "linear", "in": 3, "out": 32},
        {"type": "relu"},
        {"type": "linear", "in": 32, "out": 1, "activation_hint": "relu"},
    ]
    model = Parameter_init(config)
    print(model)

    losses = adam(model, X, y, lr=0.001, epochs=10000)

    final_loss = evaluate(model, X, y)
    print(f"Final MSE (normalized): {final_loss:.6f}")

    y_hat_norm = predict(model, X)
    y_hat = y_hat_norm * y_sig + y_mu

    mse_orig = np.mean((y_hat - y_raw) ** 2)
    rmse_orig = np.sqrt(mse_orig)
    print(f"RMSE (original scale): {rmse_orig:,.0f}")

    # --- Training loss curve ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training Loss — Housing Price (Adam)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("training_loss.png", dpi=150)
    plt.close(fig)
    print("Saved training_loss.png")

    # --- Predicted vs Actual (reuses style from linear_regression.plot_results) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_raw, y_hat, alpha=0.4, s=15)
    ax.plot(
        [y_raw.min(), y_raw.max()], [y_raw.min(), y_raw.max()],
        color="red", linewidth=2, linestyle="--", label="Perfect",
    )
    ax.set_xlabel("True Price")
    ax.set_ylabel("Predicted Price")
    ax.set_title("Predictions vs Actual")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("predictions.png", dpi=150)
    plt.close(fig)
    print("Saved predictions.png")

    # --- Per-feature scatter plots ---
    feature_names = ["Area", "Bedrooms", "Bathrooms"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, (ax, name) in enumerate(zip(axes, feature_names)):
        ax.scatter(X_raw[:, i], y_raw[:, 0], alpha=0.3, s=10, label="Actual")
        ax.scatter(X_raw[:, i], y_hat[:, 0], alpha=0.3, s=10, label="Predicted")
        ax.set_xlabel(name)
        ax.set_ylabel("Price")
        ax.set_title(f"Price vs {name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("feature_scatter.png", dpi=150)
    plt.close(fig)
    print("Saved feature_scatter.png")
