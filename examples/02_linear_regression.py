import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.gradient_descent import gradient_descent


if __name__ == "__main__":
    X = np.array([0, 0.2, 0.4, 0.8, 1.0])
    target = (5 + X).reshape(-1, 1)

    X = X.reshape(-1, 1)
    X = np.hstack([np.ones((X.shape[0], 1)), X])

    optimal_weights = gradient_descent(X, target, step_size=0.1)

    print("Optimal weights:\n", optimal_weights)
    print("Intercept:", optimal_weights[0, 0])
    print("Slope:", optimal_weights[1, 0])
    predictions = X @ optimal_weights

    plt.scatter(X[:, 1], target, color="green", label="Training data")
    plt.plot(X[:, 1], predictions, color="red", label="Learned regression line")

    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    output_path = Path(__file__).resolve().parents[1] / "images" / "linear-regression.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()