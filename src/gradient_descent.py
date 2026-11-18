
import numpy as np
from numpy.linalg import norm

from src.cost_function import cost_value, f_derivative


def gradient_descent(X, target, step_size=0.01,
                     precision=0.0001, max_iterations=1000,
                     random_state=None):

    features = X.shape[1]

    rng = np.random.default_rng(random_state)
    cur_weights = rng.random((features, 1))

    last_weights = cur_weights + 100 * precision

    iterations = 0

    while norm(cur_weights - last_weights) > precision and iterations < max_iterations:
        last_weights = cur_weights.copy()

        gradient = f_derivative(X, target, cur_weights)
        cur_weights -= step_size * gradient

        iterations += 1

    print(f"Total iterations: {iterations}")
    print(f"Optimal cost: {cost_value(X, target, cur_weights)}")

    return cur_weights
