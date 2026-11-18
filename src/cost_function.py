import numpy as np

# For the given weights, return the cost value
# Measures how wrong the model is; the goal is to minimize this cost
def cost_value(X, target, weights):
    rows = X.shape[0]  # N rows

    predictions = np.dot(X, weights)  # (N, 1)
    error = predictions - target  # (N, 1)

    # error.T ==> (1,N), (1,N) @ (N,1) = (1,1) scalar == cost
    cost = error.T.dot(error) / (2 * rows)

    return cost.item()


# Given X, target and weights, return the (D+1, 1) array
# of partial derivatives of the cost function on each weight
def f_derivative(X, target, weights):
    examples = X.shape[0]  # N

    predictions = np.dot(X, weights)  # (N, 1)
    error = predictions - target  # (N, 1)

    # For the jth weight, multiply xj by the error for all examples and sum them
    # This is equivalent to the following matrix multiplication
    gradient = X.T @ error / examples  # (D+1, 1)

    # gradient = [df/dw0, ..., df/dwD].T
    return gradient



if __name__ == "__main__":
    X = np.array([0, 0.2, 0.4, 0.8, 1.0])
    target = (5 + X).reshape(-1, 1)

    X = X.reshape(-1, 1)
    X = np.hstack([np.ones((X.shape[0], 1)), X])

    weights = np.array([[1.0], [1.0]])

    print("Cost:", cost_value(X, target, weights))
    print("Gradient:\n", f_derivative(X, target, weights))
