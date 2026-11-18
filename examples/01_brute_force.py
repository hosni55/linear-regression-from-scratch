import numpy as np
import matplotlib.pyplot as plt


# Ground truth (X, Y) training dataset: N tuples (Xi, Yi)
X = [1, 2, 3, 4, 5, 6]
Y = np.array([1, 1.5, 2, 2.5, 2.7, 3], dtype=np.float32) * 3

# Candidate parameters (possible lines)
M = [-1.5, -1, 0.5, 1, 2]
C = [-1, 0.5, 3, 5, 6, 7]


# Compute MSE for each line (y = mx + c)
def compute_cost_per_line(m, c):
    cost = 0

    for x, y_gt in zip(X, Y):
        y_pd = m * x + c
        err = y_gt - y_pd
        squared_err = err ** 2
        cost += squared_err

    return cost / len(Y)


if __name__ == "__main__":
    best_cost = float("inf")
    best_m, best_c = None, None

    # Find the best line
    for m in M:
        for c in C:
            this_line_cost = compute_cost_per_line(m, c)

            if this_line_cost < best_cost:
                best_cost = this_line_cost
                best_m, best_c = m, c

    print("Best m:", best_m)
    print("Best c:", best_c)
    print("Best MSE:", best_cost)

    # Plot all candidate lines
    x1 = np.array([min(X), max(X)])

    for m in M:
        for c in C:
            y1 = m * x1 + c
            plt.plot(x1, y1, color="gray", alpha=0.15)

    # Plot best line
    y1 = best_m * x1 + best_c
    plt.plot(
        x1, y1, "-r", linewidth=2,
        label=f"Best line: y = {best_m}x + {best_c}"
    )

    # Plot ground truth
    plt.scatter(X, Y, color="green", label="Ground truth")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.show()
