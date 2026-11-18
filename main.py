from src.data_preparation import load_data, add_bias_column
from src.gradient_descent import gradient_descent
import matplotlib.pyplot as plt
from src.visualization import plot_regression

X, y = load_data()
X = add_bias_column(X)

weights = gradient_descent(X, y, step_size=0.1)

print("Learned weights:\n", weights)
print("Intercept:", weights[0, 0])
print("Slope:", weights[1, 0])

predictions = X @ weights
plot_regression(X, y, predictions)