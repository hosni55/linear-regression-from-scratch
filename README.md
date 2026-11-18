

# Linear Regression from Scratch

A Python implementation of **Linear Regression from scratch using NumPy**, exploring the mathematical foundations and optimization process behind one of the fundamental machine learning algorithms.

The project focuses on understanding how linear regression works internally, rather than relying entirely on ready-made machine learning libraries.

## Project Objectives

- Understand the mathematics behind linear regression.
- Implement the cost function and its gradient using NumPy.
- Implement gradient descent from scratch.
- Explore brute-force parameter search.
- Apply the model to synthetic and real-world datasets.
- Evaluate the implementation against Scikit-learn.



## Mathematical Foundations

For a dataset containing \(N\) training examples and \(D\) features, linear regression predicts the target using:

\[
\hat{y} = w_0 + w_1x_1 + \cdots + w_Dx_D
\]

Where:
- \(w_0\) is the intercept.
- \(w_1, \ldots, w_D\) are the feature coefficients.
- \(\hat{y}\) is the predicted value.

### Cost Function

The model uses the Mean Squared Error-based cost function:

\[
J(w) = \frac{1}{2N}\sum_{i=1}^{N}(\hat{y}^{(i)}-y^{(i)})^2
\]

The objective is to find the weights that minimize this cost.

### Gradient Descent

Gradient descent iteratively updates the weights in the direction that reduces the cost:

\[
w_j := w_j - \alpha \frac{\partial J(w)}{\partial w_j}
\]

Where:
- \(\alpha\) is the learning rate.
- \(\frac{\partial J(w)}{\partial w_j}\) is the partial derivative of the cost with respect to weight \(w_j\).

The implementation uses NumPy matrix operations to calculate predictions, the cost, and gradients efficiently.



## Project Structure

```text
linear-regression-from-scratch/
│
├── data/
│
├── src/
│   ├── __init__.py
│   ├── data_preparation.py
│   ├── cost_function.py
│   ├── gradient_descent.py
│   └── visualization.py
│
├── examples/
│   ├── 01_brute_force.py
│   ├── 02_linear_regression.py
│   └── 03_diabetes_dataset.py
│
├── main.py
└── requirements.txt
```

### Components

- **`data_preparation.py`**: Loads datasets, prepares features, and adds the intercept column.
- **`cost_function.py`**: Implements the cost function and its gradient.
- **`gradient_descent.py`**: Implements the optimization algorithm.
- **`visualization.py`**: Plots training data and regression predictions.
- **`examples/`**: Contains experiments using brute-force search, synthetic data, and the diabetes dataset.
- **`main.py`**: Runs the basic project pipeline.
- **`requirements.txt`**: Lists the project dependencies.



## Experiments

### 1. Brute-Force Search

Explores a simple parameter search approach by evaluating candidate slopes and intercepts and selecting the line with the lowest Mean Squared Error.

### 2. Linear Regression with Gradient Descent

Implements linear regression from scratch using NumPy.

- Synthetic dataset.
- Vectorized cost and gradient calculations.
- Gradient descent optimization.
- Visualization of the learned regression line.

### 3. Diabetes Dataset

Applies the implementation to Scikit-learn's diabetes dataset.

The experiment includes:
- Training and testing split (80/20).
- Feature standardization using `StandardScaler`.
- Training with the custom gradient descent implementation.
- Evaluation using MSE and R².
- Comparison with Scikit-learn's `LinearRegression`.

#### Results

| Metric | From Scratch | Scikit-learn |
|---|---:|---:|
| Test MSE | 2900.1299 | 2900.1936 |
| Test R² | 0.452615 | 0.452603 |

The results show that the custom implementation produces predictions very close to Scikit-learn's implementation.



## Installation

### 1. Clone the repository

```bash
git clone https://github.com/hosni55/linear-regression-from-scratch.git
cd linear-regression-from-scratch
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

Run the examples from the project root directory:

```bash
# Brute-force parameter search
python -m examples.01_brute_force

# Linear regression on synthetic data
python -m examples.02_linear_regression

# Linear regression on the diabetes dataset
python -m examples.03_diabetes_dataset
```

Run the main pipeline:

```bash
python main.py
```

## Technologies

- Python
- NumPy
- Matplotlib
- Scikit-learn

