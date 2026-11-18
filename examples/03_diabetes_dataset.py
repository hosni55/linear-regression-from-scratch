from src.data_preparation import load_diabetes_data, add_bias_column
from src.gradient_descent import gradient_descent
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":
    X, y = load_diabetes_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = add_bias_column(X_train)
    X_test = add_bias_column(X_test)

    weights = gradient_descent(
        X_train, y_train,
        step_size=0.1,
        max_iterations=10000,
        random_state=42
    )

    print("Learned weights:\n", weights)
    print("Intercept:", weights[0, 0])
    print("Feature coefficients:\n", weights[1:, 0])

    predictions = X_test @ weights

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("Test MSE:", mse)
    print("Test R² Score:", r2)

    sk_model = LinearRegression()
    sk_model.fit(X_train[:, 1:], y_train)

    sk_predictions = sk_model.predict(X_test[:, 1:])

    print("\nScikit-learn results:")
    print("MSE:", mean_squared_error(y_test, sk_predictions))
    print("R² Score:", r2_score(y_test, sk_predictions))