import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("auto-mpg.csv")

df['horsepower'] = df['horsepower'].replace('?', np.nan)
df['horsepower'] = pd.to_numeric(df['horsepower'])
df['horsepower'].fillna(df['horsepower'].mean(), inplace=True)

X = df[['horsepower']]
y = df['mpg']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

degrees = [2, 3, 4]
train_errors = []
test_errors = []
models = {}

for d in degrees:
    poly = PolynomialFeatures(degree=d)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)

    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)

    train_errors.append(mse_train)
    test_errors.append(mse_test)

    models[d] = (model, poly)

    print(f"\nDegree {d} Polynomial Regression")
    print("MSE:", mse_test)
    print("RMSE:", np.sqrt(mse_test))
    print("R² Score:", r2_score(y_test, y_test_pred))

plt.figure()
plt.plot(degrees, train_errors, marker='o', label='Training Error')
plt.plot(degrees, test_errors, marker='o', label='Testing Error')
plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error")
plt.title("Training vs Testing Error Comparison")
plt.legend()
plt.show()

poly = PolynomialFeatures(degree=4)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_poly, y_train)
ridge_pred = ridge.predict(X_test_poly)

print("\nRidge Regression (Degree 4)")
print("MSE:", mean_squared_error(y_test, ridge_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, ridge_pred)))
print("R² Score:", r2_score(y_test, ridge_pred))

X_range = np.linspace(X_scaled.min(), X_scaled.max(), 100).reshape(-1, 1)

plt.figure()
plt.scatter(X_scaled, y, label='Actual Data')

for d in degrees:
    model, poly = models[d]
    X_range_poly = poly.transform(X_range)
    y_range_pred = model.predict(X_range_poly)
    plt.plot(X_range, y_range_pred, label=f'Degree {d}')

plt.xlabel("Horsepower (Scaled)")
plt.ylabel("MPG")
plt.title("Polynomial Curve Fitting")
plt.legend()
plt.show()

plt.figure()
plt.plot(degrees, train_errors, marker='o', label='Training Error')
plt.plot(degrees, test_errors, marker='o', label='Testing Error')
plt.xlabel("Polynomial Degree")
plt.ylabel("Error")
plt.title("Underfitting vs Overfitting")
plt.legend()
plt.show()
