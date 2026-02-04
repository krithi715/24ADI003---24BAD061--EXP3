import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("StudentsPerformance.csv")

le = LabelEncoder()
df['parental level of education'] = le.fit_transform(df['parental level of education'])
df['test preparation course'] = le.fit_transform(df['test preparation course'])

df['final_exam_score'] = (
    df['math score'] + df['reading score'] + df['writing score']
) / 3

np.random.seed(42)
df['study_hours_per_day'] = np.random.normal(3, 1, len(df)).clip(1, 8)
df['attendance_percentage'] = np.random.normal(85, 5, len(df)).clip(60, 100)
df['sleep_hours'] = np.random.normal(7, 1, len(df)).clip(4, 9)

df.fillna(df.mean(numeric_only=True), inplace=True)

X = df[
    [
        'study_hours_per_day',
        'attendance_percentage',
        'parental level of education',
        'test preparation course',
        'sleep_hours'
    ]
]
y = df['final_exam_score']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("RMSE:", rmse)
print("R² Score:", r2)

coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
print("\nRegression Coefficients:\n", coefficients)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
print("\nRidge R²:", r2_score(y_test, ridge_pred))

lasso = Lasso(alpha=0.05)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)
print("Lasso R²:", r2_score(y_test, lasso_pred))

plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Exam Scores")
plt.ylabel("Predicted Exam Scores")
plt.title("Predicted vs Actual Exam Scores")
plt.show()

plt.figure()
plt.bar(coefficients['Feature'], coefficients['Coefficient'])
plt.xticks(rotation=45)
plt.title("Regression Coefficients Comparison")
plt.show()

residuals = y_test - y_pred
plt.figure()
sns.histplot(residuals, kde=True)
plt.title("Residual Distribution")
plt.xlabel("Residuals")
plt.show()
