import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('./winequality-white.csv', sep=";")

y = df["quality"].values.reshape(-1, 1)

X_volatile = df['volatile acidity'].values.reshape(-1, 1)
X_tsd = df['total sulfur dioxide'].values.reshape(-1, 1)
X_fsd = df['free sulfur dioxide'].values.reshape(-1, 1)
X_acidity = df['fixed acidity'].values.reshape(-1, 1)
X_sugar = df['residual sugar'].values.reshape(-1, 1)
X_sulphates = df['sulphates'].values.reshape(-1, 1)
X_chlorides = df['chlorides'].values.reshape(-1, 1)
X_acid = df['citric acid'].values.reshape(-1, 1)
X_density = df['density'].values.reshape(-1, 1)
X_alcohol = df['alcohol'].values.reshape(-1, 1)
X_pH = df['pH'].values.reshape(-1, 1)

X_tags = [
    X_volatile,
    X_tsd,
    X_fsd,
    X_acidity,
    X_sugar,
    X_sulphates,
    X_chlorides,
    X_acid,
    X_density,
    X_alcohol,
]

X = X_pH


for tag in X_tags:
    np.hstack([tag, X])

print(X, end='\n')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(f"X_train.shape = {X_train.shape}")
print(f"X_test.shape = {X_test.shape}")

model = LinearRegression()
model.fit(X_train, y_train)

y_predict = model.predict(X_test)

mse = mean_squared_error(y_test, y_predict)

print(f"Coeficiente: {model.coef_}")
print(f"Intercepto: {model.intercept_}")
print(f"Error cuadratico medio: {mse}")

# Plot data
plt.scatter(X, y, s=10, label="Datos")
plt.plot(X_test, y_predict, color="red", label="Ajuste lineal")

plt.xlabel("Variable independiente (X)")
plt.ylabel("Variable dependiente(y)")
plt.title("Regresion lineal")
plt.legend()
plt.show()
