import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Generate random data
X = np.random.rand(100, 1)
# print(f"X = {X[:5, :]}")

y = 2 * X + 3 + np.random.rand(100, 1)
# print(f"y = {y[:5, :]}")

'''
El objetivo de tener un conjunto de datos que no haya sido
usado en el entrenamiento
'''
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
