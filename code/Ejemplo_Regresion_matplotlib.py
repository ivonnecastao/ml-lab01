import numpy as np
import matplotlib.pyplot as plt

# Generate random data
X = np.random.rand(100, 1)
# print(f"X = {X[:5, :]}")

y = 2 * X + 3 + np.random.rand(100, 1)
# print(f"y = {y[:5, :]}")

X_b = np.hstack([np.ones((X.shape[0], 1)), X])
print(f"X_b = {X_b[:5, :]}")

Theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
print(f"Theta_best = {Theta_best}")

y_predict = X_b @ Theta_best
print(f"y_predict = {y_predict}")

# Sin los 1 apliados
'''
Para que la regresion se ajuste a los datos se ha de usar
byas, offset o sesgo; ese array de 1s lo que hace es que
"mueve" en la recta de y para que se ajuste a los datos

Theta_best =  np.linalg.inv(X.T @ X) @ X.T @ y
y_predict = X @ Theta_best
'''

# Plot data
plt.scatter(X, y, s=10, label="Datos")
plt.plot(X, y_predict, color="red", label="Ajuste lineal")

plt.xlabel("Variable independiente (X)")
plt.ylabel("Variable dependiente(y)")
plt.title("Regresion lineal")
plt.legend()
plt.show()
