import numpy as np
import matplotlib.pyplot as plt
import time

# Función de costo
def costo(x):
    return x**2

# Derivada de la función de costo
def gradiente(x):
    return 2*x

# Datos de ejemplo
np.random.seed(0)
N= 100000000
x = 5 * np.random.rand(N, 1)
y = 20 + 3 * x + 4 * x**2 + 5 * np.random.randn(N, 1)

# Parámetro inicial
x = 3

# Tasa de aprendizaje
tasa_aprendizaje = 0.1  #Contrae el gradiente

""""
# Transformacion polinomial de caracteristicas
X_poly = np.hstack((x, x**2))
# Stack a column of ones
# X_poly = np.hstack((np.ones((x.shape[0], 1)), x))
X_poly = np.hstack((np.ones((x.shape[0], 1)), X_poly))
"""

# Get current time
start_time = time.time();

# Ajuste del modelo
#theta = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y

for i in range(15):  #Ciclo de entrenamiento, cuantas veces actualizazo a w
    
    cost = costo(x)  # Cuanto vale la perdida con los valores actuales de w
    grad = gradiente(x)  # cuanto vale mi gradiente para w
    x = x - tasa_aprendizaje * grad  # actualizo el w
    print(f"x = {x:.4f}", f"costo = {cost:.4f}")

stop_time = time.time()

print(f"Elapsed time: {stop_time - start_time:.4f}")


# Prediccion
#y_pred = X_poly @ theta

# Calculo del error RMSE
rmse = np.sqrt(np.mean((y - y_pred)**2))
print(f'RMSE: {rmse:.4f}')

# Calculo del error MAE
mae = np.mean(np.abs(y - y_pred))
print(f'MAE: {mae:.4f}')

# Visualizacion
plt.scatter(x, y, label='Datos originales') 
plt.scatter(x, y_pred, color='red', label='Predicciones') 
plt.legend()
plt.show()

# Create a new figure


# Crear una nueva grafica de y vs y_pred
plt.figure()
plt.scatter(y, y_pred)
plt.xlabel('y')
plt.ylabel('y_pred')
plt.show()

# Crear una nueva grafica de y vs y_pred
plt.figure()
plt.scatter(y, (y_pred-y))
plt.xlabel('y')
plt.ylabel('y_pred - y')
plt.show()