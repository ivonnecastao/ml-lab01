import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split

# Datos de ejemplo
np.random.seed(0)
N = 100
x = 5 * np.random.rand(N, 1)
y = 20 + 3 * x + 4 * x**2 + 5 * np.random.randn(N, 1)
# Transformacion polinomial de caracteristicas
X_poly = np.hstack((x, x**2))
# Stack a column of ones
# X_poly = np.hstack((np.ones((x.shape[0], 1)), x))
X_poly = np.hstack((np.ones((x.shape[0], 1)), X_poly))

# Calculo de los coeficientes usando gradiente descendente
# Inicializacion de los parametros
theta = np.random.randn(X_poly.shape[1], 1)
# Tasa de aprendizaje
alpha = 0.0001
# Numero de iteraciones
n_iterations = 300

X_poly_train, X_poly_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2)

X_poly_test, X_poly_val = train_test_split(X_poly_test, test_size=0.5)

y_test, y_val = train_test_split(y_test, test_size=0.5)

print(f"Dim Vector X_poly_train = {X_poly_train.shape}")
print(f"Dim Vector X_poly_test = {X_poly_test.shape}")
print(f"Dim Vector X_poly_val = {X_poly_val.shape}")
print(f"Dim Vector y_train = {y_train.shape}")
print(f"Dim Vector y_test = {y_test.shape}")
print(f"Dim Vector y_val = {y_val.shape}")
print(f"Dim Vector theta = {theta.shape}")

# Inicializacion del vector de costos
Loss_train = np.zeros(n_iterations)
Loss_test = np.zeros(n_iterations)
Loss_val = np.zeros(n_iterations)


#TRAIN

plt.figure()
# Gradiente descendente
for i in range(n_iterations):
    # Calculo de las predicciones
    y_pred = X_poly_train @ theta
    # Calculo del error
    error = y_pred - y_train
    # Calculo del gradiente
    grad = X_poly_train.T @ error
    # Actualizacion de los parametros
    theta = theta - alpha * grad
    # Calculo del costo
    Loss_train[i] = np.mean(error**2)
    print(f"Step {i+1}: Loss_train = {Loss_train[i]:.4f}")
    plt.scatter(np.array([X_poly_train[:,1]]), y_train, label='Datos originales') 
    plt.scatter(np.array([X_poly_train[:,1]]), y_pred, color='red', label='Predicciones') 
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Iteración: {i+1}, Costo: {Loss_train[i]:.4f}')
    plt.xlim(0, 5)
    plt.ylim(0, 150)
    # plt.show()
    plt.pause(0.1)
    plt.clf()


print(f"Dim Vector y_pred train = {y_pred.shape}")
# Visualizacion de la evolucion del costo
plt.figure()
plt.plot(Loss_train)
plt.xlabel('Step')
plt.ylabel('Loss_train')
plt.xlim(0, n_iterations)
plt.ylim(0, 300)
plt.show()



# VALIDATION

plt.figure()
# Gradiente descendente
for i in range(n_iterations):
    # Calculo de las predicciones
    y_pred = X_poly_val @ theta
    # Calculo del error
    error = y_pred - y_val
    # Calculo del gradiente
    grad = X_poly_val.T @ error
    # Actualizacion de los parametros
    theta = theta - alpha * grad
    # Calculo del costo
    Loss_val[i] = np.mean(error**2)
    print(f"Step {i+1}: Loss_val = {Loss_val[i]:.4f}")
    plt.scatter(np.array([X_poly_val[:,1]]), y_val, label='Datos originales') 
    plt.scatter(np.array([X_poly_val[:,1]]), y_pred, color='red', label='Predicciones') 
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Iteración: {i+1}, Costo: {Loss_val[i]:.4f}')
    plt.xlim(0, 5)
    plt.ylim(0, 150)
    # plt.show()
    plt.pause(0.1)
    plt.clf()

print(f"Dim Vector y_pred val= {y_pred.shape}")
# Visualizacion de la evolucion del costo
plt.figure()
plt.plot(Loss_val)
plt.xlabel('Step')
plt.ylabel('Loss_val')
plt.xlim(0, n_iterations)
plt.ylim(0, 300)
plt.show()


# TEST

plt.figure()
# Gradiente descendente
for i in range(n_iterations):
    # Calculo de las predicciones
    y_pred = X_poly_test @ theta
    # Calculo del error
    error = y_pred - y_test
    # Calculo del gradiente
    grad = X_poly_test.T @ error
    # Actualizacion de los parametros
    theta = theta - alpha * grad
    # Calculo del costo
    Loss_test[i] = np.mean(error**2)
    print(f"Step {i+1}: Loss_test = {Loss_test[i]:.4f}")
    plt.scatter(np.array([X_poly_test[:,1]]), y_test, label='Datos originales') 
    plt.scatter(np.array([X_poly_test[:,1]]), y_pred, color='red', label='Predicciones') 
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Iteración: {i+1}, Costo: {Loss_test[i]:.4f}')
    plt.xlim(0, 5)
    plt.ylim(0, 150)
    # plt.show()
    plt.pause(0.1)
    plt.clf()

print(f"Dim Vector y_pred test= {y_pred.shape}")
# Visualizacion de la evolucion del costo
plt.figure()
plt.plot(Loss_test)
plt.xlabel('Step')
plt.ylabel('Loss_test')
plt.xlim(0, n_iterations)
plt.ylim(0, 300)
plt.show()















# # Get current time
# start_time = time.time()
# # Ajuste del modelo
# theta = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y
# # Get current time
# stop_time = time.time()
# # Print elapsed time
# print(f"Elapsed time: {stop_time - start_time:.4f} seconds")

# # Prediccion
# y_pred = X_poly @ theta

# # Calculo del error RMSE
# rmse = np.sqrt(np.mean((y - y_pred)**2))
# print(f'RMSE: {rmse:.4f}')

# # Calculo del error MAE
# mae = np.mean(np.abs(y - y_pred))
# print(f'MAE: {mae:.4f}')

# # Visualizacion
# plt.scatter(x, y, label='Datos originales') 
# plt.scatter(x, y_pred, color='red', label='Predicciones') 
# plt.legend()
# plt.show()

# # Create a new figure


# # Crear una nueva grafica de y vs y_pred
# plt.figure()
# plt.scatter(y, y_pred)
# plt.xlabel('y')
# plt.ylabel('y_pred')
# plt.show()

# # Crear una nueva grafica de y vs y_pred
# plt.figure()
# plt.scatter(y, (y_pred-y))
# plt.xlabel('y')
# plt.ylabel('y_pred - y')
# plt.show()