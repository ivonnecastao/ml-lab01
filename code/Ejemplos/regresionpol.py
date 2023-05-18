import numpy as np
import matplotlib.pyplot as plt

#Datos de ejemplo
np.random.seed(0)
x = 5 * np.random.rand(100, 1)
y = 20 + 3*x + 4* x**2 + 3 * np.random.randn(100, 1)

#Transformacion polinomial de caracteristicas
X_poly = np.hstack((x, x**2))

#Stack a column of ones
#X_poly = np.hstack((np.ones((x.shape[0], 1)), x))
X_poly = np.hstack((np.ones((x.shape[0], 1)), X_poly))

#Ajuste del modelo
theta = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y

#prediccion
y_pred = X_poly @ theta

#Calculo del error RMSE
rmse = np.sqrt(np.mean((y-y_pred)**2))#y-y_pred es el residual
print(f'RMSE: {rmse:.4f}')

#Calculo del error MAE
mae = np.mean(np.abs(y-y_pred))
print(f'MAE: {mae:.4f}')

#Visualizacion
plt.scatter(x, y, label='Datos originales')
plt.scatter(x, y_pred, color='red', label='predicciones')
plt.legend()
plt.show()

#Se visualiza que tan bueno es el modelo
#Crear una nueva grafica de y vs y_pred
plt.scatter(y, y_pred)
plt.xlabel('y')
plt.ylabel('y_pred')
plt.show()

#Se visualiza el error de los datos
#Crear una nueva grafica de y vs y_pred
plt.scatter(y, (y_pred - y))
plt.xlabel('y')
plt.ylabel('y_pred')
plt.show()

##Histograma básico de los datos
#Hacer estudio de los datos
#Analisis estadistico de la dispersión

#binding afinity

#Como se extraen esos datos
