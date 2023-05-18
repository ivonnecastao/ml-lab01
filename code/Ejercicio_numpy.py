import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# PREPARACIÓN DE DATOS

#CARGUE
print("DATOS INICIALES:\n")

df = pd.read_csv('data\downloaded\winequality-white.csv', sep=";")
print(f"Dataframe:\n {df}")
print(f"Dim Dataframe = {df.shape}")  # me dice las dimensiones del dataframe

#LIMPIEZA

df = df.dropna() # Elimino datos incompletos. Limpio datos
print(f"Dim Dataframe limpio = {df.shape}")

# TRANSFORMACIÓN

mnp = df.to_numpy()  # me convierte el df en una matriz de numpy (mnp)
#print(mnp)
#print(mnp.shape)  # me dice las dimensiones de la mnp

i = 11 # columna que queremos obtener
y = np.array([mnp[:,i]]) # Extrae la columna i y la guarda en y
#print(y)
#print(y.shape)

y = np.transpose(y) # Transpongo y porque quedó como fila y necesito columna
#print(y)
#print(y.shape)

mnp = np.delete(mnp, 11, axis=1) # Elimino la columna que extraje en y
#print(mnp)
#print(mnp.shape)

vones= np.ones((y.shape[0],1)) # Creo el vector de 1's BIAS
#print(vones)
#print(vones.shape)

mnp = np.hstack([vones, mnp]) # Integro el BIAS al mnp
#print(mnp)
#print(mnp.shape)

#DATOS PREPARADOS
print(f"Matriz mnp:\n {mnp}")
print(f"Dim Matriz mnp = {mnp.shape}")
print(f"Vector y:\n {y}")
print(f"Dim Vector y = {y.shape}")


# MODELO

Theta_best = np.linalg.inv(mnp.T @ mnp) @ (mnp.T @ y)  # y= Theta 0 + x Theta 1 
print(f"Theta_best = {Theta_best}")
print(f"Dim Theta_best = {Theta_best.shape}")


y_pred = mnp @ Theta_best  # resultado, predicción
print(f"Vector y_pred = {y_pred}")
print(f"Dim Vector y_pred = {y_pred.shape}")


# EVALUACIÓN DEL MODELO

print("\nEVALUACIÓN DEL MODELO DE REGRESIÓN LINEAL CON NUMPY:\n")

# Cálculo del error MSE
mse = mean_squared_error(y, y_pred)
# Calculo del error RMSE
rmse = np.sqrt(np.mean((y - y_pred)**2))  # y_test - y_pred es el residual
# Calculo del error MAE
mae = np.mean(np.abs(y - y_pred))

print(f"MSE: {mse}")
print(f'RMSE: {rmse:.4f}')
print(f'MAE: {mae:.4f}')


# VISUALIZACIÓN

# Se visualiza que tan bueno es el modelo
plt.scatter(y, y_pred) # Crear una nueva grafica de y_test vs y_pred
#plt.plot(y_test, y_pred, color='red') 
plt.xlabel('y')
plt.ylabel('y_pred')
plt.title("Regresion lineal con sklearn: Qué tan bueno es el Modelo?")
plt.savefig('plots\ Numpy y vs y_pred.png')
plt.show()

# Se visualiza el error de los datos
plt.scatter(y, (y_pred - y))  # y_test - y_pred es el residual
#plt.plot(y_test, y_pred, color='red')
plt.xlabel('y')
plt.ylabel('y_pred')
plt.title("Regresion lineal con sklearn: Cuál es el error del Modelo?")
plt.savefig('plots\ Numpy y vs y_pred-y_test.png')
plt.show()