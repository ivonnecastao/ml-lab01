import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data\downloaded\winequality-white.csv', sep=";")
print(df)
print(df.shape) # me dice las dimensiones del dataframe

df = df.dropna()
print(df)

mnp = df.to_numpy()  # me convierte el df en una matriz de numpy (mnp)
print(mnp)
print(mnp.shape)  # me dice las dimensiones de la mnp

i = 11 # columna que queremos obtener
y = np.array([mnp[:,i]]) # Extrae la columna i y la guarda en y
print(y)
print(y.shape)

y = np.transpose(y)
print(y)
print(y.shape)

mnp = np.delete(mnp, 11, axis=1) # Elimino la columna que extraje en y
print(mnp)
print(mnp.shape)

vones= np.ones((y.shape[0],1)) # Creo el vector de 1's BIAS
print(vones)
print(vones.shape)

mnp = np.hstack([vones, mnp])
print(mnp)
print(mnp.shape)

Theta_best = np.linalg.inv(mnp.T @ mnp) @ (mnp.T @ y)  #y= x Theta 1 + Theta 0
print(f"Theta_best = {Theta_best}")
print(Theta_best.shape)

y_pred = mnp @ Theta_best  # resultado, predicción
print(y_pred)
print(y_pred.shape)


"""
# plot data
plt.scatter(y, y_pred, label='Datos')
#plt.plot(y, y_pred, color='red', label='Ajuste lineal') 
plt.xlabel('Variables independiente (mnp)')
plt.ylabel('Variable dependiente (y)')
plt.legend()
plt.title('Regresión Lineal con Numpy')
plt.show()
plt.savefig('plots\y_vs_y_pred_numpy.png')
"""

# Calculo del error RMSE
rmse = np.sqrt(np.mean((y - y_pred)**2))  # y-y_pred es el residual
print(f'RMSE: {rmse:.4f}')

# Calculo del error MAE
mae = np.mean(np.abs(y-y_pred))
print(f'MAE: {mae:.4f}')

# Se visualiza que tan bueno es el modelo
# Crear una nueva grafica de y vs y_pred
plt.scatter(y, y_pred)
plt.xlabel('y')
plt.ylabel('y_pred')
plt.show()

# Se visualiza el error de los datos
# Crear una nueva grafica de y vs y_pred
plt.scatter(y, (y_pred - y))
plt.xlabel('y')
plt.ylabel('y_pred')
plt.show()
