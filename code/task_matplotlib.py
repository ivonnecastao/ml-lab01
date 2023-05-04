import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data\downloaded\winequality-white.csv', sep=";")


print(df.shape) # me dice las dimensiones del dataframe

mnp = df.to_numpy(dtype = 'float', copy = False)  # me convierte el df en una matriz de numpy (mnp)

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

Theta_best = np.linalg.inv(mnp.T @ mnp) @ (mnp.T @ y)  #y= x Theta1 + Theta 0
print(f"Theta_best = {Theta_best}")
print(Theta_best.shape)

y_pred = mnp @ Theta_best  #resultado, predicción
print(y_pred)
print(y_pred.shape)


#"""
# plot data
#plt.scatter(mnp, y, label='Datos')
plt.plot(mnp, y_pred, color='red', label='Ajuste lineal') 
plt.xlabel('Variables independiente (x)')
plt.ylabel('Variable dependiente (y)')
plt.legend()
plt.title('Regresión Lineal con Numpy')
plt.show()
plt.savefig('plots\regresionlineal.png')
#"""


