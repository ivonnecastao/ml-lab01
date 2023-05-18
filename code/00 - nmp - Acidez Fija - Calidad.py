import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data\downloaded\winequality-white.csv', sep=";")
print('Dataframe:')
print(df)
print(df.shape) # me dice las dimensiones del dataframe

df = df.dropna()
print('Dataframe limpio:')
print(df)

mnp = df.to_numpy()  # me convierte el df en una matriz de numpy (mnp)
print('Matriz Numpy:')
print(mnp)
print(mnp.shape)  # me dice las dimensiones de la mnp

i = 0 # columna que queremos obtener
x = np.array([mnp[:,i]]) # Extrae la columna i y la guarda en x
print('Variable Independiente X: Acidez Fija')
print(x)
print(x.shape)

print('Variable Independiente X Transpuesta: Acidez Fija')
x = np.transpose(x)
print(x)
print(x.shape)

i = 11 # columna que queremos obtener
y = np.array([mnp[:,i]]) # Extrae la columna i y la guarda en y
print('Variable Dependiente Y: Calidad')
print(y)
print(y.shape)

print('Variable Dependiente Y Transpuesta: Calidad')
y = np.transpose(y)
print(y)
print(y.shape)

#mnp = np.delete(mnp, 11, axis=1) # Elimino la columna que extraje en y
#print(mnp)
#print(mnp.shape)

vones= np.ones((x.shape[0],1)) # Creo el vector de 1's BIAS
print('Vector BIAS')
print(vones)
print(vones.shape)

x_b = np.hstack([vones, x])
print('Variable Independiente X + BIAS: BIAS Acidez Fija')
print(x_b)
print(x_b.shape)

Theta_best = np.linalg.inv(x_b.T @ x_b) @ (x_b.T @ y)  #y= x Theta1 + Theta 0
print(f"Theta_best = {Theta_best}")
print(Theta_best.shape)

y_pred = x_b @ Theta_best  #resultado, predicción
print('Variable Dependiente Y Predicha: Calidad')
print(y_pred)
print(y_pred.shape)

rmse = np.sqrt(np.mean((y_pred - y)**2))
print(f"RMSE: {rmse}")

#"""

# plot data x/y
plt.scatter(x, y_pred, label='Datos')
plt.plot(x, y_pred, color='red', label='Ajuste lineal') 
plt.xlabel('Variable independiente (x)')
plt.ylabel('Variable dependiente (y)')
plt.legend()
plt.title('Regresión Lineal con Numpy')
plt.show()
plt.savefig('plots\Acidezfija_vs_calidad_numpy.png')

# plot data y/y_pred
plt.scatter(y, y_pred, label='Datos')
plt.xlabel('Datos Reales (y)')
plt.ylabel('Predicción (y_pred)')
plt.legend()
plt.title('Regresión Lineal con Numpy')
plt.show()
plt.savefig('plots\y_vs_y_pred_afq_numpy.png')
#"""


