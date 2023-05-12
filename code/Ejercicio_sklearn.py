import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#PREPARACIÓN

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

#MODELO

mnp_train, mnp_test, y_train, y_test = train_test_split(mnp, y, test_size=0.2)

print(f"mnp_train.shape = {mnp_train.shape}")
print(f"mnp_test.shape = {mnp_test.shape}")

model = LinearRegression()
model.fit(mnp_train, y_train)

y_pred = model.predict(mnp_test)
print(y_pred)
print(y_pred.shape)

mse = mean_squared_error(y_test, y_pred)

print(f"Coeficiente: {model.coef_}")
print(f"Intercepto: {model.intercept_}")
print(f"Error cuadratico medio: {mse}")

"""
# Plot data
plt.scatter(y_pred, y, label="Datos")
#plt.plot(mnp_test, y_predict, color="red", label="Ajuste lineal")

plt.xlabel("Variable independiente (y_pred)")
plt.ylabel("Variable dependiente(y)")
plt.title("Regresion lineal")
plt.legend()
plt.show()
"""