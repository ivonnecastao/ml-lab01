import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


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

mnp_train, mnp_test, y_train, y_test = train_test_split(mnp, y, test_size=0.2)

print("\nMODELO:\n")
print(f"Dim mnp_train = {mnp_train.shape}")
print(f"Dim y_train = {y_train.shape}")
print(f"Dim mnp_test = {mnp_test.shape}")
print(f"Dim y_test = {y_test.shape}")

model = LinearRegression()
model.fit(mnp_train, y_train)   # Este método se utiliza para entrenar los modelos con los datos de entrenamiento. 
                                # Es decir, se utiliza para obtener los parámetros del modelo con los que se 
                                # minimiza el error de predicción en los datos de entrenamiento. Debido a que se 
                                # emplea para entrenar el modelo fit() se suele usar sobre los datos de entrenamiento.

y_pred = model.predict(mnp_test)    # Este método se utiliza para obtener las predicciones de un modelo sobre 
                                    # un conjunto de datos de entrada desconocidos. Es decir, una vez entrenado el modelo, 
                                    # se puede utilizar el método predict() para saber los valores o etiquetas que asigna el 
                                    # modelo a una conjunto de datos. Dada la finalidad de este método, suele usarse 
                                    # con los datos de test o cualquier conjunto desconocido.
#print(f"Vector y_pred = {y_pred}")
print(f"Dim Vector y_pred = {y_pred.shape}")


# EVALUACIÓN DEL MODELO

print("\nEVALUACIÓN DEL MODELO DE REGRESIÓN LINEAL CON SKLEARN:\n")

# Cálculo del error MSE
mse = mean_squared_error(y_test, y_pred)
# Calculo del error RMSE
rmse = np.sqrt(np.mean((y_test - y_pred)**2))  # y_test - y_pred es el residual
# Calculo del error MAE
mae = np.mean(np.abs(y_test - y_pred))

print(f"Coeficiente Theta_best: {model.coef_}")
print(f"Intercepto: {model.intercept_}")
print(f"MSE: {mse}")
print(f'RMSE: {rmse:.4f}')
print(f'MAE: {mae:.4f}')


# VISUALIZACIÓN

# Se visualiza que tan bueno es el modelo
plt.scatter(y_test, y_pred) # Crear una nueva grafica de y_test vs y_pred
#plt.plot(y_test, y_pred, color='red') 
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.title("Regresion lineal con sklearn: Qué tan bueno es el Modelo?")
plt.savefig('plots\ Sklearn y_test vs y_pred.png')
plt.show()

# Se visualiza el error de los datos
plt.scatter(y_test, (y_pred - y_test))  # y_test - y_pred es el residual
#plt.plot(y_test, y_pred, color='red')
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.title("Regresion lineal con sklearn: Cuál es el error del Modelo?")
plt.savefig('plots\ Sklearn y_test vs y_pred-y_test.png')
plt.show()