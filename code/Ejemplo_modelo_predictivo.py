import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# PREPARACIÓN DE DATOS

# CARGUE
print("DATOS INICIALES:\n")

df = pd.read_csv('data\downloaded\Real estate.csv', sep=",")
print(f"Dataframe:\n {df}")
print(f"Dim Dataframe = {df.shape}")  # me dice las dimensiones del dataframe

# LIMPIEZA

df = df.dropna() # Elimino datos incompletos. Limpio datos
print(f"Dim Dataframe limpio = {df.shape}")

# TRANSFORMACIÓN

mnp = df.to_numpy()  # me convierte el df en una matriz de numpy (mnp)
print(mnp)
print(mnp.shape)  # me dice las dimensiones de la mnp

# Extraemos Y
i = 7 # columna que queremos obtener (Precio de las casas)
y = np.array([mnp[:,i]]) # Extrae la columna i y la guarda en Y
#print(y)
#print(y.shape)

y = np.transpose(y) # Transpongo Y porque quedó como fila y necesito columna
#print(y)
#print(y.shape)

# Extraemos X
 # columna que queremos obtener
i = 2 # Antiguedad de la casa
#i = 3 # Distancia a la estación del metro
#i = 4 # Número de tiendas cercanas

x = np.array([mnp[:,i]]) # Extrae la columna i y la guarda en X
#print(x)
#print(x.shape)

x = np.transpose(x) # Transpongo Y porque quedó como fila y necesito columna
#print(x)
#print(x.shape)

print(f"Vector X:\n {x}")
print(f"Dim Vector X = {x.shape}")
print(f"Vector Y:\n {y}")
print(f"Dim Vector Y = {y.shape}")

# Separación del dataframe en conjuntos de entrenamiento y evaluación
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print("\nConjuntos preparados:\n")
print(f"Dim x_train = {x_train.shape}")
print(f"Dim y_train = {y_train.shape}")
print(f"Dim x_test = {x_test.shape}")
print(f"Dim y_test = {y_test.shape}")


# MODELO

model = LinearRegression()
model.fit(x_train, y_train)     # Este método se utiliza para entrenar los modelos con los datos de entrenamiento. 
                                # Es decir, se utiliza para obtener los parámetros del modelo con los que se 
                                # minimiza el error de predicción en los datos de entrenamiento. Debido a que se 
                                # emplea para entrenar el modelo fit() se suele usar sobre los datos de entrenamiento.

y_pred = model.predict(x_test)      # Este método se utiliza para obtener las predicciones de un modelo sobre 
                                    # un conjunto de datos de entrada desconocidos. Es decir, una vez entrenado el modelo, 
                                    # se puede utilizar el método predict() para saber los valores o etiquetas que asigna el 
                                    # modelo a una conjunto de datos. Dada la finalidad de este método, suele usarse 
                                    # con los datos de test o cualquier conjunto desconocido.

print(f"Dim Vector y_pred = {y_pred.shape}")


# EVALUACIÓN DEL MODELO

print("\nEVALUACIÓN DEL MODELO DE REGRESIÓN LINEAL:\n")

# Calculo del error RMSE
rmse = np.sqrt(np.mean((y_test - y_pred)**2))  # y_test - y_pred es el residual

print(f"Mejor Coeficiente: {model.coef_}")
print(f'RMSE: {rmse:.4f}')


# VISUALIZACIÓN

# Se visualizan datos de entrenamiento
plt.scatter(x_train, y_train) # Crear una nueva grafica de x_train vs y_train
plt.xlabel('x_train')
plt.ylabel('y_train')
plt.title("Datos de Entrenamiento")
#plt.savefig('plots\ x_train vs y_train.png')
plt.show()

# Se visualizan datos de evaluación
plt.scatter(x_test, y_pred) # Crear una nueva grafica de x_test vs y_pred
plt.plot(x_test, y_pred, color='red') 
plt.xlabel('x_test')
plt.ylabel('y_pred')
plt.title("Datos de Evaluación")
#plt.savefig('plots\ x_test vs y_pred.png')
plt.show()

# Se visualiza que tan bueno es el modelo
plt.scatter(y_test, y_pred) # Crear una nueva grafica de y_test vs y_pred
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.title("Regresion lineal: Qué tan bueno es el Modelo?")
#plt.savefig('plots\ y_test vs y_pred.png')
plt.show()

# Se visualiza el error de los datos
plt.scatter(y_test, (y_pred - y_test))  # y_test - y_pred es el residual
plt.xlabel('y_test')
plt.ylabel('y_pred')
plt.title("Regresion lineal: Cuál es el error del Modelo?")
#plt.savefig('plots\ error.png')
plt.show()


# PREDECIR DATOS CON EL MODELO

entrada = [[20]]
resultado = model.predict(entrada)

print(f"Entrada: {entrada}")
print(f'Resultado de la predicción: {resultado}')


"""
"""

