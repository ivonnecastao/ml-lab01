import numpy as np
from sklearn.datasets import make_circles
import json
#from pathlib import Path
import os


# Especificar la ruta y el nombre del archivo JSON donde se guardarán los datos
archivo_json = 'pesos.json'

# Generación del dataset 
np.random.seed(0)
X, B = make_circles(n_samples=5, noise=0.05)

print(f"X:\n {X}")
print(f"Dim de X: {X.shape}")
print(f"B:\n {B.reshape(-1,1)}")
print(f"Dim de B: {B.shape}")

# Definir la matriz A y el vector 
#X  = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
#W1 = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]) 
#W2 = np.array([[1], [2], [3], [4], [5]]) 
#B1 = np.array([[1, 1, 1, 1, 1]])   
#B2 = np.array([[1]]) 

W1 = np.random.randn(2, 5)
W2 = np.random.randn(5, 1)
B1 = np.ones((1, 5))
B2 = np.ones((1, 1))

# Obtener las dimensiones de la matriz y el vector
print(f"Dim de X: {X.shape}")
print(f"Dim de W1: {W1.shape}")
print(f"Dim de B1: {B1.shape}")

XW1 = np.dot(X, W1)
print(f"X.W1: {XW1}")
print(f"Dim de X.W1: {XW1.shape}")

Z1 = XW1 + B1
print(f"Z1:\n {Z1}")
print(f"Dim de Z1: {Z1.shape}")

A1 = 1 / (1 + np.exp(-Z1))
print(f"A1:\n {A1}")
print(f"Dim de A1: {A1.shape}")

print(f"Dim de W2: {W2.shape}")
print(f"Dim de B2: {B2.shape}")

A1W2 = np.dot(A1, W2)
print(f"A1.W2: {A1W2}")
print(f"Dim de A1.W2: {A1W2.shape}")

Z2 = A1W2 + B2
print(f"Z2:\n {Z2}")
print(f"Dim de Z2: {Z2.shape}")

Y_hat = 1 / (1 + np.exp(-Z2))
print(f"Y_hat:\n {Y_hat}")
print(f"Y_hat: {Y_hat.shape}")


# Convertir la matriz en una lista de Python
W1_lista = W1.tolist()
W2_lista = W2.tolist()

# Crear un diccionario con las variables que deseas guardar
pesos = {
    'W1' : W1_lista,
    'W2' : W2_lista
}


# Borrar archivo existente

# Especifica la ruta y el nombre del archivo JSON que deseas borrar
#ruta_archivo = Path(archivo_json)

# Verifica si el archivo existe antes de borrarlo
if os.path.exists(archivo_json):
    os.remove(archivo_json)
    print("Archivo borrado exitosamente.")
else:
    print("El archivo no existe.")

# Guardar el diccionario en un archivo JSON
with open(archivo_json, 'w') as archivo:
    json.dump(pesos, archivo)
    
# Leer los datos del archivo JSON
with open(archivo_json, 'r') as archivo:
    pesos_leidos = json.load(archivo)

# Guardar las variables en otras variables
v1 = np.array(pesos_leidos['W1'])
v2 = np.array(pesos_leidos['W2'])

print(f"W1_archivo: {v1}")
print(f"W1: {W1}")
print(f"W2_archivo: {v2}")
print(f"W2: {W2}")

