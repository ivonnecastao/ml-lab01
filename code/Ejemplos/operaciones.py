import numpy as np

# Definamos dos arreglos de numpy de 2 dimensiones
x = np.array([[1, 2, 3], [4, 5, 6]])
y = np.array([[6, 5, 4], [3, 2, 1]])
print(f"x = {x}")
print(f"y = {y}")

# Suma de arreglos
z = x + y
print(f"z = {z}")

# Resta de arreglos
z = x - y
print(f"z = {z}")

# Multiplicacion de arreglos
z = x * y
print(f"z = {z}")

# Producto punto de arreglos
a = np.array([[3, 2, 3], [3, 2, 3]])
b = np.array([[4, 5], [6, 7], [8,9]])
z = np.dot(a,b) #Producto interno
print(f"z = {z}")

# Producto punto de arreglos
a = np.array([[3, 2, 3], [3, 2, 3]])
b = np.array([[4, 5], [6, 7], [8,9]])
z = a @ b 
print(f"z = {z}")

print(f"Seno de x = {np.sin(x)}")

redondeo_arriba = np.ceil(np.sin(x))
print(f"Redondeo hacia arriba de a = {redondeo_arriba}")

redondeo_abajo = np.floor(np.sin(x))
print(f"Redondeo hacia abajo de a = {redondeo_abajo}")

# Redondeo a 2 decimales
redondeo_2_decimales = np.round(np.sin(x), 2)
print(f"Redondeo a 2 decimales de a = {redondeo_2_decimales}")

# Comparación de arreglos
z = x > y
print(f"Comparación de x > y = {z}")

# Obtengo el tipo de dato de z
print(f"Tipo de dato de z = {z.dtype}")

# Obtener el promedio de los elementos de un arreglo
print(f"Promedio de los elementos de a = {np.mean(a)}")

# Obtener la moda de los elementos de un arreglo
print(f"Moda de los elementos de a = {np.median(a)}")

# Concatenación de arreglos
z = np.concatenate((x, y), axis=0)
print(f"Concatenación de x y y = {z}")

# Concatenación de arreglos
z = np.concatenate((x, y), axis=1)
print(f"Concatenación de x y y = {z}")

# Primer elemento de[ la primera fila de x
print(f"x[0, 0] = {x[-1, -1]}")

# Slice de la primera fila de x
print(f"x[0, :] = {x[0, :]}")

a = np.array([1, 2, 3, 4, 5, 6])
print(f"a = {a}")

# Reshape a un arreglo de 2x3
b = a.reshape(2, 3)
print(f"b = {b}")

#matriz de una imagen reshape para volverlo vector ANOTACIOOOOOOOOOOOON

# ------------------------- broadcasting --------------------------------
print(f"a + 1 = {a + 1}")

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [6, 7, 8, 9]])
b = np.array([1, 0, -1, 2])
print(f"b = {b}")

print(f"a + b = {a + b}")

# Calcular la descomposicion de Cholesky

A = np.array([[1, 2, 3], [2, 5, 6], [3, 6, 10]])
print(f"A = {A}")
L = np.linalg.cholesky(A)
print(f"L = {L}")
# Calcular la descomposicion QR

Q, R = np.linalg.qr(A)
print(f"Q = {Q}")
print(f"R = {R}")
# Calcular los autovalores y  autovectores de una matriz
w, v = np.linalg.eig(A)
print(f"w = {w}")
print(f"v = {v}")

# Resolver un sistema de ecuaciones lineales
b = np.array([1, 2, 3])
x = np.linalg.solve(A, b)
print(f"x = {x}")

#Calcular la inversa de una matriz
Ainv = np.linalg.inv(A)
print(f"Ainv = {Ainv}")

# Calcular la determinante de una matriz

detA = np.linalg.det(A)
print(f"detA = {detA}")


# Generar un numero aleatorio entre 0 y 1
r = np.random.randn( 4) * 2 + 10 #Desplaza la media 10 y desviación estandar 2
print(f"r = {r}")

A = np.array([[1, 2, 3], [2, 5, 6], [3, 6, 10]], dtype=np.float64)
suma_filas = np.sum(A, axis=0)
print(f"suma_filas = {suma_filas}")