import numpy as np

# Imprima la versión actual de NumPy
print(np.__version__)

# Creación de un arreglo de 1 dimensión
a = np.array([1, 2, 3, 7])

# Imprimir el arreglo
print(a)

# Creacion de un arreglo de 2 dime[nsiones
b = np.array([[1, 2, 3], [4, 5, 6]])

# Imprimir el arreglo
print(b)

#Imprimir numero de dimensiones
print(b.ndim)

# Imprimir numero de elementos
print(b.size)

# Crear un vector de ceros
c = np.zeros(10)

# Imprimir el vector
print(c)
# Crear un vector de ceros
d = np.ones(10)

# Imprimir el vector
print(d)

# Crear un vector de ceros de 2 dimensiones
e = np.zeros((2,3))

# Imprimir el vector
f = np.ones((2, 3), dtype=int)
print(f)
