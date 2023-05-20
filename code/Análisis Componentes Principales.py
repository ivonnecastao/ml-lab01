import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#% matplotlib inline


# PREPARACIÓN DE DATOS

#CARGUE
print("DATOS INICIALES:\n")

df = pd.read_csv('data\downloaded\winequality-white.csv', sep=";")
print(f"Dataframe:\n {df}")
print(f"Dim Dataframe = {df.shape}")  # me dice las dimensiones del dataframe

#LIMPIEZA

df = df.dropna() # Elimino datos incompletos. Limpio datos
print(f"Dim Dataframe limpio = {df.shape}")


# Se divide la matriz del dataset en dos partes

x = df.iloc[:,0:11].values   # la submatriz x contiene los valores de las primeras 11 columnas del dataframe y todas las filas
print(f"x:\n {x}")
print(f"Dim x = {x.shape}")

y = df.iloc[:,-1].values   # El vector y contiene los valores de la 12 columna (calidad) para todas las filas
print(f"y:\n {y}")
print(f"Dim y = {y.shape}")

#Aplicamos una transformación de los datos para poder aplicar las propiedades de la distribución normal

x_std = StandardScaler().fit_transform(x)
print(f"x_std:\n {x_std}")
print(f"Dim x_std = {x_std.shape}")

# Calculamos la matriz de covarianza de numpy

mcn = np.cov(x_std.T)
print(f"Matriz de Covarianza de Numpy: \n {mcn}")
print(f"Dim mcn = {mcn.shape}")

#Calculamos los autovalores y autovectores de la matriz y los mostramos

eig_vals, eig_vecs = np.linalg.eig(mcn)

print('Vectores Propios \n%s' %eig_vecs)
print(f"Dim Vectores Propios = {eig_vecs.shape}")
print('\nValores Propios \n%s' %eig_vals)
print(f"Dim Valores Propios = {eig_vals.shape}")

#  Hacemos una lista de parejas (autovector, autovalor) 
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
print('\nValores Propios - Vectores Propios \n%s' %eig_pairs)

# Ordenamos estas parejas den orden descendiente con la función sort
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visualizamos la lista de autovalores en orden desdenciente
print('Autovalores en orden descendiente:')
for i in eig_pairs:
    print(i[0])
    
# A partir de los autovalores, calculamos la varianza explicada
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

# Representamos en un diagrama de barras la varianza explicada por cada autovalor, y la acumulada
with plt.style.context('seaborn-pastel'):
    plt.figure(figsize=(8, 4))

    plt.bar(range(11), var_exp, alpha=0.5, align='center',
            label='Varianza individual explicada', color='g')
    plt.step(range(11), cum_var_exp, where='mid', linestyle='--', label='Varianza explicada acumulada')
    plt.ylabel('Ratio de Varianza Explicada')
    plt.xlabel('Componentes Principales')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('plots\ ACP.png')
    plt.show()


#Generamos la matríz a partir de los pares autovalor-autovector
matrix_w = np.hstack((eig_pairs[0][1].reshape(11,1),
                      eig_pairs[1][1].reshape(11,1)))

print('Matriz W:\n', matrix_w)
print(f"Dim W = {matrix_w.shape}")

y = x_std.dot(matrix_w)
print('Y:\n', y)
print(f"Dim Y = {y.shape}")


