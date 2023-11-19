import numpy as np

# Parámetros del algoritmo genético
num_generaciones = 2
tam_poblacion = 20
tasa_mutacion = 0.01
rango = (-1, 2)

# Función objetivo
def f(x):
    return x * np.sin(10 * np.pi * x) + 1.0

# Inicializar la población
poblacion = np.random.uniform(rango[0], rango[1], tam_poblacion)
print(poblacion)

for _ in range(num_generaciones):
    # Calcular la aptitud de cada individuo
    aptitud = f(poblacion)
    print(aptitud)

    # Seleccionar los padres mediante la ruleta
    probabilidad_seleccion = aptitud / sum(aptitud)
    padres_idx = np.random.choice(range(tam_poblacion), size=tam_poblacion, p=probabilidad_seleccion)
    padres = poblacion[padres_idx]
    print(padres)
    
    # Aplicar el cruce aritmético
    hijos = padres.copy()
    for i in range(0, tam_poblacion, 2):
        if np.random.rand() < 0.5:
            hijos[i], hijos[i+1] = (padres[i] + padres[i+1]) / 2, (padres[i] + padres[i+1]) / 2

    # Aplicar la mutación gaussiana
    mutaciones = np.random.rand(tam_poblacion) < tasa_mutacion
    hijos[mutaciones] += np.random.normal(0, 0.1, sum(mutaciones))

    # Reemplazar la población actual con los hijos
    poblacion = hijos

# Imprimir la mejor solución encontrada
mejor_solucion = poblacion[np.argmax(aptitud)]
print("Mejor solución: ", mejor_solucion)
print("Valor máximo de f: ", f(mejor_solucion))
