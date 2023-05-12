

# costos = [20, 20, 23, 24, 25, 12, 8, 3, 23, 15, 3, 19, 4, 21, 8, 14, 11, 7, 3, 19, 14, 11, 23, 23, 9, 14, 13, 16, 23, 22, 8, 23, 11, 8, 8, 25, 25, 5, 6, 14, 10, 5, 22, 5, 7, 10, 15, 15, 5, 23]
# volúmenes = [10, 25, 12, 9, 13, 1, 6, 20, 8, 11, 1, 1, 5, 20, 15, 15, 7, 3, 7, 7, 13, 22, 19, 4, 5, 14, 13, 19, 5, 11, 23, 22, 7, 1, 23, 19, 15, 8, 15, 25, 3, 21, 13, 24, 20, 10, 24, 23, 21, 25]
import random

# Parámetros del problema
VOLUMEN_MAXIMO = 442
NUM_OBJETOS = 50
NUM_ITERACIONES = int(input("Ingrese el número de iteraciones: "))
TAM_POBLACION = 50
PORCENTAJE_INFECTIBLES = 3

# Datos del problema de la mochila
costos = [20, 20, 23, 24, 25, 12, 8, 3, 23, 15, 3, 19, 4, 21, 8, 14, 11, 7, 3, 19, 14, 11, 23, 23, 9, 14, 13, 16, 23, 22, 8, 23, 11, 8, 8, 25, 25, 5, 6, 14, 10, 5, 22, 5, 7, 10, 15, 15, 5, 23]
volu_menes = [10, 25, 12, 9, 13, 1, 6, 20, 8, 11, 1, 1, 5, 20, 15, 15, 7, 3, 7, 7, 13, 22, 19, 4, 5, 14, 13, 19, 5, 11, 23, 22, 7, 1, 23, 19, 15, 8, 15, 25, 3, 21, 13, 24, 20, 10, 24, 23, 21, 25]


# Función de evaluación del costo
def evaluar_costo(individuo):
    return sum(gen * costo for gen, costo in zip(individuo, costos))

# Función de evaluación del volumen
def evaluar_volumen(individuo):
    return sum(gen * volumen for gen, volumen in zip(individuo, volu_menes))

# Función de reparación
def reparar_individuo(individuo):
    while evaluar_volumen(individuo) > VOLUMEN_MAXIMO:
        indices_1 = [i for i, gen in enumerate(individuo) if gen == 1]
        if not indices_1:
            break
        indice_random = random.choice(indices_1)
        individuo[indice_random] = 0
    return individuo

# Generar población inicial
poblacion = []
for _ in range(TAM_POBLACION):
    individuo = [random.randint(0, 1) for _ in range(NUM_OBJETOS)]
    individuo = reparar_individuo(individuo)
    poblacion.append(individuo)

# Ciclo principal
for iteracion in range(NUM_ITERACIONES):
    # Selección de padres
    padres = []
    for _ in range(2):
        participantes = random.sample(poblacion, 4)
        mejor_padre = max(participantes, key=evaluar_costo)
        padres.append(mejor_padre)

    # Recombinación (cruce uniforme)
    hijos = []
    for i in range(0, 2, 2):
        padre1 = padres[i]
        padre2 = padres[i + 1]
        mascara = [random.choice([0, 1]) for _ in range(NUM_OBJETOS)]
        hijo = [padre1[j] if mascara[j] == 1 else padre2[j] for j in range(NUM_OBJETOS)]
        hijo = reparar_individuo(hijo)
        hijos.append(hijo)

    # Mutación (dos puntos)
    for hijo in hijos:
        if random.random() < 0.05:  # Probabilidad de mutación: 5%
            punto1, punto2 = random.sample(range(NUM_OBJETOS), 2)
            hijo[punto1] = 1 - hijo[punto1]
            hijo[punto2] = 1 - hijo[punto2]
            hijo = reparar_individuo(hijo)

      # Reemplazo de peores individuos
    peores_individuos = sorted(poblacion, key=evaluar_costo)[:PORCENTAJE_INFECTIBLES]
    poblacion = [individuo for individuo in poblacion if individuo not in peores_individuos]
    poblacion.extend(hijos)

    # Ciclo de mejora
    for individuo in poblacion:
        while 0 in individuo:
            indice_max = max(range(NUM_OBJETOS), key=lambda i: costos[i] / volu_menes[i])
            individuo[indice_max] = 1
            individuo = reparar_individuo(individuo)

    # Obtener mejor individuo
    mejor_individuo = max(poblacion, key=evaluar_costo)
    mejor_costo = evaluar_costo(mejor_individuo)
    mejor_volumen = evaluar_volumen(mejor_individuo)

    # Imprimir resultado de la iteración actual
    print(f"Iteración {iteracion+1}: Mejor individuo: {mejor_individuo}, Costo: {mejor_costo}, Volumen: {mejor_volumen}")

# Obtener mejor individuo final
mejor_individuo = max(poblacion, key=evaluar_costo)
mejor_costo = evaluar_costo(mejor_individuo)
mejor_volumen = evaluar_volumen(mejor_individuo)

# Imprimir resultado final
print("\nResultado final:")
print(f"Mejor individuo: {mejor_individuo}")
print(f"Costo del mejor individuo: {mejor_costo}")
print(f"Volumen del mejor individuo: {mejor_volumen}")
