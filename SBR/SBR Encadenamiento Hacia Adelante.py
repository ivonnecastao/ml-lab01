
"""
SISTEMA BASADO EN REGLAS

ESTRATEGIA: Encadenamiento hacia adelante

Este código implementa un sistema basado en reglas con un motor de inferencia utilizando 
la estrategia de encadenamiento hacia adelante. Las reglas se definen como un diccionario 
donde cada clave es el número de la regla y cada valor es una lista de dos listas, donde 
la primera lista son los antecedentes y la segunda lista son los consecuentes. Los hechos 
se definen como una lista de letras. Las reglas utilizadas se almacenan en una lista. 

"""

# Definimos las reglas
reglas = {
    1: [["b", "d", "e"], ["f"]],
    2: [["d", "g"], ["a"]],
    3: [["c", "f"], ["a"]],
    4: [["b"], ["x"]],
    5: [["d"], ["e"]],
    6: [["a", "x"], ["h"]],
    7: [["c"], ["d"]],
    8: [["x", "c"], ["a"]],
    9: [["x", "b"], ["d"]]
}

# Definimos el conjunto de hechos
hechos = ["b", "c"]

# Definimos una lista para almacenar las reglas utilizadas
reglas_utilizadas = []

# Definimos la meta
meta = "h"

#-------------------------------------------------------------------------------

# Mientras haya reglas y la meta no esté en los hechos
while reglas and meta not in hechos:
    # Para cada regla en las reglas
    for r in list(reglas):
        # Si todos los antecedentes de la regla están en los hechos
        if all(i in hechos for i in reglas[r][0]):
            # Añadimos los consecuentes a los hechos
            hechos += reglas[r][1]
            # Añadimos la regla a las reglas utilizadas
            reglas_utilizadas.append(r)
            # Eliminamos la regla de las reglas
            del reglas[r]
            # Salimos del bucle for ya que el diccionario ha cambiado de tamaño
            break

print("Hechos:", hechos)
print("Reglas utilizadas:", reglas_utilizadas)
