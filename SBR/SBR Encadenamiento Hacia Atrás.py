"""
SISTEMA BASADO EN REGLAS

ESTRATEGIA: Encadenamiento hacia atrás

Este código implementa un sistema basado en reglas con un motor de inferencia utilizando 
la estrategia de encadenamiento hacia atrás. Las reglas se definen como un diccionario 
donde cada clave es el número de la regla y cada valor es una lista de dos listas, donde 
la primera lista son los antecedentes y la segunda lista son los consecuentes. Los hechos 
se definen como una lista de letras. Las reglas utilizadas se almacenan en una lista. 
La función backward_chaining implementa el encadenamiento hacia atrás. Dentro de la función, 
se verifica si la meta está en los hechos conocidos. Si es así, se retorna verdadero. 
Si no, se verifica si alguna regla puede inferir la meta. Si es así, se verifica si todas 
las condiciones de esa regla se pueden inferir utilizando el encadenamiento hacia atrás. 
Si es así, se añade el resultado a los hechos conocidos, se añade la regla a las reglas utilizadas 
y se retorna verdadero. Si no se puede inferir ninguna condición o si no hay ninguna regla que 
pueda inferir la meta, se retorna falso. Finalmente, se imprime los hechos y las reglas utilizadas.
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

#----------------------------------------------------------------------------------------------------

# Función para implementar el encadenamiento hacia atrás
def backward_chaining(goal):
    # Si la meta ya está en los hechos, retornamos verdadero
    if goal in hechos:
        return True
    # Para cada regla en las reglas
    for r in sorted(reglas.keys()):
        # Si el consecuente de la regla es la meta
        if goal in reglas[r][1]:
            # Si todos los antecedentes de la regla están en los hechos o pueden ser inferidos
            if all(backward_chaining(i) for i in reglas[r][0]):
                # Añadimos los consecuentes a los hechos
                hechos.extend(reglas[r][1])
                # Añadimos la regla a las reglas utilizadas
                reglas_utilizadas.append(r)
                return True
    return False

# Llamamos a la función de encadenamiento hacia atrás con la meta
backward_chaining(meta)

print("Hechos:", hechos)
print("Reglas utilizadas:", reglas_utilizadas)

