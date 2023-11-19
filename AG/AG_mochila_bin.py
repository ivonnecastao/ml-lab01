import numpy as np
import pygad

# Definir los pesos y valores de los productos
pesos = np.random.randint(1, 50, 50)
valores = np.random.randint(1, 100, 50)
capacidad = sum(pesos) // 2

# Función de aptitud
def fitness_func(solucion, solucion_idx):
    peso_total = sum(solucion * pesos)
    valor_total = sum(solucion * valores)
    if peso_total <= capacidad:
        return valor_total
    else:
        return 0

# Parámetros del algoritmo genético
num_generaciones = 100
tam_poblacion = 100
tasa_mutacion = 3

# Crear una instancia de GA
ga_instance = pygad.GA(
   num_generations=num_generaciones,
   num_parents_mating=2,
   fitness_func=fitness_func,
   sol_per_pop=tam_poblacion,
   num_genes=len(pesos),
   init_range_low=0,
   init_range_high=2,
   mutation_percent_genes=tasa_mutacion,
   mutation_by_replacement=True,
   random_mutation_min_val=0,
   random_mutation_max_val=1)

# Ejecutar el algoritmo genético
ga_instance.run()

# Obtener la mejor solución
mejor_solucion, mejor_aptitud = ga_instance.best_solution()
print("Mejor solución: ", mejor_solucion)
print("Peso total: ", sum(mejor_solucion * pesos))
print("Valor total: ", mejor_aptitud)
