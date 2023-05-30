import numpy as np
import matplotlib.pyplot as plt

# Función de costo
def costo(x):
    return x**2

# Derivada de la función de costo
def gradiente(x):
    return 2*x

# Parámetro inicial
x = 3

# Tasa de aprendizaje
tasa_aprendizaje = 0.1  #Contrae el gradiente

# Preparación de datos para la gráfica - ESto es sólo para el ejemplo, normalmente no es viable
x_values = np.linspace(-3, 3, 100)
y_values = costo(x_values)

plt.figure(figsize=(10, 5))

for i in range(15):  #Ciclo de entrenamiento, cuantas veces actualizazo a w
    
    cost = costo(x)  # Cuanto vale la perdida con los valores actuales de w
    grad = gradiente(x)  # cuanto vale mi gradiente para w
    
    # Dibujar la función de costo
    plt.plot(x_values, y_values)
    plt.title(f"Iteración {i+1}")
    
    # Dibujar el punto actual
    plt.plot(x, cost, 'ro')
    
    # Dibujar la flecha del gradiente
    plt.arrow(x, cost, -tasa_aprendizaje * grad, - grad * (tasa_aprendizaje * grad), head_width=0.15, head_length=0.1, fc='red', ec='red')

    x = x - tasa_aprendizaje * grad  # actualizo el w

    # Pause and update the plot
    plt.pause(1)

    # clear the previous plot
    plt.cla()
    
    print(f"x = {x:.4f}", f"costo = {cost:.4f}")

plt.show()

