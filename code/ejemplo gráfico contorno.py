import numpy as np
import matplotlib.pyplot as plt

# Datos de ejemplo
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = 3*X + Y #np.sin(np.sqrt(X**2 + Y**2))

# Gráfico de contorno relleno
plt.contourf(X, Y, Z, levels=20, cmap='coolwarm')
plt.colorbar()  # Agrega una barra de color
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Gráfico de contorno relleno')
plt.show()



# Datos de ejemplo
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

# Gráfico de contorno relleno con cmap=plt.cm.Spectral
plt.contourf(X, Y, Z, levels=20, cmap=plt.cm.Spectral)
plt.colorbar()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Gráfico de contorno relleno con cmap=plt.cm.Spectral')
plt.show()
