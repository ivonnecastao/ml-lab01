import numpy as np
from sklearn import datasets
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# Datos sintéticos

x,y = datasets.make_moons(n_samples=1000, noise=0.2, random_state=42)
x_train,x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# Funciones

def sigmoid(x):
    return 1 /(1 + np.exp(-x))

def sigmoid_derivate(x):
    return x * (1-x)

def binary_cross_entropy(y, y_hat):
    N = y.shape[0]
    loss = -1/N * np.sum((y*np.log(y_hat) + (1-y)*np.log(1-y_hat)))
    return loss

def binary_cross_entropy_derivative(y, y_hat):
    y = y.reshape(-1,1)
    m = y.shape[0]
    bce_grad = -1 / m * (y / y_hat - (1 - y) / (1 - y_hat))
    return bce_grad


# Hiperparámetros

input_size = 2   # Tamaño de la capa de entrada
hidden_size = 3  # Tamaño de la capa oculta
output_size = 1  # Tamaño de la capa de salida
lr = 0.01        # Tasa de aprendizaje
num_epochs = 1000


# Pesos

b1 = np.zeros((1, hidden_size)) # 1x3  Bias 1
W1 = np.random.randn(input_size, hidden_size) #2x3
b2 = np.zeros((1, output_size)) # 1x1  Bias 2
W2 = np.random.randn(hidden_size, output_size) #3x1


# Entrenamiento

for epoch in range(num_epochs):
    
    #forward propagation
    z1 = np.dot(x_train, W1) + b1  # Producto punto mas bias
    a1 = sigmoid(z1)  # g(1)(z1)
    z2 = np.dot(a1, W2) + b2
    y_hat = sigmoid(z2)  # g(2)(z2)

    # Calcular pérdida con Función de pérdida Binary_cross_entropy

    loss = binary_cross_entropy(y_train, y_hat)
    print(f"Epoch: {epoch},  Loss: {loss}")

    # Propagar el error hacia atrás: Back Propagation

    delta_y_hat = binary_cross_entropy_derivative(y_train, y_hat)
    delta_z2 = delta_y_hat * sigmoid_derivate(y_hat)
    delta_W2 = np.dot(a1.T, delta_z2)
    delta_b2 = np.sum(delta_z2, axis=0)                # Porque es el vector de 1's multiplicado por delta_z2, da el mismo delta_z2
    delta_a1 = np.dot(delta_z2, W2.T)
    delta_z1 = delta_a1 * sigmoid_derivate(a1)
    delta_W1 = np.dot(x_train.T, delta_z1)
    delta_b1 = np.sum(delta_z1, axis=0)

    # Actualizar pesos y bias

    W2 = W2 - lr * delta_W2  # El menos es porque vamos en el sentido contrario del gradiente
    b2 = b2 - lr * delta_b2
    W1 = W1 - lr * delta_W1 
    b1 = b1 - lr * delta_b1
