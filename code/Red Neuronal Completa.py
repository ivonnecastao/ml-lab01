import numpy as np
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------------------
# FUNCIONES GENERALES

def sintonizar_umbral(y_pred, umbral):
    y_new = np.zeros((y_pred.shape[0], 1))
    i = 0
    
    while i < y_pred.shape[0]:
        if y_pred[i] < umbral:
            y_new[i] = 0
        else:
            y_new[i] = 1
        i = i + 1      
        
    return y_new


def matriz_confusion(y_pred, y):

    # VP + VN
    v = np.sum(y_pred == y.reshape(-1, 1)) 
    # FN + FP
    f = np.sum(y_pred != y.reshape(-1, 1)) 
    
    # VP
    i = 0
    vp = 0
    while i < y_pred.shape[0]:
        vp = vp + np.sum((y_pred[i] == True) and (y[i] == True))
        i = i + 1
    
    # VN
    i = 0
    vn = 0
    while i < y_pred.shape[0]:
        vn = vn + np.sum((y_pred[i] == False) and (y[i] == False))
        i = i + 1
    
    # FP
    i = 0
    fp = 0
    while i < y_pred.shape[0]:
        fp = fp + np.sum((y_pred[i] == True) and (y[i] == False))
        i = i + 1
    
    # FN
    i = 0
    fn = 0
    while i < y_pred.shape[0]:
        fn = fn + np.sum((y_pred[i] == False) and (y[i] == True))
        i = i + 1
    
    # Total Predicciones
    m = vp + fp + vn + fn
    
    print(f"TOTAL: Instancias {m} / Instancias Verdaderas: {v} / Instancias Falsas: {f}")
    
    print(f"VN: {vn}   FP: {fp}")
    print(f"FN: {fn}   VP: {vp}")

    return vp, vn, fp, fn, m


def calculo_metricas(vp, vn, fp, fn, m):
    
    # Balanceo del Dataset
    print(f"Balanceo de Dataset: N {vn + fp:.2f} =~ P {vp + fn:.2f}")

    # Accuracy - Exactitud
    accuracy = (vn + vp) / m
    
    # Precision - Precisión
    precision = vp / (vp + fp)
    
    # Recall - Exhaustividad
    recall = vp / (vp + fn)
    
    # F1 Score - Puntaje F1
    f1score = (2 * precision * recall) / (precision + recall)
        
    return accuracy, precision, recall, f1score

def curvas_roc_pr(y_pred, y):
    
    # Generar matriz ROC-PR para los gráficos de las Curvas ROC y PR
    roc_pr = np.zeros((9, 8))
    i = 0
    paso = 0.1
    umbral = 0

    while i < roc_pr.shape[0]:
        umbral = np.round(umbral + paso, 1)
        roc_pr[i, 0] = umbral
        print(f"Umbral: {umbral} ->: ")
        y_new = sintonizar_umbral(y_pred, umbral)
        roc_pr[i, 1], roc_pr[i, 2], roc_pr[i, 3], roc_pr[i, 4], roc_pr[i, 5] = matriz_confusion(y_new, y)
        
        if roc_pr[i, 1] == 0:
            roc_pr[i, 6] = 0
            roc_pr[i, 7] = 0
        else:
            roc_pr[i, 6] = roc_pr[i, 1] / (roc_pr[i, 1] + roc_pr[i, 3]) # Precision
            roc_pr[i, 7] = roc_pr[i, 1] / (roc_pr[i, 1] + roc_pr[i, 4]) # Recall
        
        i = i + 1
    #print(roc_pr)

    # Plot Curva ROC
    plt.clf()
    plt.scatter(roc_pr[:, 3], roc_pr[:, 1], color='blue', label = roc_pr[:, 0])
    plt.plot(roc_pr[:, 3], roc_pr[:, 1], color='red') 
    plt.xlabel('Tasa FP')
    plt.ylabel('Tasa VP')
    plt.xlim(-0.5, roc_pr[:, 3].max() + 5)  # Establece los límites del eje x 
    plt.ylim(-0.5, roc_pr[:, 1].max() + 5)
    for j, label in enumerate(roc_pr[:, 0]):
        plt.text(roc_pr[j, 3], roc_pr[j, 1], label)
    plt.title("Curva ROC")
    plt.savefig('plots\Curva ROC.png')
    plt.show()

    # Plot Curva PR
    plt.clf()
    plt.scatter(roc_pr[:, 7], roc_pr[:, 6], color='blue', label = roc_pr[:, 0])
    plt.plot(roc_pr[:, 7], roc_pr[:, 6], color='red') 
    plt.xlabel('Recall / Exhaustividad')
    plt.ylabel('Precision / Precisión')
    plt.xlim(roc_pr[:, 7].min() - 1, roc_pr[:, 7].max() + 1)  # Establece los límites del eje x 
    plt.ylim(roc_pr[:, 6].min() - 1, roc_pr[:, 6].max() + 1)
    for j, label in enumerate(roc_pr[:, 0]):
        plt.text(roc_pr[j, 7], roc_pr[j, 6], label)
    plt.title("Curva PR")
    plt.savefig('plots\Curva PR.png')
    plt.show()
    
#-------------------------------------------------------------------------------------------
# ARQUITECTURA DE LA RED NEURONAL

class SimpleNN:
    def __init__(self, num_inputs, num_hidden, num_outputs):
        self.W1 = np.random.randn(num_inputs, num_hidden)
        self.b1 = np.zeros((1, num_hidden))
        self.W2 = np.random.randn(num_hidden, num_outputs)
        self.b2 = np.zeros((1, num_outputs))

    def binary_cross_entropy(self, y, y_hat):
        m = y.shape[0]
        bce = -1 / m * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        return bce

    def binary_cross_entropy_gradient(self, y, y_hat):
        m = y.shape[0]
        bce_gradient = -1 / m * (y / y_hat - (1 - y) / (1 - y_hat))
        return bce_gradient

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def feedforward(self, X):
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self.sigmoid(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        y_hat = self.sigmoid(z2)
        return z1, a1, z2, y_hat

    def backpropagation(self, X, y, z1, a1, z2, y_hat):
        m = X.shape[0]
        dz2 = self.binary_cross_entropy_gradient(y, y_hat)
        dW2 = 1 / m * np.dot(a1.T, dz2)
        db2 = 1 / m * np.sum(dz2, axis=0)
        dz1 = np.dot(dz2, self.W2.T) * (a1 * (1 - a1))
        dW1 = 1 / m * np.dot(X.T, dz1)
        db1 = 1 / m * np.sum(dz1, axis=0)
        return dW1, db1, dW2, db2

    def update_parameters(self, dW1, db1, dW2, db2, lr):
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def train(self, X, y, epochs, lr):
        
        mem = np.zeros((int(epochs/10000), 2)) # Almacena el histórico de loss x epochs
        i = 0
        y = y.reshape(-1, 1)

        # Plot the decision boundary
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        h = .02  # step size in the mesh
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))                    
        Z = self.predict2(np.c_[xx.ravel(), yy.ravel()])        
        Z = Z.reshape(xx.shape)  
                     
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
        plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap=plt.cm.Spectral)
        plt.xlabel("$X_1$")
        plt.ylabel("$X_2$")
        plt.pause(0.1)

        # Entrenamiento
        for epoch in range(epochs):
            z1, a1, z2, y_hat = self.feedforward(X)
            dW1, db1, dW2, db2 = self.backpropagation(X, y, z1, a1, z2, y_hat)
            self.update_parameters(dW1, db1, dW2, db2, lr)
            if epoch % 10000 == 0:
                loss = self.binary_cross_entropy(y, y_hat)
                print(f"Epoch: {epoch}, loss: {loss}")
                mem[i, :] = [epoch, loss]
                i = i + 1
                
                # Update decision boundary
                Z = self.predict2(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                
                # Update plot
                plt.clf()
                plt.title(f"Entrenamiento de Clasificación en Epoch {epoch}")
                plt.contourf(xx, yy, Z, cmap='RdYlBu')                
                plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap=plt.cm.Spectral)
                plt.savefig('plots\Final de Entrenamiento.png')
                plt.pause(0.1)
                plt.draw()
         
        #print(mem)
        
        # Plot Loss x Epoch
        plt.clf()
        plt.scatter(mem[:, 0], mem[:, 1], color='blue')
        plt.plot(mem[:, 0], mem[:, 1], color='red') 
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.xlim(0, mem[:, 0].max() + 5000)  # Establece los límites del eje x 
        plt.ylim(0, mem[:, 1].max() + 0.5)
        plt.title("Loss x Epoch")
        plt.savefig('plots\Loss X Epoch.png')
        plt.show()

    def predict(self, X):
        _, _, _, y_hat = self.feedforward(X)
        return y_hat

    def predict2(self, X):  # Función para preparación de gráfico de contorno
        _, _, _, y_hat = self.feedforward(X)
        return np.round(y_hat)


#-------------------------------------------------------------------------------------------
# MAIN

# Generación del dataset 
np.random.seed(0)
X, y = make_circles(n_samples=200, noise=0.05)

# División del dataset en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Parámetros Iniciales
num_inputs = X_train.shape[1]
num_hidden = 50
num_outputs = 1
epochs = 1000000
lr = 0.02

# Generación de la RN
nn = SimpleNN(num_inputs, num_hidden, num_outputs)

#...........................................................
#ENTRENAMIENTO DEL MODELO
print("ETAPA DE ENTRENAMIENTO DEL MODELO:")

# Entrenamiento de la RN
nn.train(X_train, y_train, epochs=epochs, lr=lr)

# Generación de Predicción
y_pred_train = nn.predict(X_train)
#print(y_pred_train)

# Métricas del entrenamiento

print("Matriz de Confusión - Entrenamiento ->")
vp, vn, fp, fn, m = matriz_confusion(np.round(y_pred_train), y_train)

print("Métricas Entrenamiento ->")
train_accuracy, train_precision, train_recall, train_f1score = calculo_metricas(vp, vn, fp, fn, m)

print(f"Training Accuracy: {train_accuracy * 100:.2f}% de las predicciones son verdaderas")
print(f"Training Precision: {train_precision * 100:.2f}% de las instancias clasificadas como positivas son correctas (En la predicción)")
print(f"Training Recall: {train_recall * 100:.2f}% de las instancias que realmente son positivas son correctas (En el Dataset)")
print(f"Training F1 Score: {train_f1score:.2f} / 1 en Precisión y Exhaustividad")

curvas_roc_pr(y_pred_train, y_train)


#...........................................................
#EVALUACIÓN DEL MODELO
print("ETAPA DE EVALUACIÓN DEL MODELO:")

# Evaluación
y_pred_test = nn.predict(X_test)

# Métricas del evaluación

print("Matriz de Confusión - Evaluación ->")
vp, vn, fp, fn, m = matriz_confusion(np.round(y_pred_test), y_test)

print("Métricas Evaluación ->")
test_accuracy, test_precision, test_recall, test_f1score = calculo_metricas(vp, vn, fp, fn, m)

print(f"Test Accuracy: {test_accuracy * 100:.2f}% de las instancias evaluadas son verdaderas")
print(f"Test Precision: {test_precision * 100:.2f}% de las instancias clasificadas como positivas son correctas (En el test)")
print(f"Test Recall: {test_recall * 100:.2f}% de las instancias que realmente son positivas son correctas (En el Dataset)")
print(f"Test F1 Score: {test_f1score:.2f} / 1 en Precisión y Exhaustividad")

curvas_roc_pr(y_pred_test, y_test)


