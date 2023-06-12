import numpy as np
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json
from pathlib import Path

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
    
    #print(f"TOTAL: Instancias {m} / Instancias Verdaderas: {v} / Instancias Falsas: {f}")
    
    #print(f"VN: {vn}   FP: {fp}")
    #print(f"FN: {fn}   VP: {vp}")

    return vp, vn, fp, fn, m


def calculo_metricas(vp, vn, fp, fn, m):
    
    # Balanceo del Dataset
    print(f"Balanceo de Dataset: N {vn + fp:.2f} =~ P {vp + fn:.2f}")

    # Accuracy - Exactitud
    accuracy = (vn + vp) / m
    
    # Precision - Precisión
    if vp == 0 :
        precision = 0
    else:        
        precision = vp / (vp + fp)
        
    # Recall - Exhaustividad
    if vp == 0 :
        recall = 0
    else:        
        recall = vp / (vp + fn)
    
    # F1 Score - Puntaje F1
    if precision == 0 or recall == 0 :
        f1score = 0
    else:
        f1score = (2 * precision * recall) / (precision + recall)
        
    return accuracy, precision, recall, f1score


def curvas_roc_pr(y_pred, y, etapa):
    
    # Generar matriz ROC-PR para los gráficos de las Curvas ROC y PR
    roc_pr = np.zeros((9, 8))
    i = 0
    paso = 0.1
    umbral = 0

    while i < roc_pr.shape[0]:
        umbral = np.round(umbral + paso, 1)
        roc_pr[i, 0] = umbral
        #print(f"Umbral: {umbral} ->: ")
        y_new = sintonizar_umbral(y_pred, umbral)
        roc_pr[i, 1], roc_pr[i, 2], roc_pr[i, 3], roc_pr[i, 4], roc_pr[i, 5] = matriz_confusion(y_new, y)
        
        if roc_pr[i, 1] == 0:
            roc_pr[i, 6] = 0  # Precision
            roc_pr[i, 7] = 0  # Recall
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
    plt.ylim(-0.5, roc_pr[:, 1].max() + 5)  # Establece los límites del eje y
    for j, label in enumerate(roc_pr[:, 0]):
        plt.text(roc_pr[j, 3], roc_pr[j, 1], label)
    plt.title(f"Curva ROC {etapa}")
    ruta = "plots\Curva ROC " + etapa + ".png"
    plt.savefig(ruta)
    plt.show()

    # Plot Curva PR
    plt.clf()
    plt.scatter(roc_pr[:, 7], roc_pr[:, 6], color='blue', label = roc_pr[:, 0])
    plt.plot(roc_pr[:, 7], roc_pr[:, 6], color='red') 
    plt.xlabel('Recall / Exhaustividad')
    plt.ylabel('Precision / Precisión')
    plt.xlim(roc_pr[:, 7].min() - 0.2, roc_pr[:, 7].max() + 0.5)  # Establece los límites del eje x 
    plt.ylim(roc_pr[:, 6].min() - 0.2, roc_pr[:, 6].max() + 0.5)  # Establece los límites del eje y
    for j, label in enumerate(roc_pr[:, 0]):
        plt.text(roc_pr[j, 7], roc_pr[j, 6], label)
    plt.title(f"Curva PR {etapa}")
    ruta = "plots\Curva PR " + etapa + ".png"
    plt.savefig(ruta)
    plt.show()
    
    
def encontrar_mejor_modelo(archivo_pesos):
    
    # Cargar archivo
    with open(archivo_pesos, 'r') as archivo:
        pesos_leidos = json.load(archivo)
       
    # Buscar el mejor loss
    for registro in pesos_leidos:
        loss_ant = registro['loss']
        break
     
    for registro in pesos_leidos:
        loss_new = registro['loss']
        if loss_new > loss_ant :
            break
        else:
            loss_ant = registro['loss']
        
    mejor_loss = loss_ant
     
    # Buscar el mejor modelo
    pesos_leidos = None
        
    with open(archivo_pesos, 'r') as archivo:
        pesos_leidos = json.load(archivo)
    
    for registro in pesos_leidos:
        if registro['loss'] == mejor_loss:
            registro_mejor_modelo = registro
            break
        
    # Extraer las variables del mejor modelo
    epoch = registro_mejor_modelo['epoch']
    W1 = np.array(registro_mejor_modelo['W1'])
    W2 = np.array(registro_mejor_modelo['W2'])
    loss = registro_mejor_modelo['loss']
        
    return epoch, W1, W2, loss

def guardar_modelo(epoch, W1, W2, loss):
        
        archivo_modelo = 'model\modelo.json'
        # Convertir W1 y W2 en listas
        W1_lista = W1.tolist()
        W2_lista = W2.tolist()

        # Crear un diccionario con las variables
        modelo = {
            'epoch' : epoch,
            'W1' : W1_lista,
            'W2' : W2_lista,
            'loss' : loss
        }

        # Borrar archivo existente
        ruta_archivo = Path(archivo_modelo)

        if ruta_archivo.exists():
            ruta_archivo.unlink()
        
        # Guardar el diccionario en un archivo JSON
        with open(archivo_modelo, 'w') as archivo:
            json.dump([modelo], archivo)


    
#-------------------------------------------------------------------------------------------
# ARQUITECTURA DE LA RED NEURONAL

class SimpleNN:
    def __init__(self, num_inputs, num_hidden, num_outputs):
        self.W1 = np.random.randn(num_inputs, num_hidden)
        self.b1 = np.zeros((1, num_hidden))
        self.W2 = np.random.randn(num_hidden, num_outputs)
        self.b2 = np.zeros((1, num_outputs))
        

    def iniciar_archivo_pesos(self, archivo_pesos, epoch, loss):
        
        #Inicializa el archivo en donde se guardarán los modelos por epoch, con los pesos iniciales.
        
        # Borrar archivo existente
        ruta_archivo = Path(archivo_pesos)

        if ruta_archivo.exists():
            ruta_archivo.unlink()
             
        # Convertir self.W1 y self.W2 en listas
        W1_lista = self.W1.tolist()
        W2_lista = self.W2.tolist()

        # Crear un diccionario con las variables
        pesos = {
            'epoch' : epoch,
            'W1' : W1_lista,
            'W2' : W2_lista,
            'loss' : loss
        }

        # Guardar el diccionario en un archivo JSON
        with open(archivo_pesos, 'w') as archivo:
            json.dump([pesos], archivo)
    
    
    def guardar_pesos(self, archivo_pesos, epoch, loss):
        # Convertir self.W1 y self.W2 en listas
        W1_lista = self.W1.tolist()
        W2_lista = self.W2.tolist()

        # Crear un diccionario con las variables
        pesos = {
            'epoch' : epoch,
            'W1' : W1_lista,
            'W2' : W2_lista,
            'loss' : loss
        }

        # Leer los datos existentes del archivo JSON
        with open(archivo_pesos, 'r') as archivo:
            pesos_existentes = json.load(archivo)
        
        pesos_existentes.append(pesos)
                
        # Guardar el diccionario en un archivo JSON
        with open(archivo_pesos, 'w') as archivo:
            json.dump(pesos_existentes, archivo)
            
       
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
        
    def train(self, X, y, epochs, lr, archivo):
        
        mem = np.zeros((int(epochs/10000), 2)) # Matriz para almacenar el histórico de loss x epochs
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

        # Guardar pesos iniciales
        self.iniciar_archivo_pesos(archivo, -1, epochs)
        
        # Entrenamiento
        for epoch in range(epochs):
            z1, a1, z2, y_hat = self.feedforward(X)
            dW1, db1, dW2, db2 = self.backpropagation(X, y, z1, a1, z2, y_hat)
            self.update_parameters(dW1, db1, dW2, db2, lr)
            
            if epoch % 10000 == 0:
                loss = self.binary_cross_entropy(y, y_hat)
                self.guardar_pesos(archivo, epoch, loss)
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
                plt.savefig('plots\Gráfico Final del Entrenamiento.png')
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
        plt.savefig('plots\Gráfico Loss X Epoch.png')
        plt.show()

    def predict(self, X):
        _, _, _, y_hat = self.feedforward(X)
        return y_hat

    def predict2(self, X):  # Función para preparación de gráfico de contorno
        _, _, _, y_hat = self.feedforward(X)
        return np.round(y_hat)

    def feedforward_eval(self, X, W1, W2):   # Función forward para la evaluación del mejor modelo
        z1 = np.dot(X, W1) + self.b1
        a1 = self.sigmoid(z1)
        z2 = np.dot(a1, W2) + self.b2
        y_hat = self.sigmoid(z2)
        return y_hat
    
    def predict_eval(self, X, W1, W2):   # Función predict para la evaluación del mejor modelo
        y_hat = self.feedforward_eval(X, W1, W2)
        return y_hat
    
    
#-------------------------------------------------------------------------------------------
# MAIN

# Especificar la ruta y el nombre del archivo JSON donde se guardarán los pesos
archivo_json = 'model\pesos.json'

# Generación del dataset 
np.random.seed(0)
X, y = make_circles(n_samples=200, noise=0.05)

# División del dataset en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Parámetros Iniciales
num_inputs = X_train.shape[1]
num_hidden = 92
num_outputs = 1
epochs = 1000000 #1000000
lr = 0.02

# Generación de la RN
nn = SimpleNN(num_inputs, num_hidden, num_outputs)

#...........................................................
#ENTRENAMIENTO DEL MODELO
print("\nETAPA DE ENTRENAMIENTO DEL MODELO:\n")

# Entrenamiento de la RN
nn.train(X_train, y_train, epochs=epochs, lr=lr, archivo = archivo_json)

# Generación de la predicción
y_pred_train = nn.predict(X_train)
#print(y_pred_train)

# Métricas del entrenamiento
print("\nMatriz de Confusión - Entrenamiento ->")
vp, vn, fp, fn, m = matriz_confusion(np.round(y_pred_train), y_train)

print(f"VN: {vn}   FP: {fp}")
print(f"FN: {fn}   VP: {vp}")
print(f"TOTAL: Instancias {m}")

print("\nMétricas Entrenamiento ->")
train_accuracy, train_precision, train_recall, train_f1score = calculo_metricas(vp, vn, fp, fn, m)

print(f"Training Accuracy: {train_accuracy * 100:.2f}% de las predicciones son verdaderas")
print(f"Training Precision: {train_precision * 100:.2f}% de las instancias clasificadas como positivas son correctas (En la predicción)")
print(f"Training Recall: {train_recall * 100:.2f}% de las instancias que realmente son positivas son correctas (En el Dataset)")
print(f"Training F1 Score: {train_f1score:.2f} / 1 en Precisión y Exhaustividad")

curvas_roc_pr(y_pred_train, y_train, "Entrenamiento")


#...........................................................
#EVALUACIÓN DEL MODELO
print("\nETAPA DE EVALUACIÓN DEL MODELO:\n")

# Extraer el mejor modelo
epoch, W1, W2, loss = encontrar_mejor_modelo(archivo_json)

print(f"Mejor Modelo: Epoch {epoch}")

guardar_modelo(epoch, W1, W2, loss)

# Evaluación del modelo
y_pred_test = nn.predict_eval(X_test, W1, W2)

# Métricas de la evaluación
print("\nMatriz de Confusión - Evaluación ->")
vp, vn, fp, fn, m = matriz_confusion(np.round(y_pred_test), y_test)

print(f"VN: {vn}   FP: {fp}")
print(f"FN: {fn}   VP: {vp}")
print(f"TOTAL: Instancias {m}")

print("\nMétricas Evaluación ->")
test_accuracy, test_precision, test_recall, test_f1score = calculo_metricas(vp, vn, fp, fn, m)

print(f"Test Accuracy: {test_accuracy * 100:.2f}% de las instancias evaluadas son verdaderas")
print(f"Test Precision: {test_precision * 100:.2f}% de las instancias clasificadas como positivas son correctas (En el test)")
print(f"Test Recall: {test_recall * 100:.2f}% de las instancias que realmente son positivas son correctas (En el Dataset)")
print(f"Test F1 Score: {test_f1score:.2f} / 1 en Precisión y Exhaustividad")

curvas_roc_pr(y_pred_test, y_test, "Evaluación")




