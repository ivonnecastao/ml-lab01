import numpy as np
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Generate a dataset and plot it
np.random.seed(0)
X, y = make_circles(n_samples=200, noise=0.05)

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

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
        # dz2 = y_hat - y
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
        y = y.reshape(-1, 1)

        # Plot the decision boundary
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        h = .02  # step size in the mesh
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)        
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
        plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap=plt.cm.Spectral)
        plt.xlabel("$X_1$")
        plt.ylabel("$X_2$")
        plt.pause(0.1)

        for epoch in range(epochs):
            z1, a1, z2, y_hat = self.feedforward(X)
            dW1, db1, dW2, db2 = self.backpropagation(X, y, z1, a1, z2, y_hat)
            self.update_parameters(dW1, db1, dW2, db2, lr)
            if epoch % 100 == 0:
                loss = self.binary_cross_entropy(y, y_hat)
                print(f"Epoch: {epoch}, loss: {loss}")
                # Update decision boundary
                Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                # Update plot
                plt.clf()
                plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
                plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap=plt.cm.Spectral)
                plt.pause(0.1)
                plt.draw()



    def predict(self, X):
        _, _, _, y_hat = self.feedforward(X)
        return np.round(y_hat)

num_inputs = X_train.shape[1]
num_hidden = 50
num_outputs = 1
epochs = 10000
lr = 0.02

nn = SimpleNN(num_inputs, num_hidden, num_outputs)
nn.train(X_train, y_train, epochs=epochs, lr=lr)

y_pred_train = nn.predict(X_train)
train_accuracy = np.mean(y_pred_train == y_train.reshape(-1, 1))
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

y_pred_test = nn.predict(X_test)
test_accuracy = np.mean(y_pred_test == y_test.reshape(-1, 1))
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


