import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error

df = pd.read_csv('winequality-white.csv', sep=';')
volatile = df['volatile acidity'].values.reshape(-1, 1)
citric = df['citric acid'].values.reshape(-1, 1)
residual = df['residual sugar'].values.reshape(-1, 1)
chlorides = df['chlorides'].values.reshape(-1, 1)
free_sulfur = df['free sulfur dioxide'].values.reshape(-1, 1)
total_sulfur = df['total sulfur dioxide'].values.reshape(-1, 1)
density = df['density'].values.reshape(-1, 1)
pH = df['pH'].values.reshape(-1, 1)
sulphate = df['sulphates'].values.reshape(-1, 1)
alcohol = df['alcohol'].values.reshape(-1, 1)

quality = df['quality'].values.reshape(-1, 1)

#Generate random data
print(f"x = {volatile[:5, :]}")

print(f"y = {citric[:5, :]}")

x_b = np.hstack([np.ones((volatile.shape[0], 1)), volatile])
x_b = np.hstack([np.ones((citric.shape[0], 1)), volatile])
x_b = np.hstack([np.ones((residual.shape[0], 1)), volatile])
x_b = np.hstack([np.ones((chlorides.shape[0], 1)), volatile])
x_b = np.hstack([np.ones((free_sulfur.shape[0], 1)), volatile])
x_b = np.hstack([np.ones((total_sulfur.shape[0], 1)), volatile])
x_b = np.hstack([np.ones((density.shape[0], 1)), volatile])
x_b = np.hstack([np.ones((pH.shape[0], 1)), volatile])
x_b = np.hstack([np.ones((sulphate.shape[0], 1)), volatile])
x_b = np.hstack([np.ones((alcohol.shape[0], 1)), volatile])


print(f"x_b = {x_b[:5, :]}")

Theta_best = np.linalg.inv(x_b.T @ x_b) @ (x_b.T) @ quality
print(f"Theta_best = {Theta_best}")

y_predict = x_b @ Theta_best    #Prediction.

#mse = mean_squared_error(y_test, y_predict)

#plot data
plt.scatter(volatile, quality, label='Datos')
plt.plot(volatile, y_predict, color='red', label='Ajuste lineal')
plt.ylabel('volatile acidity')
plt.xlabel('citric acid')
plt.legend()
plt.title('Regresion Lineal')
#plt.show()
plt.savefig('regresionlinealwine.png')