import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#Generate random data
x = np.random.rand(100, 1)
print(f"x = {x[:5, :]}")

y = 2 * x + np.random.rand(100, 1)
print(f"y = {y[:5, :]}")

x_b = np.hstack([np.ones((x.shape[0], 1)), x])
print(f"x_b = {x_b[:5, :]}")

Theta_best = np.linalg.inv(x_b.T @ x_b) @ (x_b.T) @ y
print(f"Theta_best = {Theta_best}")

y_predict = x_b @ Theta_best    #Prediction.

#plot data
plt.scatter(x, y, label='Datos')
plt.plot(x, y_predict, color='red', label='Ajuste lineal')
plt.xlabel('Variable independiente (X)')
plt.ylabel('Variable dependiente (Y)')
plt.legend()
plt.title('Regresion Lineal')
#plt.show()
plt.savefig('regresionlineal.png')



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print(f"x_train.shape = {x_train.shape}")
print(f"x_test.shape = {x_test.shape}")

model = LinearRegression()

model.fit(x_train, y_train)
y_predict = model.predict(x_test)

mse = mean_squared_error(y_test, y_predict)

#plot data
plt.scatter(x, y, label='Datos')
plt.plot(x_test, y_predict, color='red', label='Ajuste lineal')
plt.xlabel('Variable independiente (X)')
plt.ylabel('Variable dependiente (Y)')
plt.legend()
plt.title('Regresion Lineal con SKLearn')
#plt.show()
plt.savefig('plots\Regresionlinealsklearn.png')

