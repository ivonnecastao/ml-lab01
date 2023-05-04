import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#generate ramdom data
x = np.random.rand(100, 1)
print(f"x = {x[:5,:]}")

y = 2 * x + 3 +np.random.rand(100, 1)
print(f"y = {y[:5,:]}")

#Theta_best = np.linalg.inv(x.T @ x) @ x.T @ y #Esto es sin el bias
#print(f"Theta_best = {Theta_best}")  #Esto es sin el bias

#y_pred = x @ Theta_best  #Esto es sin el bias

x_b= np.hstack([np.ones((100,1)), x]) #Bias  multiplica el theta 0 para lograr el desplazamiento en y
print(f"x_b = {x_b[:5,:]}")
print(f"x_b.shape = {x_b.shape}")

Theta_best = np.linalg.inv(x_b.T @ x_b) @ x_b.T @ y  #y= x Theta1 + Theta 0
print(f"Theta_best = {Theta_best}")

y_pred = x_b @ Theta_best  #resultado, predicción

"""
# plot data
plt.scatter(x, y, label='Datos')
plt.plot(x, y_pred, color='red', label='Ajuste lineal') 
plt.xlabel('Variable independiente (x)')
plt.ylabel('Variable dependiente (y)')
plt.legend()
plt.title('Regresión Lineal con numpy')
plt.show()
plt.savefig('regresionlineal.png') """

np.ramdom.seed(42)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print(f"x_train.shape = {x_train.shape}")
print(f"x_test.shape = {x_test.shape}")

model= LinearRegression()  # modelo vacio sin entrenar

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)

print("coeficientes", model.coef_)
print("Intercepto", model.intercept_)
print("error cuadrático medio", mse)

plt.scatter(x, y, label='Datos')
plt.plot(x_test, y_pred, color='red', label='Ajuste lineal') 
plt.xlabel('Variable independiente (x)')
plt.ylabel('Variable dependiente (y)')
plt.legend()
plt.title('Regresión Lineal con sklearn')
plt.show()
plt.savefig('regresionlinealsklearn.png')