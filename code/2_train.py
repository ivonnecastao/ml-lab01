import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Read in the dataset
input_dir = 'data/prepared'
output_dir = 'model'
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(input_dir + '/TempSal.csv')
temperature = df[['T_degC']].values
salinity = df[['Salnty']].values

# Split the data into training/testing sets and create the model
X_train, X_test, y_train, y_test = train_test_split(salinity, temperature, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = model.predict(X_test)

# The coefficients
print('Coefficients: ', model.coef_)
# The mean squared error
print('Mean squared error: %.2f'
        % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
        % model.score(X_test, y_test))

# Plot outputs
plt.plot(X_test, y_test, 'o', markersize=0.1)
plt.plot(X_test, y_pred, color='red')
plt.ylabel('Temperature (°C)')
plt.xlabel('Salinity')
plt.savefig(output_dir + '/TempSal.png')
# plt.show()

# Save the model
import joblib
joblib.dump(model, output_dir + '/TempSal.pkl')
