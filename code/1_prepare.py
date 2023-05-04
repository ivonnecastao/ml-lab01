import pandas as pd
import matplotlib.pyplot as plt
import os

# Read in the dataset
input_dir = 'data/downloaded'
output_dir = 'data/prepared'
os.makedirs(output_dir, exist_ok=True)

# read the file with the detected encoding
df = pd.read_csv(input_dir + '/bottle.csv', encoding='iso-8859-1')
# print(df.head())
df = df[['T_degC','Salnty']]
df = df.dropna()
temperature = df[['T_degC']].values
salinity = df[['Salnty']].values
plt.plot(salinity, temperature, 'o', markersize=0.1)
plt.ylabel('Temperature (°C)')
plt.xlabel('Salinity')
plt.savefig(output_dir + '/TempSal.png')
# plt.show()
df.to_csv(output_dir + '/TempSal.csv', index=False)
