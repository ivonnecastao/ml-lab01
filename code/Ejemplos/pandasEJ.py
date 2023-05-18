import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = [1, 2, 3, 4, 5]
index = ['a', 'b', 'c', 'd', 'e']
serie = pd.Series(data, index=index)
print(serie)
print(serie['c'])

s2 = serie * 2
print(s2)

sb = serie > 2
s3 = serie[sb]
print(sb)
print(s3)

data = {
    'Columna1': [1, 2, 3, 4, 5],
    'Columna2': ['a', 'b', 'c', 'd', 'e'],
    'Columna3': [1.1, 2.2, 3.3, 4.4, 5.5]
}
df = pd.DataFrame(data)
print(df)
print(df.head(6))
print(df.tail(4))
print(df.describe())
print(df.info())

print(df['Columna2'])
