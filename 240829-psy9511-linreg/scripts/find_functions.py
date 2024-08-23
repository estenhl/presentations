import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

df = pd.read_csv(os.path.join('data', 'Auto.csv'))
X = df[['horsepower']]
y = df['mpg']
model = LinearRegression()
model.fit(X, y)
print(f'Linear regression intercept: {model.intercept_}, coef: {model.coef_[0]}')

print(f'Constant intercept: {np.mean(y)}')

df = pd.read_csv(os.path.join('data', 'Auto.csv'))
X = df[['horsepower']]
y = df['mpg']
model = LinearRegression()
model.fit(X, np.log(y))
print(f'Exponential intercept: {model.intercept_}, coef: {model.coef_[0]}')
