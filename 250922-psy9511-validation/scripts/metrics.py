import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


df = pd.read_csv('data/Auto.csv')
print(f'X: {np.amin(df["horsepower"]):.2f}-{np.amax(df["horsepower"]):.2f}')
print(f'y: {np.amin(df["mpg"]):.2f}-{np.amax(df["mpg"]):.2f}')

model = LinearRegression()
model.fit(df[['horsepower']], df['mpg'])
print(f'MSE model: {model.intercept_:.3f} + {model.coef_[0]:.3f} * x')
predictions = model.predict(df[['horsepower']])
print(f'MSE: {mean_squared_error(df["mpg"], predictions)}')

mae_model = sm.QuantReg(df['mpg'], sm.add_constant(df['horsepower']))
res = mae_model.fit(q=0.5)
mse_preds = model.predict(df[['horsepower']])
mae_preds = res.predict(sm.add_constant(df['horsepower']))
print(f'MAE model: {res.params['const']:.3f} + {float(res.params['horsepower']):.3f}')
