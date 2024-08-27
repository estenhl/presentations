
import os
import matplotlib.pyplot as plt
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

X = df[['horsepower']]
y = df['mpg']
model = LinearRegression()
model.fit(X, np.log(y))
print(f'Exponential intercept: {model.intercept_}, coef: {model.coef_[0]}')


X = df[['horsepower', 'weight']]
y = df['mpg']
model = LinearRegression()
model.fit(X, y)
print(f'Multiple regression intercept: {model.intercept_}, coef: {model.coef_}')

df['chevy'] = df['name'].apply(lambda x: 1 if 'chevrolet' in x else 0)
X = df[['horsepower', 'chevy']]
y = df['mpg']
model = LinearRegression()
model.fit(X, y)
print(f'Interaction term intercept: {model.intercept_}, coef: {model.coef_}')


horsepower = np.arange(np.amin(df['horsepower'].values), np.amax(df['horsepower'].values))

chevy_preds = model.predict(np.column_stack((horsepower, np.ones(len(horsepower)))))
other_preds = model.predict(np.column_stack((horsepower, np.zeros(len(horsepower)))))

plt.plot(horsepower, chevy_preds, c='red')
plt.plot(horsepower, other_preds, c='blue')
plt.show()

df['interaction'] = df['horsepower'] * df['chevy']
X = df[['horsepower', 'chevy', 'interaction']]
y = df['mpg']
model = LinearRegression()
model.fit(X, y)
print(f'Interaction term intercept: {model.intercept_}, coef: {model.coef_}')

chevy_preds = model.predict(np.column_stack((horsepower, np.ones(len(horsepower)), horsepower)))
other_preds = model.predict(np.column_stack((horsepower, np.zeros(len(horsepower)), np.zeros(len(horsepower)))))

plt.plot(horsepower, chevy_preds, c='red')
plt.plot(horsepower, other_preds, c='blue')
plt.show()
