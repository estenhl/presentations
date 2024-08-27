
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, LogisticRegression

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

X = np.asarray([[36, 25, 26, 32, 28, 29, 29.5, 21,
                 15, 17, 21, 14, 14.5, 16, 19, 24, 27]]).T
y = np.asarray([[1, 1, 1, 1, 1, 1, 1, 1,
                 0, 0, 0, 0, 0, 0, 0, 0, 0]]).T
model = LinearRegression()
model.fit(X, y)
print(f'Linear classification intercept: {model.intercept_}, coef: {model.coef_}')

model = LogisticRegression()
model.fit(X, y)
print(f'Logistic classification intercept: {model.intercept_}, coef: {model.coef_}')
x = np.linspace(10, 40, 100)
y = model.predict_proba(x.reshape(-1, 1))[:, 1]
plt.plot(x, y)
plt.show()
