import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsRegressor


X = np.linspace(0, 1, 100)
y = X ** 2 + np.random.normal(0, 0.1, 100)

simple = np.polyfit(X, y, 1)
medium = np.polyfit(X, y, 2)
complex = np.polyfit(X, y, 20)

idx = np.argsort(X)
X = X[idx]
y = y[idx]

k1 = KNeighborsRegressor(n_neighbors=1)
k1.fit(X[:, None], y)
k30 = KNeighborsRegressor(n_neighbors=30)
k30.fit(X[:, None], y)
k100 = KNeighborsRegressor(n_neighbors=100)
k100.fit(X[:, None], y)
approx_fit = np.polyfit(X, k30.predict(X[:, None]), 4)
approx = np.polyval(approx_fit, X)
print(f'Mean: {np.mean(y):.2f}')
print(f'Coefficients: {np.round(approx_fit, 2)}')

df = pd.DataFrame({
    'X': X,
    'y': y,
    'simple': np.polyval(simple, X),
    'medium': np.polyval(medium, X),
    'complex': np.polyval(complex, X),
    'k1': k1.predict(X[:, None]),
    'k30': k30.predict(X[:, None]),
    'k100': k100.predict(X[:, None]),
    'approx': approx
})

df.to_csv(os.path.join('data', 'flexibility.csv'), index=False)

fig, ax = plt.subplots(2, 1)

ax[0].scatter(X, y)
ax[0].plot(df['X'], df['simple'], label='Simple')
ax[0].plot(df['X'], df['medium'], label='Medium')
ax[0].plot(df['X'], df['complex'], label='Complex')


ax[1].scatter(X, y)
ax[1].plot(df['X'], df['k1'], label='K=1')
ax[1].plot(df['X'], df['k30'], label='K=30')
ax[1].plot(df['X'], df['k100'], label='K=100')
ax[1].plot(df['X'], approx, label='Approximation')

plt.show()
