import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

X1 = np.random.normal(0, 0.1, 20)
y1 = np.random.normal(0, 0.1, 20)

X2 = np.random.normal(0, 0.25, 20)
y2 = np.random.normal(0, 0.25, 20)

X3 = np.random.normal(0.25, 0.1, 20)
y3 = np.random.normal(0.25, 0.1, 20)

X4 = np.random.normal(0.25, 0.25, 20)
y4 = np.random.normal(0.25, 0.25, 20)

df = pd.DataFrame({
    'X1': X1,
    'y1': y1,
    'X2': X2,
    'y2': y2,
    'X3': X3,
    'y3': y3,
    'X4': X4,
    'y4': y4
})
df.to_csv('data/bias-variance.csv', index=False)

fig, ax = plt.subplots(2, 2)
ax[0,0].scatter(X1, y1)
ax[0,1].scatter(X2, y2)
ax[1,0].scatter(X3, y3)
ax[1,1].scatter(X4, y4)

for axis in ax.ravel():
    axis.set_xlim(-1, 1)
    axis.set_ylim(-1, 1)

plt.show()

X = np.linspace(0, 10, 100)
periodic = (np.cos(X) + 1) * 5
noisy_periodic = periodic + np.random.normal(0, 2, 100)
noisy_periodic2 = periodic + np.random.normal(0, 2, 100)

plt.scatter(X, noisy_periodic)
plt.plot(X, periodic)
plt.show()

noisy = X + np.random.normal(0, 1, 100)
noisy2 = X + np.random.normal(0, 1, 100)

plt.plot(X, noisy)
plt.show()

df = pd.DataFrame({
    'X': X,
    'periodic': periodic,
    'noisy_periodic': noisy_periodic,
    'noisy': noisy,
    'noisy_periodic2': noisy_periodic2,
    'noisy2': noisy2
})
df.to_csv('data/functions.csv', index=False)
