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
