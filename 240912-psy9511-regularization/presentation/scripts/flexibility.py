import matplotlib.pyplot as plt
import numpy as np


X = np.random.uniform(0, 1, 100)
y = X ** 2 + np.random.normal(0, 0.1, 100)

simple = np.polyfit(X, y, 1)
medium = np.polyfit(X, y, 2)
complex = np.polyfit(X, y, 20)

df = {
    'X': np.sort(X),
    'simple': np.polyval(simple, np.sort(X)),
    'medium': np.polyval(medium, np.sort(X)),
    'complex': np.polyval(complex, np.sort(X))
}


plt.scatter(X, y)
plt.plot(df['X'], df['simple'], label='Simple')
plt.plot(df['X'], df['medium'], label='Medium')
plt.plot(df['X'], df['complex'], label='Complex')
plt.show()
