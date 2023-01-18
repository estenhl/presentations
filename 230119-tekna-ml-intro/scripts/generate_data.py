import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


X = np.random.randint(30, 90, 300)
y = 82031 * X + 706495 + np.random.normal(0, 200000, 300)

df = pd.DataFrame({'X': X, 'y': y})
df.to_csv(os.path.join('data', 'data.csv'), index=False)

plt.scatter(X, y)
plt.show()
