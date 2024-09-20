import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MAX_STEPS = 100

X = np.arange(MAX_STEPS)
train_y = MAX_STEPS / X + np.random.normal(0, 2.5, MAX_STEPS)
val_y = MAX_STEPS / X + np.random.normal(0, 2.5, MAX_STEPS) + (X ** 2) / (MAX_STEPS * 2)

df = pd.DataFrame({'epoch': X, 'train': train_y, 'val': val_y})
df.to_csv(os.path.join('data', 'losses.csv'), index=False)

plt.plot(X, train_y)
plt.plot(val_y)
plt.show()
