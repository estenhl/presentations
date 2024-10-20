import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


values = np.random.uniform(0, 1, (10, 10))
values[:6, :5] = 0.8
values[:6, 5:] = 0.9
values[6:, :5] = 0.5
values[6:, 5:] = 0.1

values += np.random.normal(0, 0.025, values.shape)

indices = list(np.ndindex(values.shape))
xs = np.asarray([i for i, _ in indices])
ys = np.asarray([j for _, j in indices])

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.scatter(xs, ys, values.flatten())
plt.show()

df = pd.DataFrame({'x': xs / 10.0, 'y': ys / 10.0, 'value': values.flatten()})
df.to_csv('data/treedata.csv', index=False)
