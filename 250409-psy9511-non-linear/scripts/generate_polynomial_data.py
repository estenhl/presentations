import numpy as np

x = [0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75]
y = [0.5, 0.8, 0.3, 0.2, 0.55, 0.65, 0.1]

print(np.polyfit(x, y, 5))
