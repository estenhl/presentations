# Create a vector of 100 standard normally distributed numbers and visualize them with a histogram.
import numpy as np
import matplotlib.pyplot as plt

data = np.random.randn(100)
plt.hist(data, bins=10)
plt.show()
