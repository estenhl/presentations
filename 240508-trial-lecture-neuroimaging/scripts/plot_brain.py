import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Generate a 3D numpy array of shape (4, 4, 4) with random grayscale values (0 to 255)
np.random.seed(0)  # For reproducibility
cube = np.random.randint(0, 256, (4, 4, 4))

# Step 2: Define the block to remove, instead of setting it to NaN, we'll skip plotting these values
# Removing the top front right corner: cube[0:2, 0:2, 2:4]
x, y, z = np.indices(np.array(cube.shape) + 1).astype(float)[:-1]
x, y, z = x.flatten(), y.flatten(), z.flatten()
values = cube.flatten()

# Exclude the block from plotting
exclude_block = (x < 2) & (y < 2) & (z >= 2) & (z < 4)
values[exclude_block] = -1  # Marking values to exclude as -1 for now

# Filter out the marked values
keep_mask = values != -1

x, y, z, values = x[keep_mask], y[keep_mask], z[keep_mask], values[keep_mask]

# Step 3: Visualization using Matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scat = ax.scatter(x, y, z, c=values, cmap='gray', marker='o', s=100, edgecolor='black')
fig.colorbar(scat, ax=ax, shrink=0.6, aspect=20, label='Grayscale value')

# Setting the labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Cube Visualization with Section Removed')

plt.show()
