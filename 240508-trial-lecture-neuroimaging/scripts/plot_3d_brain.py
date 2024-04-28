import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D


# Example data
data = nib.load('/Applications/freesurfer/7.3.1/subjects/bert/mri/T1.mgz').get_fdata()

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

from scipy.ndimage import zoom
data = zoom(data, 0.5)
max_x, max_y, max_z = data.shape
mid_x, mid_y, mid_z = np.asarray(data.shape) // 2
data = np.swapaxes(data, 0, 1)
data = np.swapaxes(data, 0, 2)
data = data[::,::,::-1]
data[:,0,0] = 255
data[max_x - 1, 0, 0:mid_z] = 255
data[mid_x-1:, 0, mid_z-1] = 255
data[max_x-1, :mid_y, mid_z-1] = 255
data[mid_x-1,0,mid_z:] = 255
data[max_x-1,mid_y,mid_z-1:] = 255
data[mid_x-1,:mid_y,max_z-1] = 255
data[mid_x-1:,mid_y,max_z-1] = 255
data[max_x-1:,:,0] = 255
data[0,:,max_z-1] = 255
data[0,0,:] = 255
data[:,max_y-1,max_z-1] = 255
data[max_x-1,max_y-1,:] = 255
data[:max_x,0,max_z-1] = 255
data[max_x-1,mid_y:,max_z-1] = 255
print(data.shape)

norm = plt.Normalize(data.min(), data.max())
colormap = plt.cm.gray

# Prepare face colors with an alpha channel
facecolors = colormap(norm(data))
alpha = np.ones(data.shape, dtype=bool)
alpha[16:, :16, 16:] = False

ax.voxels(alpha, facecolors=facecolors, shade=False)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
