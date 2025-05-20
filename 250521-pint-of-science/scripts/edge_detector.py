import matplotlib.pyplot as plt
from skimage.io import imread, imsave
import nilearn.plotting
import numpy as np

img = imread('data/edgedetector.png', as_gray=True)
img = img - np.amin(img)
img = img / np.amax(img)
img = 1 - img

plt.imshow(img, cmap = 'cold_hot', clim=(-1, 1))
plt.savefig('data/edgedetector.png')
