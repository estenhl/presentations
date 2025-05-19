import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature
from skimage.io import imread

img = imread('data/bird.png', as_gray=True)
edges = feature.canny(img, sigma=1.5)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = '')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()