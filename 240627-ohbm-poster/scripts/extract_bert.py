import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from skimage.io import imsave

path = '/Applications/freesurfer/7.3.1/subjects/bert/mri/brain.mgz'
image = nib.load(path).get_fdata()
image = image[:,108]
image = np.rot90(image)
image = image[40:-20,50:-50]
image = image * 256 / np.amax(image)
image = image.astype(np.uint8)

#plt.imshow(image)
#plt.show()

imsave('data/bert.png', image)
