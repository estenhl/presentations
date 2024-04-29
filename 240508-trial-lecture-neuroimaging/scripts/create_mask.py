import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import gray2rgb
from skimage.io import imsave


img = nib.load('scripts/data/bert/mri/orig/001.mgz').get_fdata()
mask = nib.load('scripts/data/bert/mri/aparc.DKTatlas+aseg.deep.mgz').get_fdata()
binary_mask = np.zeros_like(mask)
binary_mask[mask == 17] = 1
binary_mask[mask == 53] = 1

summed = np.sum(binary_mask, axis=(0, 1))
print(np.argmax(summed))

img = np.rot90(img[:,:,121], 3)
img = gray2rgb(img).astype(np.uint8)
binary_mask = np.rot90(binary_mask[:,:,121], 3)
imsave('data/bert_coronal.png', img)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(img)

print((binary_mask == 1).shape)
print(img.shape)
img[binary_mask == 1] = np.asarray((255, 0, 0))
imsave('data/bert_coronal_marked.png', img)

ax[1].imshow(img)

plt.show()
