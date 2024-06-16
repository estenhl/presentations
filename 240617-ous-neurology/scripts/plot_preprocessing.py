import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import gray2rgb
from skimage.io import imsave
from matplotlib import colormaps



root = '/Applications/freesurfer/7.3.1/subjects/bert/mri'
slice = 125
orig = nib.load(f'{root}/orig/001.mgz').get_fdata()[slice]
rotated = nib.load(f'{root}/orig.mgz').get_fdata()[slice]
mask = nib.load(f'{root}/brainmask.mgz').get_fdata()[slice]

orig_seg = nib.load('scripts/data/bert/mri/aparc.DKTatlas+aseg.deep.mgz').get_fdata()[slice]
seg = orig_seg.copy()
unique = np.unique(seg).tolist()

for i in range(len(seg)):
    for j in range(len(seg[i])):
        seg[i][j] = unique.index(seg[i][j])


cm = plt.get_cmap('tab20')
seg = cm(seg.astype(int))
print(seg[0,0])
seg[orig_seg == 0] = [0, 0, 0, 1]
print(seg[0,0])
seg = seg[...,:3]
seg = seg * 255
seg = seg.astype(np.uint8)

fig, ax = plt.subplots(2, 2)
ax = ax.ravel()
ax[0].imshow(orig)
ax[1].imshow(rotated)
ax[2].imshow(mask)
ax[3].imshow(seg)
plt.show()

imsave('data/preprocessing_orig.png', orig.astype(np.uint8))
imsave('data/preprocessing_rotated.png', rotated.astype(np.uint8))
imsave('data/preprocessing_mask.png', mask.astype(np.uint8))
print(np.amax(seg))
imsave('data/preprocessing_seg.png', seg)
