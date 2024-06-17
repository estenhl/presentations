import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.io import imsave


disorders = ['ANX', 'ASD']

template = nib.load(f'data/averages/{disorders[0]}/template.nii.gz').get_fdata()
masks = [nib.load(f'data/averages/{disorder}/{disorder}.nii.gz').get_fdata() \
         for disorder in disorders]
mask = masks[0] + masks[1]
print(mask)

template = template[:,109]
template = np.rot90(template)
template = template / np.amax(template)

mask = mask[:,109]
mask = np.rot90(mask)
mask[template == 0] = 0

diff = mask

#diff = gaussian_filter(mask, sigma=(3, 3), order=0)
diff[template == 0] = 0
diff[diff < 0] = (diff[diff < 0] / np.amin(diff) * -np.amax(diff))
max_value = np.amax(np.abs(diff))
alpha = np.abs(diff) / max_value
template = (template * 255.0).astype(np.uint8)
template = np.stack([template, template, template], axis=-1)
diff = diff / max_value
diff = (diff * 255.0)
r = np.maximum(diff, 0).astype(np.uint8)
g = np.zeros_like(r, np.uint8)
b = np.maximum(-diff, 0).astype(np.uint8)

diff = np.stack([r, g, b], axis=-1)
alpha = alpha ** 0.7
alpha = alpha / np.amax(alpha)
alpha = np.expand_dims(alpha, -1)
image = (1 - alpha) * template + alpha * diff
image = image.astype(np.uint8)
print(image.shape)

#imsave('data/mri.png', template)
#imsave('data/heatmap.png', image)


fig, ax = plt.subplots(2, 1)
ax[0].imshow(image)
ax[1].imshow(template)
plt.show()
