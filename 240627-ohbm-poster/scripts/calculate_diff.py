import os
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from scipy.ndimage import gaussian_filter
from skimage.io import imsave


averages = os.path.join('data', 'averages')

fig, ax = plt.subplots(3, 3)
ax = ax.ravel()

for i, diagnosis in enumerate(os.listdir(averages)):
    patients = nib.load(os.path.join(averages, diagnosis,
                                     f'{diagnosis}.nii.gz'))
    patients = patients.get_fdata()
    patients = patients[:,109]
    patients = np.rot90(patients)


    controls = nib.load(os.path.join(averages, diagnosis, 'HC.nii.gz'))
    controls = controls.get_fdata()
    controls = controls[:,109]
    controls = np.rot90(controls)

    template = nib.load(os.path.join(averages, diagnosis, 'template.nii.gz'))
    template = template.get_fdata()
    template = template[:,109]
    template = np.rot90(template)
    template = template / np.amax(template)

    diff = patients - controls
    diff = gaussian_filter(diff, sigma=(3, 3), order=0)
    diff[template == 0] = 0
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
    alpha = alpha ** 1.1
    alpha = alpha / np.amax(alpha)
    alpha = np.expand_dims(alpha, -1)
    image = (1 - alpha) * template + alpha * diff
    image = image.astype(np.uint8)

    print(np.amax(alpha))

    ax[i].imshow(image)
    ax[i].set_title(diagnosis)
    imsave(os.path.join(averages, diagnosis, 'diff.png'), image)
plt.show()
