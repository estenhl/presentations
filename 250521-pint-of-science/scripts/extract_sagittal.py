import os

import nibabel as nib
from skimage.io import imsave
import numpy as np


source = os.path.join('/home', 'esten', 'Downloads', 'images')
for i, filename in enumerate(os.listdir(source)):
    filepath = os.path.join(source, filename)
    img = nib.load(filepath)
    data = img.get_fdata()
    sagittal_slice = data[155 // 2]
    sagittal_slice /= sagittal_slice.max()
    sagittal_slice = (sagittal_slice * 255).astype('uint8')
    sagittal_slice = np.rot90(sagittal_slice)
    imsave(f'data/sagittal/{i}.png', sagittal_slice)
    #sagittal_img = nib.Nifti1Image(sagittal_slice, img.affine)
    #output_path = os.path.join(source, subject, session,
    #                            'sagittal_' + filename)
    #nib.save(sagittal_img, output_path)
