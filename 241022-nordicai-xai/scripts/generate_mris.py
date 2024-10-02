import argparse
import os
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from copy import copy
from matplotlib import colormaps
from skimage.io import imsave


def generate_mris(mri: str, heatmap: str):
    mri = nib.load(mri).get_fdata()
    heatmap = nib.load(heatmap).get_fdata()
    heatmap[mri == 0] = 0

    print(heatmap.shape)
    print(np.amin(heatmap))
    print(np.amax(heatmap))

    idx = [110, 90, 110]
    names = ['sagittal', 'axial', 'coronal']
    greys = colormaps['Greys_r']
    seismic = colormaps['seismic']

    fig, ax = plt.subplots(3, 3)

    for i, axis in enumerate(names):
        slices = [slice(None)] * len(idx)
        slices[i] = idx[i]

        image_slice = copy(mri[tuple(slices)])
        image_slice /= np.amax(image_slice)
        image_rgb = greys(image_slice)[...,:3]

        if axis != 'sagittal':
            image_rgb = np.rot90(image_rgb, 3)

        image_rgb = (image_rgb * 255.0).astype(np.uint8)
        imsave(os.path.join('data', f'mri_{axis}.png'), image_rgb)

        ax[i, 0].imshow(image_rgb)

        heatmap_slice = copy(heatmap[tuple(slices)])
        heatmap_slice = np.where(heatmap_slice < 0, heatmap_slice * 2, heatmap_slice)
        heatmap_slice /= np.amax(np.abs(heatmap_slice))
        heatmap_alpha = np.abs(heatmap_slice)
        heatmap_slice /= 2
        heatmap_slice += 0.5
        heatmap_rgb = seismic(heatmap_slice)[...,:3]

        if axis != 'sagittal':
            heatmap_rgb = np.rot90(heatmap_rgb, 3)
            heatmap_alpha = np.rot90(heatmap_alpha, 3)

        heatmap_rgb = (heatmap_rgb * 255.0).astype(np.uint8)
        imsave(os.path.join('data', f'heatmap_{axis}.png'), heatmap_rgb)

        ax[i, 1].imshow(heatmap_rgb)

        heatmap_alpha = np.where(heatmap_alpha > 0.1, heatmap_alpha ** 0.5, heatmap_alpha)
        image_alpha = 1 - heatmap_alpha
        combined = image_rgb * image_alpha[..., None] + heatmap_rgb * heatmap_alpha[..., None]

        combined = combined.astype(np.uint8)
        imsave(os.path.join('data', f'combined_{axis}.png'), combined)

        ax[i, 2].imshow(combined)

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Creates PNGs of an MRI and heatmap')

    parser.add_argument('-i', '--mri', required=False,
                        default=os.path.join('data', 'mri.nii.gz'),
                        help='Path to MRI file')
    parser.add_argument('-m', '--heatmap', required=False,
                        default=os.path.join('data', 'heatmap.nii.gz'),
                        help='Path to heatmap file')

    args = parser.parse_args()

    generate_mris(args.mri, args.heatmap)
