import argparse
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import colormaps
from skimage.io import imsave


def create_segmentation(input: str):
    mask = nib.load(input).get_fdata()
    mask = mask[:,128]

    values, counts = np.unique(mask, return_counts=True)
    print(np.unique(values))
    counts = {values[i]: counts[i] for i in range(len(values))}
    values = sorted(values, key=lambda x: counts[x], reverse=True)
    cmap = colormaps['tab20c']
    rgb = np.zeros(mask.shape + (3,))

    for i in range(len(mask)):
        for j in range(len(mask[i])):
            #rgb[i,j] = cmap(values.index(mask[i,j]) / len(values))[:3]

            if mask[i,j] in [17, 53]:
                rgb[i,j] = [255, 0, 0]


    rgb[mask == 0] = [0, 0, 0]
    rgb = rgb * 255
    rgb = rgb.astype(np.uint8)
    rgb = np.rot90(rgb, 3)

    plt.imshow(rgb)
    plt.show()

    imsave('data/segmentation.png', rgb)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Creates a segmentation png')

    parser.add_argument('--input', required=False,
                        default=os.path.join('/Applications', 'freesurfer',
                                             '7.3.1', 'subjects', 'bert',
                                             'mri', 'aseg.mgz'),
                        help='Path to input segmentation mask')

    args = parser.parse_args()

    create_segmentation(args.input)
