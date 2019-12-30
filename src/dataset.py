import cv2
import numpy as np
from skimage import color

import config as cfg


def dir_data_generator(batch_size, data_range=(0, 0), outType='LAB', input_size=(64, 64)):

    names_gt = np.array(cfg.file_names)

    if data_range != (0, 0):
        names_gt = names_gt[data_range[0]:data_range[1]]

    batch_count = len(names_gt) // batch_size

    while True:
        for i in range(0, batch_count):
            images_gt = np.array([cv2.imread(f) for f in names_gt[i * batch_size:i * batch_size + batch_size]])
            images_dataset = [cv2.resize(img, input_size, interpolation=cv2.INTER_CUBIC) for img in images_gt]

            if outType == 'RGB':
                yield images_gt, images_dataset[:, :, :, np.newaxis], color.rgb2gray(images_gt)

            elif outType == 'LAB':
                yield color.rgb2lab(images_gt), images_dataset[:, :, :, np.newaxis], color.rgb2gray(images_gt)
