
import glob
import numpy as np
import cv2
from skimage import color

import config as cfg

def dir_data_generator(batch_size, data_range=(0, 0), outType='LAB'):

    names_gt = np.array(glob.glob(cfg.GT_PATH + '/*.jpg'))
    names_dataset = np.array(glob.glob(cfg.TRAIN_PATH + '/*.jpg'))

    if data_range != (0, 0):
        names_gt = names_gt[data_range[0]:data_range[1]]
        names_dataset = names_dataset[data_range[0]:data_range[1]]

    batch_count = len(names_gt) // batch_size

    while True:
        for i in range(0, batch_count):
            files_gt = np.array([cv2.imread(f) for f in names_gt[i * batch_size:i * batch_size + batch_size]])
            files_dataset = np.array([cv2.imread(f) for f in names_dataset[i * batch_size:i * batch_size + batch_size]])

            if outType == 'RGB':
                yield files_gt, files_dataset[:, :, :, np.newaxis], color.rgb2gray(files_gt)

            elif outType == 'LAB':
                yield color.rgb2lab(files_gt), files_dataset[:, :, :, np.newaxis], color.rgb2gray(files_gt)
