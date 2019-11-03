import os
import glob
import numpy as np
from scipy.misc import imread
from skimage import color
from scipy.misc import imresize


def dir_data_generator(batch_size, data_range=(0, 0), outType='YUV', upType='bicubic'):
    
    ## Up-Sampling Type = 'Bicubic'
    if upType == 'bicubic':
        PATH = '../Datasets/MyDataSet/train_RGB256x256'
        namesRGB = np.array(glob.glob(PATH + '/*.jpg'))

        PATH = '../Datasets/MyDataSet/train_IR64x64'
        namesIR64 = np.array(glob.glob(PATH + '/*.jpg'))

        PATH = '../Datasets/MyDataSet/train_IR256x256'
        namesOriginIR = np.array(glob.glob(PATH + '/*.jpg'))

        if data_range != (0, 0):
            namesRGB = namesRGB[data_range[0]:data_range[1]]
            namesIR64 = namesIR64[data_range[0]:data_range[1]]
            namesOriginIR = namesOriginIR[data_range[0]:data_range[1]]

        batch_count = len(namesRGB) // batch_size

        while True:
            for i in range(0, batch_count):
                filesRGB = np.array([imread(f) for f in namesRGB[i * batch_size:i * batch_size + batch_size]])            
                filesIR64 = np.array([imread(f) for f in namesIR64[i * batch_size:i * batch_size + batch_size]])
                filesOriginIR = np.array([imread(f) for f in namesOriginIR[i * batch_size:i * batch_size + batch_size]])

                filesIR128 = np.array([imresize(filesIR64[f, :, :], size=[64, 128], interp='bicubic', mode=None) for f in range(0, batch_size)])
                filesIR256 = np.array([imresize(filesIR64[f, :, :], size=[128, 256], interp='bicubic', mode=None) for f in range(0, batch_size)])
            
                if outType == 'YUV':
                    yield color.rgb2yuv(filesRGB), filesRGB

                elif outType == 'LAB':
                    yield color.rgb2lab(filesRGB), filesIR64[:, :, :, None], filesIR128[:, :, :, None], filesIR256[:, :, :, None], filesOriginIR[:, :, :, None]


    ## Up-Sampling Type = 'FSRCNN'
    elif upType == 'FSRCNN':
        PATH = '../Datasets/MyDataSet/train_RGB256x256'
        namesRGB = np.array(glob.glob(PATH + '/*.jpg'))

        PATH = '../Datasets/MyDataSet/train_IR64x64'
        namesIR64 = np.array(glob.glob(PATH + '/*.jpg'))

        PATH = '../Datasets/FSRCNN/train_IR64x64to128x128'
        namesIR128 = np.array(glob.glob(PATH + '/*.jpg'))

        PATH = '../datasets/FSRCNN/train_IR64x64to256x256'
        namesIR256 = np.array(glob.glob(PATH + '/*.jpg'))

        PATH = '../datasets/MyDataSet/train_IR256x256'
        namesOriginIR = np.array(glob.glob(PATH + '/*.jpg'))

        if data_range != (0, 0):
            namesRGB = namesRGB[data_range[0]:data_range[1]]
            namesIR64 = namesIR64[data_range[0]:data_range[1]]
            namesIR128 = namesIR128[data_range[0]:data_range[1]]
            namesIR256 = namesIR256[data_range[0]:data_range[1]]
            namesOriginIR = namesOriginIR[data_range[0]:data_range[1]]

        batch_count = len(namesRGB) // batch_size

        while True:
            for i in range(0, batch_count):
                filesRGB = np.array([imread(f) for f in namesRGB[i * batch_size:i * batch_size + batch_size]])
                filesIR64 = np.array([imread(f) for f in namesIR64[i * batch_size:i * batch_size + batch_size]])
                filesIR128 = np.array([imread(f) for f in namesIR64[i * batch_size:i * batch_size + batch_size]])
                filesIR256 = np.array([imread(f) for f in namesIR64[i * batch_size:i * batch_size + batch_size]])
                filesOriginIR = np.array([imread(f) for f in namesOriginIR[i * batch_size:i * batch_size + batch_size]])
            
                if outType == 'YUV':
                    yield color.rgb2yuv(filesRGB), filesRGB

                elif outType == 'LAB':
                    yield color.rgb2lab(filesRGB), filesIR64[:, :, :, None], filesIR128[:, :, :, None], filesIR256[:, :, :, None], filesOriginIR[:, :, :, None]
