
import os
import glob
import numpy as np

# TRAIN PARAMETER
EPOCHS = 500
BATCH_SIZE = 10
LEARNING_RATE = 0.0005
MOMENTUM = 0.9
LAMBDA1 = 1
LAMBDA2 = 100
INPUT_SHAPE_GEN = (64, 64, 1)
INPUT_SHAPE_ORIGIN = (256, 256, 1)
INPUT_SHAPE_DIS = (256, 256, 4)

# WEIGHTS FILE
WEIGHTS_GEN = './weightData/my_gen.hdf5'
WEIGHTS_DIS = './weightData/my_dis.hdf5'
WEIGHTS_GAN = './weightData/my_gan.hdf5'

# DATASET PARAMETER
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
TRAIN_PATH = '../Datasets/MyDataSet/train_IR64x64'
GT_PATH = '../Datasets/MyDataSet/train_RGB256x256'
TOTAL_SIZE = len(np.array(glob.glob(TRAIN_PATH + '/*.jpg')))
TRAIN_SIZE = 1600
TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
