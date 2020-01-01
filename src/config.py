
import os

from utils import create_dir

# TRAIN PARAMETER
EPOCHS = 2
BATCH_SIZE = 8
LEARNING_RATE = 0.0005
MOMENTUM = 0.9
LAMBDA1 = 1
LAMBDA2 = 100
INPUT_SHAPE_GEN = (64, 64, 1)
INPUT_SHAPE_ORIGIN = (256, 256, 1)
INPUT_SHAPE_DIS = (256, 256, 4)

# WEIGHTS FILE
WEIGHTS_PATH = './01_GRAY2LAB/'
create_dir(WEIGHTS_PATH)
WEIGHTS_GEN = os.path.join(WEIGHTS_PATH, 'gen.hdf5')
WEIGHTS_DIS = os.path.join(WEIGHTS_PATH, 'dis.hdf5')
WEIGHTS_GAN = os.path.join(WEIGHTS_PATH, 'gan.hdf5')

# DATASET
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

GT_PATH = 'K:/places365_standard/'

f = open(os.path.join(GT_PATH, 'train.txt'), 'r')
lines = f.readlines()
f.close()

file_names = []
for line in lines:
    line = os.path.join(GT_PATH, line[:-1])
    file_names.append(line)

TOTAL_SIZE = len(lines)
TRAIN_SIZE = TOTAL_SIZE // 10
TEST_SIZE = 100
