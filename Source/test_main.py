import os
import time
import glob
import numpy as np
import scipy.misc
import imageio

from processing import inference

# init parameter
learning_rate = 0.0005
momentum = 0.5
lambda_1 = 1
lambda_2 = 100
weight_path = './weightData/weights_MyDataset_64x64to256x256_gen.hdf5'

# image path
path = '../Datasets/MyDataset/test_IR64x64/'
save_path = '../Result/'

# create network
net = inference(learning_rate, momentum, lambda_1, lambda_2, weight_path)

# read image directory
fname = np.array(sorted(glob.glob(path + '*.jpg')))


for i in range(len(fname)):

	# read image
	img = imageio.imread(fname[i])

	# inference
	output = net.predict(img)

	# write image
	scipy.misc.toimage(output).save(save_path + fname[i])

