import os
import time
import scipy.misc
import cv2 as cv
import numpy as np

from processing import inference

# init parameter
weight_path = './weightData/my_gen.hdf5'

# image path
path = './sample_images/'
save_path = path + 'result/'

if not os.path.exists(save_path):
	os.mkdir(save_path)

# create network
net = inference(weight_path=weight_path)

# read image directory
included_extensions = ['jpg', 'jpeg', 'bmp', 'png', 'gif']

file_names = [fn for fn in os.listdir(path)
              if any(fn.endswith(ext) for ext in included_extensions)]

for i in range(len(file_names)):

	# read image
	img = cv.imread(path + file_names[i], cv.IMREAD_GRAYSCALE)
	img = img.astype('float')

	# inference
	tic = time.time()
	output = net.predict(img)
	toc = time.time()
	print(file_names[i], 'inference time =', toc - tic)

	# write image
	scipy.misc.toimage(output).save(save_path + file_names[i])
