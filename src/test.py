import argparse
import os
import time

import cv2
import numpy as np
from skimage import color

import utils
from model import create_model_gen


class Inference:

    def __init__(self, weight_path, init_size=(480, 640, 1)):

        self.init_size = init_size

        self.model_gen = create_model_gen(
            input_shape=self.init_size,
            output_channels=3)

        if os.path.exists(weight_path):
            self.model_gen.load_weights(weight_path)

    def process(self, img):

		# input image reshape
        input_tensor = utils.image_reshape(img, self.init_size)

		# predict image
        output_tensor = self.model_gen.predict(input_tensor)[0]

		# change LAB to RGB color space & stretching (0, 1) to (0, 255)
        result = np.clip(np.abs(color.lab2rgb(output_tensor)), 0, 1) * 255

        return result


def test(args):

	# init parameter
	weight_path = args.model
	path = args.input
	save_path = args.output

	utils.create_dir(save_path)

	# create network
	net = Inference(weight_path=weight_path)

	# read image file names
	file_names = utils.get_image_file_name(path)

	for file_name in file_names:

		# read image
		img = cv2.imread(path + file_name, cv2.IMREAD_GRAYSCALE)
		img = img.astype('float')

		# inference
		tic = time.time()
		output = net.process(img)
		toc = time.time()
		print(file_name, 'inference time(sec) =', toc - tic)

		# image write
		utils.image_write(os.path.join(save_path, file_name), output)


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--model', '-M', dest='model', default='./weightData/my_gen.hdf5', help='inference network model')
	parser.add_argument('--input', '-I', dest='input', default='./sample_images/', help='input images path')
	parser.add_argument('--output', '-O', dest='output', default='./sample_images/result/', help='output images path')
	args = parser.parse_args()

	test(args)
