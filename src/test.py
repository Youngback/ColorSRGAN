import argparse
import time
import os

import cv2

import utils
from processing import Inference


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
		output = net.predict(img)
		toc = time.time()
		print(file_name, 'inference time(sec) =', toc - tic)

		# image write
		utils.image_write(os.path.join(save_path, file_name), output)


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--model', '-m', dest='model', default='./weightData/my_gen.hdf5', help='inference network model')
	parser.add_argument('--input', '-i', dest='input', default='./sample_images/', help='input images path')
	parser.add_argument('--output', '-o', dest='output', default='./sample_images/result/', help='output images path')
	args = parser.parse_args()

	test(args)
