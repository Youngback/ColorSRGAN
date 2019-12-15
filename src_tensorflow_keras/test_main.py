import argparse
import time

import cv2

import utils
from processing import inference


def predict(args):

	# init parameter
	weight_path = args.model
	path = args.input
	save_path = args.output

	utils.createDir(save_path)

	# create network
	net = inference(weight_path=weight_path)

	# read image file names
	file_names = utils.getImageFileName(path)

	output_images = []

	for i in range(len(file_names)):

		# read image
		img = cv2.imread(path + file_names[i], cv2.IMREAD_GRAYSCALE)
		img = img.astype('float')

		# inference
		tic = time.time()
		output = net.predict(img)
		toc = time.time()
		print(file_names[i], 'inference time(sec) =', toc - tic)

		output_images.append(output)

	# image write
	utils.imageWrite(save_path, file_names, output_images)

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--model', '-m', dest='model', default='./weightData/my_gen.hdf5', help='inference network model')
	parser.add_argument('--input', '-i', dest='input', default='./sample_images/', help='input images path')
	parser.add_argument('--output', '-o', dest='output', default='./sample_images/result/', help='output images path')
	args = parser.parse_args()

	predict(args)
