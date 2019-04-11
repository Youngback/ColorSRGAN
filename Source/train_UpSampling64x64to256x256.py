import os
import time
import glob
import numpy as np
from keras.utils import generic_utils
from myModel_64x64to256x256 import create_models
from TFNetDataset_UpSampling import dir_data_generator
import matplotlib.pyplot as plt

import scipy.misc
from scipy import ndimage, misc
import imageio
from skimage import color
from scipy.misc import imresize

EPOCHS = 500
BATCH_SIZE = 10
LEARNING_RATE = 0.0005
MOMENTUM = 0.5
LAMBDA1 = 1
LAMBDA2 = 100

INPUT_SHAPE_GEN64 = (64, 64, 1)
INPUT_SHAPE_GEN128 = (128, 128, 1)
INPUT_SHAPE_GEN256 = (256, 256, 1)
INPUT_SHAPE_GEN_Origin256 = (256, 256, 1)
INPUT_SHAPE_DIS = (256, 256, 4)

WEIGHTS_GEN = './weightData/weights_MyDataset_64x64to256x256_gen.hdf5'
WEIGHTS_DIS = './weightData/weights_MyDataset_64x64to256x256_dis.hdf5'
WEIGHTS_GAN = './weightData/weights_MyDataset_64x64to256x256_gan.hdf5'

# MODE 1: Train 
# MODE 2: Inference(Up-Sampling using Bicubic) 
# MODE 3: Inference(Up-Sampling using FSRCNN)
MODE = 1

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model_gen, model_dis, model_gan = create_models(
	input_shape_gen_64=INPUT_SHAPE_GEN64,
	input_shape_gen_128=INPUT_SHAPE_GEN128,
	input_shape_gen_256=INPUT_SHAPE_GEN256,
	input_shape_gen_Origin256=INPUT_SHAPE_GEN_Origin256,
	input_shape_dis=INPUT_SHAPE_DIS,
	output_channels=3,
	lr=LEARNING_RATE,
	momentum=MOMENTUM,
	loss_weights=[LAMBDA1, LAMBDA2])

if os.path.exists(WEIGHTS_GEN):
	model_gen.load_weights(WEIGHTS_GEN)

if os.path.exists(WEIGHTS_DIS):
	model_dis.load_weights(WEIGHTS_DIS)

if os.path.exists(WEIGHTS_GAN):
	model_gan.load_weights(WEIGHTS_GAN)


PATH = '../Datasets/MyDataset/train_IR64x64'
TOTAL_SIZE = len(np.array(glob.glob(PATH + '/*.jpg')))
TRAIN_SIZE = 1600 # 1884
TEST_SIZE = 140
data_gen = dir_data_generator(batch_size=BATCH_SIZE, data_range=(0, TRAIN_SIZE), outType='LAB', upType='bicubic')
test_data_gen = dir_data_generator(batch_size=BATCH_SIZE, data_range=(TRAIN_SIZE, TOTAL_SIZE), outType='LAB', upType='bicubic')

loss_gen = []
loss_ave_gen = []
loss_dis = []

## Training Unbalanced DCGAN
if MODE == 1:
	print("Start training...")

	for e in range(EPOCHS):
		toggle = True
		batch_counter = 1
		batch_total = TRAIN_SIZE // BATCH_SIZE
		progbar = generic_utils.Progbar(batch_total * BATCH_SIZE)
		start = time.time()
		dis_res = 0
		loss_gen = []

		while batch_counter < batch_total:

			batch_counter += 1

			data_lab, data_ir64, data_ir128, data_ir256, data_Origin_ir256 = next(data_gen)

			if batch_counter % 2 == 0:
				toggle = not toggle
				if toggle:
					x_dis = np.concatenate((model_gen.predict([data_ir64, data_ir128, data_ir256]), data_Origin_ir256), axis=3)
					y_dis = np.zeros((BATCH_SIZE, 1))
					y_dis = np.ones((BATCH_SIZE, 1)) * .1
				else:
					x_dis = np.concatenate((data_lab, data_Origin_ir256), axis=3)
					y_dis = np.ones((BATCH_SIZE, 1))
					y_dis = np.ones((BATCH_SIZE, 1)) * .9

				dis_res = model_dis.train_on_batch(x_dis, y_dis)

			model_dis.trainable = False
			y_gen = np.ones((BATCH_SIZE, 1))
			x_output = data_lab
			gan_res = model_gan.train_on_batch([data_ir64, data_ir128, data_ir256, data_Origin_ir256], [y_gen, x_output])
			model_dis.trainable = True

			progbar.add(BATCH_SIZE, 
			values=[("D loss", dis_res),
				("G total loss", gan_res[0]),
				("G loss", gan_res[1]),
				("G L1", gan_res[2]),
				("pacc", gan_res[5]),
				("acc", gan_res[6])])


			if batch_counter % 1000 == 0:
				print('')
				print('Saving weights...')
				model_gen.save_weights(WEIGHTS_GEN, overwrite=True)
				model_dis.save_weights(WEIGHTS_DIS, overwrite=True)
				model_gan.save_weights(WEIGHTS_GAN, overwrite=True)

		print('')
		data_test_lab, data_test_64, data_test_128, data_test_256, data_origin_256 = next(test_data_gen)
		ev = model_gan.evaluate([data_test_64, data_test_128, data_test_256, data_origin_256], [np.ones((data_test_64.shape[0], 1)), data_test_lab])
		ev = np.round(np.array(ev), 4)
		print('Epoch %s/%s, Time: %s' % (e + 1, EPOCHS, round(time.time() - start)))
		print('G total loss: %s - G loss: %s - G L1: %s: pacc: %s - acc: %s' % (ev[0], ev[1], ev[2], ev[5], ev[6]))
		print('')

		model_gen.save_weights(WEIGHTS_GEN, overwrite=True)
		model_dis.save_weights(WEIGHTS_DIS, overwrite=True)
		model_gan.save_weights(WEIGHTS_GAN, overwrite=True)


## Up-Sampling Input Image using 'Bicubic interpolation'
elif MODE == 2:
	print("Strat testing / Up-Sampling using Bicubic interpolation ...")

	IRPath = '../Datasets/MyDataset/test_IR64x64/'

	sumTime = 0

	for i in range(0, TEST_SIZE):
		# Image read
		fnameIR = np.array(sorted(glob.glob(IRPath + '*.jpg')))
		imgIR64 = imageio.imread(fnameIR[i])
		lab_ir64 = imgIR64[:, :, None]
		ir64 = lab_ir64[None, :, :, :]
		
		# Image prediction & time
		start_time = time.time()

		imgIR128 = imresize(imgIR64, size=[128, 128], interp='bicubic', mode=None)
		imgIR256 = imresize(imgIR64, size=[256, 256], interp='bicubic', mode=None)

		lab_ir128 = imgIR128[:, :, None]
		lab_ir256 = imgIR256[:, :, None]
			
		ir128 = lab_ir128[None, :, :, :]
		ir256 = lab_ir256[None, :, :, :]
	
		lab_pred = np.array(model_gen.predict([ir64, ir128, ir256]))[0]

		end_time = time.time()
	
		sumTime = sumTime + (end_time - start_time)
	
		lab_pred = lab_pred.astype(np.float64)
		imgPred = np.clip(np.abs(color.lab2rgb(lab_pred)), 0, 1)

		scipy.misc.toimage(imgPred).save('../Result/SuperResol_TrainResult/MyDataset_64x64to256x256_BB/%.3d.jpg' % i)

	print('Image Colorization Averaging time =', sumTime/TEST_SIZE)
		

## Up-Sampling Input Image using 'Super-Resolution using FSRCNN'
elif MODE == 3:
	print("Strat testing / Up-Sampling using FSRCNN ...")
	
	IRPath64 = '../Datasets/MyDataset/test_IR64x64/'
	IRPath128 = '../Datasets/FSRCNN/test_IR64x64to128x128/'
	IRPath256 = '../Datasets/FSRCNN/test_IR64x64to256x256/'
	
	sumTime = 0

	for i in range(0, TEST_SIZE):
		# Image read
		fnameIR64 = np.array(sorted(glob.glob(IRPath64 + '*.jpg')))
		imgIR64 = imageio.imread(fnameIR64[i])
		lab_ir64 = imgIR64[:, :, None]
		ir64 = lab_ir64[None, :, :, :]

		fnameIR128 = np.array(sorted(glob.glob(IRPath128 + '*.jpg')))
		imgIR128 = imageio.imread(fnameIR128[i])
		filesIR128 = imgIR128[:, :, 0]
		ir128 = filesIR128[None, :, :, None]
		
		fnameIR256 = np.array(sorted(glob.glob(IRPath256 + '*.jpg')))
		imgIR256 = imageio.imread(fnameIR256[i])
		filesIR256 = imgIR256[:, :, 0]
		ir256 = filesIR256[None, :, :, None]

		# Image prediction & time
		start_time = time.time()

		lab_pred = np.array(model_gen.predict([ir64, ir128, ir256]))[0]

		end_time = time.time()
	
		sumTime = sumTime + (end_time - start_time)
	
		lab_pred = lab_pred.astype(np.float64)
		imgPred = np.clip(np.abs(color.lab2rgb(lab_pred)), 0, 1)

        scipy.misc.toimage(imgPred).save('../Result/SuperResol_TrainResult/MyDataset_64x64to256x256_BF/%.3d.jpg' % i)
			
	print('Image Colorization Averaging time =', sumTime/TEST_SIZE)
