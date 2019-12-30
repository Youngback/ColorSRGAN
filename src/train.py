import os
import time

import numpy as np
from tensorflow.keras.utils import Progbar

import config as cfg
from dataset import dir_data_generator
from model import create_models

print('[*] Build ColorSRGAN model ...')
model_gen, model_dis, model_gan = create_models(
	input_shape_gen=cfg.INPUT_SHAPE_GEN,
	input_shape_dis=cfg.INPUT_SHAPE_DIS,
	input_shape_origin=cfg.INPUT_SHAPE_ORIGIN,
	output_channels_gen=3,
	lr=cfg.LEARNING_RATE,
	momentum=cfg.MOMENTUM,
	loss_weights=[cfg.LAMBDA1, cfg.LAMBDA2])

print('[*] Load pre-trained weight ...')
if os.path.exists(cfg.WEIGHTS_GEN):
	model_gen.load_weights(cfg.WEIGHTS_GEN)

if os.path.exists(cfg.WEIGHTS_DIS):
	model_dis.load_weights(cfg.WEIGHTS_DIS)

if os.path.exists(cfg.WEIGHTS_GAN):
	model_gan.load_weights(cfg.WEIGHTS_GAN)

print('[*] Load dataset ...')
data_gen = dir_data_generator(batch_size=cfg.BATCH_SIZE, data_range=(0, cfg.TRAIN_SIZE), outType='LAB')
test_data_gen = dir_data_generator(batch_size=cfg.BATCH_SIZE, data_range=(cfg.TRAIN_SIZE, cfg.TOTAL_SIZE), outType='LAB')

loss_gen = []
loss_ave_gen = []
loss_dis = []

print('[*] Start train ...')
for e in range(cfg.EPOCHS):

	toggle = True
	batch_counter = 1
	batch_total = cfg.TRAIN_SIZE // cfg.BATCH_SIZE
	progbar = Progbar(batch_total * cfg.BATCH_SIZE)
	start = time.time()
	dis_res = 0
	loss_gen = []

	while batch_counter < batch_total:

		batch_counter += 1

		data_gt, data, data_gray = next(data_gen)

		if batch_counter % 2 == 0:
			toggle = not toggle

			if toggle:
				x_dis = np.concatenate((model_gen.predict(data), data_gray), axis=3)
				y_dis = np.ones((cfg.BATCH_SIZE, 1)) * .1
			else:
				x_dis = np.concatenate((data_gt, data_gray), axis=3)
				y_dis = np.ones((cfg.BATCH_SIZE, 1)) * .9

			dis_res = model_dis.train_on_batch(x_dis, y_dis)

		model_dis.trainable = False
		y_gen = np.ones((cfg.BATCH_SIZE, 1))
		x_output = data_gt
		gan_res = model_gan.train_on_batch([data, data_gt], [y_gen, x_output])
		model_dis.trainable = True

		progbar.add(
			cfg.BATCH_SIZE,
			values=[
				("D loss", dis_res),
				("G total loss", gan_res[0]),
				("G loss", gan_res[1]),
				("G L1", gan_res[2]),
				("pacc", gan_res[5]),
				("acc", gan_res[6])
			]
		)

		if batch_counter % 1000 == 0:
			print('')
			print('Saving weights ...')
			model_gen.save_weights(cfg.WEIGHTS_GEN, overwrite=True)
			model_dis.save_weights(cfg.WEIGHTS_DIS, overwrite=True)
			model_gan.save_weights(cfg.WEIGHTS_GAN, overwrite=True)

	print('')
	data_test_gt, data_test, data_test_gray = next(test_data_gen)
	ev = model_gan.evaluate([data_test, data_test_gt], [np.ones((data_test.shape[0], 1)), data_test_gt])
	ev = np.round(np.array(ev), 4)
	print('Epoch %s/%s, Time: %s' % (e + 1, cfg.EPOCHS, round(time.time() - start)))
	print('G total loss: %s - G loss: %s - G L1: %s: pacc: %s - acc: %s' % (ev[0], ev[1], ev[2], ev[5], ev[6]))
	print('')

	model_gen.save_weights(cfg.WEIGHTS_GEN, overwrite=True)
	model_dis.save_weights(cfg.WEIGHTS_DIS, overwrite=True)
	model_gan.save_weights(cfg.WEIGHTS_GAN, overwrite=True)

print('[!] Successful training.')