import os
import cv2 as cv
import numpy as np
from skimage import color

from myModel import create_models

class inference:
    def __init__(self, weight_path, learning_rate=0.0005, momentum=0.5, lambda_1=1, lambda_2=100, init_size=64):
        '''
        init parameter and create network
        :param learning_rate: when training, learning rate, default 0.0005
        :param momentum: when training, momentum optimization hyper-parameter, default 0.5
        :param lambda_1: default 1
        :param lambda_2: default 100
        :param weight_path: pre-trained network weight path
        :param init_size: input image width, height size, default 64
        '''

        self.model_gen, self.model_dis, self.model_gan = create_models(
            input_shape_gen_64=(init_size, init_size, 1),
            input_shape_gen_128=(init_size*2, init_size*2, 1),
            input_shape_gen_256=(init_size*4, init_size*4, 1),
            input_shape_gen_Origin256=(init_size*4, init_size*4, 1),
            input_shape_dis=(init_size*4, init_size*4, 4),
            output_channels=3,
            lr=learning_rate,
            momentum=momentum,
            loss_weights=[lambda_1, lambda_2])

        if os.path.exists(weight_path):
            self.model_gen.load_weights(weight_path)


    def img_reshape(self, img, scale):
        '''
        image reshape
        :param img: original input image
        :param scale: output scale relative to input
        :return: reshaped output image
        '''
        temp = cv.resize(img, dsize=(img.shape[0]*scale, img.shape[1]*scale), interpolation=cv.INTER_CUBIC)
        output = temp[None, :, : , None]
        return output


    def predict(self, img):
        '''
        inference network
        :param img: original input image
        :return: output image
        '''
        img_1 = self.img_reshape(img, 1)
        img_2 = self.img_reshape(img, 2)
        img_3 = self.img_reshape(img, 4)

        output = self.model_gen.predict([img_1, img_2, img_3])[0]

        result = np.clip(np.abs(color.lab2rgb(output)), 0, 1)

        return result
