import os
import cv2 as cv
import numpy as np
from skimage import color

from my_model import create_models

class inference:
    def __init__(self, weight_path, learning_rate=0.0005, momentum=0.5, init_size=64):
        '''
        init parameter and create network
        :param learning_rate: when training, learning rate, default 0.0005
        :param momentum: when training, momentum optimization hyper-parameter, default 0.5
        :param weight_path: pre-trained network weight path
        :param init_size: input image width, height size, default 64
        '''

        self.model_gen = create_models(
            shape_input_img=(init_size, init_size, 1),
            output_ch=3,
            lr=learning_rate,
            momentum=momentum)

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
        input_tensor = self.img_reshape(img, 1)

        # predict image
        output_tensor = self.model_gen.predict(input_tensor)[0]

        # extract network output
        result = np.clip(np.abs(color.lab2rgb(output_tensor)), 0, 1)

        return result
