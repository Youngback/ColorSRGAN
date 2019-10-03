import os
import cv2 as cv
import numpy as np
from skimage import color

from myModel import create_models

class inference:
    def __init__(self, learning_rate, momentum, lambda_1, lambda_2, weight_path):
        self.init_size = 64

        self.model_gen, self.model_dis, self.model_gan = create_models(
            input_shape_gen_64=(self.init_size, self.init_size, 1),
            input_shape_gen_128=(self.init_size*2, self.init_size*2, 1),
            input_shape_gen_256=(self.init_size*4, self.init_size*4, 1),
            input_shape_gen_Origin256=(self.init_size*4, self.init_size*4, 1),
            input_shape_dis=(self.init_size*4, self.init_size*4, 4),
            output_channels=3,
            lr=learning_rate,
            momentum=momentum,
            loss_weights=[lambda_1, lambda_2])

        if os.path.exists(weight_path):
            self.model_gen.load_weights(weight_path)

    def img_reshape(self, img, scale):
        temp = cv.imresize(img, scale, interp='bicubic', mode=None)
        temp = temp[:, : , None]
        output = temp[None, :, :, :]
        return output

    def predict(self, img):
        img_1 = self.img_reshape(img, 1.0)
        img_2 = self.img_reshape(img, 2.0)
        img_3 = self.img_reshape(img, 4.0)

        lab_pred = np.array(self.model_gen.predict([img_1, img_2, img_3]))[0]

        lab_pred = lab_pred.astype(np.float64)
        imgPred = np.clip(np.abs(color.lab2rgb(lab_pred)), 0, 1)

        return imgPred
