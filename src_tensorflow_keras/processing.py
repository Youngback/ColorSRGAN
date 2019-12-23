import os

import numpy as np
from skimage import color

import utils
from model import create_models


class Inference:

    def __init__(self, weight_path, learning_rate=0.0005, momentum=0.5, init_size=(640, 480)):

        self.init_size = init_size

        self.model_gen = create_models(
            shape_input_img=(self.init_size[1], self.init_size[0], 1),
            shape_output_img=3,
            lr=learning_rate,
            momentum=momentum)

        if os.path.exists(weight_path):
            self.model_gen.load_weights(weight_path)


    def predict(self, img):

        input_tensor = utils.image_reshape(img, self.init_size)

        # predict image
        output_tensor = self.model_gen.predict(input_tensor)[0]

        # extract network output
        result = np.clip(np.abs(color.lab2rgb(output_tensor)), 0, 1) * 255

        return result
